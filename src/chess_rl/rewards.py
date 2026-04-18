"""
rewards.py — Three reward functions for chess move quality assessment.

R1  reward_format(response)          -> float   syntax/format check
R2  reward_legality(response, board) -> float   legality via python-chess
R3  reward_strategic(response, board, analyst) -> float   centipawn diff (DIFF-ONLY, no +C)

compute_reward chains all three with early-exit on failure.

Module-level analyst singleton (lazy-init, atexit close) is exposed via get_analyst().
Callers (S3, S9) that already hold a chess.engine.SimpleEngine can pass it directly to
reward_strategic / compute_reward; get_analyst() is provided for callers that need the
module-managed singleton.
"""

import atexit
import os
import re
import sys
from pathlib import Path
from typing import Optional

import chess
import chess.engine

# ---------------------------------------------------------------------------
# Module-level analyst singleton
# ---------------------------------------------------------------------------

_analyst: Optional[chess.engine.SimpleEngine] = None
_atexit_registered: bool = False


def _find_stockfish_binary() -> Path:
    """
    Locate Stockfish binary.
    Prefer ./stockfish/stockfish.exe (Windows) or ./stockfish/stockfish (Linux/Mac).
    Mirrors the logic in stockfish.py so this module works standalone.
    """
    base = Path("./stockfish")
    if sys.platform == "win32" or os.name == "nt":
        candidate = base / "stockfish.exe"
    else:
        candidate = base / "stockfish"

    if candidate.exists():
        return candidate

    # Fallback: any stockfish file in the dir
    if base.exists():
        for p in sorted(base.iterdir()):
            if p.name.lower().startswith("stockfish") and p.is_file():
                if sys.platform == "win32" and p.suffix.lower() == ".exe":
                    return p
                elif sys.platform != "win32" and not p.suffix:
                    return p

    raise FileNotFoundError(
        f"Stockfish binary not found in {base.resolve()}. "
        "Run setup_stockfish.sh first."
    )


def _close_analyst() -> None:
    """atexit handler: quit the module-level analyst engine."""
    global _analyst
    if _analyst is not None:
        try:
            _analyst.quit()
        except Exception:
            pass
        _analyst = None


def _get_analyst() -> chess.engine.SimpleEngine:
    """
    Lazy-initialise the module-level Stockfish analyst engine.
    Registers an atexit handler on first call.
    """
    global _analyst, _atexit_registered
    if _analyst is None:
        binary = _find_stockfish_binary()
        _analyst = chess.engine.SimpleEngine.popen_uci(str(binary))
        if not _atexit_registered:
            atexit.register(_close_analyst)
            _atexit_registered = True
    return _analyst


def get_analyst() -> chess.engine.SimpleEngine:
    """
    Public accessor for the module-level Stockfish analyst singleton.
    Used by S3, S9, and any other caller that needs a ready engine without
    constructing a full StockfishManager.
    """
    return _get_analyst()


# ---------------------------------------------------------------------------
# R1 — Format / syntax reward
# ---------------------------------------------------------------------------

# SAN regex: optionally piece letter, optionally source file/rank, optional capture,
# destination square, optional promotion, optional check/mate marker.
# Also handles pawn promotions like a8=Q+, b1=R#, exd8=Q, etc.
_SAN_PATTERN = re.compile(
    r'[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](=[QRBN])?[+#]?'
)
_MOVE_TAG = re.compile(r'<move>(.*?)</move>', re.IGNORECASE)


def extract_san(response: str) -> str:
    """Pull move from <move>...</move> tag; fall back to full stripped response."""
    m = _MOVE_TAG.search(response)
    if m:
        return m.group(1).strip().rstrip(".")
    return response.strip().rstrip(".")


def reward_format(response: str) -> float:
    """
    R1: Check that the response contains a <move>SAN</move> tag with valid SAN.

    Returns:
        +0.1  if <move> tag present and content is valid SAN or castling.
        -1.0  otherwise (no tag, empty tag, non-SAN content).
    """
    if not _MOVE_TAG.search(response):
        return -1.0

    san = extract_san(response)

    if san in ("O-O", "O-O-O"):
        return 0.1

    if re.fullmatch(r'[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](=[QRBN])?[+#]?', san):
        return 0.1

    return -1.0


# ---------------------------------------------------------------------------
# R2 — Legality reward
# ---------------------------------------------------------------------------

def reward_legality(response: str, board: chess.Board) -> float:
    """
    R2: Check that the response is a legal move in the given position.

    Returns:
        +0.5  if the move parses and is in board.legal_moves.
        -5.0  if parsing fails or the move is illegal.
    """
    response = extract_san(response).rstrip(".")

    try:
        move = board.parse_san(response)
        if move in board.legal_moves:
            return 0.5
        return -5.0
    except (
        chess.InvalidMoveError,
        chess.IllegalMoveError,
        chess.AmbiguousMoveError,
        ValueError,
    ):
        return -5.0


# ---------------------------------------------------------------------------
# R3 — Strategic reward (DIFF-ONLY, no additive constant)
# ---------------------------------------------------------------------------

def reward_strategic(
    response: str,
    board: chess.Board,
    analyst: chess.engine.SimpleEngine,
) -> float:
    """
    R3: Pure centipawn-difference reward (no additive constant).

    Algorithm:
      1. Evaluate the CURRENT board with Stockfish; score relative to the side to
         move → best_score (centipawns, mover's POV).
      2. Parse the LLM's move and push it onto a COPY of the board.
      3. Evaluate the post-move board; the perspective has flipped to the opponent,
         so we negate → model_score (back to mover's POV).
      4. delta = (model_score - best_score) / 100.0
         Clamped to [-10.0, 0.0]: best move ≈ 0, worse moves go negative.

    Returns:
        A float in [-10.0, 0.0].
        Returns 0.0 without penalty if the move cannot be parsed (R2 already
        penalises illegality; compute_reward gates R3 on R2 success anyway).
    """
    response = extract_san(response).rstrip(".")

    try:
        # Step 1: evaluate current position (mover's POV)
        info_pre = analyst.analyse(board, chess.engine.Limit(time=0.1))
        best_score = info_pre["score"].relative.score(mate_score=10000)
        assert best_score is not None

        # Step 2: push LLM move onto a copy
        board_copy = board.copy()
        move = board_copy.parse_san(response)
        board_copy.push(move)

        # Step 3: evaluate post-move position (opponent's POV); negate → mover's POV
        info_post = analyst.analyse(board_copy, chess.engine.Limit(time=0.1))
        opponent_score = info_post["score"].relative.score(mate_score=10000)
        assert opponent_score is not None
        model_score = -opponent_score

        # Step 4: diff, normalise, clamp
        delta = (model_score - best_score) / 100.0
        return max(-10.0, min(0.0, delta))

    except Exception:
        # Parse failure, engine error, etc. — no double-penalty (R2 already fired)
        return 0.0


# ---------------------------------------------------------------------------
# Combined reward
# ---------------------------------------------------------------------------

def compute_reward(
    response: str,
    board: chess.Board,
    analyst: chess.engine.SimpleEngine,
) -> float:
    """
    Chain R1 → R2 → R3 with early exit on failure.

    Early-exit semantics:
      - Bad format  → return r1          (skip R2, R3)
      - Illegal     → return r1 + r2     (skip R3)
      - Legal       → return r1 + r2 + r3

    Typical good-move range: [0.1 + 0.5 + (-small)] ≈ [0.5, 0.7]
    Typical illegal range:   [0.1 + (-5.0)]          = -4.9
    Bad format:              -1.0
    """
    r1 = reward_format(response)
    if r1 < 0:
        return r1

    r2 = reward_legality(response, board)
    if r2 < 0:
        return r1 + r2

    r3 = reward_strategic(response, board, analyst)
    return r1 + r2 + r3
