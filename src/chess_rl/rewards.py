"""
rewards.py — Four reward functions for chess move quality assessment.

R1  reward_format(response)                         -> float          think+move tag check
R2  reward_legality(response, board)                -> float          legality via python-chess
R3  reward_strategic(response, board, analyst)      -> Optional[float] centipawn delta (None=unparseable)
R4  reward_optimal(delta)                           -> float          +1.0 bonus when delta > -0.1

compute_reward evaluates all four with NO early exits. Total = R1 + R2 + (R3 or 0.0) + R4.

Module-level analyst singleton (lazy-init, atexit close) is exposed via get_analyst().
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
    """Lazy-initialise the module-level Stockfish analyst engine."""
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
    Used by train.py and any other caller that needs a ready engine.
    """
    return _get_analyst()


# ---------------------------------------------------------------------------
# Shared regex patterns
# ---------------------------------------------------------------------------

_MOVE_TAG    = re.compile(r'<move>(.*?)</move>', re.IGNORECASE)
_THINK_START = re.compile(r'<think>',            re.IGNORECASE)
_THINK_END   = re.compile(r'</think>',           re.IGNORECASE)

_SAN_PATTERN = re.compile(
    r'[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](=[QRBN])?[+#]?'
)


def extract_san(response: str) -> str:
    """Pull move from <move>...</move> tag; fall back to full stripped response."""
    m = _MOVE_TAG.search(response)
    if m:
        return m.group(1).strip().rstrip(".")
    return response.strip().rstrip(".")


# ---------------------------------------------------------------------------
# R1 — Format reward
# ---------------------------------------------------------------------------

def reward_format(response: str) -> float:
    """
    R1: Validate <think>...</think><move>SAN</move> structure.

    All four conditions must pass:
      1. <think> is present
      2. </think> is present
      3. <move>SAN</move> is present with valid SAN content
      4. Tag order: <think> < </think> < <move>

    Returns:
        +0.2  all conditions satisfied
        -1.0  any condition fails
    """
    think_start_m = _THINK_START.search(response)
    think_end_m   = _THINK_END.search(response)
    move_m        = _MOVE_TAG.search(response)

    if not think_start_m or not think_end_m or not move_m:
        return -1.0

    # Tag order: <think> must appear before </think>, which must appear before <move>
    if not (think_start_m.start() < think_end_m.start() < move_m.start()):
        return -1.0

    san = move_m.group(1).strip().rstrip(".")
    if san in ("O-O", "O-O-O"):
        return 0.2

    if re.fullmatch(r'[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](=[QRBN])?[+#]?', san):
        return 0.2

    return -1.0


# ---------------------------------------------------------------------------
# R2 — Legality reward
# ---------------------------------------------------------------------------

def reward_legality(response: str, board: chess.Board) -> float:
    """
    R2: Check that the extracted move is legal in the given position.

    Returns:
        +0.5  move parses and is in board.legal_moves
        -5.0  parsing fails or move is illegal
    """
    san = extract_san(response).rstrip(".")

    try:
        move = board.parse_san(san)
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
# R3 — Strategic reward
# ---------------------------------------------------------------------------

def reward_strategic(
    response: str,
    board: chess.Board,
    analyst: chess.engine.SimpleEngine,
) -> Optional[float]:
    """
    R3: Centipawn-difference reward relative to Stockfish best.

    Algorithm:
      1. Parse the LLM's move — return None immediately if it fails.
      2. Evaluate current board → best_score (mover POV, centipawns).
      3. Push LLM move onto a copy; evaluate post-move → negate → model_score.
      4. delta = (model_score - best_score) / 100, clamped to [-10.0, 0.0].

    Returns:
        float in [-10.0, 0.0]  move parses and Stockfish evaluates successfully.
        None                   move cannot be parsed (R2 already penalises this;
                               returning None prevents reward_optimal from firing spuriously).
    """
    san = extract_san(response).rstrip(".")

    # Validate parse before touching Stockfish
    try:
        board_copy = board.copy()
        move = board_copy.parse_san(san)
    except Exception:
        return None

    try:
        # Step 1: evaluate current position (mover's POV)
        info_pre = analyst.analyse(board, chess.engine.Limit(time=0.1))
        best_score = info_pre["score"].relative.score(mate_score=10000)
        assert best_score is not None

        # Step 2: push LLM move; evaluate post-move (opponent's POV); negate → mover's POV
        board_copy.push(move)
        info_post = analyst.analyse(board_copy, chess.engine.Limit(time=0.1))
        opponent_score = info_post["score"].relative.score(mate_score=10000)
        assert opponent_score is not None
        model_score = -opponent_score

        # Step 3: diff, normalise, clamp
        delta = (model_score - best_score) / 100.0
        return max(-10.0, min(0.0, delta))

    except Exception:
        return None


# ---------------------------------------------------------------------------
# R4 — Optimal bonus
# ---------------------------------------------------------------------------

def reward_optimal(delta: Optional[float]) -> float:
    """
    R4: +1.0 bonus when the move is within 10 centipawns of Stockfish best.

    Args:
        delta: Value from reward_strategic. None means the move was unparseable.

    Returns:
        +1.0  delta is not None and delta > -0.1
         0.0  otherwise (suboptimal or unparseable)
    """
    if delta is None:
        return 0.0
    return 1.0 if delta > -0.1 else 0.0


# ---------------------------------------------------------------------------
# Combined reward — no early exits
# ---------------------------------------------------------------------------

def compute_reward(
    response: str,
    board: chess.Board,
    analyst: chess.engine.SimpleEngine,
) -> float:
    """
    Evaluate all four reward components unconditionally. Total = R1 + R2 + R3 + R4.

    R3 returns Optional[float]; None is treated as 0.0 in the sum.
    R4 receives R3's raw value so it returns 0.0 when R3 is None (no spurious bonus).

    Reward ranges:
      Optimal move, good format   → +0.2 +0.5 +0.0 +1.0 = +1.7
      Good move −50 cp, good fmt  → +0.2 +0.5 −0.5 +0.0 = +0.2
      Blunder −300 cp, good fmt   → +0.2 +0.5 −3.0 +0.0 = −2.3
      Illegal move, good format   → +0.2 −5.0 +0.0 +0.0 = −4.8
      Bad format, legal move      → −1.0 +0.5 delta bonus varies
      Bad format, illegal move    → −1.0 −5.0 +0.0 +0.0 = −6.0
    """
    r1 = reward_format(response)
    r2 = reward_legality(response, board)
    r3_raw = reward_strategic(response, board, analyst)
    r3 = r3_raw if r3_raw is not None else 0.0
    r4 = reward_optimal(r3_raw)
    return r1 + r2 + r3 + r4
