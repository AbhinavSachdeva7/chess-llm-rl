from __future__ import annotations
import random
from dataclasses import dataclass, field
from typing import Callable, Literal, Optional

import chess

from .prompts import build_messages
from .rewards import extract_san


@dataclass
class GameMetrics:
    illegal: int = 0
    legal: int = 0
    format_fail: int = 0
    no_legal_fallback: int = 0
    centipawn_losses: list[float] = field(default_factory=list)


class ChessEnvironment:
    def __init__(self, config: dict, stockfish_mgr, tokenizer):
        self.config = config
        self.stockfish = stockfish_mgr
        self.tokenizer = tokenizer
        self.board: chess.Board = chess.Board()
        self.pgn_san: list[str] = []
        self.llm_color: bool = chess.WHITE
        self.metrics: GameMetrics = GameMetrics()
        self.arm: str = config.get("training", {}).get("arm", "fen_pgn")
        self.max_moves: int = config.get("game", {}).get("max_moves", 200)

    def reset(self, llm_color: Optional[bool] = None) -> None:
        self.board = chess.Board()
        self.pgn_san = []
        self.metrics = GameMetrics()
        if llm_color is None:
            self.llm_color = random.choice([chess.WHITE, chess.BLACK])
        else:
            self.llm_color = llm_color

    def is_llm_turn(self) -> bool:
        return self.board.turn == self.llm_color

    def get_messages(self) -> list[dict]:
        return build_messages(self.board, self.pgn_san, self.arm,
                              llm_color=(self.llm_color == chess.WHITE))

    def apply_llm_move(self, san: str) -> bool:
        """Parse SAN and push. Returns True on legal push, False on any failure.
        Increments metrics accordingly.
        """
        response = extract_san(san).rstrip(".")
        try:
            move = self.board.parse_san(response)
        except (chess.InvalidMoveError, chess.IllegalMoveError,
                chess.AmbiguousMoveError, ValueError):
            self.metrics.illegal += 1
            return False
        if move not in self.board.legal_moves:
            self.metrics.illegal += 1
            return False
        san_clean = self.board.san(move)
        self.pgn_san.append(san_clean)
        self.board.push(move)
        self.metrics.legal += 1
        return True

    def apply_stockfish_move(self) -> None:
        mv = self.stockfish.play(
            self.board,
            time_limit=self.config.get("stockfish", {}).get("opponent_time", 0.1),
        )
        # Record SAN BEFORE push, since SAN depends on pre-push position.
        self.pgn_san.append(self.board.san(mv))
        self.board.push(mv)

    def is_game_over(self) -> bool:
        return (
            self.board.is_game_over(claim_draw=True)
            or len(self.pgn_san) >= self.max_moves
        )

    def get_result(self) -> Literal["win", "loss", "draw"]:
        outcome = self.board.outcome(claim_draw=True)
        if outcome is None:
            return "draw"  # max-move cap
        winner = outcome.winner
        if winner is None:
            return "draw"
        return "win" if winner == self.llm_color else "loss"


def play_full_game(env: ChessEnvironment,
                   generate_fn: Callable[[list[dict]], str]) -> tuple[str, GameMetrics]:
    """Run a complete game; returns (result, metrics).

    `generate_fn(messages) -> raw_text` is the user-supplied LLM callable.
    If LLM output fails to parse/apply, a random legal move is pushed and
    `metrics.no_legal_fallback` is incremented.
    """
    while not env.is_game_over():
        if env.is_llm_turn():
            msgs = env.get_messages()
            raw = generate_fn(msgs)
            ok = env.apply_llm_move(raw)
            if not ok:
                legal = list(env.board.legal_moves)
                if not legal:
                    break
                mv = random.choice(legal)
                env.pgn_san.append(env.board.san(mv))
                env.board.push(mv)
                env.metrics.no_legal_fallback += 1
        else:
            env.apply_stockfish_move()
    return env.get_result(), env.metrics
