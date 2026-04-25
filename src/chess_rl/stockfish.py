from __future__ import annotations

import atexit
import os
import random
import sys
from pathlib import Path
from typing import Optional

import chess
import chess.engine


class StockfishManager:
    """
    Manages Stockfish engine lifecycle for opponent play and position analysis.
    Supports Elo-based strength limiting for the opponent.
    """

    def __init__(self, path: Optional[str] = None):
        self.path = Path(path) if path else self._find_binary()
        self.opponent_engine: Optional[chess.engine.SimpleEngine] = None
        self.analyst_engine: Optional[chess.engine.SimpleEngine] = None
        self.elo = 400
        
        # Initialize engines
        self._ensure_opponent()
        # analyst_engine is lazy-loaded if analyze() is called
        
        atexit.register(self.close)

    def _find_binary(self) -> Path:
        """Locate Stockfish binary in ./stockfish/ directory."""
        base = Path("./stockfish")
        if sys.platform == "win32" or os.name == "nt":
            candidate = base / "stockfish.exe"
        else:
            candidate = base / "stockfish"
        
        if candidate.exists():
            return candidate
        
        # Fallback search
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

    def _ensure_opponent(self):
        if self.opponent_engine is None:
            self.opponent_engine = chess.engine.SimpleEngine.popen_uci(str(self.path))
            self.set_opponent_elo(self.elo)

    def _ensure_analyst(self):
        if self.analyst_engine is None:
            self.analyst_engine = chess.engine.SimpleEngine.popen_uci(str(self.path))

    def set_opponent_elo(self, elo: int):
        """Set opponent strength using UCI options or Skill Level."""
        self.elo = elo
        self._ensure_opponent()
        
        if elo >= 1320:
            # Native Elo limiting
            self.opponent_engine.configure({
                "UCI_LimitStrength": True,
                "UCI_Elo": elo
            })
        elif elo >= 1200:
            self.opponent_engine.configure({"Skill Level": 10})
        elif elo >= 1000:
            self.opponent_engine.configure({"Skill Level": 5})
        else:
            # For very low Elo, we use Skill Level 0 and handle randomness in play()
            self.opponent_engine.configure({"Skill Level": 0})

    def play(self, board: chess.Board, time_limit: float = 0.1) -> chess.Move:
        """Play a move as the opponent. Handles sub-800 Elo with explicit randomness."""
        self._ensure_opponent()
        
        # Handle sub-800 Elo randomness (per plan.md / steps.md)
        if self.elo < 800:
            random_chance = 0.5 if self.elo <= 400 else 0.3
            if random.random() < random_chance:
                legal = list(board.legal_moves)
                if legal:
                    return random.choice(legal)
        
        result = self.opponent_engine.play(board, chess.engine.Limit(time=time_limit))
        return result.move

    def analyze(self, board: chess.Board, time_limit: float = 0.1) -> int:
        """Analyze position and return centipawn score from relative POV."""
        self._ensure_analyst()
        info = self.analyst_engine.analyse(board, chess.engine.Limit(time=time_limit))
        score = info["score"].relative
        # Never return None, use 10000 for mate
        return score.score(mate_score=10000)

    def close(self):
        """Quit engines."""
        if self.opponent_engine is not None:
            try:
                self.opponent_engine.quit()
            except Exception:
                pass
            self.opponent_engine = None
            
        if self.analyst_engine is not None:
            try:
                self.analyst_engine.quit()
            except Exception:
                pass
            self.analyst_engine = None
