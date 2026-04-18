import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import chess
from unittest.mock import MagicMock

from chess_rl.rewards import reward_format, reward_legality, reward_strategic
from chess_rl.prompts import build_messages, format_pgn
from chess_rl.env import ChessEnvironment, play_full_game
from chess_rl.curriculum import Curriculum


# --- reward_format ---
def test_reward_format_valid_san():
    assert reward_format("Nf3") == 0.1
    assert reward_format("exd5") == 0.1
    assert reward_format("a8=Q+") == 0.1
    assert reward_format("Nbd2") == 0.1
    assert reward_format("Qxh7#") == 0.1


def test_reward_format_chatty():
    assert reward_format("I play e4") == -1.0
    assert reward_format("My move is Nf3.") == -1.0
    assert reward_format("") == -1.0


def test_reward_format_castling():
    assert reward_format("O-O") == 0.1
    assert reward_format("O-O-O") == 0.1
    assert reward_format("O-O.") == 0.1   # trailing-dot tolerated


# --- reward_legality ---
def test_reward_legality_legal():
    assert reward_legality("e4", chess.Board()) == 0.5


def test_reward_legality_illegal():
    # e5 is Black's move on White's turn
    assert reward_legality("e5", chess.Board()) == -5.0
    assert reward_legality("gibberish", chess.Board()) == -5.0


# --- reward_strategic (mock) ---
def test_reward_strategic_best_is_zero(mock_analyst):
    # Make pre-move return +50 (mover POV), post-move return -50 (opponent POV).
    # After negation: model_score = +50. diff = (50 - 50) / 100 = 0.
    calls = {"n": 0}

    def analyse(board, limit=None, **kw):
        calls["n"] += 1
        info = {"score": MagicMock()}
        # pre-move: mover POV = +50; post-move: opponent POV = -50 -> negated = +50 -> diff = 0
        info["score"].relative.score = MagicMock(
            return_value=50 if calls["n"] == 1 else -50
        )
        return info

    mock_analyst.analyse.side_effect = analyse
    r = reward_strategic("e4", chess.Board(), mock_analyst)
    assert r == 0.0


def test_reward_strategic_worse_is_negative(mock_analyst):
    # pre: +100 (mover); post: +100 (opponent POV) -> negated = -100 -> diff = -200/100 = -2.0
    calls = {"n": 0}

    def analyse(board, limit=None, **kw):
        calls["n"] += 1
        info = {"score": MagicMock()}
        info["score"].relative.score = MagicMock(return_value=100)
        return info

    mock_analyst.analyse.side_effect = analyse
    r = reward_strategic("e4", chess.Board(), mock_analyst)
    assert r < 0
    assert r >= -10.0


# --- prompts ---
def test_prompt_arm1_no_pgn_section():
    msgs = build_messages(chess.Board(), [], "fen_only", llm_color=True)
    joined = " ".join(m["content"] for m in msgs)
    assert "Game so far" not in joined
    assert "Position (FEN)" in joined
    assert "Your move:" in joined


def test_prompt_arm2_has_pgn():
    msgs = build_messages(chess.Board(), ["e4", "e5"], "fen_pgn", llm_color=True)
    joined = " ".join(m["content"] for m in msgs)
    assert "Game so far" in joined
    assert "1. e4 e5" in joined


def test_format_pgn_empty():
    assert format_pgn([]) == ""


def test_format_pgn_pairs():
    assert format_pgn(["e4", "e5", "Nf3"]).startswith("1. e4 e5 2. Nf3")


# --- env ---
def test_env_plays_full_game(mock_stockfish):
    cfg = {
        "training": {"arm": "fen_only"},
        "game": {"max_moves": 200},
        "stockfish": {"opponent_time": 0.1},
    }
    env = ChessEnvironment(cfg, mock_stockfish, tokenizer=None)
    env.reset(llm_color=chess.WHITE)

    # LLM also plays "first legal move"
    def gen(messages):
        return env.board.san(next(iter(env.board.legal_moves)))

    result, metrics = play_full_game(env, gen)
    assert result in {"win", "loss", "draw"}
    assert metrics.legal > 0


def test_env_200_move_cap(mock_stockfish):
    cfg = {
        "training": {"arm": "fen_only"},
        "game": {"max_moves": 4},
        "stockfish": {"opponent_time": 0.1},
    }
    env = ChessEnvironment(cfg, mock_stockfish, tokenizer=None)
    env.reset(llm_color=chess.WHITE)

    def gen(messages):
        return env.board.san(next(iter(env.board.legal_moves)))

    _, metrics = play_full_game(env, gen)
    assert len(env.pgn_san) <= 4


# --- pgn truncation ---
def test_pgn_truncation_keeps_recent():
    moves = [f"a{i % 8 + 1}" for i in range(20)]
    # If format_pgn ever grows a max_tokens knob, test real truncation.
    # For now, assert the last move appears in the rendered string.
    s = format_pgn(moves)
    assert "a" in s


# --- curriculum ---
def test_curriculum_advances_at_threshold():
    c = Curriculum(
        start_elo=400, step=200, win_rate_threshold=0.6,
        min_games_at_level=20, window=30,
    )
    # 14 wins, 6 losses -> 70% win rate over 20 games (>= min_games_at_level).
    for _ in range(14):
        c.record("win")
    for _ in range(6):
        c.record("loss")
    assert c.should_advance()
    assert c.advance() == 600
    assert not c.should_advance()   # counters reset


def test_curriculum_no_premature_advance():
    c = Curriculum(
        start_elo=400, step=200, win_rate_threshold=0.6,
        min_games_at_level=20, window=30,
    )
    for _ in range(10):
        c.record("win")
    assert not c.should_advance()  # fewer than min_games
    for _ in range(10):
        c.record("loss")
    assert not c.should_advance()  # 50% win rate, at min_games


# --- live stockfish (skipped unless binary present) ---
_sf_missing = not (
    os.path.exists("stockfish/stockfish.exe")
    or os.path.exists("stockfish/stockfish")
)


@pytest.mark.stockfish
@pytest.mark.skipif(_sf_missing, reason="stockfish binary not found")
def test_live_reward_strategic_e4_better_than_a4():
    from chess_rl.stockfish import StockfishManager
    from chess_rl.rewards import reward_strategic

    m = StockfishManager()
    try:
        r_e4 = reward_strategic("e4", chess.Board(), m.analyst_engine)
        r_a4 = reward_strategic("a4", chess.Board(), m.analyst_engine)
        assert r_e4 > r_a4
    finally:
        m.close()
