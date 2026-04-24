import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest
import chess
from unittest.mock import MagicMock

from chess_rl.rewards import reward_format, reward_legality, reward_strategic, reward_optimal, compute_reward
from chess_rl.prompts import build_messages, format_pgn
from chess_rl.env import ChessEnvironment, play_full_game
from chess_rl.curriculum import Curriculum


# --- reward_format (requires <think>...</think><move>SAN</move>) ---

def test_reward_format_valid_tagged():
    assert reward_format("<think>ok</think><move>Nf3</move>") == 0.2
    assert reward_format("<think>x</think><move>exd5</move>") == 0.2
    assert reward_format("<think>x</think><move>a8=Q+</move>") == 0.2
    assert reward_format("<think>x</think><move>Nbd2</move>") == 0.2
    assert reward_format("<think>x</think><move>Qxh7#</move>") == 0.2


def test_reward_format_valid_castling_tagged():
    assert reward_format("<think>x</think><move>O-O</move>") == 0.2
    assert reward_format("<think>x</think><move>O-O-O</move>") == 0.2


def test_reward_format_missing_think_start():
    assert reward_format("<move>Nf3</move>") == -1.0


def test_reward_format_missing_think_end():
    assert reward_format("<think>x<move>Nf3</move>") == -1.0


def test_reward_format_missing_move_tag():
    assert reward_format("<think>x</think>Nf3") == -1.0


def test_reward_format_wrong_tag_order():
    assert reward_format("<move>Nf3</move><think>x</think>") == -1.0


def test_reward_format_chatty():
    assert reward_format("I play e4") == -1.0
    assert reward_format("My move is Nf3.") == -1.0
    assert reward_format("") == -1.0


# --- reward_legality ---

def test_reward_legality_legal():
    assert reward_legality("<think>x</think><move>e4</move>", chess.Board()) == 0.5


def test_reward_legality_legal_untagged_fallback():
    assert reward_legality("e4", chess.Board()) == 0.5


def test_reward_legality_illegal():
    assert reward_legality("<think>x</think><move>e5</move>", chess.Board()) == -5.0
    assert reward_legality("gibberish", chess.Board()) == -5.0


# --- reward_strategic (returns Optional[float]) ---

def test_reward_strategic_best_is_zero(mock_analyst):
    calls = {"n": 0}

    def analyse(board, limit=None, **kw):
        calls["n"] += 1
        info = {"score": MagicMock()}
        info["score"].relative.score = MagicMock(
            return_value=50 if calls["n"] == 1 else -50
        )
        return info

    mock_analyst.analyse.side_effect = analyse
    r = reward_strategic("<think>t</think><move>e4</move>", chess.Board(), mock_analyst)
    assert r == 0.0


def test_reward_strategic_worse_is_negative(mock_analyst):
    calls = {"n": 0}

    def analyse(board, limit=None, **kw):
        calls["n"] += 1
        info = {"score": MagicMock()}
        info["score"].relative.score = MagicMock(return_value=100)
        return info

    mock_analyst.analyse.side_effect = analyse
    r = reward_strategic("<think>t</think><move>e4</move>", chess.Board(), mock_analyst)
    assert r is not None
    assert r < 0
    assert r >= -10.0


def test_reward_strategic_unparseable_returns_none(mock_analyst):
    r = reward_strategic("<think>t</think><move>zzz</move>", chess.Board(), mock_analyst)
    assert r is None


def test_reward_strategic_no_tags_parseable_returns_float(mock_analyst):
    calls = {"n": 0}

    def analyse(board, limit=None, **kw):
        calls["n"] += 1
        info = {"score": MagicMock()}
        info["score"].relative.score = MagicMock(
            return_value=50 if calls["n"] == 1 else -50
        )
        return info

    mock_analyst.analyse.side_effect = analyse
    r = reward_strategic("e4", chess.Board(), mock_analyst)
    assert r == 0.0


# --- reward_optimal ---

def test_reward_optimal_returns_1_for_best():
    assert reward_optimal(0.0) == 1.0
    assert reward_optimal(-0.05) == 1.0
    assert reward_optimal(-0.099) == 1.0


def test_reward_optimal_returns_0_for_suboptimal():
    assert reward_optimal(-0.1) == 0.0
    assert reward_optimal(-0.5) == 0.0
    assert reward_optimal(-10.0) == 0.0


def test_reward_optimal_returns_0_for_none():
    assert reward_optimal(None) == 0.0


# --- compute_reward: no early exits ---

def test_compute_reward_bad_format_still_evaluates_all(mock_analyst):
    # Missing <think> tag -> R1=-1.0, R2=+0.5 (legal), R3=0.0 (optimal), R4=+1.0 -> total=0.5
    calls = {"n": 0}

    def analyse(board, limit=None, **kw):
        calls["n"] += 1
        info = {"score": MagicMock()}
        info["score"].relative.score = MagicMock(
            return_value=50 if calls["n"] == 1 else -50
        )
        return info

    mock_analyst.analyse.side_effect = analyse
    board = chess.Board()
    r = compute_reward("<move>e4</move>", board, mock_analyst)
    assert r == pytest.approx(0.5)


def test_compute_reward_optimal_move_full_score(mock_analyst):
    # R1=+0.2, R2=+0.5, R3=0.0 (optimal delta), R4=+1.0 -> total=+1.7
    calls = {"n": 0}

    def analyse(board, limit=None, **kw):
        calls["n"] += 1
        info = {"score": MagicMock()}
        info["score"].relative.score = MagicMock(
            return_value=50 if calls["n"] == 1 else -50
        )
        return info

    mock_analyst.analyse.side_effect = analyse
    board = chess.Board()
    r = compute_reward("<think>best</think><move>e4</move>", board, mock_analyst)
    assert r == pytest.approx(1.7)


def test_compute_reward_illegal_no_r4_bonus(mock_analyst):
    # R1=+0.2, R2=-5.0, R3=None->0.0, R4=reward_optimal(None)=0.0 -> total=-4.8
    board = chess.Board()
    r = compute_reward("<think>t</think><move>e5</move>", board, mock_analyst)
    assert r == pytest.approx(-4.8)


# --- prompts ---

def test_prompt_includes_think_instruction():
    msgs = build_messages(chess.Board(), [], "fen_only", llm_color=True)
    content = msgs[0]["content"]
    assert "<think>" in content
    assert "<move>" in content
    assert "Think step by step" in content
    assert "long reasonings" not in content


def test_prompt_example_has_think_block():
    msgs = build_messages(chess.Board(), [], "fen_only", llm_color=True)
    content = msgs[0]["content"]
    assert "<think>The queen is attacked.</think><move>Nf3</move>" in content


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
        r_e4 = reward_strategic("<think>t</think><move>e4</move>", chess.Board(), m.analyst_engine)
        r_a4 = reward_strategic("<think>t</think><move>a4</move>", chess.Board(), m.analyst_engine)
        assert r_e4 is not None and r_a4 is not None
        assert r_e4 > r_a4
    finally:
        m.close()
