import pytest
from unittest.mock import MagicMock
import chess


@pytest.fixture
def mock_analyst():
    """Returns a stub SimpleEngine whose .analyse() returns a fake score.

    By default, returns +25 cp (from side-to-move POV) for every position,
    so (model_score - best_score) = 0 -> r3 = 0. Override .analyse per-test
    for negative-diff cases.
    """
    eng = MagicMock()
    info = {"score": MagicMock()}
    info["score"].relative.score = MagicMock(return_value=25)
    eng.analyse.return_value = info
    return eng


@pytest.fixture
def mock_stockfish():
    """Stub StockfishManager-like object. .play() returns the first legal move."""
    m = MagicMock()

    def _play(board, time_limit=0.1):
        return next(iter(board.legal_moves))

    m.play.side_effect = _play
    m.analyst_engine = MagicMock()
    return m
