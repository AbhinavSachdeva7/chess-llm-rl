from collections import deque


class Curriculum:
    def __init__(self, start_elo=400, step=200, win_rate_threshold=0.6,
                 min_games_at_level=20, window=30):
        self.elo = start_elo
        self.step = step
        self.threshold = win_rate_threshold
        self.min_games = min_games_at_level
        self.window_size = window
        self._results = deque(maxlen=window)
        self._games_at_level = 0

    def record(self, result: str) -> None:
        # result in {"win","loss","draw"}; only "win" counts for win_rate
        self._results.append(result)
        self._games_at_level += 1

    def win_rate(self) -> float:
        if not self._results:
            return 0.0
        return sum(1 for r in self._results if r == "win") / len(self._results)

    def should_advance(self) -> bool:
        return (self._games_at_level >= self.min_games
                and len(self._results) >= self.min_games
                and self.win_rate() > self.threshold)

    def advance(self) -> int:
        self.elo += self.step
        self._results.clear()
        self._games_at_level = 0
        return self.elo
