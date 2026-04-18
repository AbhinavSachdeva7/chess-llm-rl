import sys; sys.path.insert(0, 'src')
import chess
from chess_rl.stockfish import StockfishManager
from chess_rl.rewards import reward_format, reward_legality, reward_strategic, compute_reward

assert reward_format("Nf3") == 0.1
assert reward_format("O-O") == 0.1
assert reward_format("I play e4") == -1.0

b = chess.Board()
assert reward_legality("e4", b) == 0.5
assert reward_legality("e5", b) == -5.0

m = StockfishManager()
r_e4 = reward_strategic("e4", b, m.analyst_engine)
r_a4 = reward_strategic("a4", b, m.analyst_engine)
print("e4 strategic:", r_e4, "a4 strategic:", r_a4)
assert r_e4 > r_a4, "e4 must be closer to best than a4"
assert r_e4 > -1.0, "e4 is near-best"
assert r_a4 < -0.2, "a4 should be visibly worse"

print("combined e4:", compute_reward("e4", b, m.analyst_engine))
print("combined a4:", compute_reward("a4", b, m.analyst_engine))
m.close()
