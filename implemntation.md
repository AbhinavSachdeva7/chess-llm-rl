# Implementation Plan: Chess LLM RL with GRPO (Simplified)

## Context

Extend Chess-R1 (Hwang et al., 2025). Hypothesis: full-game play with PGN history beats static puzzle RL. Two-arm ablation: FEN-only vs FEN+PGN. Same model, same loop, one variable.

**Decisions:**
- Model: Gemma 3 ~3B only. No benchmarking, no model selection.
- Deps: `requirements.txt`, not pyproject.toml.
- Config: one YAML, ~30 lines with comments.
- Package layout: flat, minimal modules.
- GRPO: start with trl `GRPOTrainer`. Write custom only if trl breaks.
- Pretrain: try Chess-R1 repo code if reusable; else skip.
- SFT warmup: fallback only if <50% legality after 100 games.
- Local dev + Kaggle T4. W&B for logging.

---

## Directory Structure

```
chess-llm-rl/
├── requirements.txt
├── .gitignore
├── config.yaml                  # one config, all hyperparams
├── plan.md
├── implemntation.md
├── CLAUDE.md
├── setup_stockfish.sh
├── src/chess_rl/
│   ├── __init__.py
│   ├── stockfish.py             # engine lifecycle + Elo + reward analysis
│   ├── model.py                 # Unsloth load + LoRA
│   ├── prompts.py               # system/user prompt for both arms
│   ├── rewards.py               # R1 format + R2 legal + R3 centipawn
│   ├── env.py                   # ChessEnvironment: board, PGN, step
│   ├── train.py                 # GRPO loop + curriculum + checkpoints + W&B
│   └── eval.py                  # Elo ladder evaluation
├── tests/
│   └── test_core.py             # rewards + prompts + env in one file
├── kaggle_notebook.ipynb
└── checkpoints/
```

Merge reasoning: `pgn_manager` lives inside `env.py` (just a list + formatter). `metrics` inside `train.py`. `curriculum` inside `train.py` (it's ~30 lines). `config.py` gone — load YAML into a dataclass inline. `sft_warmup.py` added only if needed.

---

## Step-by-Step (Importance-Ordered, Bottom-Up)

Build the reward signal first. Then the model. Then glue.

---

### Step 1: Stockfish works + gives centipawn reward

**Why first:** Without working reward, nothing else matters. Test in isolation.

**Files:** `setup_stockfish.sh`, `src/chess_rl/stockfish.py`, `src/chess_rl/rewards.py`

**Do:**
- Install Stockfish binary (script auto-detects Linux/Windows).
- `stockfish.py`: two `SimpleEngine` handles — opponent (Elo-limited) and analyst (full strength). `atexit` cleanup. Skill Level mapping for sub-1320 Elo (table in plan.md §5).
- `rewards.py`: R1 format (regex), R2 legal (`board.parse_san`), R3 centipawn delta (analyze pre-move + post-move, negate post, clamp −10). Singleton analyst engine.

**Verify standalone:**
```python
board = chess.Board()
assert reward_format("Nf3") == 0.1
assert reward_format("I play e4") == -1.0
assert reward_legality("e4", board) == 0.5
assert reward_strategic("e4", board, engine) > -1.0   # e4 near-best
assert reward_strategic("a4", board, engine) < -0.2   # a4 weak
```
No model, no prompt, no loop. Just: does Stockfish give the number we want?

**Pitfalls:** `board.san(move)` BEFORE `board.push`. `score.relative.score(mate_score=10000)` — without `mate_score`, mate positions return None.

## Feedback
Stockfish doesn't give the number. What happens is we are planning the reward based on two things:
1. the best move thought of by the Stockfish model of a specific Elo, like let's say the Stockfish model has 25 Elo and it thinks that the best move is e4, but the LLM gets the move to be e3
2. we calculate the difference between the senti score for the move made by Stockfish and the senti score we calculated for the move made by the AI module
We find the difference. Let's say when Stockfish made e4, the senti score was 0.2, and when the LLM made the move e3, it was 0.1, so the difference is -0.1, and that's what the reward goes for the LLM to tell that, "Okay, this was not the best move, but it was close to the best move." For the best move, the reward would be 0. Now, as the LLM requires positive reinforcement, we could add a certain number, a constant, to the whole reward so that all the rewards stay positive but the reward for the best move is a positive one instead of a zero. What we can do is just add one or try some other trick so that the rewards scale up not so much that there is then a negative reward for a really bad move, but they're not scaled up so little that the LLM thinks that the reward it is getting is enough and it doesn't learn anything. 

---

### Step 2: Model loads + generates

**Why next:** Second critical path. If Unsloth + Gemma 3 doesn't load on T4, whole project dead.

**Files:** `src/chess_rl/model.py`

**Do:**
- `FastLanguageModel.from_pretrained(gemma-3-3b, load_in_4bit=True, fast_inference=False)`.
- LoRA: r=32, alpha=64, target all attn + MLP projections. `use_gradient_checkpointing="unsloth"`.
- `tokenizer.padding_side = "left"` (required for GRPO). Set pad_token = eos_token if None.

**Verify:** Load model, feed one raw chess prompt, generate 10 tokens, print output, `torch.cuda.memory_summary()`. Must fit in T4 16GB.

**Pitfall:** No `fast_inference=True` on T4 — eats VRAM. Exact Gemma 3 HF ID must be verified before writing config.

## Feedback
we can switch to a smaller parameter gemma model if the largest model does not run but we will only be working with one llm. 

---

### Step 3: Prompts

**Files:** `src/chess_rl/prompts.py`, `config.yaml`

**Do:**
- `build_prompt(board, pgn, arm) -> list[dict]`. System: "You are a chess engine playing as {color}. Output ONLY the SAN move."
- Arm 1: FEN + legal moves (sorted SAN, comma-sep) + "Your move:".
- Arm 2: PGN + FEN + legal moves + "Your move:".
- Use `tokenizer.apply_chat_template()`. No hardcoded chat tokens.
- `config.yaml` created now — single file, ~30 lines, comments explain each knob.

**Verify:** Print prompt for 3 positions (opening, middlegame, endgame), eyeball FEN/legal-moves/PGN presence.

---

### Step 4: Env — one game end to end (no training)

**Files:** `src/chess_rl/env.py`

**Do:**
- `ChessEnvironment`: board, PGN list (SAN strings), llm_color.
- `reset()`: fresh board, random color.
- `step()`: build prompt → generate → parse → push (or fallback random).
- `apply_stockfish_move()` when opponent's turn.
- Terminate on `is_game_over(claim_draw=True)` OR move count ≥ 200.
- PGN: list of SAN, formatted on demand. Truncate last N tokens if over budget (sliding window from end, `"..."` prefix).

**Verify:** Play one full game (greedy decoding, no GRPO). Print PGN. Must be valid, must terminate, must have a result.

---

### Step 5: GRPO training loop

**Files:** `src/chess_rl/train.py`

**Do:**
- Experience replay pattern: play 10 games greedy → collect LLM-turn positions into HF Dataset `{prompt, fen, legal_moves_san}` → run `GRPOTrainer.train()` 1 epoch → repeat.
- GRPOConfig: `num_generations=4`, `max_completion_length=10` (CRITICAL — not 256), `per_device_train_batch_size=4`, `beta=0.0`, `lr=1e-6`, `temperature=1.0`, `use_vllm=False`.
- Reward function signature: `def chess_reward(completions, fen, legal_moves_san, **kwargs) -> list[float]`. Reconstructs board from FEN per call.
- Curriculum inline: rolling 30-game win rate >60% AND ≥20 games at level → Elo += 200. Reset window on advancement. Draws ≠ wins.
- W&B inline: `wandb.init()` once, log per-game metrics (result, length, legality, format, avg cp loss, current_elo).
- Checkpoint every 50 games: LoRA adapter + tokenizer + training state JSON (games, elo, rolling window, wandb run id). Push to HF Hub for Kaggle survival.

**Verify:** Run 1 iteration on tiny model / CPU. Game collection → dataset → GRPOTrainer init → 1 step completes.

**Pitfalls:** Recreating GRPOTrainer every iteration re-copies ref model → OOM. Try creating once and swapping dataset. If still OOM, write custom GRPO with `model.disable_adapter_layers()` trick for ref logits (no deep copy).

---

### Step 6: Timing benchmark — CRITICAL GATE

Run 5 full games on Kaggle T4. Extrapolate games/hour. Need ≥15/hr. If not:
1. `num_generations` 4→2
2. Stockfish `time_limit` 0.1→0.05
3. `max_completion_length` 10→6

Do NOT start full training until this passes.

---

### Step 7: Run two arms

Both arms same seed, same base checkpoint, same curriculum, same game count. Only `arm` flag differs. Separate W&B runs, same project.

Budget: ~20 games/hr × 9hr Kaggle session = 180 games/session. 3 sessions/arm. 1–2 weeks total.

---

### Step 8: Evaluation

**Files:** `src/chess_rl/eval.py`

Base (no LoRA) vs Arm 1 vs Arm 2. Elo ladder: 400, 600, 800, 1000, 1200, 1400. 50 games/level (25 W, 25 B). Greedy. Record win/draw/loss, avg cp loss, avg length. Ceiling = highest level with >50% wins.

---

### Step 9: Tests (critical path only)

Single file `tests/test_core.py`. Mock Stockfish fixture. Real-Stockfish tests marked `@pytest.mark.stockfish`, skipped if binary absent.

Cases: all reward functions on known positions; prompts contain expected fields per arm; env plays full game + terminates; PGN truncation keeps recent moves; curriculum advances at correct threshold.

---

### Step 10: Pretrain / SFT fallback (conditional)

**Only if** legality rate <50% after 100 games.

Plan A: Check [Chess-R1 repo](https://github.com/krafton-ai/Chess-R1) for reusable pretrain code. If usable → adopt.

Plan B: Unsloth SFTTrainer on 5k (position, stockfish-best-move) pairs across game phases. 500–1000 steps. Resume RL from SFT checkpoint.

Plan C: Skip pretrain, accept longer RL.

## Feedback
If free training were to be done, how much time would it take if it was to be done using plan A or plan B? 
---

### Step 11: Kaggle deploy

`kaggle_notebook.ipynb`: install deps, download Stockfish, clone repo + `pip install -r requirements.txt`, W&B login via Secrets, resume from HF Hub checkpoint, run `train.py`, push checkpoint back.

Pin versions. Enable internet. `!nvidia-smi` first cell.

---

### Step 12: Failure analysis + writeup

20–30 losses near Elo ceiling → Stockfish analysis → categorize (tactical / strategic / format). Compare arms. Write.

---

## Dependency Graph

```
Step 1 (stockfish+rewards)  ──┐
Step 2 (model load)           ├─→ Step 3 (prompts) → Step 4 (env) → Step 5 (GRPO loop)
                              │                                         │
                              │                                         ▼
                              │                                   Step 6 (timing gate)
                              │                                         │
                              │                                         ▼
                              │                                   Step 7 (two arms)
                              │                                         │
                              │                                         ▼
                              │                                   Step 8 (eval) + Step 12
                              │
Step 9 (tests) — alongside each module, not end
Step 10 (pretrain) — only if legality stuck
Step 11 (Kaggle) — parallel w/ Step 5 onward
```

## Simplifications vs old plan

| Old | New |
|-----|-----|
| 3 YAML configs | 1 `config.yaml` |
| `pyproject.toml` + extras | `requirements.txt` |
| 14 src modules | 7 src modules |
| Separate pgn_manager, metrics, curriculum, checkpointing modules | Folded into env.py / train.py |
| Model benchmark (Step 16) | Dropped — Gemma 3 only |
| Docker (Step 24) | Dropped |
| Config schema module (Step 2) | Dataclass inline |
| Build scaffolding first, model last | Reward+model first, scaffolding last |
