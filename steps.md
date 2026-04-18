# steps.md — Opus-Orchestrated Execution Script

Companion to [plan.md](plan.md) and [implemntation.md](implemntation.md). Where the three documents disagree, **this file wins** for execution details; `plan.md` wins for hypothesis and reward math.

---

## Orchestrator Protocol (read first, Opus)

**You (Opus) are the orchestrator.** Do not execute steps yourself except those assigned to `opus`. Dispatch each step via the `Agent` tool with `subagent_type: general-purpose` and `model: sonnet` to a Sonnet 4.6 subagent. Pull the step back to yourself only when `escalate_if` fires.

### Dispatch rules

1. Walk the step list top to bottom. A step is eligible when every id in its `depends_on` has status `done`.
2. All eligible steps with disjoint `files` sets run in **parallel** — issue one assistant message with multiple `Agent` tool calls.
3. Subagent prompt template (fill from the step body):
   ```
   You are implementing step {id} — {title} — of project chess-llm-rl.
   Working directory: c:\Users\abhin\Desktop\chess-llm-rl
   Read: plan.md (hypothesis + reward math), implemntation.md (pitfalls), steps.md (your step below).

   GOAL: {goal}
   FILES YOU MAY CREATE/EDIT: {files}
   INPUTS FROM PRIOR STEPS: {inputs}
   DO:
   {do}
   DELIVERABLE:
   {deliverable}
   VERIFY (run these commands, paste their raw output into your return message):
   {verify}
   ESCALATE: If you encounter any of these, STOP and return with a summary — do not work around them:
   {escalate_if}

   Report back in under 300 words: what you did, verify output, and any surprises.
   ```
4. On return: read the subagent's verify output. If it satisfies the step, mark `done`. If it's incomplete or the subagent escalated, Opus takes over directly — no second Sonnet attempt on the same failure.
5. **Never delegate decisions, only execution.** Reward tuning, pretrain path, arm allocation: Opus calls these.

### Global `escalate_if` (applies to every step)

- OOM or CUDA errors during model load, generation, or GRPO step.
- `trl.GRPOTrainer` raises an internal error, or installed `trl` version lacks a referenced field.
- Unsloth version/CUDA pin conflict that prevents `FastLanguageModel.from_pretrained` from running.
- Need to write a custom GRPO inner loop (LoRA `disable_adapter_layers` trick).
- Reward signal appears dead: >30% of positions return identical rewards across all 4 candidates in a sample check.
- Sub-1320 Stockfish Elo mapping behaves oddly (Skill Level produces non-monotonic strength).
- Ambiguity in the Chess-R1 pretrain decision (Step S15) — always Opus.
- Any destructive git op or force push.

### Verification rule

Every subagent MUST run the step's `verify` block and paste raw output. A claim of success without output is a failed step; Opus re-dispatches with "you skipped verify — run it and paste output."

### Global constants (load from config.yaml once it exists)

- Model: `unsloth/gemma-3-4b-it` (fallback `unsloth/gemma-3-1b-it` if 4B OOMs on T4). One model only.
- GRPO: `num_generations=4`, `max_completion_length=10`, `beta=0.0`, `lr=1e-6`, `temperature=1.0`, `per_device_train_batch_size=4`, `use_vllm=False`.
- Stockfish analyst `time_limit=0.1`. Opponent same.
- Curriculum start Elo: 400. Step: +200 on rolling-30 win rate >60% with ≥20 games at level.
- Checkpoint every 50 games.
- Max moves per game: 200.
- Seed: 42.

---

## Step list

---

### S0 — Repo scaffolding

- `assignee`: sonnet
- `depends_on`: []
- `goal`: Create project skeleton ready for code.
- `files`: `requirements.txt`, `.gitignore`, `src/chess_rl/__init__.py`, `tests/__init__.py`, `config.yaml` (empty placeholder), `checkpoints/.gitkeep`
- `inputs`: Directory structure from [implemntation.md §Directory Structure](implemntation.md)
- `do`:
  - `requirements.txt`: `python-chess>=1.11`, `torch>=2.1`, `transformers>=4.45`, `trl>=0.16`, `peft`, `bitsandbytes`, `wandb`, `pydantic>=2.0`, `pyyaml`, `datasets`, `pytest`. Do NOT add `unsloth` (GPU-only).
  - `.gitignore`: standard Python + `checkpoints/*` (keep `.gitkeep`), `wandb/`, `*.pt`, `stockfish*` (binary), `.env`, `__pycache__/`, `.venv/`.
  - `src/chess_rl/__init__.py`: empty.
  - `tests/__init__.py`: empty.
  - `config.yaml`: single-line comment header only; real content arrives in S6.
- `deliverable`: Files created. `pip install -r requirements.txt` succeeds on a clean venv (optional if env already set up).
- `verify`:
  ```bash
  ls -la src/chess_rl tests requirements.txt .gitignore config.yaml
  python -c "import chess_rl" || echo "OK — empty package import tolerated"
  ```
- `escalate_if`: (global only)

---

### S1 — Stockfish install + engine wrapper

- `assignee`: sonnet
- `depends_on`: [S0]
- `goal`: Two working Stockfish processes (opponent, analyst) with Elo control.
- `files`: `setup_stockfish.sh`, `src/chess_rl/stockfish.py`
- `inputs`: Elo-mapping table from [plan.md §5](plan.md) and [implemntation.md Step 3](implemntation.md).
- `do`:
  - `setup_stockfish.sh`: detect `uname`; download `stockfish-ubuntu-x86-64-avx2` on Linux, `stockfish-windows-x86-64-avx2.exe` on Windows, extract to `./stockfish/`.
  - `stockfish.py`: `class StockfishManager` with `opponent_engine` and `analyst_engine` fields. Methods: `set_opponent_elo(elo: int)`, `analyze(board, time_limit=0.1) -> cp_score`, `play(board, time_limit=0.1) -> move`, `close()`. Register `atexit` handler that calls `close()`.
  - Elo mapping (sub-1320):
    | Target | Strategy |
    |---|---|
    | 400 | Skill Level 0 + 50% random move |
    | 600 | Skill Level 0 + 30% random |
    | 800 | Skill Level 0 |
    | 1000 | Skill Level 5 |
    | 1200 | Skill Level 10 |
    | 1320+ | `UCI_LimitStrength=True`, `UCI_Elo=target` |
  - `analyze()` must use `score.relative.score(mate_score=10000)` — never `None`.
- `deliverable`: Can start two engines, set Elo, play 3 moves from startpos, shutdown without orphan processes.
- `verify`:
  ```bash
  bash setup_stockfish.sh
  python -c "
  from src.chess_rl.stockfish import StockfishManager
  import chess
  m = StockfishManager()
  m.set_opponent_elo(800)
  b = chess.Board()
  for _ in range(3):
      mv = m.play(b); b.push(mv); print(b.fen())
  print('analyst cp:', m.analyze(b))
  m.close()
  "
  ```
- `escalate_if`: Stockfish binary fails to run on platform; `UCI_Elo` silently rejected.

---

### S2 — Rewards module (DIFF-ONLY R3)

- `assignee`: sonnet
- `depends_on`: [S1]
- `goal`: Three reward functions with the user's corrected R3 semantics.
- `files`: `src/chess_rl/rewards.py`
- `inputs`: Rewards spec from [plan.md §5.5](plan.md); user feedback in [implemntation.md](implemntation.md) Step 1 feedback block.
- `do`:
  - `reward_format(response: str) -> float`:
    ```python
    response = response.strip().rstrip(".")
    if response in ("O-O", "O-O-O"): return 0.1
    if re.fullmatch(r'[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](=[QRBN])?[+#]?', response): return 0.1
    return -1.0
    ```
  - `reward_legality(response, board) -> float`: parse SAN, check `in board.legal_moves`. Return `+0.5` / `-5.0`. Catch `chess.InvalidMoveError`, `IllegalMoveError`, `AmbiguousMoveError`, `ValueError`.
  - `reward_strategic(response, board, analyst) -> float` — **DIFF-ONLY, no additive constant**:
    ```python
    # 1. analyst evaluates current board -> best_score (from mover's POV)
    # 2. push the LLM's move
    # 3. analyst evaluates post-move -> negate (perspective flipped) -> model_score
    # 4. return clamp((model_score - best_score) / 100.0, -10.0, 0.0)
    ```
    Best move ≈ 0. Worse → more negative. Do **not** add `+C`.
  - `compute_reward(response, board, analyst) -> float`: early-exit chain.
    - `r1 = reward_format(response)`; if `r1 < 0`: return `r1`.
    - `r2 = reward_legality(response, board)`; if `r2 < 0`: return `r1 + r2`.
    - `r3 = reward_strategic(response, board, analyst)`; return `r1 + r2 + r3`.
  - Singleton analyst engine inside module (not passed around), lazy-init + atexit close.
- `deliverable`: Module importable; all four functions work.
- `verify`: (runs in S3)
- `escalate_if`: (global only)

---

### S3 — Reward standalone verification

- `assignee`: sonnet
- `depends_on`: [S2]
- `goal`: Prove the reward signal is correct before touching the model.
- `files`: none (ad-hoc script or `tests/test_rewards_live.py`)
- `inputs`: S2 rewards module.
- `do`: Run this as a script, paste output:
  ```python
  import chess
  from src.chess_rl.stockfish import StockfishManager
  from src.chess_rl.rewards import reward_format, reward_legality, reward_strategic, compute_reward

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
  ```
- `deliverable`: All asserts pass. Print values logged.
- `verify`: Paste script output into return message.
- `escalate_if`: Stockfish returns `None` for any position; `a4` scores higher than `e4`.

---

### S4 — Model loader (Unsloth Gemma 3 + LoRA)

- `assignee`: **opus** — Unsloth/trl/bitsandbytes version pinning is fragile; LoRA target-module list varies across Gemma families; a Sonnet attempt is likely to wedge on library mismatches.
- `depends_on`: [S0]
- `goal`: Load Gemma 3 in 4-bit with LoRA adapter, padding ready for GRPO.
- `files`: `src/chess_rl/model.py`
- `inputs`: [implemntation.md Step 10](implemntation.md).
- `do`:
  ```python
  from unsloth import FastLanguageModel
  def load_model(model_name="unsloth/gemma-3-4b-it", max_seq_length=2048):
      model, tok = FastLanguageModel.from_pretrained(
          model_name=model_name,
          max_seq_length=max_seq_length,
          load_in_4bit=True,
          fast_inference=False,
      )
      model = FastLanguageModel.get_peft_model(
          model, r=32,
          target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
          lora_alpha=64,
          use_gradient_checkpointing="unsloth",
      )
      tok.padding_side = "left"
      if tok.pad_token is None: tok.pad_token = tok.eos_token
      return model, tok
  ```
  - On `torch.cuda.OutOfMemoryError` during `from_pretrained`, retry once with `unsloth/gemma-3-1b-it`. Record the choice in `config.yaml` under `model.name`.
- `deliverable`: `load_model()` returns `(model, tokenizer)` on T4. VRAM ≤ 12 GB headroom.
- `verify`: Performed in S5.
- `escalate_if`: Both 4B and 1B OOM; model IDs do not exist on HF; Unsloth import fails.

---

### S5 — Model smoke generate

- `assignee`: sonnet
- `depends_on`: [S4]
- `goal`: Model generates a plausible response to a raw chess prompt.
- `files`: none (script)
- `inputs`: S4 `load_model`.
- `do`:
  ```python
  from src.chess_rl.model import load_model
  import torch
  model, tok = load_model()
  prompt = "Position (FEN): rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1\nLegal moves: a3,a4,b3,b4,c3,c4,d3,d4,e3,e4,f3,f4,g3,g4,h3,h4,Na3,Nc3,Nf3,Nh3\nYour move:"
  ids = tok(prompt, return_tensors="pt").to(model.device)
  out = model.generate(**ids, max_new_tokens=10, do_sample=False)
  print(tok.decode(out[0][ids['input_ids'].shape[1]:], skip_special_tokens=True))
  print(torch.cuda.memory_summary(abbreviated=True))
  ```
- `deliverable`: Decoded output printed. VRAM summary printed.
- `verify`: Paste both prints.
- `escalate_if`: Generation hangs >60s; output is empty string.

---

### S6 — Prompts + config.yaml

- `assignee`: sonnet
- `depends_on`: [S5]
- `goal`: Prompt builders for both arms + full config file.
- `files`: `src/chess_rl/prompts.py`, `config.yaml`
- `inputs`: [plan.md §5.4](plan.md), [implemntation.md Step 3 & Step 5](implemntation.md), Global constants above.
- `do`:
  - `prompts.py`:
    - `build_messages(board: chess.Board, pgn_san: list[str], arm: str, llm_color: bool) -> list[dict]`.
    - System: `"You are a chess engine playing as {White|Black}. Given the current board state, output your next move in Standard Algebraic Notation (SAN). Output ONLY the move, nothing else. Example output: Nf3"`
    - Legal moves: `sorted(board.san(m) for m in board.legal_moves)`, comma-joined.
    - Arm 1 (`"fen_only"`): FEN + legal moves + `"Your move:"`.
    - Arm 2 (`"fen_pgn"`): PGN + FEN + legal moves + `"Your move:"`. PGN built from `pgn_san` list.
    - Return conversational list; do NOT apply chat template here — caller does.
    - `apply_template(tok, messages) -> str`: thin wrapper around `tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)`.
    - `format_pgn(pgn_san: list[str]) -> str`: standard `"1. e4 e5 2. Nf3 ..."` formatting.
  - `config.yaml` (≈30 lines, commented):
    ```yaml
    # chess-llm-rl single config
    model:
      name: unsloth/gemma-3-4b-it        # fallback: unsloth/gemma-3-1b-it
      max_seq_length: 2048
    grpo:
      num_generations: 4
      max_completion_length: 10          # chess moves are 2-7 chars
      per_device_train_batch_size: 4
      learning_rate: 1.0e-6
      beta: 0.0                           # no KL penalty
      temperature: 1.0
      use_vllm: false                     # no vLLM on T4
    stockfish:
      path: ./stockfish/stockfish
      analyst_time: 0.1
      opponent_time: 0.1
    curriculum:
      start_elo: 400
      step: 200
      win_rate_threshold: 0.6
      min_games_at_level: 20
      window: 30
    game:
      max_moves: 200
      pgn_max_tokens: 512
    training:
      arm: fen_pgn                        # or fen_only
      games_per_iteration: 10
      checkpoint_every: 50
      seed: 42
    wandb:
      project: chess-llm-rl
      mode: online                        # offline for local CPU
    paths:
      checkpoints: ./checkpoints
    ```
- `deliverable`: Prompt builder returns valid chat-format; config loads via `yaml.safe_load`.
- `verify`:
  ```python
  import chess, yaml
  from src.chess_rl.prompts import build_messages, apply_template
  from src.chess_rl.model import load_model
  _, tok = load_model()   # or skip model, use bare tokenizer
  b = chess.Board()
  for arm in ("fen_only", "fen_pgn"):
      msgs = build_messages(b, ["e4","e5"], arm, llm_color=True)
      print("---", arm, "---"); print(apply_template(tok, msgs))
  print(yaml.safe_load(open("config.yaml")))
  ```
- `escalate_if`: `apply_chat_template` raises on Gemma 3 tokenizer.

---

### S7 — Chess environment

- `assignee`: sonnet
- `depends_on`: [S6, S1]
- `goal`: Single-class wrapper that runs full games with a generation callable.
- `files`: `src/chess_rl/env.py`
- `inputs`: [implemntation.md Step 7](implemntation.md).
- `do`:
  - `class ChessEnvironment`:
    - `__init__(config, stockfish_mgr, tokenizer)`.
    - `reset(llm_color=None)`: fresh `chess.Board()`, random color, empty `pgn_san` list, reset `GameMetrics`.
    - `is_llm_turn() -> bool`: `self.board.turn == self.llm_color`.
    - `get_messages() -> list[dict]`: delegate to `prompts.build_messages`.
    - `apply_llm_move(san: str) -> bool`: parse_san, push, append SAN to `pgn_san`. Return success. On fail, record `metrics.illegal += 1`.
    - `apply_stockfish_move()`: call `stockfish_mgr.play(self.board)`, append SAN **before push**, push.
    - `is_game_over() -> bool`: `self.board.is_game_over(claim_draw=True) or len(self.pgn_san) >= config.game.max_moves`.
    - `get_result() -> Literal["win","loss","draw"]` from LLM POV.
    - `GameMetrics` dataclass: illegal, legal, format_fail, no_legal_fallback, centipawn_losses: list[float].
  - Also a `play_full_game(env, generate_fn)` helper for S8.
- `deliverable`: Module importable; no circular imports.
- `verify`: Performed in S8.
- `escalate_if`: (global only)

---

### S8 — One full game greedy, no training

- `assignee`: sonnet
- `depends_on`: [S7]
- `goal`: End-to-end sanity — model plays a complete game vs Stockfish-400 with no GRPO.
- `files`: none (script `scripts/smoke_game.py` optional)
- `inputs`: S4 model, S6 prompts, S7 env.
- `do`: Construct env with Stockfish at Elo 400, greedy generate (`do_sample=False`, `max_new_tokens=10`), play until termination. If model's output fails to parse, fall back to random legal move and increment counter. Print final PGN, result, metrics.
- `deliverable`: One complete game with a result (win/loss/draw).
- `verify`: Paste PGN + final metrics dict. Confirm: PGN parses with `chess.pgn.read_game`, game terminated correctly (not stuck at 200-move cap unless genuinely drawn out).
- `escalate_if`: Model never produces a legal move (>95% fallback rate) — this signals a prompt bug, not a training problem.

---

### S9 — GRPO training loop

- `assignee`: **opus** — trl `GRPOTrainer` ref-model OOM is the main project risk; requires judgment on create-once-vs-per-iteration, and possible fall-through to S10.
- `depends_on`: [S8, S2]
- `goal`: End-to-end training iteration: experience collection → GRPO update → curriculum advance → checkpoint.
- `files`: `src/chess_rl/train.py`
- `inputs`: [implemntation.md Step 11, Step 13, Step 14](implemntation.md).
- `do`:
  - `collect_game_experience(model, tok, stockfish, cfg, n_games) -> Dataset`: play `n_games` greedy; at each LLM turn capture `{prompt, fen, legal_moves_san}` BEFORE the model plays. Continue the game by greedy-generating + parsing + pushing (or random fallback).
  - `chess_reward_func(completions, fen, legal_moves_san, **kwargs) -> list[float]`: rebuild `chess.Board(fen)`, call `compute_reward` from S2 per completion. Use module-level analyst engine.
  - `train_iteration(model, tok, stockfish, cfg)`:
    - dataset = collect_game_experience(...)
    - trainer = GRPOTrainer(model=model, args=GRPOConfig(**cfg.grpo, num_train_epochs=1), reward_funcs=[chess_reward_func], train_dataset=dataset)
    - trainer.train()
    - **Try creating the trainer once and swapping its dataset each iteration** — if that's awkward in the installed trl version, create per iteration but monitor VRAM. OOM → escalate to S10.
  - Curriculum inline: `class Curriculum` with `record(result)` and `should_advance()`. Rolling window = `cfg.curriculum.window`. Advance + clear window when threshold met and min games reached.
  - W&B inline: `wandb.init(project=cfg.wandb.project, config=cfg)`, log per game: `result, length, legality, format_compliance, avg_cp_loss, current_elo, win_rate_30`.
  - Checkpointing inline: every `cfg.training.checkpoint_every` games, save LoRA (`model.save_pretrained(path)`), tokenizer, JSON state `{games, elo, window, wandb_run_id, arm}`. Push to HF Hub if `HF_TOKEN` set.
  - Top-level `main(config_path, resume_from=None)`.
- `deliverable`: Running `python -m chess_rl.train --config config.yaml` completes 1 iteration (10 games + 1 train pass) locally (CPU or GPU) without crashes.
- `verify`: Paste end-of-iteration W&B summary + checkpoint directory `ls`.
- `escalate_if`: OOM from ref-model copy; trl API mismatch (e.g. `GRPOConfig` field missing); reward-function arguments rejected by trl.

---

### S10 — Custom GRPO fallback (conditional)

- `assignee`: **opus** — algorithmic correctness matters; LoRA adapter-toggle ref policy is easy to get wrong.
- `depends_on`: [S9] **only executes if S9 escalates on OOM or trl API issues**
- `goal`: Standalone GRPO loop that doesn't rely on trl's trainer.
- `files`: `src/chess_rl/custom_grpo.py`
- `inputs`: [implemntation.md Step 12](implemntation.md).
- `do`:
  - Per position:
    1. Sample G=4 completions at T=1.0.
    2. Compute rewards via `compute_reward`.
    3. Advantages: `A_i = (r_i - mean(r)) / (std(r) + 1e-8)`.
    4. Ref logprobs: `model.disable_adapter_layers(); ref = model(ids).logits; model.enable_adapter_layers()`.
    5. Current logprobs: `cur = model(ids).logits`.
    6. Clipped PPO loss: `-min(ratio * A, clip(ratio, 1-eps, 1+eps) * A)` with `eps=0.2`.
    7. Gradient accumulate over 4–8 positions before `optimizer.step()`.
- `deliverable`: `train_on_positions(model, positions) -> model` with finite loss.
- `verify`: Run 3 positions on CPU with tiny model, confirm loss is finite, advantage has nonzero std.
- `escalate_if`: `disable_adapter_layers` not available on the PEFT version in use.

---

### S11 — Critical-path tests

- `assignee`: sonnet
- `depends_on`: [S7]
- `goal`: Pytest suite for rewards, prompts, env, curriculum, PGN.
- `files`: `tests/test_core.py`, `tests/conftest.py`
- `inputs`: [implemntation.md Step 21](implemntation.md).
- `do`:
  - `conftest.py`: `@pytest.fixture mock_analyst` — returns a stub object with `.analyse()` returning a fake score.
  - `test_core.py`:
    - `test_reward_format_valid_san`, `test_reward_format_chatty`, `test_reward_format_castling`.
    - `test_reward_legality_legal`, `test_reward_legality_illegal`.
    - `test_reward_strategic_best_is_zero` (with mock), `test_reward_strategic_worse_is_negative`.
    - `test_prompt_arm1_no_pgn_section`, `test_prompt_arm2_has_pgn`.
    - `test_env_plays_full_game` (with mock Stockfish).
    - `test_env_200_move_cap`.
    - `test_pgn_truncation_keeps_recent`.
    - `test_curriculum_advances_at_threshold`, `test_curriculum_no_premature_advance`.
  - Live-Stockfish tests marked `@pytest.mark.stockfish`, skipped if binary missing.
- `deliverable`: `pytest tests/ -v` passes on CPU (with `@pytest.mark.stockfish` skipped).
- `verify`: Paste pytest output.
- `escalate_if`: (global only)

---

### S12 — Timing benchmark (critical gate)

- `assignee`: sonnet
- `depends_on`: [S9]
- `goal`: Measure games/hour on Kaggle T4. Gate for full training.
- `files`: `scripts/timing_benchmark.py` (optional)
- `inputs`: Running `train.py` from S9.
- `do`: Run 5 full training iterations (50 games total) on Kaggle T4. Record wall-clock. Compute games/hour. If <15/hr, apply in order:
  1. `num_generations` 4 → 2
  2. `stockfish.analyst_time` 0.1 → 0.05
  3. `max_completion_length` 10 → 6
- `deliverable`: JSON report `{games_per_hour, per_move_time_ms, per_iteration_s, applied_levers}`.
- `verify`: Paste the JSON.
- `escalate_if`: <10 games/hr even after all 3 levers applied.

---

### S13 — Kaggle notebook

- `assignee`: sonnet
- `depends_on`: [S9]
- `goal`: Self-contained Kaggle entrypoint that survives 9-hour session timeouts.
- `files`: `notebooks/kaggle_training.ipynb`
- `inputs`: [implemntation.md Step 19](implemntation.md).
- `do`: Cells in order:
  1. `!nvidia-smi`.
  2. `!pip install -q -r requirements.txt && pip install -q unsloth`.
  3. Download Stockfish binary to `/kaggle/working/stockfish/`.
  4. Clone repo + `!pip install -e .` (or symlink src).
  5. W&B login: `wandb.login(key=UserSecretsClient().get_secret("WANDB_API_KEY"))`.
  6. HF login via Kaggle Secrets.
  7. Resume from HF Hub if prior checkpoint exists; else start fresh.
  8. `python -m chess_rl.train --config config.yaml` (or inline call).
  9. Push final checkpoint to HF Hub.
- `deliverable`: Notebook runs cells 1–6 to completion on Kaggle.
- `verify`: Paste Kaggle execution log of cells 1–6.
- `escalate_if`: Unsloth install fails on Kaggle's CUDA; HF Hub push 403s.

---

### S14 — Two-arm launch

- `assignee`: sonnet
- `depends_on`: [S12, S13]
- `goal`: Launch Arm 1 (fen_only) and Arm 2 (fen_pgn) under identical conditions.
- `files`: `scripts/run_training.py` or dual `config_arm1.yaml` / `config_arm2.yaml` overlays
- `inputs`: `config.yaml`; W&B project from S9.
- `do`:
  - Start from same base checkpoint, same seed=42.
  - Only difference: `training.arm`.
  - Two separate W&B runs in the same project, named `arm1-fen_only` and `arm2-fen_pgn`.
  - Target: ~180 games per Kaggle session, 3 sessions per arm.
- `deliverable`: Both arms have ≥1 session of training logged to W&B, with checkpoints pushed.
- `verify`: W&B run URLs for both arms + HF Hub checkpoint URLs.
- `escalate_if`: After 100 games an arm has <50% legality rate — trigger S15.

---

### S15 — Pretrain decision (conditional, Opus-only)

- `assignee`: **opus** — strategic decision, branches the project.
- `depends_on`: [S14] **only if legality <50% after 100 games**
- `goal`: Choose between Plan A (reuse Chess-R1 code), Plan B (local SFT warmup), or Plan C (skip).

**Time estimates (answer to user's feedback question):**

| Plan | What | Time on T4 | Risk |
|---|---|---|---|
| **A** — Reuse Chess-R1 pretrain | Clone https://github.com/krafton-ai/Chess-R1 , port their pretrain code (~3M Lichess puzzles) to Unsloth + T4 | **~15-40 GPU-hrs** (they used multi-GPU; single-T4 extrapolation). Requires 2-5 Kaggle sessions. Plus ~4 hrs porting. | Distributed-training code may not trivially downsize; dataset download large. |
| **B** — Local SFT warmup (recommended) | 5k (FEN, Stockfish-best-move) pairs → Unsloth `SFTTrainer` 500–1000 steps | Dataset gen: **~30-45 min** (Stockfish 0.1s × 5000 + overhead). SFT train: **~1-2 hrs**. Total: **~1.5-2.5 hrs**. | Low. Data generation is embarrassingly parallel. |
| **C** — Skip | Accept slower RL convergence, or concede it may never converge. | 0 hrs | May invalidate the comparison if base model can't output legal SAN. |

- `do`:
  1. Read Chess-R1 README + `train.py`; assess porting cost honestly (1 hour budget for this eval).
  2. If Plan A is ≤1 Kaggle session of porting plus ≤2 sessions of training: go with A.
  3. Else: go with B (default). Implement `src/chess_rl/sft_warmup.py`:
     - Generate 5k pairs: random games to random depths, then Stockfish best-move as label.
     - Unsloth `SFTTrainer`, max_seq_length=512, 500–1000 steps, batch=4.
     - After: resume RL from SFT checkpoint. Reset GRPO ref model to SFT checkpoint.
  4. If neither A nor B is feasible (compute exhausted): go with C, document in write-up.
- `deliverable`: Decision document (added to `steps.md` as appendix) + if A or B, a working checkpoint.
- `verify`: After SFT, run 50 random positions → legality rate must be >60%.
- `escalate_if`: N/A (this step IS the escalation).

---

### S16 — Evaluation harness

- `assignee`: sonnet
- `depends_on`: [S14]
- `goal`: Elo ladder evaluation of base, Arm 1, Arm 2.
- `files`: `src/chess_rl/eval.py`, `scripts/run_evaluation.py`
- `inputs`: [implemntation.md Step 20](implemntation.md).
- `do`:
  - For each model in `[base_no_lora, arm1_checkpoint, arm2_checkpoint]`:
    - For each Elo in `[400, 600, 800, 1000, 1200, 1400]`:
      - Play 50 games (25 White, 25 Black). Greedy (`temperature=0`).
      - Record win/draw/loss, avg centipawn loss, avg length.
  - Save results to `results.json`.
  - Elo ceiling = highest level with win rate > 0.5.
  - No GRPO updates; pure inference.
- `deliverable`: `results.json` + a Markdown table comparing three models across six Elo levels.
- `verify`: Paste the table.
- `escalate_if`: Base model (no LoRA) cannot be loaded without adapter — fix the load path.

---

### S17 — Failure analysis + write-up

- `assignee`: **opus** — qualitative synthesis.
- `depends_on`: [S16]
- `goal`: Categorize 20-30 losses near Elo ceiling; compare arms; draft write-up.
- `files`: `writeup/analysis.md`, `writeup/paper.md`
- `inputs`: Games from S16, W&B logs.
- `do`: Per [implemntation.md Step 23](implemntation.md) and [plan.md §12](plan.md).
- `deliverable`: Draft paper / blog with learning curves, Elo chart, error categorization, arm comparison.
- `verify`: Opus self-review.
- `escalate_if`: N/A.

---

## Appendix — Decision log

_Opus appends entries here as S4, S9, S10, S15 are resolved. Format:_

```
[YYYY-MM-DD] S<id>: <short decision> — <one-line reason>
```
