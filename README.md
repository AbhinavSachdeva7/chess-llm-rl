# Chess LLM-RL: Teaching Language Models to Play Chess via Full-Game Reinforcement Learning

A reinforcement learning pipeline that trains small open-source LLMs to play complete games of chess using Group Relative Policy Optimization (GRPO), designed to run on a single consumer GPU.

---

## Background: What Chess-R1 Found

[Chess-R1 (Hwang et al., 2025)](https://arxiv.org/abs/2507.00726) is the closest prior work to this project. They trained Qwen2.5 and Llama3.1 models to predict the best next move in chess using GRPO, with rewards from a pre-trained action-value network. Their key findings:

- Dense rewards (from a neural reward model) substantially outperform sparse binary rewards
- All models plateau at **25–30% puzzle accuracy** — well below the 60–80% accuracy of 1800-Elo engines
- Adding chain-of-thought reasoning traces from OpenAI o3 before RL training did **not** overcome the plateau
- LLMs score **0% on board-state comprehension tasks** (predicting the resulting FEN after a sequence of moves), suggesting they lack a reliable internal chess simulator
- Their failure analysis concludes: the plateau stems from *insufficient chess domain knowledge in pretraining*, and RL alone cannot overcome impoverished domain knowledge

Their conclusion is essentially pessimistic: if the model doesn't already understand chess deeply from pretraining, RL cannot instill that understanding from scratch.

---

## Hypothesis

We believe Chess-R1's conclusion conflates two distinct variables: the *training paradigm* (static puzzles) and the *model's chess knowledge*. Their experiment trains on isolated position-action pairs — snapshots with no before or after. Even if a model predicts the right move in a puzzle, it never experiences what happens next.

Our hypothesis is that the plateau is partly a property of the **static puzzle paradigm itself**, not solely of domain knowledge limits. A model trained on full games must learn that its move choices have consequences that unfold over the next 10, 20, 50 moves. This forces a qualitatively different kind of learning: not pattern-matching a position to a known tactical motif, but understanding *why* a move is good in the context of a specific game trajectory.

**Experimental design:** Two arms, identical in every way except prompt format.

- **Arm 1 (FEN-only):** Model sees only the current board position (replicates Chess-R1's prompt format)
- **Arm 2 (FEN+PGN):** Model sees the current board *plus* the full PGN history of the ongoing game

If the FEN+PGN arm significantly outperforms FEN-only, the trajectory context is doing meaningful work. If both plateau similarly to Chess-R1's puzzle-based results, then the domain knowledge explanation holds and the paradigm is not the bottleneck.

---

## How This Differs from Chess-R1

### 1. Full Games vs. Static Puzzles

Chess-R1 trains on the **Lichess puzzle dataset** — 19,200 position-action pairs, each an isolated tactical position with a forced winning sequence. The model sees a FEN string, picks a move, and that's a complete training sample. There is no before or after: no opening that led to the position, no endgame that follows.

We train on **complete games played against Stockfish**. Every LLM move is part of an ongoing game. The model experiences:
- Openings and how they shape piece development
- Middlegame plans that take multiple moves to execute
- Endgames won or lost based on decisions made 40 moves earlier
- The compounding effect of repeated small inaccuracies

This is the difference between learning vocabulary from flashcards and learning it from conversation. Puzzles are flashcards.

### 2. The PGN Question: Why Chess-R1's Ablation Doesn't Answer Our Question

Chess-R1 did test PGN vs FEN in Appendix C.1 (Figure 8) and found: *"board representation choice was not critical — all variants achieved similar performance."* One might conclude PGN adds nothing. But their ablation was conducted in the **static puzzle setting** — each sample is still an isolated position, and the PGN they fed was the puzzle's preamble, not the agent's own accumulated game history.

In our full-game setting, the PGN is categorically different: it is the agent's *own* trajectory. It shows what the model played 10 moves ago, how the opponent responded, and what the current position is a consequence of. This is information that a FEN string cannot encode — FEN is a snapshot, PGN is a story.

### 3. Reward Engine: Stockfish vs. Neural Q-Network

Chess-R1 uses a **270M-parameter pre-trained action-value network** (Qθ(s,a)) as its reward model. This network was trained on 15B Stockfish-annotated state-action pairs and achieves 2299 Elo and 95.4% puzzle accuracy on its own. The reward it provides is a **predicted win probability** in [0, 1].

We use **Stockfish directly** at search depth, returning centipawn evaluations. The difference matters:

**Chess-R1's Q-network is a distillation of Stockfish.** Any approximation errors the network makes are baked into the reward signal. The network is a frozen snapshot; it cannot adapt to positions far outside its training distribution. It predicts win probability, which is a coarse scalar — a move that avoids an immediate blunder scores nearly as well as the best positional move if the win probability doesn't visibly change.

**Stockfish at full search depth is the ground truth.** Its centipawn evaluation is computed via alpha-beta search to 20+ plies, explicitly calculating concrete lines many moves ahead. It catches tactics the Q-network might miss. And critically, our reward directly measures *how far the model's move is from Stockfish's best move* in centipawns — not an absolute win probability, but a relative deviation:

```
R3 = clamp((model_score − best_score) / 100.0, −10.0, 0.0)
```

This means the reward signal is calibrated to the difficulty of the position. In a sharp tactical position where only one move maintains equality, the reward gradient is steep. In a quiet positional position, many moves score similarly. Stockfish's tree search encodes long-term consequences precisely because it is literally computing those consequences, not approximating them.

Chess-R1's reward is a soft signal that compresses all of this into a single probability. Ours preserves the granularity.

### 4. Curriculum Learning

Chess-R1 trains against a fixed static dataset. They do not play games against an adaptive opponent. We train against a **Stockfish opponent whose Elo advances with the model** — starting at 400, stepping up 200 points whenever the model holds a 60%+ win rate over 30 games. This prevents overfitting to a fixed difficulty level and ensures the model is always being challenged at the frontier of its current capability.

### 5. Hardware and Scale

Chess-R1 trained on **4× A100 80GB GPUs** for approximately 14 hours per run. We train on a **single Kaggle T4 (16 GB VRAM)**. This required 4-bit quantization, LoRA adaptation, and Hub-based checkpointing to survive session resets — but it also means this research is accessible to anyone with a free Kaggle account.

---

## Methodology

### Model

We use **Gemma 4 E2B** (a 31B-parameter instruction-tuned model) loaded in 4-bit quantization via Unsloth, with LoRA adapters (r=32, α=64) targeting the attention and MLP projection layers. Unlike Chess-R1 which primarily used base models, we use the instruction-tuned variant — Chess-R1's own ablations showed instruct models achieve better scores due to superior formatting compliance.

### RL Algorithm: GRPO

We use Group Relative Policy Optimization (GRPO) from HuggingFace TRL. For each board position encountered during gameplay:

1. Generate **8 candidate moves** at temperature 1.0
2. Score each with the reward function (see below)
3. Compute **group-relative advantages** — normalize rewards by the group's mean and standard deviation
4. Update LoRA weights via a clipped PPO-style loss with a minimal KL penalty (β=0.01)

This is the same GRPO formulation Chess-R1 uses, allowing fair comparison of our paradigm differences while controlling for the RL algorithm.

### Reward Function

Rewards are composed of four components that apply in sequence with early-exit logic:

| Component | Signal | Range |
|-----------|--------|-------|
| **R1: Format** | Valid SAN notation | +0.1 / −1.0 |
| **R2: Legality** | Move is legal on the board | +0.5 / −5.0 |
| **R3: Strategic** | Centipawn loss vs. Stockfish best move | −10.0 to 0.0 |
| **R4: Optimal bonus** | Move within 0.1 cp of best | +1.0 / 0.0 |

**R1 and R2** address the same fundamental problem Chess-R1 identified: LLMs often don't know basic chess rules and will output hallucinated or illegal moves. Chess-R1 found (Figure 6) that without explicit legal moves in the prompt, models cannot learn at all. We provide legal moves in the prompt and add explicit penalties for illegality.

**R3** is the key strategic signal. Rather than a neural network's win-probability approximation, it is Stockfish's centipawn evaluation at full search depth, measuring exactly how much worse the model's move is than the engine's best. The formula `clamp((model_score − best_score) / 100.0, −10.0, 0.0)` produces a reward of 0 for the best move and increasingly negative values for worse moves, with sensitivity proportional to the actual strategic cost.

**R4** provides a discrete bonus for moves that match or near-match Stockfish's top choice, reinforcing convergence to strong play once R3 pulls the distribution in the right direction.

If format validation fails, legality and strategic checks are skipped — the chain prioritizes teaching rule compliance before strategy.

### Prompt Format

Following Chess-R1's ablation findings (SAN outperforms UCI; legal moves in prompt are required), both arms use SAN notation and include the list of legal moves. The arms diverge only in the board state representation:

**Arm 1 (FEN-only):**
```
You are a chess engine. Position (FEN): <fen>
Legal moves: e4, d4, Nf3, ...
Your move:
```

**Arm 2 (FEN+PGN):**
```
You are a chess engine. Game so far: 1. e4 e5 2. Nf3
Position (FEN): <fen>
Legal moves: Nc6, d6, Nf6, ...
Your move:
```

The PGN in Arm 2 is always the current game's actual move history, not a static context — it grows with each turn and reflects the agent's own prior decisions.

### Opponent and Curriculum

The opponent is a local **Stockfish** instance with configurable Elo via UCI parameters. Training begins at Elo 400 and advances by 200 points whenever the model achieves a 60%+ win rate over a rolling 30-game window (minimum 20 games at the current level).

### Game Collection

At each training iteration:
- Play 8 games in parallel (batch of 4) using **greedy decoding** (temperature=0)
- Collect every LLM turn as a `(prompt, FEN, legal_moves)` training sample
- ~240–320 samples per iteration
- If the model outputs an illegal or unparseable move, a random legal fallback is played and logged

Greedy collection followed by GRPO's exploratory multi-sample generation separates *exploitation* (coherent gameplay) from *exploration* (policy update).

### Checkpointing

Checkpoints save every 24 games: LoRA adapter + tokenizer + `state.json` (games count, current Elo, win-rate window). When a HuggingFace token is present, checkpoints are pushed to the Hub — essential for surviving Kaggle session timeouts.

---

## Constraints

### Hardware

The entire pipeline was designed around **free-tier cloud GPU constraints**:

- Single NVIDIA T4 (16 GB VRAM)
- ~9-hour session budget on Kaggle
- No persistent storage between sessions

These constraints drove several design decisions:
- 4-bit quantization via Unsloth (model fits in ~10 GB)
- LoRA rather than full fine-tuning
- Hub-based checkpoint survival
- No vLLM (T4-incompatible); sequential GRPO generation instead

### Throughput

The target throughput gate is **≥15 games/hour** on a T4. Adjustment levers if throughput falls short:

1. Reduce `num_generations` from 8 → 2
2. Lower Stockfish analysis time: 0.1s → 0.05s per position
3. Reduce `max_completion_length` from 1024 → 256 tokens

### Scale

Each arm targets ~540 games across 3 Kaggle sessions. At ~30–40 LLM decisions per game, this yields ~16,000–21,000 training samples per arm.

---

## Evaluation

After training, models are evaluated on an **Elo ladder** (400 → 600 → 800 → 1000 → 1200 → 1400), 50 games per level (25 as White, 25 as Black), using greedy play. Metrics: win/draw/loss rates, average centipawn loss, game length. Three models compared: base Gemma, Arm 1 (FEN-only), Arm 2 (FEN+PGN).

Failure analysis on 20–30 losses near each arm's Elo ceiling categorizes errors as tactical blunders vs. strategic misevaluations — the same diagnostic lens Chess-R1 applied.

---

## Expected Outcomes

| Outcome | Interpretation |
|---------|----------------|
| FEN+PGN reaches Elo 200+ above FEN-only | Trajectory context is the missing ingredient; puzzle paradigm was limiting Chess-R1 |
| Both arms improve over base; FEN+PGN slight edge | Full-game helps, PGN marginal benefit; publishable result |
| Both plateau similarly to Chess-R1 (~25–30%) | Domain knowledge deficit is fundamental; paradigm is not the bottleneck |
| Model learns >90% legality, beats Stockfish 400 | Pipeline works; engineering contribution regardless of plateau |

---

## Results

This pipeline **successfully runs on a Kaggle free-tier T4 single GPU**. The 4-bit quantized Gemma 4 E2B model fits comfortably within the 16 GB VRAM budget, the training loop maintains target throughput, and the Hub-based checkpointing survives session resets without data loss.

Further results — including the full ablation comparison between the FEN-only and FEN+PGN arms, final Elo ratings, and learning curve analysis — require more compute. A multi-GPU or A100/H100 setup would enable larger batch sizes, faster GRPO rollouts, and more games per session, making it possible to run the full experiment at the scale needed to confirm or refute the trajectory-context hypothesis with statistical confidence.

---

## Setup

```bash
pip install -r requirements.txt
# Install Stockfish (Kaggle: apt install stockfish)
```

Set environment variables:
```bash
WANDB_API_KEY=...     # Weights & Biases logging
HF_TOKEN=...          # HuggingFace Hub checkpoint uploads
HF_REPO_ID=...        # Target repo for checkpoints
```

Training is launched via the Kaggle notebook (`train.ipynb`), which handles Stockfish installation, optional checkpoint resumption from the Hub, and the full GRPO loop.

---

## Stack

| Component | Library |
|-----------|---------|
| Model loading + quantization | [Unsloth](https://github.com/unslothai/unsloth) |
| LoRA | [PEFT](https://github.com/huggingface/peft) |
| GRPO | [TRL](https://github.com/huggingface/trl) |
| Chess logic | [python-chess](https://python-chess.readthedocs.io/) |
| Engine / reward oracle | [Stockfish](https://stockfishchess.org/) |
| Experiment tracking | [Weights & Biases](https://wandb.ai/) |
| Checkpoint storage | [HuggingFace Hub](https://huggingface.co/) |

---

## Reference

Hwang, D., Lee, H., Choo, J., Park, D., & Park, J. (2025). *Can Large Language Models Develop Strategic Reasoning? Post-training Insights from Learning Chess.* Workshop on Test-time Scaling and Reasoning Models at COLM 2025. [arXiv:2507.00726](https://arxiv.org/abs/2507.00726)
