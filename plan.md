# Project Plan: Teaching Strategic Reasoning to LLMs via Chess Reinforcement Learning

## Context for LLM Assistants
This document is a complete project specification for building a Reinforcement Learning pipeline that trains an open-source LLM to play full games of chess. It was developed through iterative discussion and captures all design decisions, rationale, and implementation details. If you are an LLM helping with code implementation, treat this document as the authoritative source of truth for the project's goals, architecture, and constraints.

---

## 1. Project Summary

Train a small open-source LLM (Gemma 3 or Qwen 3.5, ~3B parameters) to play full games of chess using Group Relative Policy Optimization (GRPO). The model receives both FEN (board state) and PGN (move history) as input and outputs algebraic moves. A curriculum learning approach ramps up opponent difficulty as the model improves. The project directly extends the findings of the Chess-R1 paper (Hwang et al., COLM 2025 SCALR Workshop) which showed that RL on static puzzles plateaus — this project tests whether full-game RL can push past that ceiling.

---

## 2. Core Hypothesis

Chess-R1 (Krafton AI, 2025) demonstrated that training LLMs on static chess puzzles with GRPO and dense rewards leads to a performance plateau far below expert levels. Their conclusion was that this limitation stems from insufficient chess knowledge in the pretrained model, which RL alone cannot overcome.

**Our hypothesis is that full-game play — not static puzzles — is a critical missing ingredient.** Specifically:

1. **Full-game play forces sequential accountability.** The model must live with the consequences of its own moves. A bad pawn structure on move 12 leads to worse positions (and worse Stockfish evaluations) on moves 15-22. Over many games, the model should learn that certain early decisions lead to consistently worse downstream rewards. This is fundamentally different from isolated puzzle training where each position is independent.

2. **PGN history enables trajectory conditioning.** The FEN provides the current board state, but the PGN provides a record of the model's own prior commitments. With PGN, the model can learn statistical patterns like "when I've been pushing kingside pawns, continuing that direction gets rewarded more than switching." Without PGN, every position looks like a fresh puzzle — which is exactly the failure mode we are trying to escape.

3. **Stockfish's dense evaluation already encodes strategic value.** Stockfish's centipawn score is the product of a deep search tree (20+ moves deep). When it evaluates the model's move, it implicitly assesses long-term consequences — a quiet positional move that sets up an attack in five moves will score well even though nothing tactically exciting happened. This means the reward signal already penalizes planless play without needing an explicit "plan" field in the prompt. No separate plan generation or chain-of-thought reward is needed.

**What we are NOT claiming:** We are not claiming the model will develop general reasoning or transfer strategic thinking to other domains. We claim that within the closed system of chess, repeated full-game experience plus a strong evaluative signal will push the model beyond what static puzzle training achieves. This is a testable, scoped hypothesis.

---

## 3. Relationship to Prior Work

### 3.1 Chess-R1 (Primary Reference)
- **Paper:** "Can Large Language Models Develop Strategic Reasoning? Post-training Insights from Learning Chess" (Hwang et al., 2025)
- **Venue:** Accepted at SCALR Workshop, COLM 2025
- **Code:** https://github.com/krafton-ai/Chess-R1
- **What they did:** Trained Qwen2.5-3B, Qwen2.5-7B, and Llama3.1-8B on Lichess puzzle datasets using GRPO with dense rewards from a chess-pretrained action-value network.
- **Key findings:**
  - Dense rewards outperform sparse binary rewards.
  - Sparse rewards completely fail for Qwen2.5-3B and Llama3.1-8B.
  - All models plateau far below expert levels regardless of reward type.
  - SFT on OpenAI o3 reasoning traces before RL did NOT help break the plateau (Llama3.1-8B actually got worse).
  - PGN and UCI notation did not yield improvements — BUT this was tested in the static puzzle context where PGN is meaningless (no game continuity).
  - Models cannot learn meaningfully when legal moves are not provided in the prompt.
  - SAN notation significantly outperforms UCI notation.
- **How our project extends this:** We test whether the puzzle-paradigm ceiling can be overcome by switching to full-game play with trajectory context (FEN+PGN). Chess-R1 left this question unanswered.

### 3.2 Other Related Work
- **Xiangqi-R1:** Similar approach applied to Chinese Chess with multi-stage training (SFT → strategic annotations → GRPO). Used 5M board-move pairs. Showed 18% improvement in move legality and 22% boost in analysis accuracy.
- **ChessArena:** Evaluation framework for LLM chess. Found no model could beat Maia-1100 (amateur level). Fine-tuned Qwen3-8B approached larger reasoning models.
- **MATE Dataset:** Showed that strategy + tactic annotations improve move selection by 24.2% over commercial LLMs when both are provided.

---

## 4. Experimental Design

### 4.1 Two-Arm Ablation Study

This project runs a controlled two-arm ablation to isolate the effect of trajectory context:

| Arm | Input Representation | Purpose |
|-----|---------------------|---------|
| **Arm 1: FEN-only** | Current board state as FEN string | Baseline — equivalent to the static puzzle paradigm but in a full-game setting |
| **Arm 2: FEN + PGN** | Current FEN + full PGN move history | Tests whether trajectory context helps the model leverage Stockfish's implicit strategic signal |

Both arms use identical training loops, reward functions, curriculum schedules, and compute budgets. The only variable is the input representation. Run both arms for the same number of games and compare:
- Illegal move rate over training
- Average centipawn loss over training
- Win rate against fixed Stockfish Elo levels
- Elo ceiling (highest Stockfish level where the model maintains >50% win rate)

**Even negative results are informative.** If FEN-only performs equally well, that itself is a finding worth reporting — it would mean PGN doesn't help even in a full-game context.

### 4.2 Why No Chain-of-Thought / Plan Arm

We considered and rejected adding a third arm with explicit plan generation. Reasons:
1. Stockfish cannot generate plan explanations — it works via search trees, not natural language reasoning. This creates an unsolvable reward design problem: you can reward the move but not the plan.
2. At low Elo, plans change constantly, so the plan field would likely add noise rather than signal.
3. The FEN+PGN+Stockfish combination already forms a closed loop: FEN provides the current state, PGN provides trajectory, and Stockfish's deep evaluation already rewards moves that contribute to sound long-term plans.

---

## 5. Technical Architecture

### 5.1 Hardware and Budget
- **Primary:** Kaggle Free Tier (dual T4 GPUs, ~30 hours/week)
- **Backup:** Google Colab Free Tier (T4 GPU) or RunPod/Vast.ai ($10 budget, RTX 3090/4090)
- **Constraint:** All design decisions must fit within this budget. Checkpoint aggressively (every 50 games) to survive session timeouts.

### 5.2 Software Stack
- **Training framework:** `Unsloth` (4-bit quantized training for memory efficiency)
- **RL algorithm:** GRPO via HuggingFace `trl`
- **Chess environment:** `python-chess` (board state management, move legality validation)
- **Reward oracle:** `Stockfish` (local binary, configurable Elo via UCI `Skill Level` and `UCI_LimitStrength`)
- **Model:** Gemma 3 (~3B) or Qwen 3.5 (~3B) — choose whichever has better baseline chess knowledge (see Section 6, Step 0)

### 5.3 Model Selection Criteria
Before committing to a model, test both Gemma 3 and Qwen 3.5 base models:
- Prompt each with 10-20 chess positions (FEN strings) and ask for a move.
- Measure: (a) rate of legal moves, (b) rate of syntactically valid outputs, (c) average centipawn loss of legal moves.
- Start with whichever model performs better out of the box. Better pretrained chess knowledge means faster learning and more interesting results within the compute budget.

### 5.4 Prompt Structure

#### System Prompt
```
You are a chess engine playing as {White|Black}. Given the current board state and game history, output your next move in Standard Algebraic Notation (SAN). Output ONLY the move, nothing else. Example output: Nf3
```

#### Arm 1 (FEN-only) User Prompt
```
Current position (FEN): {fen_string}
Legal moves: {comma_separated_legal_moves}
Your move:
```

#### Arm 2 (FEN+PGN) User Prompt
```
Game history (PGN): {pgn_string}
Current position (FEN): {fen_string}
Legal moves: {comma_separated_legal_moves}
Your move:
```

**Key design notes:**
- Legal moves are provided in the prompt. Chess-R1 found that LLMs cannot learn anything meaningful without them.
- SAN notation is used, not UCI. Chess-R1 found SAN substantially outperforms UCI, likely due to pretraining data distribution.
- No chain-of-thought output. The model outputs only the move to maximize training speed and save compute.
- The legal moves list serves as both a constraint hint and a way to reduce the action space the model needs to search over.

### 5.5 Reward Functions

Three reward components, applied to each of the N candidate moves generated by GRPO:

#### R1: Format/Syntax Reward
```python
def reward_format(response: str, legal_moves: list[str]) -> float:
    """
    Checks if the model output is a clean, single move string
    without conversational text or extra tokens.
    """
    response = response.strip()
    # Check if response is a single token matching move pattern
    if re.match(r'^[KQRBN]?[a-h]?[1-8]?x?[a-h][1-8](=[QRBN])?[+#]?$', response):
        return 0.1  # Valid format
    if response == "O-O" or response == "O-O-O":
        return 0.1  # Castling
    return -1.0  # Contains conversational text or invalid format
```

#### R2: Legality Reward
```python
def reward_legality(response: str, board: chess.Board) -> float:
    """
    Checks if the move is legal in the current position.
    Uses python-chess for validation.
    """
    try:
        move = board.parse_san(response.strip())
        if move in board.legal_moves:
            return 0.5  # Legal move
    except (chess.InvalidMoveError, chess.IllegalMoveError, chess.AmbiguousMoveError, ValueError):
        pass
    return -5.0  # Illegal move — heavy penalty
```

#### R3: Strategic Reward (Dense, Stockfish-based)
```python
def reward_strategic(response: str, board: chess.Board, engine: chess.engine.SimpleEngine, time_limit: float = 0.1) -> float:
    """
    If the move is legal, evaluate its quality relative to the best move.
    Returns a normalized score based on centipawn difference.
    
    Stockfish's evaluation already encodes long-term strategic value
    because it searches 20+ moves deep. This implicitly rewards
    moves that contribute to sound plans and penalizes planless play.
    """
    try:
        move = board.parse_san(response.strip())
        if move not in board.legal_moves:
            return 0.0  # Not legal, no strategic reward
        
        # Evaluate best move
        best_info = engine.analyse(board, chess.engine.Limit(time=time_limit))
        best_score = best_info["score"].relative.score(mate_score=10000)
        
        # Evaluate the model's move
        board.push(move)
        model_info = engine.analyse(board, chess.engine.Limit(time=time_limit))
        model_score = -model_info["score"].relative.score(mate_score=10000)  # Negate because perspective flipped
        board.pop()
        
        # Centipawn delta, normalized
        delta = (model_score - best_score) / 100.0  # Will be <= 0
        return max(delta, -10.0)  # Clamp to prevent extreme penalties
        
    except Exception:
        return 0.0
```

#### Combined Reward
```python
def compute_reward(response: str, board: chess.Board, engine, legal_moves: list[str]) -> float:
    r1 = reward_format(response, legal_moves)
    if r1 < 0:
        return r1  # Bad format, don't bother checking further
    
    r2 = reward_legality(response, board)
    if r2 < 0:
        return r1 + r2  # Illegal move
    
    r3 = reward_strategic(response, board, engine)
    return r1 + r2 + r3  # Format OK + Legal + Strategic quality
```

### 5.6 GRPO Configuration
- **Candidates per position (N):** 4 (reduced from typical 8 to save compute)
- **GRPO update:** Standard group-relative advantage. Moves scoring above the group mean get positive advantage; below get negative. Model weights update accordingly.
- **Selection:** After GRPO update, the highest-scoring legal candidate move is pushed to the board for actual game play. If no legal move is generated, a random legal move is played (and this is tracked as a failure metric).

### 5.7 Game Loop (Full-Game Training)

```
For each training game:
    1. Initialize a new chess.Board()
    2. Set Stockfish opponent to current curriculum Elo level
    3. Randomly assign LLM as White or Black (50/50)
    4. While game is not over (and move count < 200):
        a. If it's the LLM's turn:
            i.   Construct prompt (FEN-only or FEN+PGN depending on arm)
            ii.  Include list of legal moves in prompt
            iii. GRPO generates N=4 candidate moves
            iv.  Score all candidates with reward functions
            v.   GRPO updates model weights
            vi.  Select best legal candidate; push to board
            vii. If no legal candidate, play random legal move (log this)
        b. If it's Stockfish's turn:
            i.   Stockfish selects move at current Elo level
            ii.  Push to board
        c. Update PGN history
    5. Record game result (win/loss/draw), total moves, average centipawn loss, illegal move rate
    6. Every 50 games: save model checkpoint
```

---

## 6. Implementation Steps

### Step 0: Model Selection Benchmark (1-2 hours)
- Prompt Gemma 3 and Qwen 3.5 base models with 20 chess positions.
- Measure legal move rate, syntax validity, and average centipawn loss.
- Select the better-performing model.
- **Deliverable:** A short log showing which model was selected and why.

### Step 1: Environment Setup (2-3 hours)
- Set up Kaggle notebook with Unsloth, trl, python-chess, and Stockfish binary.
- Write the chess environment class that maintains board state, tracks PGN, generates prompts, and interfaces with Stockfish.
- Write the reward functions (R1, R2, R3) and unit test them against known positions.
- **Deliverable:** A working `chess_env.py` and `rewards.py` that can be imported by the training script.

### Step 2: Training Loop Integration (3-4 hours)
- Integrate the chess environment with GRPO via Unsloth/trl.
- Implement the full game loop (Section 5.7).
- Implement curriculum logic: track win rate over rolling window; if win rate > 60% over last 30 games, increase Stockfish Elo by 100-200 points.
- Implement checkpointing every 50 games.
- Implement metric logging: illegal move rate, average centipawn loss, win rate, current Elo level, all over training steps.
- **Deliverable:** A working `train.py` that can run end-to-end on a single game.

### Step 3: Timing Benchmark (1 hour)
- Run 5-10 complete games and measure wall-clock time.
- Extrapolate: how many games can we run in a 20-hour Kaggle session?
- **Critical gate:** If the extrapolation shows fewer than 200 games per arm, consider reducing candidate count to 2 or shortening Stockfish analysis time. Do NOT proceed to full training without this benchmark.
- **Deliverable:** A timing report with games-per-hour estimate.

### Step 4: Full Training Run (15-20 hours per arm)
- Run Arm 1 (FEN-only) for as many games as compute allows.
- Run Arm 2 (FEN+PGN) for the same number of games.
- Monitor metrics on rolling basis. If one arm clearly dominates early, you can make a judgment call to allocate remaining compute to the more interesting arm.
- **Deliverable:** Two trained model checkpoints and complete training logs.

### Step 5: Elo Evaluation (2-3 hours)
- For each trained model (and the base model as a third baseline):
  - Play 50 games against Stockfish at each Elo level: 400, 600, 800, 1000, 1200, 1400.
  - Record win rate at each level.
  - Identify the Elo ceiling (highest level where win rate > 50%).
- **Deliverable:** Elo curve chart for all three models (base, Arm 1, Arm 2).

### Step 6: Failure Analysis (2-3 hours)
- Take games where the trained model lost, especially near its Elo ceiling.
- Run the game PGNs through Stockfish analysis.
- Categorize errors: tactical blunders (missed captures, hung pieces) vs. strategic misevaluations (bad pawn structure, poor piece placement, planless play).
- This tells you what kind of chess understanding the model did and didn't develop.
- **Deliverable:** A qualitative error analysis section for the write-up.

### Step 7: Write-up and Publication (3-5 hours)
- Frame as: "Chess-R1 showed static puzzle RL plateaus. We tested whether full-game RL with trajectory context can overcome this limitation."
- Include: learning curves (illegal move rate, centipawn loss, win rate over training), Elo evaluation curves, ablation comparison (FEN-only vs FEN+PGN), failure analysis.
- Target: arXiv preprint + LinkedIn post with charts. Optionally submit to a workshop (NeurIPS, ICML, or COLM workshop on reasoning/RL/games).
- **Deliverable:** A 6-10 page paper or technical blog post.

---

## 7. Key Metrics to Track

| Metric | What It Tells You | Track Every |
|--------|-------------------|-------------|
| Illegal move rate | Is the model learning basic chess rules? | Game |
| Format compliance rate | Is the model outputting clean move strings? | Game |
| Average centipawn loss | How far are the model's moves from optimal? | Move |
| Win rate (rolling 30 games) | Is the model improving at the current Elo? | Game |
| Current curriculum Elo | How far has the model progressed? | Curriculum step |
| No-legal-move fallback rate | How often does GRPO fail to produce any legal move? | Game |
| Game length (moves) | Is the model surviving longer / playing full games? | Game |

---

## 8. Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| **Kaggle session timeout** | Checkpoint every 50 games. Save model weights, optimizer state, curriculum Elo, and game counter. Resume seamlessly. |
| **Compute too slow** | Run timing benchmark (Step 3) before committing. Reduce candidates from 4 to 2 if needed. Reduce Stockfish analysis time to 0.05s. |
| **Model learns Stockfish-specific exploits** | Test against different Stockfish configurations (varied time controls, contempt settings, randomized opening books) during evaluation. |
| **PGN grows too long** | By move 30+, PGN may be 200+ tokens. If this causes OOM or speed issues, truncate to last 15-20 moves as a sliding window. |
| **Model never learns legal moves** | Chess-R1 found legal move lists in the prompt are essential. Our prompt includes them. If illegal move rate stays above 80% after 100 games, check prompt formatting and reward magnitudes. |
| **Full-game RL shows no improvement over puzzles** | This is a valid negative result. Report it honestly — it means the puzzle plateau is a fundamental limitation, not a training paradigm issue. |

---

## 9. Overfitting Considerations

Playing hundreds of games against a single Elo level is unlikely to cause memorization-style overfitting because:
- Chess has enormous state space diversity. Even at 400 Elo, every game generates a unique trajectory.
- The model sees thousands of unique board states across hundreds of games.
- Stockfish doesn't play deterministically (especially at low Elo with randomization).

The real risk is **policy narrowing** — the model learns to exploit specific weaknesses of low-Elo Stockfish (e.g., "just grab hanging pieces") rather than developing sound general play. The curriculum approach directly mitigates this: as Elo ramps up, exploit strategies stop working and the model must develop deeper understanding.

**Curriculum threshold:** Advance to the next Elo level when win rate exceeds 60% over the last 30 games. Allow 20-30 games of exposure at each new level before evaluating. Do NOT advance based on a single win.

---

## 10. Success Criteria

| Outcome | What It Means |
|---------|---------------|
| **Strong positive:** Full-game FEN+PGN model reaches Elo ceiling 200+ points above FEN-only baseline | Full-game play with trajectory context teaches something static puzzles cannot. Major result. |
| **Moderate positive:** Both arms improve over base model; FEN+PGN has slight edge | Full-game play helps, but PGN's contribution is marginal. Still publishable. |
| **Informative negative:** Both arms plateau at similar Elo ceiling as Chess-R1 found | The puzzle plateau is fundamental, not paradigm-dependent. Valuable negative result. |
| **Minimum viable:** Model learns to play legal chess at >90% legality rate and beats Stockfish 400 | The pipeline works. Even this alone is a useful engineering contribution. |

---

## 11. Timeline

| Phase | Duration | Constraint |
|-------|----------|-----------|
| Step 0: Model selection | Day 1 | Before spending any GPU time |
| Steps 1-2: Environment + training loop | Days 1-3 | Can be done partly on CPU |
| Step 3: Timing benchmark | Day 3 | MUST complete before full training |
| Step 4: Full training | Days 4-14 | Limited by Kaggle weekly GPU quota |
| Steps 5-6: Evaluation + analysis | Days 14-16 | Can overlap with final training |
| Step 7: Write-up | Days 16-20 | Timebox. Don't chase perfection. |

**Hard rule: Timebox the entire project to 3 weeks maximum.** The value is in the demonstrated capability and clarity of write-up, not in squeezing out another 50 Elo points. Ship it and move on.

---

## 12. Presentation Strategy (Job Search)

This project serves as a portfolio piece for ML/AI job applications. Key framing:

**LinkedIn post angle:** "I ran a controlled experiment testing whether full-game experience helps LLMs learn long-term chess planning via GRPO — here are the learning curves, here's what worked, here's what didn't."

**What the project demonstrates to employers:**
- RL implementation beyond textbook definitions (built a working training loop)
- Reward function engineering (a core skill for alignment and applied ML roles)
- Experiment design with controlled ablations (thinks like a researcher)
- Working under real compute constraints (engineering maturity)
- Building on published work with proper scientific framing

**Include in the post:** Elo evaluation chart comparing all models, a few example game PGNs annotated with interesting moments, link to code repository. Keep the write-up clear and results-focused.
