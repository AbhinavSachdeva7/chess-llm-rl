# Positive-Reinforcement Training Redesign

**Date:** 2026-04-23  
**Status:** Approved  
**Scope:** `src/chess_rl/rewards.py`, `src/chess_rl/prompts.py`, `src/chess_rl/train.py`, `config.yaml`

---

## Motivation

The current training apparatus uses a punishment-dominant signal: the best possible move scores `+0.6` (nearly identical to a merely-legal move), overlong responses receive `−1.0`, and there is no KL anchor or entropy regularisation. GRPO struggles to find a clear gradient direction because the ceiling is flat and the reward landscape near the top is compressed.

This redesign shifts to a positive-reinforcement design: the optimal move receives a clear spike above other legal moves, overlong responses are silently dropped (no negative signal), thinking is enabled, and the KL prior is restored.

---

## Design Decisions Summary

| Parameter | Before | After |
|---|---|---|
| Token budget | 64 | 1024 |
| Thinking | Off (`enable_thinking=False`) | On (`enable_thinking=True`) |
| Output format | `<move>SAN</move>` | `<think>...</think><move>SAN</move>` |
| Overlong handling | `−1.0` reward | Silent discard, retry up to 3× |
| KL penalty (`beta`) | 0.0 | 0.01 |
| Entropy coeff | None | Not added (not native to TRL; KL handles collapse) |
| `num_generations` | 4 | 8 |
| Early exits in reward | Yes | No — all four components always evaluated |
| Optimal move bonus | None (delta ceiling = 0) | `+1.0` when delta > −0.1 |

---

## Section 1: Format Schema

### Output format

```
<think>
[model reasoning — any length up to budget]
</think>
<move>SAN</move>
```

Gemma 4 uses `<think>`/`</think>` natively when `enable_thinking=True`. The `<move>` tag is kept (not renamed to `<answer>`) to preserve compatibility with existing tests, `extract_san`, and `apply_llm_move`.

### Prompt change (`prompts.py`)

The instruction line in `build_messages` changes to:

> *"Think step by step inside `<think>...</think>`, then respond with your move in `<move>SAN</move>`. Example: `<think>The queen is attacked.</think><move>Nf3</move>`."*

The existing prohibition on "long reasonings, explanations, and analysis" is removed — it contradicts `enable_thinking=True`.

### Tokenizer change (`prompts.py::apply_template`)

```python
# Before
enable_thinking=False

# After
enable_thinking=True
```

### Format validation (`rewards.py::reward_format`)

Validates three conditions in order:
1. `<think>` is present
2. `</think>` is present
3. `<move>SAN</move>` is present
4. Tag order: `<think>` before `</think>` before `<move>`

All must pass to receive the format reward.

---

## Section 2: Reward Architecture

All four components are **always evaluated** — no early exits. R3 and R4 return `0.0` when the move is unparseable (R2 already captures that penalty). The total reward is always `R1 + R2 + R3 + R4`.

### R1 — Format `reward_format(response) -> float`

| Condition | Value |
|---|---|
| All tags present and correctly ordered | `+0.2` |
| Any tag missing or out of order | `−1.0` |

### R2 — Legality `reward_legality(response, board) -> float`

| Condition | Value |
|---|---|
| Move parses and is in `board.legal_moves` | `+0.5` |
| Parse failure or illegal move | `−5.0` |

### R3 — Strategic delta `reward_strategic(response, board, analyst) -> float`

Unchanged Stockfish centipawn delta logic. Range: `[−10.0, 0.0]`.  
Returns `0.0` if the move cannot be parsed (no double-penalty on top of R2).

```
delta = (model_score - best_score) / 100.0
return max(-10.0, min(0.0, delta))
```

### R4 — Optimal bonus `reward_optimal(delta) -> float`

New component. Returns `+1.0` if the move is within 10 centipawns of the Stockfish best move (`delta > −0.1`), otherwise `0.0`. Computed from the delta already produced by R3 — no extra engine call.

```python
def reward_optimal(delta: float) -> float:
    return 1.0 if delta > -0.1 else 0.0
```

### Total reward ranges

| Situation | R1 | R2 | R3 | R4 | Total |
|---|---|---|---|---|---|
| Optimal move, good format | +0.2 | +0.5 | 0.0 | +1.0 | **+1.7** |
| Good move −50cp, good format | +0.2 | +0.5 | −0.5 | 0.0 | **+0.2** |
| Blunder −300cp, good format | +0.2 | +0.5 | −3.0 | 0.0 | **−2.3** |
| Illegal move, good format | +0.2 | −5.0 | 0.0 | 0.0 | **−4.8** |
| Bad format, legal move | −1.0 | +0.5 | delta | bonus | varies |
| Bad format, illegal move | −1.0 | −5.0 | 0.0 | 0.0 | **−6.0** |

The optimal move now clearly spikes above all other legal moves (`+1.7` vs `+0.2` for a good-but-not-best move). GRPO advantage computation will strongly prefer it.

---

## Section 3: Overlong Handling — Retry Loop

### Responsibility split

| Who handles it | Condition | Outcome |
|---|---|---|
| **Retry loop** (in `train.py`) | Response ≥ 1024 tokens | Retry up to 3×; drop position if all fail |
| **R1** | Within limit, missing/bad tags | `−1.0` format penalty |
| **R2** | Within limit, illegal/unparseable move | `−5.0` legality penalty |
| **R3/R4** | Move unparseable | `0.0` — no evaluation, no penalty |

### Retry loop logic

The retry loop wraps completion generation in `train.py`. For each position, up to 3 attempts are made per completion slot to fill `num_generations=8` valid (non-overlong) completions. If a position cannot be filled after `MAX_RETRIES=3` attempts per slot, it is dropped silently — no reward assigned, no gradient contribution.

```
MAX_RETRIES = 3
for each position:
    valid_completions = []
    for attempt in range(MAX_RETRIES):
        completion = generate(prompt, max_new_tokens=1024)
        if token_length(completion) < 1024:
            valid_completions.append(completion)
        if len(valid_completions) == num_generations:
            break
    if len(valid_completions) < num_generations:
        drop position   # skip entirely
    else:
        add to batch
```

This is philosophically consistent: the model is never told "being long is wrong." It simply never receives reward for overlong responses. Only complete, well-formed responses influence the gradient.

---

## Section 4: Config Changes

### `config.yaml`

```yaml
model:
  max_seq_length: 4096          # was 2048; needed for 1024-token completions + prompt

grpo:
  num_generations: 8            # was 4; more rollouts = better advantage estimate
  max_completion_length: 1024   # was 64
  beta: 0.01                    # was 0.0; restores KL anchor to reference model
  # entropy_coeff: not added    # not native to TRL; KL + diverse positions sufficient
```

### `train.py` — `GRPOConfig`

```python
GRPOConfig(
    max_completion_length=grpo_cfg["max_completion_length"],  # 1024
    num_generations=grpo_cfg["num_generations"],              # 8
    beta=grpo_cfg["beta"],                                    # 0.01
    max_grad_norm=1.0,                                        # unchanged
    fp16=True,                                                # unchanged
    ...
)
```

---

## Files Changed

| File | Change |
|---|---|
| `config.yaml` | `max_seq_length`, `max_completion_length`, `num_generations`, `beta` |
| `src/chess_rl/prompts.py` | Instruction text, `enable_thinking=True` |
| `src/chess_rl/rewards.py` | `reward_format` (think tag validation), remove early exits, add `reward_optimal` |
| `src/chess_rl/train.py` | Retry loop around completion generation |

---

## What Is Not Changed

- `env.py` — game loop, Stockfish opponent, `apply_llm_move` all unchanged
- `curriculum.py` — win-rate tracking and Elo advancement unchanged
- `model.py` — model loading unchanged
- Test structure — existing tests will need updating for new format but test logic is sound

---

## Out of Scope

- Language detection reward (skipped — model is English instruct, overhead not justified)
- Explicit entropy coefficient (not native to TRL `GRPOConfig`)
- Switching `<move>` to `<answer>` (unnecessary churn)
- Q-value tables (live Stockfish delta is superior for this use case)
