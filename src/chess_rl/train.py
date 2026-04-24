"""GRPO training loop for chess-llm-rl.

Entry point: python -m chess_rl.train --config config.yaml [--resume path]
Verify requires CUDA (Unsloth + bitsandbytes). Runs on Kaggle T4.
"""
from __future__ import annotations

import argparse
import atexit
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import chess
import yaml

from .curriculum import Curriculum
from .env import ChessEnvironment, GameMetrics
from .prompts import apply_template, build_messages
from .rewards import compute_reward, get_analyst
from .stockfish import StockfishManager


# ---------------------------------------------------------------------------
# Batched + parallel generation
# ---------------------------------------------------------------------------

def _batch_generate(model, tok, prompts: list[str], max_new_tokens: int) -> list[str]:
    """Greedy-decode a batch of prompts in one forward pass.

    Left-pads so completions are right-aligned. Returns one decoded string per prompt.
    """
    import torch
    orig_side = tok.padding_side
    tok.padding_side = "left"
    try:
        inputs = tok(
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        ).to(model.device)
        with torch.inference_mode():
            out = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tok.pad_token_id,
            )
        prompt_len = inputs["input_ids"].shape[1]
        return [
            tok.decode(out[i][prompt_len:], skip_special_tokens=True).strip()
            for i in range(len(prompts))
        ]
    finally:
        tok.padding_side = orig_side


# Reusable thread pool: PyTorch releases the GIL during CUDA work, so threads
# on different devices run genuinely in parallel. No shared mutable state inside
# _batch_generate, so no locks needed.
_EXECUTOR: Optional[ThreadPoolExecutor] = None


def _get_executor(n_workers: int) -> ThreadPoolExecutor:
    global _EXECUTOR
    if _EXECUTOR is None:
        _EXECUTOR = ThreadPoolExecutor(max_workers=n_workers)
        atexit.register(lambda: _EXECUTOR.shutdown(wait=False))
    return _EXECUTOR


def _parallel_generate(
    models: list,
    tok,
    prompts_per_model: list[list[str]],
    max_new_tokens: int,
) -> list[list[str]]:
    """Dispatch _batch_generate to each model concurrently. Result order matches input."""
    if len(models) == 1:
        return [_batch_generate(models[0], tok, prompts_per_model[0], max_new_tokens)]
    executor = _get_executor(len(models))
    active = [(g, m, ps) for g, (m, ps) in enumerate(zip(models, prompts_per_model)) if ps]
    futures = {g: executor.submit(_batch_generate, m, tok, ps, max_new_tokens)
               for g, m, ps in active}
    return [futures[g].result() if g in futures else [] for g in range(len(models))]


# ---------------------------------------------------------------------------
# Inference replica (extra GPUs)
# ---------------------------------------------------------------------------

def load_inference_replica(primary_model, device: str):
    """Create an inference-only replica of primary on `device`.

    Saves primary's PEFT adapter to a temp dir, loads a plain HF base + PEFT
    adapter from it on `device`. Replica is eval-mode and frozen — used only
    by _batch_generate during collection.
    """
    import tempfile
    import torch
    from transformers import AutoModelForCausalLM
    from peft import PeftModel, PeftConfig

    with tempfile.TemporaryDirectory() as tmpdir:
        primary_model.save_pretrained(tmpdir)
        peft_cfg = PeftConfig.from_pretrained(tmpdir)
        base = AutoModelForCausalLM.from_pretrained(
            peft_cfg.base_model_name_or_path,
            torch_dtype=torch.float16,
        ).to(device)
        replica = PeftModel.from_pretrained(base, tmpdir, is_trainable=False).to(device)
    replica.eval()
    for p in replica.parameters():
        p.requires_grad = False
    return replica


def sync_inference_replica(primary, replica) -> None:
    """Copy primary's LoRA weights into replica in place."""
    import torch
    primary_state = primary.state_dict()
    adapter_state = {k: v for k, v in primary_state.items() if "lora" in k.lower()}
    if not adapter_state:
        return
    replica_device = next(replica.parameters()).device
    adapter_state = {k: v.to(replica_device) for k, v in adapter_state.items()}
    replica.load_state_dict(adapter_state, strict=False)


def resolve_inference_devices(cfg: dict) -> list[str]:
    """Return configured inference devices, filtered to those actually available."""
    import torch
    devices = cfg.get("grpo", {}).get("inference_devices")
    if not devices:
        return ["cuda:0"] if torch.cuda.is_available() else ["cpu"]
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    resolved = []
    for d in devices:
        if d == "cpu":
            resolved.append(d)
        elif d.startswith("cuda:") and int(d.split(":")[1]) < n_gpus:
            resolved.append(d)
    return resolved or (["cuda:0"] if n_gpus > 0 else ["cpu"])


# ---------------------------------------------------------------------------
# Experience collection — parallel games across all inference models
# ---------------------------------------------------------------------------

def collect_game_experience(models: list, tok, stockfish: StockfishManager,
                            cfg: dict, n_games: int, curriculum: Curriculum):
    """Play n_games in parallel waves across all provided inference models.

    Each step, every LLM-turn environment is batched with others on the same
    GPU and generated in one forward pass. If len(models) > 1, the batches
    run concurrently on different GPUs via a thread pool.

    Overlong responses (>= max_collection_length tokens) are retried up to
    max_retries times; positions that exhaust all retries are dropped and
    the game advances with a random legal move.
    """
    from datasets import Dataset

    grpo = cfg["grpo"]
    max_new = grpo.get("max_collection_length", grpo["max_completion_length"])
    max_retries = grpo.get("max_retries", 3)
    per_gpu_bs = grpo.get("collection_batch_size", 4)
    n_gpus = len(models)
    wave_size = per_gpu_bs * n_gpus

    samples: list[dict] = []
    per_game_metrics: list[GameMetrics] = []
    per_game_results: list[str] = []

    stockfish.set_opponent_elo(curriculum.elo)

    remaining = n_games
    while remaining > 0:
        wave_count = min(wave_size, remaining)
        envs = [ChessEnvironment(cfg, stockfish, tokenizer=tok) for _ in range(wave_count)]
        for env in envs:
            env.reset()
        retry_counts = [0] * wave_count
        pending_meta: list[Optional[dict]] = [None] * wave_count

        while any(not env.is_game_over() for env in envs):
            # 1) Advance every env currently on Stockfish's turn.
            for env in envs:
                if not env.is_game_over() and not env.is_llm_turn():
                    env.apply_stockfish_move()

            # 2) Gather envs that now need an LLM move.
            llm_idx = [
                i for i, env in enumerate(envs)
                if not env.is_game_over() and env.is_llm_turn()
            ]
            if not llm_idx:
                continue

            # 3) Build / reuse prompts (pending retries keep their prompt).
            prompts_flat: list[str] = []
            metas: list[dict] = []
            for i in llm_idx:
                if pending_meta[i] is None:
                    legal_san = sorted(envs[i].board.san(mv) for mv in envs[i].board.legal_moves)
                    fen = envs[i].board.fen()
                    msgs = envs[i].get_messages()
                    prompt_str = apply_template(tok, msgs)
                    pending_meta[i] = {
                        "idx": i, "fen": fen,
                        "legal_san": legal_san, "prompt_str": prompt_str,
                    }
                    retry_counts[i] = 0
                m = pending_meta[i]
                prompts_flat.append(m["prompt_str"])
                metas.append(m)

            # 4) Round-robin split across GPUs → keeps per-GPU batch sizes balanced.
            prompts_per_gpu: list[list[str]] = [[] for _ in range(n_gpus)]
            metas_per_gpu: list[list[dict]] = [[] for _ in range(n_gpus)]
            for j, (p, m) in enumerate(zip(prompts_flat, metas)):
                g = j % n_gpus
                prompts_per_gpu[g].append(p)
                metas_per_gpu[g].append(m)

            # 5) Parallel generate: each GPU runs its own batch; threads release
            # the GIL during CUDA work so both GPUs compute concurrently.
            responses_per_gpu = _parallel_generate(models, tok, prompts_per_gpu, max_new)

            # 6) Route responses back to envs; handle retries and fallbacks.
            for g_responses, g_metas in zip(responses_per_gpu, metas_per_gpu):
                for resp, meta in zip(g_responses, g_metas):
                    i = meta["idx"]
                    token_count = len(tok(resp, add_special_tokens=False)["input_ids"])
                    if token_count >= max_new:
                        retry_counts[i] += 1
                        if retry_counts[i] < max_retries:
                            continue  # keep pending_meta, retry next loop iter
                        raw = None
                    else:
                        raw = resp

                    pending_meta[i] = None
                    retry_counts[i] = 0
                    env = envs[i]

                    if raw is not None:
                        samples.append({
                            "prompt": meta["prompt_str"],
                            "fen": meta["fen"],
                            "legal_moves_san": meta["legal_san"],
                        })
                        ok = env.apply_llm_move(raw)
                    else:
                        ok = False

                    if not ok:
                        legal = list(env.board.legal_moves)
                        if not legal:
                            env.board.clear()  # force game-over
                        else:
                            mv = random.choice(legal)
                            env.pgn_san.append(env.board.san(mv))
                            env.board.push(mv)
                            env.metrics.no_legal_fallback += 1

        per_game_metrics.extend(env.metrics for env in envs)
        per_game_results.extend(env.get_result() for env in envs)
        remaining -= wave_count

    return Dataset.from_list(samples), per_game_metrics, per_game_results


# ---------------------------------------------------------------------------
# Reward function for trl GRPOTrainer
# ---------------------------------------------------------------------------

def make_chess_reward_func():
    """trl calls reward_funcs[i](completions=[...], **dataset_kwargs).

    dataset columns (fen, legal_moves_san) arrive as kwargs aligned to completions.
    """
    analyst = get_analyst()

    def chess_reward_func(completions, **kwargs) -> list[float]:
        fens = kwargs.get("fen")
        rewards = []
        for i, comp in enumerate(completions):
            # comp may be str or chat-format list; normalise to str
            if isinstance(comp, list):
                text = comp[0].get("content", "") if comp else ""
            else:
                text = str(comp)
            board = chess.Board(fens[i])
            rewards.append(compute_reward(text, board, analyst))
        return rewards

    return chess_reward_func


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def _checkpoint_dir(base: str, games_seen: int) -> Path:
    p = Path(base) / f"ckpt_games_{games_seen:06d}"
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_checkpoint(model, tok, cfg: dict, state: dict, games_seen: int) -> Path:
    path = _checkpoint_dir(cfg["paths"]["checkpoints"], games_seen)
    model.save_pretrained(str(path))
    tok.save_pretrained(str(path))
    with open(path / "state.json", "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    _maybe_push_hf_hub(path, cfg, state)
    return path


def _maybe_push_hf_hub(path: Path, cfg: dict, state: dict) -> None:
    token = os.environ.get("HF_TOKEN")
    repo = os.environ.get("HF_REPO")  # e.g. "user/chess-llm-rl"
    if not token or not repo:
        return
    try:
        from huggingface_hub import HfApi
        api = HfApi(token=token)
        api.create_repo(repo, exist_ok=True, private=True)
        api.upload_folder(
            folder_path=str(path),
            repo_id=repo,
            path_in_repo=path.name,
            commit_message=f"ckpt games={state.get('games_seen')} elo={state.get('elo')}",
        )
    except Exception as e:
        print(f"[hub] push skipped: {e}")


# ---------------------------------------------------------------------------
# W&B logging
# ---------------------------------------------------------------------------

def _log_games(wb, metrics_list, results, curriculum: Curriculum, games_seen: int) -> None:
    if wb is None:
        return
    for m, r in zip(metrics_list, results):
        total = max(1, m.legal + m.illegal)
        wb.log({
            "result": {"win": 1, "draw": 0, "loss": -1}[r],
            "length": m.legal + m.illegal,
            "legality": m.legal / total,
            "format_compliance": 1.0 - (m.format_fail / total),
            "no_legal_fallback": m.no_legal_fallback,
            "current_elo": curriculum.elo,
            "win_rate_30": curriculum.win_rate(),
            "games_seen": games_seen,
        })


# ---------------------------------------------------------------------------
# Training iteration
# ---------------------------------------------------------------------------

def train_iteration(model, tok, stockfish, cfg, curriculum, state, wb, inference_models):
    from trl import GRPOConfig, GRPOTrainer

    n_games = cfg["training"]["games_per_iteration"]
    dataset, metrics_list, results = collect_game_experience(
        inference_models, tok, stockfish, cfg, n_games, curriculum,
    )
    state["games_seen"] += n_games
    for r in results:
        curriculum.record(r)
    _log_games(wb, metrics_list, results, curriculum, state["games_seen"])

    if curriculum.should_advance():
        new_elo = curriculum.advance()
        print(f"[curriculum] advanced to Elo {new_elo}")
        if wb is not None:
            wb.log({"curriculum_advance": new_elo, "games_seen": state["games_seen"]})

    if len(dataset) == 0:
        print("[warn] empty dataset this iteration — skipping GRPO update")
        return

    grpo_cfg = cfg["grpo"]
    args = GRPOConfig(
        output_dir=str(Path(cfg["paths"]["checkpoints"]) / "grpo_runtime"),
        num_generations=grpo_cfg["num_generations"],
        max_completion_length=grpo_cfg["max_completion_length"],
        per_device_train_batch_size=grpo_cfg["per_device_train_batch_size"],
        learning_rate=float(grpo_cfg["learning_rate"]),
        beta=grpo_cfg["beta"],
        temperature=grpo_cfg["temperature"],
        use_vllm=grpo_cfg["use_vllm"],
        num_train_epochs=1,
        logging_steps=1,
        report_to=["wandb"] if wb is not None else [],
        remove_unused_columns=False,
        save_strategy="no",
        max_grad_norm=1.0,           # clip gradient spikes (grad_norm hit 20+ previously)
        fp16=True,                   # explicit float16 for T4; prevents silent mixed-precision NaN
    )
    trainer = GRPOTrainer(
        model=model,
        processing_class=tok,
        reward_funcs=[make_chess_reward_func()],
        args=args,
        train_dataset=dataset,
    )
    trainer.train()

    # Sync updated LoRA weights to each replica so the next collection wave
    # generates from the post-training model.
    for replica in inference_models[1:]:
        sync_inference_replica(model, replica)

    ckpt_every = cfg["training"]["checkpoint_every"]
    if state["games_seen"] // ckpt_every > state.get("last_ckpt_bucket", -1):
        state["last_ckpt_bucket"] = state["games_seen"] // ckpt_every
        save_checkpoint(model, tok, cfg, {
            "games_seen": state["games_seen"],
            "elo": curriculum.elo,
            "window": list(curriculum._results),
            "wandb_run_id": state.get("wandb_run_id"),
            "arm": cfg["training"]["arm"],
        }, state["games_seen"])


# ---------------------------------------------------------------------------
# Main entry
# ---------------------------------------------------------------------------

def _init_wandb(cfg: dict, state: dict):
    try:
        import wandb
    except ImportError:
        return None
    wb_cfg = cfg.get("wandb", {})
    if wb_cfg.get("mode", "online") == "disabled":
        return None
    run = wandb.init(
        project=wb_cfg.get("project", "chess-llm-rl"),
        mode=wb_cfg.get("mode", "online"),
        config=cfg,
        id=state.get("wandb_run_id"),
        resume="allow" if state.get("wandb_run_id") else None,
    )
    state["wandb_run_id"] = run.id
    return wandb


def _load_resume(resume_from: Optional[str]):
    if not resume_from:
        return None
    p = Path(resume_from)
    sj = p / "state.json"
    if not sj.exists():
        print(f"[resume] no state.json at {sj}")
        return None
    with open(sj, "r", encoding="utf-8") as f:
        return json.load(f)


def main(config_path: str, resume_from: Optional[str] = None) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    random.seed(cfg["training"]["seed"])

    from .model import load_model
    model, tok = load_model(cfg["model"]["name"], cfg["model"]["max_seq_length"])

    if resume_from:
        from peft import PeftModel  # noqa: F401
        model.load_adapter(resume_from, adapter_name="default")

    stockfish = StockfishManager()

    # Build inference model list: primary + replicas on extra devices.
    devices = resolve_inference_devices(cfg)
    inference_models = [model]
    for d in devices[1:]:
        print(f"[inference] loading replica on {d}")
        inference_models.append(load_inference_replica(model, d))
    print(f"[inference] active devices: {devices}")

    prior = _load_resume(resume_from) or {}
    curriculum = Curriculum(
        start_elo=prior.get("elo", cfg["curriculum"]["start_elo"]),
        step=cfg["curriculum"]["step"],
        win_rate_threshold=cfg["curriculum"]["win_rate_threshold"],
        min_games_at_level=cfg["curriculum"]["min_games_at_level"],
        window=cfg["curriculum"]["window"],
    )
    for r in prior.get("window", []):
        curriculum.record(r)

    state = {
        "games_seen": prior.get("games_seen", 0),
        "wandb_run_id": prior.get("wandb_run_id"),
        "last_ckpt_bucket": prior.get("games_seen", 0) // cfg["training"]["checkpoint_every"],
    }
    wb = _init_wandb(cfg, state)

    target_games = int(os.environ.get("TARGET_GAMES", "180"))
    try:
        while state["games_seen"] < target_games:
            train_iteration(model, tok, stockfish, cfg, curriculum, state, wb, inference_models)
    finally:
        save_checkpoint(model, tok, cfg, {
            "games_seen": state["games_seen"],
            "elo": curriculum.elo,
            "window": list(curriculum._results),
            "wandb_run_id": state.get("wandb_run_id"),
            "arm": cfg["training"]["arm"],
        }, state["games_seen"])
        stockfish.close()
        if wb is not None:
            wb.finish()


def _cli() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()
    main(args.config, args.resume)


if __name__ == "__main__":
    _cli()
