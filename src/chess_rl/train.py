"""GRPO training loop for chess-llm-rl.

Entry point: python -m chess_rl.train --config config.yaml [--resume path]
"""
from __future__ import annotations

import argparse
import json
import os
import random
from dataclasses import asdict
from pathlib import Path
from typing import Optional

import chess
import yaml
import torch

from .curriculum import Curriculum
from .env import ChessEnvironment, GameMetrics
from .prompts import apply_template, build_messages
from .rewards import compute_reward, get_analyst
from .stockfish import StockfishManager


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _generate_batch(model, tok, prompts: list[str], max_new_tokens: int) -> list[str]:
    """Greedy-decode a batch of prompts sequentially on one GPU."""
    inputs = tok.tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=False,
        padding_side="left",
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


# ---------------------------------------------------------------------------
# Experience collection
# ---------------------------------------------------------------------------

def collect_game_experience(model, tok, stockfish: StockfishManager,
                            cfg: dict, n_games: int, curriculum: Curriculum):
    """Play n_games sequentially using a single model."""
    from datasets import Dataset

    grpo = cfg["grpo"]
    max_new = grpo.get("max_collection_length", grpo["max_completion_length"])
    max_retries = grpo.get("max_retries", 3)
    batch_size = grpo.get("collection_batch_size", 4)

    samples: list[dict] = []
    per_game_metrics: list[GameMetrics] = []
    per_game_results: list[str] = []

    stockfish.set_opponent_elo(curriculum.elo)

    remaining = n_games
    while remaining > 0:
        wave_count = min(batch_size, remaining)
        envs = [ChessEnvironment(cfg, stockfish, tokenizer=tok) for _ in range(wave_count)]
        for env in envs:
            env.reset()
        retry_counts = [0] * wave_count
        pending_meta: list[Optional[dict]] = [None] * wave_count

        while any(not env.is_game_over() for env in envs):
            # 1) Advance Stockfish turns
            for env in envs:
                if not env.is_game_over() and not env.is_llm_turn():
                    env.apply_stockfish_move()

            # 2) Gather LLM turns
            llm_idx = [i for i, env in enumerate(envs) if not env.is_game_over() and env.is_llm_turn()]
            if not llm_idx:
                continue

            # 3) Build prompts
            prompts_to_gen: list[str] = []
            metas_to_gen: list[dict] = []
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
                prompts_to_gen.append(pending_meta[i]["prompt_str"])
                metas_to_gen.append(pending_meta[i])

            # 4) Sequential generation
            responses = _generate_batch(model, tok, prompts_to_gen, max_new)

            # 5) Process responses
            for resp, meta in zip(responses, metas_to_gen):
                i = meta["idx"]
                token_count = len(tok.tokenizer(resp, add_special_tokens=False)["input_ids"])
                
                if token_count >= max_new:
                    retry_counts[i] += 1
                    if retry_counts[i] < max_retries:
                        continue
                    raw = None
                else:
                    raw = resp

                pending_meta[i] = None
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
                        env._force_quit = True
                    else:
                        mv = random.choice(legal)
                        env.pgn_san.append(env.board.san(mv))
                        env.board.push(mv)
                        env.metrics.no_legal_fallback += 1

        print(f"[wave] finished batch of {wave_count} games")
        per_game_metrics.extend(env.metrics for env in envs)
        per_game_results.extend(env.get_result() for env in envs)
        remaining -= wave_count

    return Dataset.from_list(samples), per_game_metrics, per_game_results


# ---------------------------------------------------------------------------
# Reward function
# ---------------------------------------------------------------------------

def make_chess_reward_func():
    analyst = get_analyst()

    def chess_reward_func(completions, **kwargs) -> list[float]:
        fens = kwargs.get("fen")
        rewards = []
        for i, comp in enumerate(completions):
            text = comp[0].get("content", "") if isinstance(comp, list) else str(comp)
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
    return path

def _load_resume(resume_from: Optional[str]):
    if not resume_from:
        return None
    p = Path(resume_from)
    sj = p / "state.json"
    if not sj.exists():
        return None
    with open(sj, "r", encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_iteration(model, tok, stockfish, cfg, curriculum, state, wb):
    from trl import GRPOConfig, GRPOTrainer

    n_games = cfg["training"]["games_per_iteration"]
    dataset, metrics_list, results = collect_game_experience(
        model, tok, stockfish, cfg, n_games, curriculum,
    )
    
    state["games_seen"] += n_games
    for r in results:
        curriculum.record(r)

    if wb is not None:
        for m, r in zip(metrics_list, results):
            total = max(1, m.legal + m.illegal)
            wb.log({
                "result": {"win": 1, "draw": 0, "loss": -1}[r],
                "legality": m.legal / total,
                "current_elo": curriculum.elo,
                "win_rate_30": curriculum.win_rate(),
                "games_seen": state["games_seen"],
            })

    if curriculum.should_advance():
        curriculum.advance()

    if len(dataset) == 0:
        return

    grpo_cfg = cfg["grpo"]
    args = GRPOConfig(
        output_dir=str(Path(cfg["paths"]["checkpoints"]) / "grpo_runtime"),
        num_generations=grpo_cfg["num_generations"],
        max_completion_length=grpo_cfg["max_completion_length"],
        per_device_train_batch_size=grpo_cfg["per_device_train_batch_size"],
        learning_rate=float(grpo_cfg["learning_rate"]),
        beta=grpo_cfg["beta"],
        num_train_epochs=1,
        logging_steps=1,
        report_to=["wandb"] if wb is not None else [],
        save_strategy="no",
        fp16=True,
    )
    
    trainer = GRPOTrainer(
        model=model,
        processing_class=tok,
        reward_funcs=[make_chess_reward_func()],
        args=args,
        train_dataset=dataset,
    )
    trainer.train()

    ckpt_every = cfg["training"]["checkpoint_every"]
    if state["games_seen"] // ckpt_every > state.get("last_ckpt_bucket", -1):
        state["last_ckpt_bucket"] = state["games_seen"] // ckpt_every
        save_checkpoint(model, tok, cfg, {
            "games_seen": state["games_seen"],
            "elo": curriculum.elo,
            "window": list(curriculum._results),
            "wandb_run_id": state.get("wandb_run_id"),
        }, state["games_seen"])


def main(config_path: str, resume_from: Optional[str] = None) -> None:
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    random.seed(cfg["training"]["seed"])

    from .model import load_model
    model, tok = load_model(cfg["model"]["name"], cfg["model"]["max_seq_length"])

    prior = _load_resume(resume_from) or {}
    if resume_from:
        from peft import set_peft_model_state_dict
        from safetensors.torch import load_file as _load_safetensors
        adapter_path = Path(resume_from) / "adapter_model.safetensors"
        if not adapter_path.exists():
            raise FileNotFoundError(f"adapter_model.safetensors not found in {resume_from}")
        sd = _load_safetensors(str(adapter_path))
        set_peft_model_state_dict(model, sd)

    stockfish = StockfishManager()
    
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
    
    import wandb
    wb = wandb.init(
        project=cfg["wandb"].get("project", "chess-llm-rl"),
        id=state.get("wandb_run_id"),
        resume="allow" if state.get("wandb_run_id") else None,
        config=cfg
    )
    state["wandb_run_id"] = wb.id

    target_games = int(os.environ.get("TARGET_GAMES", "180"))
    try:
        while state["games_seen"] < target_games:
            train_iteration(model, tok, stockfish, cfg, curriculum, state, wb)
    finally:
        save_checkpoint(model, tok, cfg, {
            "games_seen": state["games_seen"],
            "elo": curriculum.elo,
            "window": list(curriculum._results),
            "wandb_run_id": state.get("wandb_run_id"),
        }, state["games_seen"])
        stockfish.close()
        wb.finish()

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()
    main(args.config, args.resume)
