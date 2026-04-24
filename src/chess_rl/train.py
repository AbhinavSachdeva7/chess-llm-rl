"""GRPO training loop for chess-llm-rl.

Entry point: python -m chess_rl.train --config config.yaml [--resume path]
Verify requires CUDA (Unsloth + bitsandbytes). Runs on Kaggle T4.
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

from .curriculum import Curriculum
from .env import ChessEnvironment, GameMetrics
from .prompts import apply_template, build_messages
from .rewards import compute_reward, get_analyst
from .stockfish import StockfishManager


# ---------------------------------------------------------------------------
# Experience collection
# ---------------------------------------------------------------------------

def _greedy_generate(model, tok, prompt_str: str, max_new_tokens: int) -> str:
    import torch
    ids = tok(text=prompt_str, return_tensors="pt").to(model.device)
    with torch.inference_mode():
        out = model.generate(
            **ids,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
    new_tokens = out[0][ids["input_ids"].shape[1]:]
    return tok.decode(new_tokens, skip_special_tokens=True).strip()


def collect_game_experience(model, tok, stockfish: StockfishManager,
                            cfg: dict, n_games: int, curriculum: Curriculum):
    """Play n_games greedy vs Stockfish at current curriculum Elo.

    At every LLM turn: attempt up to max_retries greedy generations.
    Completions >= max_completion_length tokens are overlong and discarded.
    If all retries are overlong the position is dropped (not added to samples)
    and the game continues with a random legal move.
    """
    from datasets import Dataset

    arm = cfg["training"]["arm"]
    max_new = cfg["grpo"]["max_completion_length"]
    max_retries = cfg["grpo"].get("max_retries", 3)
    samples: list[dict] = []
    per_game_metrics: list[GameMetrics] = []
    per_game_results: list[str] = []

    stockfish.set_opponent_elo(curriculum.elo)

    for _ in range(n_games):
        env = ChessEnvironment(cfg, stockfish, tokenizer=tok)
        env.reset()
        while not env.is_game_over():
            if env.is_llm_turn():
                legal_san = sorted(env.board.san(m) for m in env.board.legal_moves)
                fen = env.board.fen()
                msgs = env.get_messages()
                prompt_str = apply_template(tok, msgs)

                raw = None
                for _ in range(max_retries):
                    candidate = _greedy_generate(model, tok, prompt_str, max_new)
                    token_count = len(
                        tok(candidate, add_special_tokens=False)["input_ids"]
                    )
                    if token_count < max_new:
                        raw = candidate
                        break

                if raw is not None:
                    samples.append({
                        "prompt": prompt_str,
                        "fen": fen,
                        "legal_moves_san": legal_san,
                    })
                    ok = env.apply_llm_move(raw)
                else:
                    ok = False

                if not ok:
                    legal = list(env.board.legal_moves)
                    if not legal:
                        break
                    mv = random.choice(legal)
                    env.pgn_san.append(env.board.san(mv))
                    env.board.push(mv)
                    env.metrics.no_legal_fallback += 1
            else:
                env.apply_stockfish_move()
        per_game_metrics.append(env.metrics)
        per_game_results.append(env.get_result())

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

def train_iteration(model, tok, stockfish, cfg, curriculum, state, wb):
    from trl import GRPOConfig, GRPOTrainer

    n_games = cfg["training"]["games_per_iteration"]
    dataset, metrics_list, results = collect_game_experience(
        model, tok, stockfish, cfg, n_games, curriculum,
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
            train_iteration(model, tok, stockfish, cfg, curriculum, state, wb)
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
