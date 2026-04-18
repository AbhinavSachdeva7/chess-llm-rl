"""Model loader for Gemma 4 + LoRA via Unsloth (GPU-only).

Imports `unsloth` at module top so `import chess_rl.model` triggers Unsloth's
patching before trl/transformers/peft get imported — otherwise Unsloth emits
a perf warning and some fused kernels are skipped.
"""
from __future__ import annotations

import os
from typing import Tuple

import unsloth  # noqa: F401  — must precede trl/transformers/peft imports
from unsloth import FastModel

_DEFAULT_MODEL = "unsloth/gemma-4-E2B-it"


def _load_once(model_name: str, max_seq_length: int):
    model, tok = FastModel.from_pretrained(
        model_name=model_name,
        dtype=None,
        max_seq_length=max_seq_length,
        load_in_4bit=True,
        full_finetuning=False,
    )
    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers=False,     # text-only task — skip image tower
        finetune_language_layers=True,
        finetune_attention_modules=True,  # good for GRPO
        finetune_mlp_modules=True,
        r=32,
        lora_alpha=64,
        lora_dropout=0,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return model, tok


def load_model(
    model_name: str = _DEFAULT_MODEL,
    max_seq_length: int = 2048,
) -> Tuple[object, object]:
    model, tok = _load_once(model_name, max_seq_length)
    _record_choice(model_name)
    return model, tok


def _record_choice(name: str) -> None:
    """Persist the selected model name into config.yaml under model.name.

    Best-effort: silently no-op if pyyaml unavailable or file missing.
    """
    path = os.path.join(os.getcwd(), "config.yaml")
    if not os.path.exists(path):
        return
    try:
        import yaml
    except ImportError:
        return
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    try:
        data = yaml.safe_load(raw) or {}
    except yaml.YAMLError:
        return
    if not isinstance(data, dict):
        return
    model_cfg = data.setdefault("model", {})
    if model_cfg.get("name") == name:
        return
    model_cfg["name"] = name
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, sort_keys=False)
