#!/usr/bin/env python
"""S6 verification script — run on Kaggle (where torch + jinja2 are available)."""
import sys, os

# Ensure src/ is on the path when running from repo root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

import chess
import yaml
from transformers import AutoTokenizer
from chess_rl.prompts import build_messages, apply_template, format_pgn

tok = AutoTokenizer.from_pretrained('unsloth/gemma-3-4b-it')
if tok.pad_token is None:
    tok.pad_token = tok.eos_token

b = chess.Board()
# simulate 2 half-moves of history
for san in ('e4', 'e5'):
    b.push_san(san)

print('PGN:', format_pgn(['e4', 'e5']))

for arm in ('fen_only', 'fen_pgn'):
    msgs = build_messages(b, ['e4', 'e5'], arm, llm_color=True)
    print('---', arm, '---')
    print(apply_template(tok, msgs))

cfg = yaml.safe_load(open(os.path.join(os.path.dirname(__file__), '..', 'config.yaml')))
print('config training.arm:', cfg['training']['arm'])
print('config model.name:', cfg['model']['name'])
print('ALL CHECKS PASSED')
