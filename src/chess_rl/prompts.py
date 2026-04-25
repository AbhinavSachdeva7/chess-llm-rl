"""Prompt builders for both FEN-only and FEN+PGN arms."""

import chess


def format_pgn(pgn_san: list[str]) -> str:
    """Produce '1. e4 e5 2. Nf3 Nc6 ...' from a flat list of SAN moves.

    Odd-indexed entries (0-based) are White, even are Black.
    Returns empty string for empty list.
    """
    if not pgn_san:
        return ""

    parts = []
    for i, move in enumerate(pgn_san):
        if i % 2 == 0:
            # White move — start a new move number
            parts.append(f"{i // 2 + 1}. {move}")
        else:
            # Black move — append to last entry
            parts[-1] = parts[-1] + f" {move}"

    return " ".join(parts)


def build_messages(
    board: chess.Board,
    pgn_san: list[str],
    arm: str,
    llm_color: bool,
) -> list[dict]:
    """Build chat messages for the LLM.

    Args:
        board: Current board position.
        pgn_san: Flat list of SAN moves played so far.
        arm: Either "fen_only" or "fen_pgn".
        llm_color: True = playing White, False = playing Black.

    Returns:
        [{"role": "user", "content": ...}]

    Raises:
        ValueError: If arm is not "fen_only" or "fen_pgn".

    Note:
        Gemma 4 instruct is trained on user/model turns only; system role is
        out-of-distribution. Instructions are merged into the first user turn.
    """
    if arm not in ("fen_only", "fen_pgn"):
        raise ValueError(f"arm must be 'fen_only' or 'fen_pgn', got {arm!r}")

    color_str = "White" if llm_color else "Black"
    instructions = (
        f"You are a chess engine playing as {color_str}. "
        "Think step by step inside <think>...</think>, then respond with your move in <move>SAN</move>. "
        "Example: <think>The queen is attacked.</think><move>Nf3</move>."
    )

    fen = board.fen()
    legal = ",".join(sorted(board.san(m) for m in board.legal_moves))

    if arm == "fen_only":
        position_block = (
            f"Position (FEN): {fen}\n"
            f"Legal moves: {legal}\n"
            "Your move:"
        )
    else:  # fen_pgn
        pgn = format_pgn(pgn_san) or "(no moves yet)"
        position_block = (
            f"Game so far: {pgn}\n"
            f"Position (FEN): {fen}\n"
            f"Legal moves: {legal}\n"
            "Your move:"
        )

    user_content = f"{instructions}\n\n{position_block}"

    return [
        {"role": "user", "content": user_content},
    ]


def apply_template(tok, messages: list[dict]) -> str:
    """Apply the tokenizer's chat template to messages.

    Args:
        tok: A HuggingFace tokenizer with apply_chat_template.
        messages: List of {"role": ..., "content": ...} dicts.

    Returns:
        Formatted string with generation prompt appended.
    """
    return tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
