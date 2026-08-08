#!/usr/bin/env python3
"""Project radar tokens into a Qwen3-VL embedding space and splice them in.

Qwen3-VL already carries its own vision tower, so no separate image encoder is
needed -- `<|video_pad|>` (151656) marks the positions the tower's outputs get
scattered into. Radar follows exactly that pattern: reserve a run of placeholder
positions in the prompt, then overwrite their embeddings with projected radar
tokens. Nothing about the language model changes, which is what lets the same
connector serve all three tiers.

Hidden sizes differ per tier, so the projector is built from the checkpoint's own
config rather than hardcoded:

  Qwen3-VL-8B        4096
  Qwen3-VL-32B       5120
  Qwen3-VL-30B-A3B   2048

The projector is deliberately small -- two linear layers with a GELU, the LLaVA
arrangement. The radar encoder has already done the representation work; asking a
deep connector to also learn it would just duplicate capacity and slow the
alignment stage, which is the one stage where the language model is frozen and
the connector is the only thing learning.
"""

import json
import os

import torch
import torch.nn as nn

# Reusing the video placeholder would collide with real video tokens in the same
# prompt, so radar gets its own added token.
RADAR_PAD = "<|radar_pad|>"
RADAR_START = "<|radar_start|>"
RADAR_END = "<|radar_end|>"
RADAR_SPECIAL_TOKENS = [RADAR_START, RADAR_PAD, RADAR_END]


def llm_hidden_size(model_dir):
    config = json.load(open(os.path.join(model_dir, "config.json")))
    text = config.get("text_config", config)
    return text["hidden_size"]


class RadarConnector(nn.Module):
    """radar encoder tokens (B, N, d_radar) -> LLM embeddings (B, N, d_llm)."""

    def __init__(self, d_radar, d_llm, dropout=0.0):
        super().__init__()
        self.norm = nn.LayerNorm(d_radar)
        self.project = nn.Sequential(
            nn.Linear(d_radar, d_llm), nn.GELU(), nn.Linear(d_llm, d_llm))
        self.dropout = nn.Dropout(dropout)
        # Marks these positions as radar rather than text or vision. Without it
        # the model has only position to go on, and position is already carrying
        # temporal meaning.
        self.modality = nn.Parameter(torch.zeros(1, 1, d_llm))
        self.d_radar, self.d_llm = d_radar, d_llm

    def forward(self, radar_tokens):
        x = self.project(self.norm(radar_tokens))
        return self.dropout(x + self.modality)


def add_radar_tokens(tokenizer, model=None):
    """Register the radar placeholders and resize embeddings if a model is given.

    Returns the id of the pad token, which is what the splice looks for.
    """
    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": RADAR_SPECIAL_TOKENS})
    if model is not None and added:
        model.resize_token_embeddings(len(tokenizer))
    return tokenizer.convert_tokens_to_ids(RADAR_PAD)


def radar_prompt_block(n_tokens):
    """The literal string to drop into a prompt where radar should go."""
    return RADAR_START + RADAR_PAD * n_tokens + RADAR_END


def splice(embeddings, input_ids, radar_embeddings, radar_pad_id):
    """Overwrite placeholder positions with the projected radar embeddings.

    Scatters per sequence because the number of placeholders can differ between
    rows once prompts are padded, and a mismatch here is silent -- the model
    trains happily on whatever ends up in those slots. So it is checked.
    """
    output = embeddings.clone()
    for b in range(input_ids.shape[0]):
        positions = (input_ids[b] == radar_pad_id).nonzero(as_tuple=True)[0]
        if positions.numel() == 0:
            continue
        n = min(positions.numel(), radar_embeddings.shape[1])
        if positions.numel() != radar_embeddings.shape[1]:
            raise ValueError(
                f"sequence {b} has {positions.numel()} radar placeholders but "
                f"{radar_embeddings.shape[1]} radar tokens were produced")
        output[b, positions[:n]] = radar_embeddings[b, :n].to(output.dtype)
    return output


def self_test():
    torch.manual_seed(0)
    for name, d_llm in (("8B", 4096), ("32B", 5120), ("30B-A3B", 2048)):
        connector = RadarConnector(384, d_llm)
        tokens = torch.randn(2, 256, 384)
        projected = connector(tokens)
        params = sum(p.numel() for p in connector.parameters())
        print(f"  {name:9s} d_llm {d_llm:5d}  out {tuple(projected.shape)}  "
              f"{params/1e6:.1f} M params")

    # Splice, including the guard against a placeholder-count mismatch.
    B, L, d = 2, 64, 4096
    pad_id = 151999
    input_ids = torch.full((B, L), 100)
    input_ids[:, 10:14] = pad_id
    embeddings = torch.zeros(B, L, d)
    radar = torch.ones(B, 4, d)
    out = splice(embeddings, input_ids, radar, pad_id)
    placed = (out.sum(-1) != 0).sum(1).tolist()
    print(f"  splice placed {placed} tokens per sequence (expected [4, 4])")
    try:
        splice(embeddings, input_ids, torch.ones(B, 8, d), pad_id)
        print("  !! mismatch went undetected")
    except ValueError as exc:
        print(f"  mismatch raised as expected: {str(exc)[:60]}...")


if __name__ == "__main__":
    self_test()
