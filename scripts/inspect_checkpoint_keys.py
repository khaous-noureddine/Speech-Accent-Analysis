#!/usr/bin/env python3
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

import torch


def get_state_dict(ckpt):
    if isinstance(ckpt, dict):
        for key in ["model", "state_dict", "model_state_dict", "module"]:
            if key in ckpt and isinstance(ckpt[key], dict):
                return ckpt[key], key
    return ckpt, "root"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--max_keys", type=int, default=80)
    args = parser.parse_args()

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    state, state_source = get_state_dict(ckpt)

    print(f"\nCheckpoint: {args.checkpoint}")
    print(f"Top-level type: {type(ckpt)}")

    if isinstance(ckpt, dict):
        print(f"Top-level keys: {list(ckpt.keys())}")

    print(f"State dict source: {state_source}")
    print(f"State dict type: {type(state)}")
    print(f"Number of tensors/entries: {len(state)}")

    keys = list(state.keys())

    print("\nFirst keys:")
    for k in keys[: args.max_keys]:
        v = state[k]
        shape = tuple(v.shape) if torch.is_tensor(v) else type(v)
        print(f"  {k:80s} {shape}")

    prefix_counts = Counter()
    for k in keys:
        parts = k.split(".")
        prefix_counts[parts[0]] += 1

    print("\nPrefix counts:")
    for p, c in prefix_counts.most_common():
        print(f"  {p:30s} {c}")

    important_prefixes = [
        "backbone",
        "hubert",
        "wav2vec2",
        "wavlm",
        "projection",
        "projector",
        "ctc_head",
        "lm_head",
    ]

    print("\nImportant prefix matches:")
    for pref in important_prefixes:
        matched = [k for k in keys if k.startswith(pref + ".")]
        print(f"  {pref:12s}: {len(matched)}")

    print("\nPotential backbone keys:")
    for k in keys:
        if (
            k.startswith("backbone.")
            or k.startswith("hubert.")
            or k.startswith("model.hubert.")
            or k.startswith("model.backbone.")
        ):
            v = state[k]
            shape = tuple(v.shape) if torch.is_tensor(v) else type(v)
            print(f"  {k:80s} {shape}")
            if sum(
                1 for kk in keys
                if kk.startswith("backbone.")
                or kk.startswith("hubert.")
                or kk.startswith("model.hubert.")
                or kk.startswith("model.backbone.")
            ) > args.max_keys:
                break


if __name__ == "__main__":
    main()
