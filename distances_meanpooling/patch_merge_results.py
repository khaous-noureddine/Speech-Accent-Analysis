#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
import pickle
import numpy as np


def _to_list(x):
    # we expect lists per layer; keep as-is if already list
    if isinstance(x, list):
        return x
    return [x]


def _nan_list(n):
    return [np.nan] * n


def patch_and_merge(old_pkl: Path, intra_pkl: Path, out_pkl: Path, model_name: str | None = None):
    with open(old_pkl, "rb") as f:
        old = pickle.load(f)

    with open(intra_pkl, "rb") as f:
        intra = pickle.load(f)

    # ---- old format -> new naming
    layers = old.get("layers", [])
    n = len(layers)

    merged = {}

    merged["model_name"] = model_name or old.get("model_name", intra.get("model_name", ""))
    merged["normalization"] = old.get("normalization", intra.get("normalization", ""))
    merged["layers"] = layers

    # Rename old L2->inter
    merged["mean_inter"] = _to_list(old.get("mean_L2", _nan_list(n)))
    merged["ci_inter_lower"] = _to_list(old.get("ci_L2_lower", _nan_list(n)))
    merged["ci_inter_upper"] = _to_list(old.get("ci_L2_upper", _nan_list(n)))

    # Keep native baseline from old
    merged["mean_native"] = _to_list(old.get("mean_native", _nan_list(n)))
    merged["ci_native_lower"] = _to_list(old.get("ci_native_lower", _nan_list(n)))
    merged["ci_native_upper"] = _to_list(old.get("ci_native_upper", _nan_list(n)))

    # per-L1 old -> per_L1_inter (same structure but renamed)
    if "per_L1" in old:
        merged["per_L1_inter"] = old["per_L1"]
    else:
        merged["per_L1_inter"] = {}

    # ---- inject intra from intra_pkl
    # We assume intra pickle is either:
    # - single-layer lists aligned with its own `layers`
    # - or multi-layer lists aligned with its own `layers`
    intra_layers = intra.get("layers", [])
    if not intra_layers:
        raise ValueError("intra_pkl has no 'layers' key.")

    # Build mapping layer -> index
    intra_idx = {L: i for i, L in enumerate(intra_layers)}

    mean_intra = []
    ci_intra_lower = []
    ci_intra_upper = []

    for L in layers:
        if L in intra_idx:
            i = intra_idx[L]
            mean_intra.append(_to_list(intra.get("mean_intra", []))[i])
            ci_intra_lower.append(_to_list(intra.get("ci_intra_lower", []))[i])
            ci_intra_upper.append(_to_list(intra.get("ci_intra_upper", []))[i])
        else:
            mean_intra.append(np.nan)
            ci_intra_lower.append(np.nan)
            ci_intra_upper.append(np.nan)

    merged["mean_intra"] = mean_intra
    merged["ci_intra_lower"] = ci_intra_lower
    merged["ci_intra_upper"] = ci_intra_upper

    # per_L1_intra: expected dict lang -> {mean:[...], ci_lower:[...], ci_upper:[...]}
    merged["per_L1_intra"] = intra.get("per_L1_intra", {})

    # Optional: if intra pickle already contains per_L1_inter (new format), you can prefer it
    # but normally old.per_L1 is the inter per L1 (distance-to-native).
    # merged["per_L1_inter"] = merged["per_L1_inter"] or intra.get("per_L1_inter", {})

    out_pkl.parent.mkdir(parents=True, exist_ok=True)
    with open(out_pkl, "wb") as f:
        pickle.dump(merged, f)

    print(f"✅ Wrote merged pickle: {out_pkl}")
    print(f"Layers: {len(layers)} | intra-covered layers: {sum(L in intra_idx for L in layers)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old_pkl", type=Path, required=True, help="Old results pkl (mean_L2 + mean_native already computed)")
    ap.add_argument("--intra_pkl", type=Path, required=True, help="New intra/inter pkl (contains mean_intra etc.)")
    ap.add_argument("--out_pkl", type=Path, required=True, help="Output merged pkl")
    ap.add_argument("--model_name", type=str, default=None)
    args = ap.parse_args()

    patch_and_merge(args.old_pkl, args.intra_pkl, args.out_pkl, model_name=args.model_name)


if __name__ == "__main__":
    main()