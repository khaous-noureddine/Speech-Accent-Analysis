"""Recompute utterance geometry from saved embeddings after mean centering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from accented_asr.representation_analysis.metrics import mean_center
from accented_asr.representation_analysis.run_utterance import evaluate


SPACE_MARKER = "_backbone_layer_"


def embedding_identity(path: Path) -> tuple[str, str]:
    """Recover the model and encoder-space names from an analysis NPZ path."""
    stem = path.stem
    if SPACE_MARKER not in stem:
        raise ValueError(f"Unexpected embedding filename: {path.name}")
    model, layer = stem.rsplit(SPACE_MARKER, maxsplit=1)
    return model, f"backbone_layer_{layer}"


def metric_row(
    *, accent: str, model: str, space: str, condition: str, metrics: dict
) -> dict[str, str | float]:
    alignment = metrics["alignment"]
    retrieval = metrics["retrieval"]
    return {
        "accent": accent,
        "model": model,
        "space": space,
        "condition": condition,
        "positive_cosine_distance": alignment["positive_cosine_distance"],
        "negative_cosine_distance": alignment["negative_cosine_distance"],
        "alignment_ratio": alignment["alignment_ratio"],
        "recall_at_1": retrieval["recall_at_1"],
        "recall_at_5": retrieval["recall_at_5"],
        "map": retrieval["map"],
    }


def recompute(input_root: Path, output_dir: Path, seed: int) -> pd.DataFrame:
    rows = []
    details: dict[str, dict] = {}
    fold_dirs = sorted(path.parent for path in input_root.glob("*/analysis_sample.parquet"))
    if not fold_dirs:
        raise FileNotFoundError(f"No completed fold embeddings found under {input_root}")

    for fold_dir in fold_dirs:
        accent = fold_dir.name
        frame = pd.read_parquet(fold_dir / "analysis_sample.parquet")
        details[accent] = {}
        for path in sorted(fold_dir.glob("*_backbone_layer_*.npz")):
            model, space = embedding_identity(path)
            with np.load(path) as archive:
                embeddings = np.asarray(archive["embeddings"], dtype=np.float64)
            if len(embeddings) != len(frame):
                raise ValueError(
                    f"{path} contains {len(embeddings)} embeddings for {len(frame)} rows"
                )

            conditions = {
                "raw": embeddings,
                "mean_centered": mean_center(embeddings, embeddings),
            }
            details[accent].setdefault(model, {})[space] = {}
            for condition, values in conditions.items():
                metrics = evaluate(
                    values,
                    frame,
                    seed,
                    probe=False,
                    bootstrap_replicates=0,
                )
                details[accent][model][space][condition] = metrics
                rows.append(
                    metric_row(
                        accent=accent,
                        model=model,
                        space=space,
                        condition=condition,
                        metrics=metrics,
                    )
                )

    output_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output_dir / "metrics_by_fold_raw_and_centered.csv", index=False)
    macro = frame.groupby(["condition", "model", "space"], as_index=False).mean(
        numeric_only=True
    )
    macro.to_csv(output_dir / "metrics_macro_raw_and_centered.csv", index=False)
    (output_dir / "metrics_raw_and_centered.json").write_text(
        json.dumps(details, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output_dir / "_SUCCESS_CENTERED").write_text("completed\n", encoding="utf-8")
    return macro


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()
    input_root = args.input_root.resolve()
    output_dir = args.output_dir.resolve() if args.output_dir else input_root
    recompute(input_root, output_dir, args.seed)


if __name__ == "__main__":
    main()
