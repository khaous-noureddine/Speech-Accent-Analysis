"""Analyze accent structure on the fixed Speech Accent Archive passage."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from accented_asr.representation_analysis.metrics import (
    accent_neighborhood,
    accent_separation,
    fixed_split_linear_probe,
    mean_center,
)
from accented_asr.representation_analysis.run import Collator, json_dump, sha256
from accented_asr.representation_analysis.run_utterance import (
    L2TestDataset,
    build_model,
    extract,
)


def select_saa_sample(
    frame: pd.DataFrame,
    *,
    min_speakers_per_l1: int,
    max_speakers_per_l1: int,
    max_l1: int,
    seed: int,
) -> pd.DataFrame:
    required = {"audio_path", "speaker_id", "native_language", "prompt_id"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Missing SAA columns: {sorted(missing)}")
    sample = frame.dropna(subset=list(required)).copy()
    sample["native_language"] = (
        sample["native_language"].astype(str).str.strip().str.casefold()
    )
    sample = sample.loc[sample["native_language"].ne("")]
    counts = sample.groupby("native_language")["speaker_id"].nunique()
    eligible = counts[counts >= min_speakers_per_l1].sort_values(
        ascending=False, kind="stable"
    )
    selected_l1 = eligible.head(max_l1).index
    if len(selected_l1) < 2:
        raise ValueError("SAA analysis requires at least two eligible L1 classes.")
    rng = np.random.default_rng(seed)
    parts = []
    for language in sorted(selected_l1):
        group = sample.loc[sample["native_language"].eq(language)].copy()
        if group["speaker_id"].duplicated().any():
            raise ValueError("SAA analysis expects one passage per speaker.")
        if len(group) > max_speakers_per_l1:
            group = group.iloc[
                np.sort(rng.choice(len(group), max_speakers_per_l1, replace=False))
            ]
        parts.append(group)
    return pd.concat(parts).sort_values(
        ["native_language", "speaker_id"]
    ).reset_index(drop=True)


def add_probe_split(frame: pd.DataFrame, *, test_size: float, seed: int) -> pd.DataFrame:
    indices = np.arange(len(frame))
    train, test = train_test_split(
        indices,
        test_size=test_size,
        random_state=seed,
        stratify=frame["native_language"].to_numpy(),
    )
    result = frame.copy()
    result["probe_split"] = ""
    result.loc[train, "probe_split"] = "train"
    result.loc[test, "probe_split"] = "test"
    return result


def evaluate(embeddings: np.ndarray, frame: pd.DataFrame, *, seed: int) -> list[dict]:
    labels = frame["native_language"].astype(str).to_numpy()
    train = frame["probe_split"].eq("train").to_numpy()
    test = frame["probe_split"].eq("test").to_numpy()
    conditions = {"raw": embeddings}
    conditions["mean_centered"] = mean_center(embeddings, embeddings[train])
    rows = []
    for condition, values in conditions.items():
        rows.append({
            "condition": condition,
            **accent_separation(values, labels, seed=seed),
            **accent_neighborhood(values, labels),
            **{
                f"probe_{key}": value
                for key, value in fixed_split_linear_probe(
                    values[train], values[test], labels[train], labels[test], seed=seed
                ).items()
            },
        })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--fold", required=True)
    parser.add_argument("--models", nargs="+")
    args = parser.parse_args()

    root = args.repository_root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))[
        "saa_accent_geometry"
    ]
    if not torch.cuda.is_available():
        raise RuntimeError("SAA representation analysis requires CUDA.")
    if args.fold not in config["folds"]:
        raise ValueError(f"Unknown fold: {args.fold}")
    seed = int(config["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda")
    selection = config["selection"]
    frame = select_saa_sample(
        pd.read_parquet(root / config["saa_parquet"]),
        min_speakers_per_l1=int(selection["min_speakers_per_l1"]),
        max_speakers_per_l1=int(selection["max_speakers_per_l1"]),
        max_l1=int(selection["max_l1"]),
        seed=seed,
    )
    frame = add_probe_split(
        frame, test_size=float(config["metrics"]["probe_test_size"]), seed=seed
    )
    output_dir = root / config["output_dir"] / args.fold
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_dir / "analysis_sample.parquet", index=False)

    fold = config["folds"][args.fold]
    models = {"asr_only": config["baseline_checkpoint"], **fold["models"]}
    if args.models:
        unknown = set(args.models) - set(models)
        if unknown:
            raise ValueError(f"Unknown models: {sorted(unknown)}")
        models = {name: path for name, path in models.items() if name in args.models}
    layers = [int(value) for value in config["extraction"]["layers"]]
    loader = None
    rows, metadata = [], {}
    for name, relative_checkpoint in models.items():
        checkpoint = root / relative_checkpoint
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        model, extractor, checkpoint_metadata = build_model(
            config, checkpoint, root, device
        )
        if loader is None:
            loader = DataLoader(
                L2TestDataset(frame, root, int(config["extraction"]["sample_rate"])),
                batch_size=int(config["extraction"]["batch_size"]),
                shuffle=False,
                num_workers=int(config["extraction"]["num_workers"]),
                collate_fn=Collator(extractor, int(config["extraction"]["sample_rate"])),
                pin_memory=True,
            )
        spaces = extract(model, loader, layers, device)
        metadata[name] = {
            "checkpoint": relative_checkpoint,
            "sha256": sha256(checkpoint),
            "metadata": checkpoint_metadata,
        }
        for space, embeddings in spaces.items():
            np.savez_compressed(output_dir / f"{name}_{space}.npz", embeddings=embeddings)
            for row in evaluate(embeddings, frame, seed=seed):
                rows.append({"fold": args.fold, "model": name, "space": space, **row})
        del model
        torch.cuda.empty_cache()

    pd.DataFrame(rows).to_csv(output_dir / "saa_accent_geometry.csv", index=False)
    json_dump(output_dir / "run_metadata.json", {
        "config": str(config_path.relative_to(root)),
        "fold": args.fold,
        "seed": seed,
        "sample_examples": len(frame),
        "sample_l1_classes": frame["native_language"].nunique(),
        "models": metadata,
    })
    (output_dir / "_SUCCESS").write_text("completed\n", encoding="utf-8")


if __name__ == "__main__":
    main()
