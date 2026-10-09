"""Evaluate L2-ARCTIC held-out-to-seen accent geometry and accent probes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import DataLoader

from accented_asr.representation_analysis.metrics import (
    directional_cross_accent_alignment,
    directional_cross_accent_retrieval,
    fixed_split_linear_probe,
    mean_center,
)
from accented_asr.representation_analysis.run import Collator, json_dump, sha256
from accented_asr.representation_analysis.run_utterance import (
    L2TestDataset,
    build_model,
    extract,
)


def analysis_frame(
    inventory_path: Path,
    manifest_path: Path,
    *,
    probe_prompts: int,
    seed: int,
) -> tuple[pd.DataFrame, str]:
    inventory = pd.read_parquet(inventory_path).copy()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    heldout = str(manifest["heldout_l1"])
    test_prompts = sorted(map(str, manifest["prompt_splits"]["test"]))
    roles = manifest["speaker_roles"]
    query_speakers = set(roles[heldout]["test"])
    gallery_speakers = {
        speaker
        for accent, accent_roles in roles.items()
        if accent != heldout
        for speaker in accent_roles["dev"]
    }
    cross_speakers = query_speakers | gallery_speakers
    rng = np.random.default_rng(seed)
    selected_prompts = sorted(
        rng.choice(
            test_prompts, min(probe_prompts, len(test_prompts)), replace=False
        ).tolist()
    )
    probe_split = {}
    for _accent, group in inventory.groupby("native_language"):
        speakers = sorted(group["speaker_id"].astype(str).unique())
        if len(speakers) != 4:
            raise ValueError("Accent probes require exactly four speakers per accent.")
        probe_split.update({speaker: "train" for speaker in speakers[:2]})
        probe_split.update({speaker: "test" for speaker in speakers[2:]})
    prompt_values = inventory["prompt_id"].astype(str)
    speaker_values = inventory["speaker_id"].astype(str)
    cross_mask = prompt_values.isin(test_prompts) & speaker_values.isin(cross_speakers)
    probe_mask = prompt_values.isin(selected_prompts)
    frame = inventory.loc[cross_mask | probe_mask].copy()
    frame["cross_role"] = ""
    frame.loc[
        frame["prompt_id"].astype(str).isin(test_prompts)
        & frame["speaker_id"].astype(str).isin(query_speakers),
        "cross_role",
    ] = "query"
    frame.loc[
        frame["prompt_id"].astype(str).isin(test_prompts)
        & frame["speaker_id"].astype(str).isin(gallery_speakers),
        "cross_role",
    ] = "gallery"
    frame["probe_split"] = ""
    selected_probe = frame["prompt_id"].astype(str).isin(selected_prompts)
    frame.loc[selected_probe, "probe_split"] = (
        frame.loc[selected_probe, "speaker_id"].astype(str).map(probe_split)
    )
    frame = frame.sort_values(["prompt_id", "speaker_id"]).reset_index(drop=True)

    expected_cross = len(test_prompts) * len(cross_speakers)
    if int(frame["cross_role"].ne("").sum()) != expected_cross:
        raise ValueError("Cross-accent sample is missing speaker/prompt recordings.")
    return frame, heldout.casefold()


def geometry_metrics(
    embeddings: np.ndarray, frame: pd.DataFrame, *, seed: int
) -> list[dict]:
    query = frame["cross_role"].eq("query").to_numpy()
    gallery = frame["cross_role"].eq("gallery").to_numpy()
    prompts = frame["prompt_id"].astype(str).to_numpy()
    conditions = {"raw": embeddings}
    conditions["mean_centered"] = mean_center(embeddings, embeddings[gallery])
    rows = []
    for condition, values in conditions.items():
        alignment = directional_cross_accent_alignment(
            values[query], values[gallery], prompts[query], prompts[gallery], seed=seed
        )
        retrieval = directional_cross_accent_retrieval(
            values[query], values[gallery], prompts[query], prompts[gallery]
        )
        rows.append({"condition": condition, **alignment, **retrieval})
    return rows


def probe_metrics(
    embeddings: np.ndarray, frame: pd.DataFrame, *, seed: int
) -> list[dict]:
    train = frame["probe_split"].eq("train").to_numpy()
    test = frame["probe_split"].eq("test").to_numpy()
    accents = frame["native_language"].astype(str).to_numpy()
    conditions = {"raw": embeddings}
    conditions["mean_centered"] = mean_center(embeddings, embeddings[train])
    return [
        {
            "condition": condition,
            **fixed_split_linear_probe(
                values[train], values[test], accents[train], accents[test], seed=seed
            ),
        }
        for condition, values in conditions.items()
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--fold", choices=(
        "arabic", "chinese", "hindi", "korean", "spanish", "vietnamese"
    ))
    parser.add_argument("--models", nargs="+")
    args = parser.parse_args()

    root = args.repository_root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))[
        "cross_accent_representation_analysis"
    ]
    if not torch.cuda.is_available():
        raise RuntimeError("Cross-accent representation analysis requires CUDA.")
    device = torch.device("cuda")
    seed = int(config["seed"])
    folds = {args.fold: config["folds"][args.fold]} if args.fold else config["folds"]
    layers = [int(value) for value in config["extraction"]["layers"]]
    output_root = root / config["output_dir"]

    for accent, fold in folds.items():
        fold_dir = output_root / accent
        fold_dir.mkdir(parents=True, exist_ok=True)
        frame, heldout = analysis_frame(
            root / config["inventory_parquet"],
            root / fold["manifest"],
            probe_prompts=int(config["metrics"]["probe_prompts"]),
            seed=seed,
        )
        if heldout != accent:
            raise ValueError(f"Manifest held-out accent {heldout!r} != {accent!r}.")
        frame.to_parquet(fold_dir / "analysis_sample.parquet", index=False)
        models = {"ctc_only": config["baseline_checkpoint"], **fold["models"]}
        if args.models:
            requested = {"ctc_only", *args.models}
            missing = requested - set(models)
            if missing:
                raise ValueError(f"Unknown models: {sorted(missing)}")
            models = {name: path for name, path in models.items() if name in requested}

        geometry_rows, probe_rows, metadata = [], [], {}
        loader = None
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
                    collate_fn=Collator(
                        extractor, int(config["extraction"]["sample_rate"])
                    ),
                    pin_memory=True,
                )
            spaces = extract(model, loader, layers, device)
            metadata[name] = {
                "checkpoint": relative_checkpoint,
                "sha256": sha256(checkpoint),
                "metadata": checkpoint_metadata,
            }
            for space, embeddings in spaces.items():
                np.savez_compressed(
                    fold_dir / f"{name}_{space}.npz", embeddings=embeddings
                )
                for row in geometry_metrics(embeddings, frame, seed=seed):
                    geometry_rows.append(
                        {"accent": accent, "model": name, "space": space, **row}
                    )
                for row in probe_metrics(embeddings, frame, seed=seed):
                    probe_rows.append(
                        {"accent": accent, "model": name, "space": space, **row}
                    )
            del model
            torch.cuda.empty_cache()

        pd.DataFrame(geometry_rows).to_csv(fold_dir / "geometry.csv", index=False)
        pd.DataFrame(probe_rows).to_csv(fold_dir / "accent_probe.csv", index=False)
        json_dump(fold_dir / "run_metadata.json", {
            "config": str(config_path.relative_to(root)),
            "fold": accent,
            "seed": seed,
            "models": metadata,
        })
        (fold_dir / "_SUCCESS").write_text("completed\n", encoding="utf-8")


if __name__ == "__main__":
    main()
