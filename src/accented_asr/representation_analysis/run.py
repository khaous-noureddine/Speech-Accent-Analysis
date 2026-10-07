"""Run a reproducible word-level analysis of SupCon representation geometry."""

from __future__ import annotations

import argparse
import hashlib
import html
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
import yaml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from torch.utils.data import DataLoader, Dataset
from transformers import AutoFeatureExtractor, Wav2Vec2CTCTokenizer

from accented_asr.asr.model import build_asr_model
from accented_asr.joint.model import ProjectionHead, masked_mean
from accented_asr.joint.train import load_config as load_joint_config
from accented_asr.representation_analysis.data import balanced_word_sample
from accented_asr.representation_analysis.metrics import (
    alignment_metrics,
    cross_accent_retrieval,
    grouped_bootstrap_alignment,
    l2_normalize,
    linear_probe,
)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    checkpoint: Path
    training_config: Path


class SampleDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, root: Path, sample_rate: int, max_duration_s: float):
        self.frame = frame
        self.root = root
        self.sample_rate = sample_rate
        self.max_samples = round(sample_rate * max_duration_s)

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.frame.iloc[index]
        path = Path(str(row["audio_path"]))
        path = path if path.is_absolute() else self.root / path
        info = sf.info(path)
        start = max(0, round(float(row["start_s"]) * info.samplerate))
        stop = min(info.frames, round(float(row["end_s"]) * info.samplerate))
        waveform, source_rate = sf.read(
            path, start=start, stop=stop, dtype="float32", always_2d=True
        )
        audio = torch.from_numpy(waveform).mean(dim=1)
        if source_rate != self.sample_rate:
            audio = torchaudio.functional.resample(audio, source_rate, self.sample_rate)
        return {"audio": audio[: self.max_samples], "index": index}


class Collator:
    def __init__(self, extractor, sample_rate: int):
        self.extractor = extractor
        self.sample_rate = sample_rate

    def __call__(self, batch: list[dict[str, Any]]) -> dict[str, Any]:
        values = self.extractor(
            [item["audio"].numpy() for item in batch],
            sampling_rate=self.sample_rate,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_values": values.input_values,
            "attention_mask": values.attention_mask,
            "indices": [item["index"] for item in batch],
        }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def json_dump(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def metrics_table(results: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    def visit(model: str, space: str, prefix: str, value: Any) -> None:
        if isinstance(value, dict):
            for key, nested in value.items():
                visit(model, space, f"{prefix}.{key}" if prefix else key, nested)
        elif isinstance(value, (int, float)) and not isinstance(value, bool):
            rows.append({"model": model, "space": space, "metric": prefix, "value": value})

    for model, model_result in results.items():
        for space, space_result in model_result["spaces"].items():
            visit(model, space, "", space_result)
    return pd.DataFrame(rows)


def resolve(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def load_document(path: Path, root: Path) -> tuple[dict[str, Any], list[ModelSpec]]:
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if set(document) != {"representation_analysis"}:
        raise ValueError("Config requires exactly one representation_analysis section.")
    config = document["representation_analysis"]
    required = {"dataset", "sampling", "extraction", "metrics", "visualization", "models", "output_dir"}
    missing = required - set(config)
    if missing:
        raise ValueError(f"Missing analysis config sections: {sorted(missing)}")
    models = [
        ModelSpec(
            name=str(name),
            checkpoint=resolve(root, values["checkpoint"]),
            training_config=resolve(root, values["training_config"]),
        )
        for name, values in config["models"].items()
    ]
    if len(models) < 2:
        raise ValueError("At least two checkpoints are required for comparison.")
    return config, models


def build_model(spec: ModelSpec, root: Path, device: torch.device):
    for path in (spec.checkpoint, spec.training_config):
        if not path.is_file():
            raise FileNotFoundError(path)
    training = load_joint_config(spec.training_config)
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(root / training.tokenizer_path)
    model, _ = build_asr_model(
        backbone_name=training.backbone_name,
        vocab_size=len(tokenizer),
        pad_token_id=tokenizer.pad_token_id,
        stage2_checkpoint=None,
        gradient_checkpointing=False,
        mask_time_prob=0.05,
        mask_time_length=10,
        mask_feature_prob=0.008,
        mask_feature_length=64,
        layerdrop=0.1,
        activation_dropout=0.1,
    )
    checkpoint = torch.load(spec.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    projection = ProjectionHead(
        model.config.hidden_size,
        training.projection_hidden_size,
        training.projection_size,
    )
    projection.load_state_dict(checkpoint["projection_state_dict"], strict=True)
    model.to(device).eval()
    projection.to(device).eval()
    extractor = AutoFeatureExtractor.from_pretrained(training.backbone_name)
    return model, projection, extractor, training, checkpoint.get("metadata", {})


def extract_embeddings(
    model,
    projection,
    loader: DataLoader,
    *,
    device: torch.device,
    layers: list[int],
    include_projection: bool,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    backbone: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    projected: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            input_values = batch["input_values"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            outputs = model.wav2vec2(
                input_values=input_values,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )
            lengths = model._get_feat_extract_output_lengths(
                attention_mask.sum(dim=1)
            ).long()
            for layer in layers:
                hidden = outputs.hidden_states[layer]
                pooled = torch.nn.functional.normalize(masked_mean(hidden, lengths), dim=-1)
                backbone[layer].append(pooled.cpu().numpy())
            if include_projection:
                projected.append(
                    projection(outputs.last_hidden_state, lengths).cpu().numpy()
                )
    return (
        {f"backbone_layer_{layer}": np.concatenate(values) for layer, values in backbone.items()},
        np.concatenate(projected) if projected else np.empty((len(loader.dataset), 0)),
    )


def evaluate_space(
    embeddings: np.ndarray,
    frame: pd.DataFrame,
    *,
    seed: int,
    bootstrap_replicates: int,
    probe_test_size: float,
) -> dict[str, Any]:
    words = frame["normalized_word"].astype(str).to_numpy()
    accents = frame["accent"].astype(str).to_numpy()
    speakers = frame["speaker_id"].astype(str).to_numpy()
    embeddings = l2_normalize(embeddings)
    return {
        "alignment": alignment_metrics(
            embeddings, words, accents, speakers, seed=seed
        ),
        "alignment_bootstrap_95": grouped_bootstrap_alignment(
            embeddings, words, accents, speakers,
            seed=seed, replicates=bootstrap_replicates,
        ),
        "cross_accent_retrieval": cross_accent_retrieval(
            embeddings, words, accents, speakers
        ),
        "word_probe": linear_probe(
            embeddings, words, speakers, seed=seed, test_size=probe_test_size
        ),
        "accent_probe": linear_probe(
            embeddings, accents, speakers, seed=seed, test_size=probe_test_size
        ),
    }


def make_tsne(
    model_embeddings: dict[str, np.ndarray],
    frame: pd.DataFrame,
    output_dir: Path,
    *,
    seed: int,
    max_words: int,
    perplexity: float,
) -> None:
    words = sorted(frame["normalized_word"].astype(str).unique())[:max_words]
    keep = frame["normalized_word"].astype(str).isin(words).to_numpy()
    names = list(model_embeddings)
    combined = np.concatenate([model_embeddings[name][keep] for name in names])
    dimensions = min(50, combined.shape[1], len(combined) - 1)
    reduced = PCA(n_components=dimensions, random_state=seed).fit_transform(combined)
    coordinates = TSNE(
        n_components=2,
        perplexity=min(perplexity, (len(reduced) - 1) / 3),
        init="pca",
        learning_rate="auto",
        random_state=seed,
    ).fit_transform(reduced)
    rows_per_model = int(keep.sum())
    labels = frame.loc[keep, ["normalized_word", "accent", "speaker_id"]].reset_index(drop=True)
    coordinate_frames = []
    for column, name in enumerate(names):
        coords = coordinates[column * rows_per_model : (column + 1) * rows_per_model]
        current = labels.copy()
        current["model"] = name
        current["x"] = coords[:, 0]
        current["y"] = coords[:, 1]
        coordinate_frames.append(current)
    coordinates_frame = pd.concat(coordinate_frames, ignore_index=True)
    coordinates_frame.to_parquet(output_dir / "tsne_coordinates.parquet", index=False)
    write_svg_scatter(coordinates_frame, names, output_dir / "tsne_comparison.svg")


def write_svg_scatter(frame: pd.DataFrame, model_names: list[str], path: Path) -> None:
    """Write a dependency-free 2xN scatter figure for words and accents."""

    width_per_panel, height_per_panel, margin = 520, 440, 45
    width, height = width_per_panel * len(model_names), height_per_panel * 2
    palette = [
        "#4477AA", "#EE6677", "#228833", "#CCBB44", "#66CCEE", "#AA3377",
        "#BBBBBB", "#000000", "#EE8866", "#44AA99", "#999933", "#882255",
    ]
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
    ]
    for column, model_name in enumerate(model_names):
        model_frame = frame.loc[frame["model"] == model_name]
        for row, field in enumerate(("normalized_word", "accent")):
            x0, y0 = column * width_per_panel, row * height_per_panel
            x = model_frame["x"].to_numpy(float)
            y = model_frame["y"].to_numpy(float)
            xspan = max(float(np.ptp(x)), 1e-9)
            yspan = max(float(np.ptp(y)), 1e-9)
            px = x0 + margin + (x - x.min()) / xspan * (width_per_panel - 2 * margin)
            py = y0 + height_per_panel - margin - (y - y.min()) / yspan * (height_per_panel - 2 * margin)
            labels = model_frame[field].astype(str).to_numpy()
            categories = sorted(set(labels))
            colors = {label: palette[index % len(palette)] for index, label in enumerate(categories)}
            elements.append(
                f'<text x="{x0 + width_per_panel / 2}" y="{y0 + 24}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="16">{html.escape(model_name)} — {html.escape(field)}</text>'
            )
            elements.append(
                f'<rect x="{x0 + margin}" y="{y0 + margin}" width="{width_per_panel - 2 * margin}" '
                f'height="{height_per_panel - 2 * margin}" fill="none" stroke="#cccccc"/>'
            )
            for x_value, y_value, label in zip(px, py, labels):
                elements.append(
                    f'<circle cx="{x_value:.2f}" cy="{y_value:.2f}" r="3" '
                    f'fill="{colors[label]}" fill-opacity="0.72"><title>{html.escape(label)}</title></circle>'
                )
    elements.append("</svg>")
    path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    root = args.repository_root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    config, model_specs = load_document(config_path, root)
    seed = int(config.get("seed", 13))
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda" and not args.smoke:
        raise RuntimeError("Full representation analysis requires CUDA.")

    dataset_cfg = config["dataset"]
    sampling = config["sampling"]
    parquet = resolve(root, dataset_cfg["parquet"])
    frame = balanced_word_sample(
        parquet,
        split=str(dataset_cfg["split"]),
        min_accents_per_word=int(sampling["min_accents_per_word"]),
        examples_per_word_accent=1 if args.smoke else int(sampling["examples_per_word_accent"]),
        max_words=min(4, int(sampling["max_words"])) if args.smoke else int(sampling["max_words"]),
        seed=seed,
    )
    output_dir = resolve(root, config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output_dir / "analysis_sample.parquet", index=False)

    extraction = config["extraction"]
    layers = [int(layer) for layer in extraction["layers"]]
    include_projection = bool(extraction.get("include_projection", False))
    all_results: dict[str, Any] = {}
    visual_embeddings: dict[str, np.ndarray] = {}
    for spec in model_specs:
        model, projection, extractor, training, metadata = build_model(spec, root, device)
        loader = DataLoader(
            SampleDataset(
                frame, root, int(extraction["sample_rate"]),
                float(extraction["max_duration_s"]),
            ),
            batch_size=int(extraction["batch_size"]),
            shuffle=False,
            num_workers=int(extraction["num_workers"]),
            collate_fn=Collator(extractor, int(extraction["sample_rate"])),
            pin_memory=device.type == "cuda",
        )
        backbone, projected = extract_embeddings(
            model,
            projection,
            loader,
            device=device,
            layers=layers,
            include_projection=include_projection,
        )
        spaces = dict(backbone)
        if include_projection:
            spaces["projection"] = projected
        model_dir = output_dir / spec.name
        model_dir.mkdir(exist_ok=True)
        for space, embeddings in spaces.items():
            np.savez_compressed(model_dir / f"{space}.npz", embeddings=embeddings)
        metrics_cfg = config["metrics"]
        all_results[spec.name] = {
            "checkpoint": str(spec.checkpoint.relative_to(root)),
            "checkpoint_sha256": sha256(spec.checkpoint),
            "checkpoint_metadata": metadata,
            "spaces": {
                space: evaluate_space(
                    embeddings,
                    frame,
                    seed=seed,
                    bootstrap_replicates=(10 if args.smoke else int(metrics_cfg["bootstrap_replicates"])),
                    probe_test_size=float(metrics_cfg["probe_test_size"]),
                )
                for space, embeddings in spaces.items()
            },
        }
        visual_embeddings[spec.name] = backbone[f"backbone_layer_{layers[-1]}"]
        del model, projection
        if device.type == "cuda":
            torch.cuda.empty_cache()

    json_dump(output_dir / "metrics.json", all_results)
    metrics_table(all_results).to_csv(output_dir / "metrics.csv", index=False)
    visualization = config["visualization"]
    make_tsne(
        visual_embeddings,
        frame,
        output_dir,
        seed=seed,
        max_words=int(visualization["max_words"]),
        perplexity=float(visualization["perplexity"]),
    )
    run_metadata = {
        "config": str(config_path.relative_to(root)),
        "git_commit": subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True,
            capture_output=True, check=False,
        ).stdout.strip(),
        "seed": seed,
        "device": str(device),
        "examples": len(frame),
        "words": int(frame["normalized_word"].nunique()),
        "accents": sorted(frame["accent"].astype(str).unique()),
    }
    json_dump(output_dir / "run_metadata.json", run_metadata)
    (output_dir / "_SUCCESS").write_text("completed\n", encoding="utf-8")


if __name__ == "__main__":
    main()
