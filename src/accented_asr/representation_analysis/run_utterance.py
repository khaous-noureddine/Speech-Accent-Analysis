"""Analyze held-out-accent utterance representations on L2-ARCTIC."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import torchaudio
import yaml
from torch.utils.data import DataLoader, Dataset
from transformers import AutoFeatureExtractor, Wav2Vec2CTCTokenizer

from accented_asr.asr.model import build_asr_model
from accented_asr.joint.model import masked_mean
from accented_asr.representation_analysis.metrics import (
    cross_speaker_alignment,
    cross_speaker_retrieval,
    grouped_bootstrap_cross_speaker,
    linear_probe,
)
from accented_asr.representation_analysis.run import Collator, json_dump, make_tsne, sha256


class L2TestDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, root: Path, sample_rate: int):
        self.frame = frame.reset_index(drop=True)
        self.root = root
        self.sample_rate = sample_rate

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        path = Path(str(row["audio_path"]))
        path = path if path.is_absolute() else self.root / path
        waveform, source_rate = sf.read(path, dtype="float32", always_2d=True)
        audio = torch.from_numpy(waveform).mean(dim=1)
        if source_rate != self.sample_rate:
            audio = torchaudio.functional.resample(audio, source_rate, self.sample_rate)
        return {"audio": audio, "index": index}


def load_test_frame(path: Path, accent: str) -> pd.DataFrame:
    frame = pd.read_parquet(path)
    required = {"audio_path", "native_language", "prompt_id", "speaker_id", "split"}
    missing = required - set(frame)
    if missing:
        raise ValueError(f"Missing L2-ARCTIC columns: {sorted(missing)}")
    frame = frame.loc[frame["split"] == "test"].copy()
    found = set(frame["native_language"].astype(str).str.casefold())
    if found != {accent.casefold()}:
        raise ValueError(f"Fold {accent} test contains accents {sorted(found)}")
    if frame["speaker_id"].nunique() < 2 or frame["prompt_id"].nunique() < 2:
        raise ValueError(f"Fold {accent} lacks repeated prompts across speakers.")
    return frame.sort_values(["prompt_id", "speaker_id"]).reset_index(drop=True)


def build_model(config: dict, checkpoint_path: Path, root: Path, device: torch.device):
    tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(root / config["tokenizer_path"])
    model, _ = build_asr_model(
        backbone_name=config["backbone_name"],
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
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.to(device).eval()
    extractor = AutoFeatureExtractor.from_pretrained(config["backbone_name"])
    return model, extractor, checkpoint.get("metadata", {})


def extract(model, loader: DataLoader, layers: list[int], device: torch.device):
    values = {layer: [] for layer in layers}
    with torch.inference_mode():
        for batch in loader:
            audio = batch["input_values"].to(device)
            mask = batch["attention_mask"].to(device)
            output = model.wav2vec2(
                input_values=audio, attention_mask=mask, output_hidden_states=True
            )
            lengths = model._get_feat_extract_output_lengths(mask.sum(dim=1)).long()
            for layer in layers:
                pooled = masked_mean(output.hidden_states[layer], lengths)
                pooled = torch.nn.functional.normalize(pooled, dim=-1)
                values[layer].append(pooled.cpu().numpy())
    return {f"backbone_layer_{layer}": np.concatenate(parts) for layer, parts in values.items()}


def evaluate(
    embeddings: np.ndarray,
    frame: pd.DataFrame,
    seed: int,
    probe: bool,
    bootstrap_replicates: int,
):
    prompts = frame["prompt_id"].astype(str).to_numpy()
    speakers = frame["speaker_id"].astype(str).to_numpy()
    result = {
        "alignment": cross_speaker_alignment(embeddings, prompts, speakers, seed=seed),
        "retrieval": cross_speaker_retrieval(embeddings, prompts, speakers),
    }
    if probe:
        result["prompt_probe"] = linear_probe(
            embeddings, prompts, speakers, seed=seed, test_size=0.25
        )
        result["alignment_bootstrap_95"] = grouped_bootstrap_cross_speaker(
            embeddings, prompts, speakers, seed=seed,
            replicates=bootstrap_replicates,
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    root = args.repository_root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    document = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    config = document["utterance_representation_analysis"]
    seed = int(config["seed"])
    torch.manual_seed(seed)
    np.random.seed(seed)
    if not torch.cuda.is_available():
        raise RuntimeError("Utterance representation analysis requires CUDA.")
    device = torch.device("cuda")
    output_dir = root / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    layers = [int(layer) for layer in config["extraction"]["layers"]]
    probe_layer = f"backbone_layer_{int(config['metrics']['probe_layer'])}"
    results = {}
    flat_rows = []
    for accent, fold in config["folds"].items():
        frame = load_test_frame(root / fold["parquet"], accent)
        fold_dir = output_dir / accent
        fold_dir.mkdir(exist_ok=True)
        frame.to_parquet(fold_dir / "analysis_sample.parquet", index=False)
        model_embeddings = {}
        results[accent] = {}
        fold_models = fold.get("models")
        if fold_models is None:
            fold_models = {"utterance_supcon": fold["supcon_checkpoint"]}
        model_checkpoints = {
            "ctc_only": config["baseline_checkpoint"], **fold_models,
        }
        missing_checkpoints = [
            value for value in model_checkpoints.values() if not (root / value).is_file()
        ]
        if missing_checkpoints:
            raise FileNotFoundError(
                "Missing analysis checkpoints:\n" + "\n".join(missing_checkpoints)
            )
        for name, checkpoint_value in model_checkpoints.items():
            checkpoint_path = root / checkpoint_value
            model, extractor, metadata = build_model(config, checkpoint_path, root, device)
            loader = DataLoader(
                L2TestDataset(frame, root, int(config["extraction"]["sample_rate"])),
                batch_size=int(config["extraction"]["batch_size"]),
                shuffle=False,
                num_workers=int(config["extraction"]["num_workers"]),
                collate_fn=Collator(extractor, int(config["extraction"]["sample_rate"])),
                pin_memory=True,
            )
            spaces = extract(model, loader, layers, device)
            model_embeddings[name] = spaces[probe_layer]
            results[accent][name] = {
                "checkpoint": checkpoint_value,
                "checkpoint_sha256": sha256(checkpoint_path),
                "checkpoint_metadata": metadata,
                "spaces": {},
            }
            for space, embeddings in spaces.items():
                metrics = evaluate(
                    embeddings, frame, seed, probe=space == probe_layer,
                    bootstrap_replicates=int(config["metrics"]["bootstrap_replicates"]),
                )
                results[accent][name]["spaces"][space] = metrics
                np.savez_compressed(fold_dir / f"{name}_{space}.npz", embeddings=embeddings)
                alignment = metrics["alignment"]
                retrieval = metrics["retrieval"]
                flat_rows.append({
                    "accent": accent, "model": name, "space": space,
                    "positive_cosine_distance": alignment["positive_cosine_distance"],
                    "negative_cosine_distance": alignment["negative_cosine_distance"],
                    "alignment_ratio": alignment["alignment_ratio"],
                    "recall_at_1": retrieval["recall_at_1"],
                    "recall_at_5": retrieval["recall_at_5"],
                    "map": retrieval["map"],
                })
            del model
            torch.cuda.empty_cache()
        visual_frame = pd.DataFrame({
            "normalized_word": frame["prompt_id"].astype(str),
            "accent": frame["native_language"].astype(str),
            "speaker_id": frame["speaker_id"].astype(str),
        })
        make_tsne(
            model_embeddings, visual_frame, fold_dir, seed=seed,
            max_words=int(config["visualization"]["max_prompts"]),
            perplexity=float(config["visualization"]["perplexity"]),
        )
    metrics = pd.DataFrame(flat_rows)
    metrics.to_csv(output_dir / "metrics_by_fold.csv", index=False)
    macro = metrics.groupby(["model", "space"], as_index=False).mean(numeric_only=True)
    macro.to_csv(output_dir / "metrics_macro.csv", index=False)
    json_dump(output_dir / "metrics.json", results)
    json_dump(output_dir / "run_metadata.json", {
        "config": str(config_path.relative_to(root)), "seed": seed,
        "folds": list(config["folds"]), "layers": layers,
    })
    (output_dir / "_SUCCESS").write_text("completed\n", encoding="utf-8")


if __name__ == "__main__":
    main()
