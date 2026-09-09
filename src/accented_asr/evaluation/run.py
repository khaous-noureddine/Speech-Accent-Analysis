"""Run auditable greedy or 4-gram LM CTC evaluation for one checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import pandas as pd
import soundfile as sf
import torch
import torchaudio
import yaml
from torch.utils.data import DataLoader, Dataset
from transformers import AutoFeatureExtractor, Wav2Vec2CTCTokenizer, Wav2Vec2ForCTC

from accented_asr.asr.model import build_asr_model
from accented_asr.asr.train import load_config as load_stage3_config
from accented_asr.evaluation.metrics import (
    NORMALIZATION_VERSION,
    aggregate_edit_counts,
    score_utterance,
)


@dataclass(frozen=True)
class EvaluationConfig:
    name: str
    model: str
    objective: str
    fold: str | None
    seed: int | str
    checkpoint: str | None
    stage3_config: str | None
    dataset: str
    split: str
    parquet: str
    raw_dir: str
    decoder: str
    output_dir: str
    batch_size: int
    num_workers: int
    device: str
    source: str = "stage3"
    hf_model: str | None = None
    hf_revision: str | None = None


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_config(
    path: Path, dataset_name: str, decoder_override: str | None = None
) -> EvaluationConfig:
    document = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if set(document) != {"evaluation"}:
        raise ValueError("Evaluation config requires exactly one evaluation section.")
    section = dict(document["evaluation"])
    datasets = section.pop("datasets", None)
    if not isinstance(datasets, dict) or not datasets:
        raise ValueError("Evaluation config requires a non-empty datasets mapping.")
    if dataset_name not in datasets:
        raise ValueError(
            f"Unknown evaluation dataset {dataset_name!r}; expected {sorted(datasets)}."
        )
    config = EvaluationConfig(
        **section, dataset=dataset_name, **datasets[dataset_name]
    )
    if decoder_override is not None:
        config = replace(config, decoder=decoder_override)
    if config.decoder not in {"greedy", "beam_4gram"}:
        raise ValueError("decoder must be greedy or beam_4gram.")
    if config.source not in {"stage3", "huggingface"}:
        raise ValueError("source must be stage3 or huggingface.")
    if config.source == "huggingface" and not (
        config.hf_model and config.hf_revision
    ):
        raise ValueError("A Hugging Face source requires hf_model and hf_revision.")
    if config.split != "test":
        raise ValueError("Final evaluation configs must select split=test.")
    if (
        config.dataset == "l2_arctic"
        and config.fold is not None
        and config.fold.lower() not in config.parquet.lower()
    ):
        raise ValueError("Evaluation parquet must match the held-out accent fold.")
    if config.batch_size <= 0 or config.num_workers < 0:
        raise ValueError("Invalid evaluation loader parameters.")
    return config


def load_4gram_decoder(lm_dir: Path, tokenizer: Wav2Vec2CTCTokenizer):
    """Load the published Hugging Face 4-gram decoder after vocabulary checks."""
    from pyctcdecode import build_ctcdecoder

    alphabet_path = lm_dir / "alphabet.json"
    model_path = lm_dir / "language_model" / "4-gram.bin"
    attrs_path = lm_dir / "language_model" / "attrs.json"
    unigrams_path = lm_dir / "language_model" / "unigrams.txt"
    for required in (alphabet_path, model_path, attrs_path, unigrams_path):
        if not required.is_file():
            raise FileNotFoundError(required)

    labels = json.loads(alphabet_path.read_text(encoding="utf-8"))["labels"]
    tokenizer_labels = [None] * len(tokenizer)
    for token, index in tokenizer.get_vocab().items():
        tokenizer_labels[index] = token
    tokenizer_labels[tokenizer.pad_token_id] = ""
    tokenizer_labels[tokenizer.unk_token_id] = "⁇"
    tokenizer_labels[tokenizer.word_delimiter_token_id] = " "
    if tokenizer_labels != labels:
        raise ValueError(
            "The 4-gram decoder alphabet does not match the Stage 3 CTC vocabulary."
        )

    attrs = json.loads(attrs_path.read_text(encoding="utf-8"))
    unigrams = unigrams_path.read_text(encoding="utf-8").splitlines()
    decoder = build_ctcdecoder(
        labels=labels,
        kenlm_model_path=str(model_path),
        unigrams=unigrams,
        alpha=float(attrs["alpha"]),
        beta=float(attrs["beta"]),
        unk_score_offset=float(attrs["unk_score_offset"]),
        lm_score_boundary=bool(attrs["score_boundary"]),
    )
    return decoder, attrs, (alphabet_path, model_path, attrs_path, unigrams_path)


class EvaluationDataset(Dataset):
    def __init__(self, parquet: Path, *, split: str, root: Path, validate_audio: bool):
        frame = pd.read_parquet(parquet)
        required = {"audio_path", "transcript", "split", "speaker_id", "utterance_id"}
        missing = required - set(frame.columns)
        if missing:
            raise ValueError(f"Missing evaluation columns: {sorted(missing)}")
        frame = frame.loc[frame["split"] == split].copy().reset_index(drop=True)
        if frame.empty:
            raise ValueError(f"No rows for split={split!r} in {parquet}.")
        frame["resolved_audio_path"] = frame["audio_path"].astype(str).map(
            lambda value: str(Path(value) if Path(value).is_absolute() else root / value)
        )
        if validate_audio:
            missing_audio = [path for path in frame["resolved_audio_path"] if not Path(path).is_file()]
            if missing_audio:
                raise ValueError(
                    f"Found {len(missing_audio)} missing evaluation audio files: "
                    f"{missing_audio[:5]}"
                )
        self.frame = frame

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> dict:
        row = self.frame.iloc[index]
        audio, sample_rate = sf.read(
            row["resolved_audio_path"], dtype="float32", always_2d=True
        )
        waveform = torch.from_numpy(audio).mean(dim=1)
        if sample_rate != 16_000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16_000)
        metadata = {
            key: row[key] for key in self.frame.columns if key != "resolved_audio_path"
        }
        return {"audio": waveform, "metadata": metadata}


class EvaluationCollator:
    def __init__(self, feature_extractor):
        self.feature_extractor = feature_extractor

    def __call__(self, batch: list[dict]) -> dict:
        inputs = self.feature_extractor(
            [item["audio"].numpy() for item in batch],
            sampling_rate=16_000,
            padding=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        return {
            "input_values": inputs.input_values,
            "attention_mask": inputs.attention_mask,
            "metadata": [item["metadata"] for item in batch],
        }


def git_commit(root: Path) -> str | None:
    git_executable = shutil.which("git")
    if git_executable is None:
        return None
    result = subprocess.run(
        [git_executable, "rev-parse", "HEAD"], cwd=root, text=True,
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def json_value(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--decoder", choices=("greedy", "beam_4gram"))
    parser.add_argument("--lm-dir", type=Path)
    parser.add_argument("--beam-width", type=int, default=100)
    args = parser.parse_args()

    root = args.repository_root.resolve()
    config_path = (root / args.config).resolve() if not args.config.is_absolute() else args.config
    config = load_config(config_path, args.dataset, args.decoder)
    if args.beam_width <= 0:
        raise ValueError("beam-width must be positive.")
    parquet_path = (root / config.parquet).resolve()
    if not parquet_path.is_file():
        raise FileNotFoundError(parquet_path)

    checkpoint_metadata = {}
    if config.source == "stage3":
        if config.checkpoint is None or config.stage3_config is None:
            raise ValueError("A Stage 3 source requires checkpoint and stage3_config.")
        checkpoint_path = (root / config.checkpoint).resolve()
        stage3_config_path = (root / config.stage3_config).resolve()
        for required in (checkpoint_path, stage3_config_path):
            if not required.is_file():
                raise FileNotFoundError(required)
        stage3 = load_stage3_config(stage3_config_path)
        if (stage3.objective, stage3.fold) != (config.objective, config.fold):
            raise ValueError("Evaluation identity does not match the Stage 3 config.")
        if config.seed not in stage3.seeds:
            raise ValueError("Evaluation seed is not declared by the Stage 3 config.")
        tokenizer_path = root / stage3.tokenizer_path
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(tokenizer_path)
        feature_extractor = AutoFeatureExtractor.from_pretrained(stage3.backbone_name)
        model, _ = build_asr_model(
            backbone_name=stage3.backbone_name, vocab_size=len(tokenizer),
            pad_token_id=tokenizer.pad_token_id, stage2_checkpoint=None,
            gradient_checkpointing=False, mask_time_prob=stage3.mask_time_prob,
            mask_time_length=stage3.mask_time_length,
            mask_feature_prob=stage3.mask_feature_prob,
            mask_feature_length=stage3.mask_feature_length,
            layerdrop=stage3.layerdrop, activation_dropout=stage3.activation_dropout,
        )
        checkpoint_stat_before = checkpoint_path.stat()
        checkpoint_hash = sha256(checkpoint_path)
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        checkpoint_stat_after = checkpoint_path.stat()
        if (
            checkpoint_stat_before.st_size != checkpoint_stat_after.st_size
            or checkpoint_stat_before.st_mtime_ns != checkpoint_stat_after.st_mtime_ns
        ):
            raise RuntimeError(
                "The checkpoint changed while it was being read. Retry evaluation "
                "after training finishes writing checkpoint_best.pt."
            )
        if "model_state_dict" not in checkpoint:
            raise ValueError("Stage 3 checkpoint has no model_state_dict.")
        checkpoint_metadata = checkpoint.get("metadata", {})
        expected_metadata = {
            "objective": config.objective, "fold": config.fold, "seed": config.seed,
        }
        mismatches = {
            key: (checkpoint_metadata.get(key), expected)
            for key, expected in expected_metadata.items()
            if checkpoint_metadata.get(key) != expected
        }
        if mismatches:
            raise ValueError(f"Stage 3 checkpoint identity mismatch: {mismatches}")
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        checkpoint_reference = config.checkpoint
        tokenizer_hash = sha256(tokenizer_path / "vocab.json")
    else:
        tokenizer = Wav2Vec2CTCTokenizer.from_pretrained(
            config.hf_model, revision=config.hf_revision
        )
        feature_extractor = AutoFeatureExtractor.from_pretrained(
            config.hf_model, revision=config.hf_revision
        )
        model = Wav2Vec2ForCTC.from_pretrained(
            config.hf_model, revision=config.hf_revision
        )
        checkpoint_hash = None
        checkpoint_reference = f"{config.hf_model}@{config.hf_revision}"
        tokenizer_hash = hashlib.sha256(
            json.dumps(tokenizer.get_vocab(), sort_keys=True).encode("utf-8")
        ).hexdigest()

    lm_decoder = None
    lm_attrs = None
    lm_artifacts: tuple[Path, ...] = ()
    if config.decoder == "beam_4gram":
        if args.lm_dir is None:
            raise ValueError("beam_4gram decoding requires --lm-dir.")
        lm_dir = args.lm_dir if args.lm_dir.is_absolute() else root / args.lm_dir
        lm_decoder, lm_attrs, lm_artifacts = load_4gram_decoder(lm_dir, tokenizer)
    requested_device = config.device
    if requested_device == "auto":
        requested_device = "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Evaluation requests CUDA but it is unavailable.")
    device = torch.device(requested_device)
    model.to(device).eval()

    dataset = EvaluationDataset(
        parquet_path, split=config.split, root=root, validate_audio=True
    )
    if args.smoke:
        dataset.frame = dataset.frame.iloc[:8].copy().reset_index(drop=True)
    loader = DataLoader(
        dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=0 if args.smoke else config.num_workers,
        collate_fn=EvaluationCollator(feature_extractor), pin_memory=device.type == "cuda",
    )

    predictions = []
    with torch.inference_mode():
        for batch in loader:
            inputs = {
                "input_values": batch["input_values"].to(device),
                "attention_mask": batch["attention_mask"].to(device),
            }
            logits = model(**inputs).logits
            if config.decoder == "greedy":
                hypotheses = tokenizer.batch_decode(logits.argmax(dim=-1))
            else:
                output_lengths = model._get_feat_extract_output_lengths(
                    batch["attention_mask"].sum(dim=-1)
                ).tolist()
                hypotheses = [
                    lm_decoder.decode(
                        logits[index, :length].float().cpu().numpy(),
                        beam_width=args.beam_width,
                    )
                    for index, length in enumerate(output_lengths)
                ]
            for metadata, hypothesis in zip(batch["metadata"], hypotheses):
                score = score_utterance(str(metadata["transcript"]), hypothesis)
                predictions.append({
                    **{key: json_value(value) for key, value in metadata.items()},
                    "reference_raw": str(metadata["transcript"]),
                    "hypothesis_raw": hypothesis,
                    **score,
                    "model": config.model,
                    "objective": config.objective,
                    "fold": config.fold,
                    "seed": config.seed,
                    "decoder": config.decoder,
                    "checkpoint_path": checkpoint_reference,
                    "checkpoint_sha256": checkpoint_hash,
                })

    totals = aggregate_edit_counts(predictions)
    manifest_hashes = sorted({
        str(row["split_manifest_sha256"])
        for row in predictions if row.get("split_manifest_sha256")
    })
    if len(manifest_hashes) > 1:
        raise ValueError(f"Expected at most one split manifest hash, got {manifest_hashes}.")
    manifest_hash = manifest_hashes[0] if manifest_hashes else None
    metrics = {
        **totals,
        "dataset": config.dataset,
        "split": config.split,
        "model": config.model,
        "objective": config.objective,
        "fold": config.fold,
        "seed": config.seed,
        "decoder": config.decoder,
        "checkpoint_path": checkpoint_reference,
        "checkpoint_sha256": checkpoint_hash,
        "checkpoint_global_step": checkpoint_metadata.get("global_step"),
        "split_manifest_sha256": manifest_hash,
        "normalization_version": NORMALIZATION_VERSION,
        "smoke": args.smoke,
        "beam_width": args.beam_width if config.decoder == "beam_4gram" else None,
        "language_model": lm_attrs,
    }
    output_dir = (
        root / config.output_dir / f"seed={config.seed}" /
        config.decoder / config.dataset
    )
    if args.smoke:
        output_dir = output_dir / "smoke"
    output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(predictions).to_parquet(output_dir / "predictions.parquet", index=False)
    (output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    resolved = {
        "evaluation": asdict(config),
        "runtime": {
            "code_commit": git_commit(root),
            "normalization_version": NORMALIZATION_VERSION,
            "evaluated_utterances": len(predictions),
            "smoke": args.smoke,
        },
        "artifacts": {
            "checkpoint_sha256": checkpoint_hash,
            "tokenizer_vocab_sha256": tokenizer_hash,
            "parquet_sha256": sha256(parquet_path),
            "split_manifest_sha256": manifest_hash,
            "language_model_sha256": {
                str(path.relative_to(lm_dir)): sha256(path) for path in lm_artifacts
            },
        },
    }
    (output_dir / "config.resolved.json").write_text(
        json.dumps(resolved, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(metrics, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
