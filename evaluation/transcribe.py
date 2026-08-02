"""
eval/transcribe.py

Transcription pipeline for ASR evaluation.

For each (model, dataset) pair defined in the config, loads the model once,
transcribes all audio files in the dataset parquet, and saves a CSV with
the transcriptions alongside the reference transcripts and metadata.

Model config fields (shortcut mode)
------------------------------------
  shortcut        : str   Short label from MODEL_REGISTRY (e.g. "wav2vec2-base-960h")
  label           : str   Display name in results table (optional, defaults to shortcut)
  from_checkpoint : bool  Override: load weights from checkpoint_path
  checkpoint_path : str   Override: path to local checkpoint

Model config fields (full mode — backward compatible)
------------------------------------------------------
  name            : str   HuggingFace model id (e.g. "facebook/wav2vec2-base-960h")
  label           : str   Display name in results table (optional, defaults to name)
  from_checkpoint : bool  If True, load weights from checkpoint_path after HF init
  checkpoint_path : str   Path to local checkpoint (only used if from_checkpoint=True)

Dataset config fields
---------------------
  name            : str   Identifier for this dataset
  parquet         : str   Path to parquet file
  audio_col       : str   Column name for audio paths (default: "audio_path")
  transcript_col  : str   Column name for reference transcripts (default: "transcript")
  group_col       : str   Column for per-group WER breakdown (e.g. "native_language")
  filter_col      : str   Optional column to filter on (e.g. "split")
  filter_val      : str   Value to keep (e.g. "eval")

Usage
-----
  python evaluation/transcribe.py --config configs/evaluation/eval_config.yaml
"""

from __future__ import annotations

import argparse
import sys
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
import yaml
from loguru import logger
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

SAMPLE_RATE = 16_000


# ═══════════════════════════════════════════════════════════════════════════
# Model Registry — shortcut → (family, hf_model_name)
# ═══════════════════════════════════════════════════════════════════════════

MODEL_REGISTRY = {
    # ── Wav2Vec2 ───────────────────────────────────────────────────────────
    "wav2vec2-base":           ("wav2vec2", "facebook/wav2vec2-base"),
    "wav2vec2-large":          ("wav2vec2", "facebook/wav2vec2-large"),
    "wav2vec2-base-960h":      ("wav2vec2", "facebook/wav2vec2-base-960h"),
    "wav2vec2-large-960h":     ("wav2vec2", "facebook/wav2vec2-large-960h"),
    "wav2vec2-large-xlsr53":   ("wav2vec2", "facebook/wav2vec2-large-xlsr-53"),
    "xls-r-300m":              ("wav2vec2", "facebook/wav2vec2-xls-r-300m"),
    "xls-r-1b":                ("wav2vec2", "facebook/wav2vec2-xls-r-1b"),
    "xls-r-2b":                ("wav2vec2", "facebook/wav2vec2-xls-r-2b"),

    "xlsr-53-english":          ("wav2vec2", "jonatasgrosman/wav2vec2-large-xlsr-53-english"),
    # ── HuBERT ─────────────────────────────────────────────────────────────
    "hubert-base":             ("hubert",   "facebook/hubert-base-ls960"),
    "hubert-large":            ("hubert",   "facebook/hubert-large-ls960-ft"),
    # ── Whisper (multilingual) ─────────────────────────────────────────────
    "whisper-large-v3":        ("whisper",  "openai/whisper-large-v3"),
    "whisper-large-v2":        ("whisper",  "openai/whisper-large-v2"),
    "whisper-medium":          ("whisper",  "openai/whisper-medium"),
    "whisper-small":           ("whisper",  "openai/whisper-small"),
    "whisper-base":            ("whisper",  "openai/whisper-base"),
    "whisper-tiny":            ("whisper",  "openai/whisper-tiny"),
    # ── Whisper (English-only) ─────────────────────────────────────────────
    "whisper-medium.en":       ("whisper",  "openai/whisper-medium.en"),
    "whisper-small.en":        ("whisper",  "openai/whisper-small.en"),
    "whisper-base.en":         ("whisper",  "openai/whisper-base.en"),
    "whisper-tiny.en":         ("whisper",  "openai/whisper-tiny.en"),
}


# ═══════════════════════════════════════════════════════════════════════════
# Audio loading
# ═══════════════════════════════════════════════════════════════════════════

def load_audio(path: str, target_sr: int = SAMPLE_RATE) -> np.ndarray:
    """Load audio file as float32 mono array at target_sr."""
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != target_sr:
        import resampy
        audio = resampy.resample(audio, sr, target_sr)
    return audio


# ═══════════════════════════════════════════════════════════════════════════
# ASR Model classes
# ═══════════════════════════════════════════════════════════════════════════

class ASRModel(ABC):
    """Common interface for all ASR models."""

    def __init__(
        self,
        name:            str,
        label:           str,
        from_checkpoint: bool       = False,
        checkpoint_path: str | None = None,
        device:          str        = "cuda",
    ) -> None:
        self.name            = name
        self.label           = label
        self.from_checkpoint = from_checkpoint
        self.checkpoint_path = checkpoint_path
        self.device          = device

    @abstractmethod
    def load(self) -> None: ...

    @abstractmethod
    def transcribe_batch(self, audios: list[np.ndarray], sr: int = SAMPLE_RATE) -> list[str]: ...

    def transcribe(self, audio: np.ndarray, sr: int = SAMPLE_RATE) -> str:
        return self.transcribe_batch([audio], sr)[0]


class Wav2Vec2Model(ASRModel):

    def load(self) -> None:
        from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC

        if self.from_checkpoint and self.checkpoint_path:
            ckpt_path = Path(self.checkpoint_path)

            if ckpt_path.is_dir():
                logger.info(f"Loading Wav2Vec2ForCTC from local dir: {ckpt_path}")
                self.processor = Wav2Vec2Processor.from_pretrained(str(ckpt_path))
                self.model     = Wav2Vec2ForCTC.from_pretrained(str(ckpt_path)).to(self.device)

            elif ckpt_path.suffix == ".pt":
                logger.info(f"Init Wav2Vec2ForCTC from HF: {self.name}, loading weights from {ckpt_path}")
                self.processor = Wav2Vec2Processor.from_pretrained("facebook/wav2vec2-base-960h")
                self.model     = Wav2Vec2ForCTC.from_pretrained(
                    self.name,
                    vocab_size=len(self.processor.tokenizer),
                    ctc_loss_reduction="mean",
                    pad_token_id=self.processor.tokenizer.pad_token_id,
                    ignore_mismatched_sizes=True,
                ).to(self.device)

                # ckpt  = torch.load(ckpt_path, map_location=self.device)
                ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
                state = ckpt["model"] if "model" in ckpt else ckpt
                self.model.load_state_dict(state, strict=True)

            else:
                raise ValueError(f"Unknown checkpoint format: {ckpt_path}")
        else:
            logger.info(f"Loading Wav2Vec2ForCTC from HF: {self.name}")
            self.processor = Wav2Vec2Processor.from_pretrained(self.name)
            self.model     = Wav2Vec2ForCTC.from_pretrained(self.name).to(self.device)

        self.model.eval()
        logger.info(f"  [{self.label}] Wav2Vec2 loaded — vocab_size={len(self.processor.tokenizer)}")

    @torch.inference_mode()
    def transcribe_batch(self, audios: list[np.ndarray], sr: int = SAMPLE_RATE) -> list[str]:
        inputs = self.processor(
            audios, sampling_rate=sr, return_tensors="pt",
            padding=True, return_attention_mask=True,
        )
        logits   = self.model(
            input_values=inputs.input_values.to(self.device),
            attention_mask=inputs.attention_mask.to(self.device),
        ).logits
        pred_ids = torch.argmax(logits, dim=-1)
        return [t.strip() for t in self.processor.batch_decode(pred_ids)]


class HubertModel(ASRModel):

    def load(self) -> None:
        from transformers import Wav2Vec2Processor, HubertForCTC

        if self.from_checkpoint and self.checkpoint_path:
            ckpt_path = Path(self.checkpoint_path)
            if ckpt_path.is_dir():
                logger.info(f"Loading HuBERT from local dir: {ckpt_path}")
                self.processor = Wav2Vec2Processor.from_pretrained(str(ckpt_path))
                self.model     = HubertForCTC.from_pretrained(str(ckpt_path)).to(self.device)
            else:
                raise ValueError(f"HuBERT checkpoint must be a directory, got: {ckpt_path}")
        else:
            logger.info(f"Loading HuBERT from HF: {self.name}")
            self.processor = Wav2Vec2Processor.from_pretrained(self.name)
            self.model     = HubertForCTC.from_pretrained(self.name).to(self.device)

        self.model.eval()
        logger.info(f"  [{self.label}] HuBERT loaded — vocab_size={len(self.processor.tokenizer)}")

    @torch.inference_mode()
    def transcribe_batch(self, audios: list[np.ndarray], sr: int = SAMPLE_RATE) -> list[str]:
        inputs = self.processor(
            audios, sampling_rate=sr, return_tensors="pt",
            padding=True, return_attention_mask=True,
        )
        logits   = self.model(
            input_values=inputs.input_values.to(self.device),
            attention_mask=inputs.attention_mask.to(self.device),
        ).logits
        pred_ids = torch.argmax(logits, dim=-1)
        return [t.strip() for t in self.processor.batch_decode(pred_ids)]


class WhisperModel(ASRModel):

    def load(self) -> None:
        from transformers import AutoProcessor, AutoModelForSpeechSeq2Seq

        torch_dtype = torch.float16 if "cuda" in self.device else torch.float32

        if self.from_checkpoint and self.checkpoint_path:
            ckpt_path = Path(self.checkpoint_path)
            if ckpt_path.is_dir():
                logger.info(f"Loading Whisper from local dir: {ckpt_path}")
                self.processor = AutoProcessor.from_pretrained(str(ckpt_path))
                self.model     = AutoModelForSpeechSeq2Seq.from_pretrained(
                    str(ckpt_path), torch_dtype=torch_dtype, low_cpu_mem_usage=True,
                ).to(self.device)
            else:
                raise ValueError(f"Whisper checkpoint must be a directory, got: {ckpt_path}")
        else:
            logger.info(f"Loading Whisper from HF: {self.name}")
            self.processor = AutoProcessor.from_pretrained(self.name)
            self.model     = AutoModelForSpeechSeq2Seq.from_pretrained(
                self.name, torch_dtype=torch_dtype,
                low_cpu_mem_usage=True, use_safetensors=True,
            ).to(self.device)

        self.model.eval()
        self.torch_dtype = torch_dtype
        logger.info(f"  [{self.label}] Whisper loaded — dtype={torch_dtype}")

    @torch.inference_mode()
    def transcribe_batch(self, audios: list[np.ndarray], sr: int = SAMPLE_RATE) -> list[str]:
        inputs = self.processor(audios, sampling_rate=sr, return_tensors="pt", padding=True)
        input_features = inputs.input_features.to(self.device, dtype=self.torch_dtype)

        forced_decoder_ids = self.processor.get_decoder_prompt_ids(
            language="english", task="transcribe",
        )
        predicted_ids = self.model.generate(
            input_features, forced_decoder_ids=forced_decoder_ids,
        )
        return [t.strip() for t in self.processor.batch_decode(predicted_ids, skip_special_tokens=True)]


# ═══════════════════════════════════════════════════════════════════════════
# Model family → class mapping
# ═══════════════════════════════════════════════════════════════════════════

MODEL_CLASSES = {
    "wav2vec2": Wav2Vec2Model,
    "hubert":   HubertModel,
    "whisper":  WhisperModel,
}


def detect_family(name: str) -> str:
    name_lower = name.lower()
    if "whisper" in name_lower:
        return "whisper"
    if "hubert" in name_lower:
        return "hubert"
    return "wav2vec2"


# ═══════════════════════════════════════════════════════════════════════════
# Config resolution + model loading
# ═══════════════════════════════════════════════════════════════════════════

def resolve_model_cfg(model_cfg: dict) -> dict:
    """
    Resolve a model config entry.

    Shortcut mode (recommended):
        - shortcut: "wav2vec2-base-960h"
          label: "wav2vec2_base_960h"

    Shortcut + checkpoint override:
        - shortcut: "wav2vec2-large-xlsr53"
          label: "condition_C_baseline"
          from_checkpoint: true
          checkpoint_path: "checkpoints/conditions/C/stage_3/best_model"

    Full mode (backward compatible):
        - name: "facebook/wav2vec2-base-960h"
          label: "wav2vec2_base_960h"
          from_checkpoint: false
    """
    if "model" in model_cfg:
        key = model_cfg["model"]
        if key not in MODEL_REGISTRY:
            raise ValueError(
                f"Unknown shortcut '{key}'. "
                f"Valid: {sorted(MODEL_REGISTRY.keys())}"
            )
        family, hf_name = MODEL_REGISTRY[key]
        return {
            "name":            hf_name,
            "label":           model_cfg.get("label", key),
            "family":          family,
            "from_checkpoint": model_cfg.get("from_checkpoint", False),
            "checkpoint_path": model_cfg.get("checkpoint_path", None),
        }
    return model_cfg


def load_model(model_cfg: dict, device: str = "cuda") -> ASRModel:
    """Instantiate and load an ASR model from a (possibly shortcut) config."""
    cfg = resolve_model_cfg(model_cfg)

    name            = cfg["name"]
    label           = cfg.get("label", name)
    from_checkpoint = cfg.get("from_checkpoint", False)
    checkpoint_path = cfg.get("checkpoint_path", None)
    family          = cfg.get("family", detect_family(name))

    cls = MODEL_CLASSES.get(family)
    if cls is None:
        raise ValueError(f"Unknown model family '{family}'. Valid: {list(MODEL_CLASSES.keys())}")

    model = cls(
        name=name, label=label,
        from_checkpoint=from_checkpoint,
        checkpoint_path=checkpoint_path,
        device=device,
    )
    model.load()
    return model


# ═══════════════════════════════════════════════════════════════════════════
# Transcription
# ═══════════════════════════════════════════════════════════════════════════

def transcribe_dataset(
    model:          ASRModel,
    df:             pd.DataFrame,
    audio_col:      str = "audio_path",
    transcript_col: str = "transcript",
    batch_size:     int = 8,
) -> pd.DataFrame:
    predictions: list[str] = []
    n = len(df)

    for start in tqdm(range(0, n, batch_size), desc=f"  Transcribing ({model.label})"):
        end   = min(start + batch_size, n)
        batch = df.iloc[start:end]

        audios = []
        for _, row in batch.iterrows():
            try:
                audio = load_audio(row[audio_col])
                audios.append(audio)
            except Exception as e:
                logger.warning(f"Failed to load {row[audio_col]}: {e}")
                audios.append(np.zeros(SAMPLE_RATE, dtype=np.float32))

        try:
            preds = model.transcribe_batch(audios)
        except Exception as e:
            logger.error(f"Batch transcription failed: {e}")
            preds = ["[ERROR]"] * len(audios)

        predictions.extend(preds)

    result = df.copy()
    result["prediction"] = predictions
    result["reference"]  = result[transcript_col].astype(str).str.upper().str.strip()
    result["prediction"] = result["prediction"].str.upper().str.strip()
    result["model"]      = model.label

    return result


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()

    with open(args.config) as f:
        full_config = yaml.safe_load(f)

    # ── Experiment-aware output/checkpoint paths ─────────────────────────────
    if "experiment" in full_config and "name" in full_config["experiment"]:
        exp_name = full_config["experiment"]["name"]
        exp_dir = Path("experiments") / exp_name

        # Prefix evaluation output_dir
        if "evaluation" in full_config and "output_dir" in full_config["evaluation"]:
            full_config["evaluation"]["output_dir"] = str(
                exp_dir / full_config["evaluation"]["output_dir"]
            )

        # Prefix local Stage 3 checkpoint paths used by evaluation
        for model_cfg in full_config.get("evaluation", {}).get("models", []):
            ckpt_path = model_cfg.get("checkpoint_path", None)

            if ckpt_path and not str(ckpt_path).startswith(("experiments/", "/")):
                model_cfg["checkpoint_path"] = str(exp_dir / ckpt_path)

    eval_cfg     = full_config["evaluation"]
    model_cfgs   = eval_cfg["models"]
    dataset_cfgs = eval_cfg["datasets"]
    batch_size   = eval_cfg.get("batch_size", 8)
    device       = eval_cfg.get("device", "cuda")
    device       = device if torch.cuda.is_available() else "cpu"
    output_dir   = Path(eval_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Device: {device}")
    logger.info(f"Models: {len(model_cfgs)} | Datasets: {len(dataset_cfgs)}")

    if not model_cfgs:
        raise ValueError("No models in config")

    if not dataset_cfgs:
        raise ValueError("No datasets in config")

    # ── Loop models × datasets ─────────────────────────────────────────────
    for model_cfg in model_cfgs:
        resolved = resolve_model_cfg(model_cfg)
        model_label = resolved.get("label", resolved["name"])
        safe_label = model_label.replace("/", "_").replace(" ", "_")

        logger.info(f"═══ Model: {model_label} ═══")

        # ── Check which datasets are still missing for this model ──────────
        pending_datasets = []

        for ds_cfg in dataset_cfgs:
            ds_name = ds_cfg["name"]
            ds_dir   = output_dir / "transcriptions" / ds_name
            csv_path = ds_dir / f"{safe_label}.csv"

            if csv_path.exists():
                logger.info(f"  Already exists, will skip → {csv_path}")
            else:
                pending_datasets.append(ds_cfg)

        # If all CSVs for this model already exist, do not even load the model
        if not pending_datasets:
            logger.info(f"  All transcriptions already exist for {model_label}; skipping model load.")
            continue

        logger.info(f"═══ Loading model: {model_label} ═══")

        try:
            model = load_model(model_cfg, device=device)
        except Exception:
            logger.exception(f"Failed to load model '{model_label}'")
            raise

        for ds_cfg in pending_datasets:
            ds_name        = ds_cfg["name"]
            parquet_path   = ds_cfg["parquet"]
            audio_col      = ds_cfg.get("audio_col", "audio_path")
            transcript_col = ds_cfg.get("transcript_col", "transcript")
            group_col      = ds_cfg.get("group_col", None)
            filter_col     = ds_cfg.get("filter_col", None)
            filter_val     = ds_cfg.get("filter_val", None)

            logger.info(f"  ── Dataset: {ds_name} ──")

            # ── Output path ────────────────────────────────────────────────
            ds_dir = output_dir / "transcriptions" / ds_name
            ds_dir.mkdir(parents=True, exist_ok=True)
            csv_path = ds_dir / f"{safe_label}.csv"

            # Safety check, useful if file was created while script was running
            if csv_path.exists():
                logger.info(f"  Skipping existing transcription → {csv_path}")
                continue

            # ── Load parquet ───────────────────────────────────────────────
            if not Path(parquet_path).exists():
                raise FileNotFoundError(f"Parquet not found: {parquet_path}")

            df = pd.read_parquet(parquet_path)
            logger.info(f"  {len(df):,} rows loaded")

            # ── Optional filter ────────────────────────────────────────────
            if filter_col and filter_val and filter_col in df.columns:
                df = df[df[filter_col] == filter_val].reset_index(drop=True)
                logger.info(f"  After filtering {filter_col}='{filter_val}' → {len(df):,} rows")

            # ── Transcribe ─────────────────────────────────────────────────
            result_df = transcribe_dataset(
                model=model,
                df=df,
                audio_col=audio_col,
                transcript_col=transcript_col,
                batch_size=batch_size,
            )

            # ── Select output columns ──────────────────────────────────────
            out_cols = ["prediction", "reference", "model"]

            if audio_col in result_df.columns:
                out_cols.insert(0, audio_col)

            if group_col and group_col in result_df.columns:
                out_cols.insert(1, group_col)

            for col in ["speaker_id", "utterance_id", "native_language", "country", "accent"]:
                if col in result_df.columns and col not in out_cols:
                    out_cols.append(col)

            # ── Save ───────────────────────────────────────────────────────
            result_df[out_cols].to_csv(csv_path, index=False)
            logger.info(f"  Saved → {csv_path}  ({len(result_df):,} rows)")

        del model

        if "cuda" in device:
            torch.cuda.empty_cache()

    logger.info("All transcriptions complete.")

if __name__ == "__main__":
    main()