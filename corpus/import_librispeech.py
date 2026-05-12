# LibriSpeech train-clean-100 

"""
import_librispeech.py

Import a LibriSpeech subset into the project format.
Works for any split that shares the same directory structure:
    train-clean-100 / dev-clean / test-clean / ...

All rows are assigned the split label given by --split, so:
  - Call with --split train  for train-clean-100
  - Call with --split eval   for dev-clean

Follows the same conventions as import_arctic.py and import_l2_arctic.py:
  - One row per utterance
  - FLAC files converted to WAV and copied to --audio_dir
  - Parquet output with the canonical schema
  - Optional CSV mirror

LibriSpeech directory structure (after extracting the tar.gz):
    {subset}/
        {speaker_id}/
            {chapter_id}/
                {speaker_id}-{chapter_id}.trans.txt
                {speaker_id}-{chapter_id}-{utt_idx}.flac

Parquet schema (mirrors arctic / l2_arctic)
-------------------------------------------
speaker_id    str    e.g. "LS_1272"
gender        str    "male" | "female" | "unknown"
split         str    value of --split ("train" | "eval")
utterance_id  str    e.g. "1272-128104-0000"
transcript    str    normalised upper-case text
audio_path    str    absolute path to the converted WAV file
duration_s    float  audio duration in seconds

Usage
-----
    # train-clean-100 → split="train"
    python corpus/librispeech/import_librispeech.py \\
        --corpus_dir  ~/data/raw/LibriSpeech/train-clean-100 \\
        --output_parquet data/processed/librispeech/train/corpus.parquet \\
        --audio_dir      data/processed/librispeech/train/wavs \\
        --split train

    # dev-clean → split="eval"
    python corpus/librispeech/import_librispeech.py \\
        --corpus_dir  ~/data/raw/LibriSpeech/dev-clean \\
        --output_parquet data/processed/librispeech/eval/corpus.parquet \\
        --audio_dir      data/processed/librispeech/eval/wavs \\
        --split eval

    # Optional: restrict to a subset of speakers (useful for dev runs)
    python corpus/librispeech/import_librispeech.py \\
        --corpus_dir  ~/data/raw/LibriSpeech/train-clean-100 \\
        --output_parquet data/processed/librispeech/train/corpus.parquet \\
        --audio_dir      data/processed/librispeech/train/wavs \\
        --split train \\
        --speakers 1272 1462 1673
"""

from __future__ import annotations

import sys
import argparse
from pathlib import Path

import pandas as pd
import soundfile as sf
from loguru import logger
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from utils import normalize_transcript


# ---------------------------------------------------------------------------
# SPEAKERS.TXT parser
# ---------------------------------------------------------------------------

def load_speaker_metadata(corpus_dir: Path) -> dict[str, str]:
    """
    Parse SPEAKERS.TXT and return {speaker_id_str: gender}.

    The file lives either inside the subset dir or one level above,
    depending on how the tar.gz was extracted.

    Format (after the comment header):
        ID  |SEX| SUBSET           | MINUTES | NAME
        14  | F | train-clean-360  | ...     | ...
    """
    candidates = [
        corpus_dir / "SPEAKERS.TXT",
        corpus_dir.parent / "SPEAKERS.TXT",
    ]
    speakers_txt = next((p for p in candidates if p.exists()), None)

    if speakers_txt is None:
        logger.warning(
            "SPEAKERS.TXT not found (looked in {} and {}) — "
            "gender will be 'unknown' for all speakers.",
            candidates[0], candidates[1],
        )
        return {}

    gender_map: dict[str, str] = {}
    with open(speakers_txt, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            parts = [p.strip() for p in line.split("|")]
            if len(parts) < 2:
                continue
            spk_id, sex = parts[0], parts[1].upper()
            gender_map[spk_id] = "female" if sex == "F" else "male"

    logger.info("SPEAKERS.TXT: {} entries loaded from {}", len(gender_map), speakers_txt)
    return gender_map


# ---------------------------------------------------------------------------
# Transcript parser
# ---------------------------------------------------------------------------

def parse_trans_file(trans_path: Path) -> dict[str, str]:
    """
    Parse a .trans.txt file.

    Each line:  UTTERANCE-ID transcription text here
    Returns:    {"1272-128104-0000": "MISTER QUILTER ...", ...}
    """
    transcripts: dict[str, str] = {}
    for line in trans_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        utt_id, _, text = line.partition(" ")
        transcripts[utt_id] = normalize_transcript(text)
    return transcripts


# ---------------------------------------------------------------------------
# FLAC → WAV conversion
# ---------------------------------------------------------------------------

def flac_to_wav(flac_path: Path, wav_path: Path) -> float:
    """
    Convert a FLAC to WAV (LibriSpeech is already 16 kHz mono — no resampling).
    Returns the duration in seconds.
    Skips conversion if the WAV already exists.
    """
    if wav_path.exists():
        return sf.info(wav_path).duration

    audio, sr = sf.read(flac_path, dtype="float32")
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(wav_path, audio, sr)
    return len(audio) / sr


# ---------------------------------------------------------------------------
# Main import
# ---------------------------------------------------------------------------

def import_corpus(
    corpus_dir:     Path,
    output_parquet: Path,
    audio_dir:      Path,
    split:          str,
    output_csv:     Path | None  = None,
    speakers:       list[str] | None = None,
) -> pd.DataFrame:
    """
    Walk a LibriSpeech subset directory, convert FLACs to WAVs, build parquet.

    Parameters
    ----------
    corpus_dir      Root of the subset (e.g. train-clean-100/ or dev-clean/)
    output_parquet  Destination .parquet file
    audio_dir       Directory where WAVs will be written
    split           Label assigned to every row — "train" or "eval"
    output_csv      Optional CSV mirror
    speakers        If given, only import these speaker IDs (strings)
    """
    corpus_dir     = Path(corpus_dir)
    output_parquet = Path(output_parquet)
    audio_dir      = Path(audio_dir)

    audio_dir.mkdir(parents=True, exist_ok=True)
    output_parquet.parent.mkdir(parents=True, exist_ok=True)

    # ── Speaker gender map ─────────────────────────────────────────────────
    gender_map = load_speaker_metadata(corpus_dir)

    # ── Discover speaker directories ───────────────────────────────────────
    all_speaker_dirs = sorted([d for d in corpus_dir.iterdir() if d.is_dir()])
    if not all_speaker_dirs:
        raise ValueError(f"No speaker directories found in {corpus_dir}")

    if speakers is not None:
        requested = set(str(s) for s in speakers)
        unknown   = requested - {d.name for d in all_speaker_dirs}
        if unknown:
            raise ValueError(f"Requested speakers not found in corpus: {sorted(unknown)}")
        speaker_dirs = [d for d in all_speaker_dirs if d.name in requested]
    else:
        speaker_dirs = all_speaker_dirs

    logger.info(
        "Importing {} / {} speakers from {} as split='{}'",
        len(speaker_dirs), len(all_speaker_dirs), corpus_dir.name, split,
    )

    # ── Walk speaker → chapter → utterances ───────────────────────────────
    rows: list[dict] = []

    for spk_dir in tqdm(speaker_dirs, desc="speakers"):
        spk_id  = spk_dir.name
        gender  = gender_map.get(spk_id, "unknown")
        spk_tag = f"LS_{spk_id}"            # e.g. "LS_1272"  mirrors US_AWB style

        for chap_dir in sorted(d for d in spk_dir.iterdir() if d.is_dir()):
            trans_file = chap_dir / f"{spk_id}-{chap_dir.name}.trans.txt"
            if not trans_file.exists():
                logger.warning("Trans file missing: {}", trans_file)
                continue

            transcripts = parse_trans_file(trans_file)

            for utt_id, transcript in transcripts.items():
                flac_path = chap_dir / f"{utt_id}.flac"
                if not flac_path.exists():
                    logger.warning("FLAC missing: {}", flac_path)
                    continue

                wav_path = audio_dir / f"{spk_tag}_{utt_id}.wav"

                try:
                    duration = flac_to_wav(flac_path, wav_path)
                except Exception as exc:
                    logger.warning("Conversion failed {}: {}", flac_path.name, exc)
                    continue

                rows.append({
                    "speaker_id":   spk_tag,
                    "gender":       gender,
                    "split":        split,
                    "utterance_id": utt_id,
                    "transcript":   transcript,
                    "audio_path":   str(wav_path),
                    "duration_s":   round(duration, 3),
                })

    df = pd.DataFrame(rows)

    # ── Summary ────────────────────────────────────────────────────────────
    total_h = df["duration_s"].sum() / 3600
    logger.info("Total utterances   : {:,}", len(df))
    logger.info("Total duration     : {:.1f} h", total_h)
    logger.info("Speakers           : {}", df["speaker_id"].nunique())
    logger.info("Split              : {}", split)

    # ── Save ───────────────────────────────────────────────────────────────
    df.to_parquet(output_parquet, index=False)
    logger.info("Parquet → {}", output_parquet)

    if output_csv is not None:
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        logger.info("CSV    → {}", output_csv)

    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Import a LibriSpeech subset into the project format."
    )
    parser.add_argument(
        "--corpus_dir", type=Path, required=True,
        help="Path to the extracted subset directory (e.g. train-clean-100/ or dev-clean/).",
    )
    parser.add_argument(
        "--output_parquet", type=Path, required=True,
        help="Destination parquet file.",
    )
    parser.add_argument(
        "--audio_dir", type=Path, required=True,
        help="Directory where converted WAV files will be written.",
    )
    parser.add_argument(
        "--split", type=str, required=True, choices=["train", "eval", "test"],
        help="Split label assigned to all rows in this import.",
    )
    parser.add_argument(
        "--output_csv", type=Path, default=None,
        help="Optional CSV mirror of the parquet (for human inspection).",
    )
    parser.add_argument(
        "--speakers", nargs="+", default=None,
        help="Restrict import to these speaker IDs (default: all). E.g. --speakers 1272 1462",
    )

    args = parser.parse_args()

    import_corpus(
        corpus_dir     = args.corpus_dir,
        output_parquet = args.output_parquet,
        audio_dir      = args.audio_dir,
        split          = args.split,
        output_csv     = args.output_csv,
        speakers       = args.speakers,
    )