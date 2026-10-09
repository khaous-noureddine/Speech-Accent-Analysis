"""Prepare canonical evaluation corpora from raw LibriSpeech, AESRC, SAA, or EDACC."""

from __future__ import annotations

import argparse
import random
import re
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import soundfile as sf


SAMPLE_RATE = 16_000
AESRC_TEST_ACCENTS = {"Canadian", "Spanish"}
AESRC_COUNTRIES = {
    "british english speech data": "British",
    "american english speech data": "American",
    "russian speaking english speech data": "Russian",
    "korean speaking english speech data": "Korean",
    "canadian speaking english speech data": "Canadian",
    "portuguese speaking english speech data": "Portuguese",
    "japanese speaking english speech data": "Japanese",
    "spanish speaking english speech data": "Spanish",
    "india english speech data": "Indian",
    "chinese speaking english speech data": "Chinese",
    "russian english speech data": "Russian",
    "korean english speech data": "Korean",
    "canadian english speech data": "Canadian",
    "portuguese english speech data": "Portuguese",
    "japanese english speech data": "Japanese",
    "spanish english speech data": "Spanish",
    "chinese english speech data": "Chinese",
    "indian english speech data": "Indian",
}


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def portable(path: Path, root: Path) -> str:
    path = path.resolve()
    try:
        return str(path.relative_to(root.resolve()))
    except ValueError as error:
        raise ValueError(f"Processed audio must live below repository root: {path}") from error


def write_audio(source: Path, destination: Path, target_rate: int = SAMPLE_RATE) -> float:
    if not destination.exists():
        audio, _ = librosa.load(source, sr=target_rate, mono=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        sf.write(destination, audio, target_rate)
    info = sf.info(destination)
    if info.samplerate != target_rate or info.channels != 1:
        raise ValueError(f"Invalid processed audio format: {destination}")
    return float(info.duration)


def parse_libri_speakers(corpus_dir: Path) -> dict[str, str]:
    candidates = (corpus_dir / "SPEAKERS.TXT", corpus_dir.parent / "SPEAKERS.TXT")
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        return {}
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip() or line.startswith(";"):
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) >= 2:
            result[parts[0]] = "female" if parts[1].upper() == "F" else "male"
    return result


def prepare_librispeech(
    raw: Path,
    output: Path,
    root: Path,
    dataset_name: str = "librispeech_test_clean",
) -> pd.DataFrame:
    genders = parse_libri_speakers(raw)
    audio_dir = output.parent / "wavs"
    rows = []
    for transcript_file in sorted(raw.glob("*/*/*.trans.txt")):
        speaker = transcript_file.parent.parent.name
        for line in transcript_file.read_text(encoding="utf-8").splitlines():
            utterance, separator, transcript = line.strip().partition(" ")
            source = transcript_file.parent / f"{utterance}.flac"
            if not separator or not source.is_file():
                raise ValueError(f"Malformed LibriSpeech entry for {utterance}: {source}")
            destination = audio_dir / f"LS_{speaker}_{utterance}.wav"
            rows.append({
                "dataset": dataset_name,
                "speaker_id": f"LS_{speaker}",
                "gender": genders.get(speaker, "unknown"),
                "split": "test",
                "utterance_id": utterance,
                "prompt_id": utterance,
                "transcript": transcript.strip(),
                "audio_path": portable(destination, root),
                "duration_s": write_audio(source, destination),
            })
    return pd.DataFrame(rows)


def parse_aesrc_metadata(path: Path) -> dict[str, str]:
    result = {}
    if not path.is_file():
        return result
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        parts = line.strip().split(None, 1)
        if parts and not parts[0].startswith("CMT"):
            result[parts[0]] = parts[1].strip() if len(parts) == 2 else ""
    return result


def aesrc_split(country: str, speakers: list[Path], seed: int) -> dict[str, str]:
    if country in AESRC_TEST_ACCENTS:
        return {speaker.name: "test" for speaker in speakers}
    names = [speaker.name for speaker in speakers]
    shuffled = names.copy()
    random.Random(seed).shuffle(shuffled)
    dev_count = max(1, min(len(names) - 1, round(len(names) * 0.10))) if len(names) >= 2 else 0
    dev = set(shuffled[:dev_count])
    return {name: "dev" if name in dev else "train" for name in names}


def prepare_aesrc(raw: Path, output: Path, root: Path, seed: int) -> pd.DataFrame:
    audio_dir = output.parent / "wavs"
    rows = []
    for country_dir in sorted(path for path in raw.iterdir() if path.is_dir()):
        country = AESRC_COUNTRIES.get(country_dir.name.lower().strip(), country_dir.name)
        speakers = sorted(path for path in country_dir.iterdir() if path.is_dir())
        split_by_speaker = aesrc_split(country, speakers, seed)
        for speaker_dir in speakers:
            speaker_id = f"AESRC_{slug(country)}_{slug(speaker_dir.name)}"
            for source in sorted(speaker_dir.glob("*.wav")):
                transcript_path = source.with_suffix(".txt")
                if not transcript_path.is_file():
                    raise ValueError(f"Missing AESRC transcript: {transcript_path}")
                metadata = parse_aesrc_metadata(source.with_suffix(".metadata"))
                utterance_id = f"{speaker_id}_{slug(source.stem)}"
                destination = audio_dir / f"{utterance_id}.wav"
                rows.append({
                    "dataset": "aesrc",
                    "speaker_id": speaker_id,
                    "gender": metadata.get("SEX", "").lower(),
                    "age": pd.to_numeric(metadata.get("AGE"), errors="coerce"),
                    "accent": metadata.get("ACT", country) or country,
                    "country": country,
                    "split": split_by_speaker[speaker_dir.name],
                    "utterance_id": utterance_id,
                    "prompt_id": slug(source.stem),
                    "transcript": transcript_path.read_text(
                        encoding="utf-8", errors="replace"
                    ).strip(),
                    "audio_path": portable(destination, root),
                    "duration_s": write_audio(source, destination),
                })
    frame = pd.DataFrame(rows)
    seen = set(frame.loc[frame.split.isin(["train", "dev"]), "transcript"])
    frame = frame.loc[~((frame.split == "test") & frame.transcript.isin(seen))].copy()
    return frame.reset_index(drop=True)


def prepare_speech_accent(raw: Path, output: Path, root: Path) -> pd.DataFrame:
    metadata_path = raw / "speakers_all.csv"
    passage_path = raw / "reading-passage.txt"
    recordings = raw / "recordings" / "recordings"
    for path in (metadata_path, passage_path, recordings):
        if not path.exists():
            raise FileNotFoundError(path)
    metadata = pd.read_csv(metadata_path)
    if "file_missing?" in metadata:
        missing = (
            metadata["file_missing?"]
            .astype(str)
            .str.strip()
            .str.upper()
            .isin({"TRUE", "YES", "1"})
        )
        metadata = metadata.loc[~missing].copy()
    transcript = passage_path.read_text(encoding="utf-8").strip()
    audio_dir = output.parent / "wavs"
    rows = []
    missing_sources = []
    for _, row in metadata.iterrows():
        filename = str(row["filename"])
        source = recordings / f"{filename}.mp3"
        if not source.is_file():
            missing_sources.append(source)
            continue
        destination = audio_dir / f"{filename}.wav"
        rows.append({
            "dataset": "speech_accent_archive",
            "speaker_id": f"SAA_{int(row['speakerid'])}",
            "gender": row.get("sex"),
            "accent": row.get("native_language"),
            "native_language": row.get("native_language"),
            "country": row.get("country"),
            "age": row.get("age"),
            "age_onset": row.get("age_onset"),
            "birthplace": row.get("birthplace"),
            "split": "test",
            "utterance_id": filename,
            "prompt_id": "stella_passage",
            "transcript": transcript,
            "audio_path": portable(destination, root),
            "duration_s": write_audio(source, destination),
        })
    if missing_sources:
        print(
            f"Skipped {len(missing_sources)} Speech Accent Archive rows whose "
            "MP3 file is absent."
        )
        for source in missing_sources[:10]:
            print(f"Missing MP3: {source}")
    return pd.DataFrame(rows)


def prepare_edacc(raw: Path, output: Path, root: Path) -> pd.DataFrame:
    try:
        from datasets import Audio, load_dataset
    except ImportError as error:
        raise RuntimeError("EDACC preparation requires the 'datasets' dependency.") from error
    dataset = load_dataset("edinburghcstr/edacc", split="test", cache_dir=str(raw))
    dataset = dataset.cast_column("audio", Audio(sampling_rate=SAMPLE_RATE))
    audio_dir = output.parent / "wavs"
    rows = []
    for index, example in enumerate(dataset):
        speaker = str(example["speaker"])
        utterance = f"EDACC_{slug(speaker)}_{index:06d}"
        destination = audio_dir / f"{utterance}.wav"
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            sf.write(destination, np.asarray(example["audio"]["array"], dtype=np.float32), SAMPLE_RATE)
        rows.append({
            "dataset": "edacc",
            "speaker_id": speaker,
            "gender": example["gender"],
            "accent": example["accent"],
            "raw_accent": example["raw_accent"],
            "native_language": example["l1"],
            "split": "test",
            "utterance_id": utterance,
            "prompt_id": utterance,
            "transcript": example["text"],
            "audio_path": portable(destination, root),
            "duration_s": float(sf.info(destination).duration),
        })
    return pd.DataFrame(rows)


def save(frame: pd.DataFrame, output: Path, root: Path, dataset: str) -> None:
    if frame.empty:
        raise ValueError(f"No rows prepared for {dataset}.")
    if frame["utterance_id"].duplicated().any() or frame["audio_path"].duplicated().any():
        raise ValueError(f"{dataset} contains duplicate utterance IDs or audio paths.")
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=(
        "librispeech_test_clean", "librispeech_test_other", "aesrc",
        "speech_accent_archive", "edacc"
    ))
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--output-parquet", required=True, type=Path)
    parser.add_argument("--repository-root", type=Path, default=Path.cwd())
    parser.add_argument("--seed", type=int, default=20260817)
    args = parser.parse_args()
    root = args.repository_root.resolve()
    raw = (root / args.raw_dir).resolve() if not args.raw_dir.is_absolute() else args.raw_dir
    output = (root / args.output_parquet).resolve()
    if not raw.exists():
        raise FileNotFoundError(f"Missing raw {args.dataset} data: {raw}")
    builders = {
        "librispeech_test_clean": lambda: prepare_librispeech(
            raw, output, root, "librispeech_test_clean"
        ),
        "librispeech_test_other": lambda: prepare_librispeech(
            raw, output, root, "librispeech_test_other"
        ),
        "aesrc": lambda: prepare_aesrc(raw, output, root, args.seed),
        "speech_accent_archive": lambda: prepare_speech_accent(raw, output, root),
        "edacc": lambda: prepare_edacc(raw, output, root),
    }
    save(builders[args.dataset](), output, root, args.dataset)


if __name__ == "__main__":
    main()
