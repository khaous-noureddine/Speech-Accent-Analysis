from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import soundfile as sf

from accented_asr.data.prepare_evaluation_data import (
    prepare_aesrc,
    prepare_librispeech,
    save,
)
from accented_asr.data.validate_evaluation_data import validate


def write_audio(path: Path, sample_rate: int = 16_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.zeros(sample_rate // 10, dtype=np.float32), sample_rate)


def test_librispeech_preparation_is_portable_and_valid(tmp_path):
    raw = tmp_path / "raw" / "test-clean" / "1" / "2"
    raw.mkdir(parents=True)
    (raw / "1-2.trans.txt").write_text("1-2-0000 HELLO WORLD\n", encoding="utf-8")
    write_audio(raw / "1-2-0000.flac")
    output = tmp_path / "data" / "processed" / "librispeech_test" / "corpus.parquet"
    frame = prepare_librispeech(raw.parent.parent, output, tmp_path)
    save(frame, output, tmp_path, "librispeech_test_clean")
    assert frame.loc[0, "audio_path"].startswith("data/processed/")
    report = validate(
        output, dataset="librispeech_test_clean", split="test", root=tmp_path
    )
    assert report["rows"] == 1
    assert report["valid"] is True


def test_aesrc_names_include_country_and_speaker_to_prevent_collisions(tmp_path):
    country = tmp_path / "raw" / "canadian speaking english speech data"
    for speaker in ("speaker_a", "speaker_b"):
        source = country / speaker / "shared.wav"
        write_audio(source)
        source.with_suffix(".txt").write_text("THE SAME PROMPT", encoding="utf-8")
    output = tmp_path / "data" / "processed" / "aesrc" / "corpus.parquet"
    frame = prepare_aesrc(tmp_path / "raw", output, tmp_path, seed=13)
    assert frame["utterance_id"].nunique() == 2
    assert frame["audio_path"].nunique() == 2
    assert set(frame["split"]) == {"test"}


def test_validator_rejects_missing_audio(tmp_path):
    parquet = tmp_path / "corpus.parquet"
    pd.DataFrame([{
        "dataset": "example", "speaker_id": "s1", "split": "test",
        "utterance_id": "u1", "transcript": "HELLO",
        "audio_path": "data/processed/example/wavs/missing.wav",
    }]).to_parquet(parquet, index=False)
    with pytest.raises(ValueError, match="missing audio files"):
        validate(parquet, dataset="example", split="test", root=tmp_path)
