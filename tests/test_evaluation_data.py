from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from accented_asr.data.prepare_evaluation_data import (
    prepare_aesrc,
    prepare_librispeech,
    prepare_speech_accent,
    save,
)


def write_audio(path: Path, sample_rate: int = 16_000) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, np.zeros(sample_rate // 10, dtype=np.float32), sample_rate)


def test_librispeech_preparation_is_portable_and_valid(tmp_path):
    raw = tmp_path / "raw" / "test-clean" / "1" / "2"
    raw.mkdir(parents=True)
    (raw / "1-2.trans.txt").write_text("1-2-0000 HELLO WORLD\n", encoding="utf-8")
    write_audio(raw / "1-2-0000.flac")
    output = tmp_path / "data" / "processed" / "librispeech_test_clean" / "corpus.parquet"
    frame = prepare_librispeech(raw.parent.parent, output, tmp_path)
    save(frame, output, tmp_path, "librispeech_test_clean")
    assert frame.loc[0, "audio_path"].startswith("data/processed/")
    saved = pd.read_parquet(output)
    assert len(saved) == 1
    assert (tmp_path / saved.loc[0, "audio_path"]).is_file()


def test_aesrc_names_include_country_and_speaker_to_prevent_collisions(tmp_path):
    raw = tmp_path / "data" / "raw" / "aesrc" / "data"
    country = raw / "canadian speaking english speech data"
    for speaker in ("speaker_a", "speaker_b"):
        source = country / speaker / "shared.wav"
        write_audio(source)
        source.with_suffix(".txt").write_text("THE SAME PROMPT", encoding="utf-8")
    output = tmp_path / "data" / "processed" / "aesrc" / "corpus.parquet"
    frame = prepare_aesrc(raw, output, tmp_path, seed=13)
    assert frame["utterance_id"].nunique() == 2
    assert frame["audio_path"].nunique() == 2
    assert set(frame["split"]) == {"test"}


def test_speech_accent_archive_skips_metadata_rows_without_mp3(tmp_path, monkeypatch):
    raw = tmp_path / "data" / "raw" / "speech_accent_archive"
    recordings = raw / "recordings" / "recordings"
    recordings.mkdir(parents=True)
    (raw / "reading-passage.txt").write_text("PLEASE CALL STELLA", encoding="utf-8")
    pd.DataFrame([
        {"filename": "available", "speakerid": 1, "file_missing?": False},
        {"filename": "nicaragua", "speakerid": 2, "file_missing?": False},
    ]).to_csv(raw / "speakers_all.csv", index=False)
    (recordings / "available.mp3").touch()
    monkeypatch.setattr(
        "accented_asr.data.prepare_evaluation_data.write_audio",
        lambda source, destination: 1.0,
    )
    output = tmp_path / "data" / "processed" / "speech_accent_archive" / "corpus.parquet"
    frame = prepare_speech_accent(raw, output, tmp_path)
    assert frame["utterance_id"].tolist() == ["available"]
