from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from accented_asr.data.prepare_evaluation_data import (
    prepare_aesrc,
    prepare_librispeech,
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
