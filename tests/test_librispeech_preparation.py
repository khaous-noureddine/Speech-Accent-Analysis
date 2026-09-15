from pathlib import Path

import numpy as np
import soundfile as sf

from accented_asr.data.prepare_librispeech import build_inventory, save_inventory


def make_subset(root: Path, subset: str, speaker: str, utterance: str) -> Path:
    chapter = root / "data/raw/librispeech/train/LibriSpeech" / subset / speaker / "1"
    chapter.mkdir(parents=True)
    (chapter / f"{speaker}-1.trans.txt").write_text(
        f"{utterance}  a   short sentence \n", encoding="utf-8"
    )
    sf.write(chapter / f"{utterance}.flac", np.zeros(1600, dtype=np.float32), 16000)
    return chapter.parents[1]


def test_multiple_librispeech_subsets_share_one_portable_manifest(tmp_path):
    clean = make_subset(tmp_path, "train-clean-100", "1", "1-1-0001")
    other = make_subset(tmp_path, "train-other-500", "2", "2-1-0001")
    frame = build_inventory([clean, other], tmp_path)
    assert frame["subset"].tolist() == ["train-clean-100", "train-other-500"]
    assert frame["transcript"].tolist() == ["A SHORT SENTENCE"] * 2
    assert all(not Path(path).is_absolute() for path in frame["audio_path"])

    output = tmp_path / "data/processed/librispeech_960/corpus.parquet"
    save_inventory(frame, output, tmp_path)
    assert output.is_file()
    assert (output.parent / "inventory_report.json").is_file()
