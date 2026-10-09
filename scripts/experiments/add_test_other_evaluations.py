#!/usr/bin/env python3
"""Add LibriSpeech test-other to the evaluation configs used in the paper."""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EXPERIMENTS = ROOT / "experiments"
PATTERNS = (
    "baselines/internal/librispeech-100h/wav2vec2-large-lv60/ctc-only/full-transformer/evaluation.yaml",
    "baselines/prior-methods/librispeech-100h/wav2vec2-large-lv60/multidomain-ctc/*/full-transformer/evaluation.yaml",
    "joint/librispeech-100h/wav2vec2-large-lv60/utterance-supcon/*/full-transformer/evaluation.yaml",
    "joint/librispeech-100h/wav2vec2-large-lv60/word-supcon/mswc-common-voice-50h/full-transformer/evaluation.yaml",
    "joint/librispeech-100h/wav2vec2-large-lv60/md-ft-cp-supcon/**/full-transformer/evaluation.yaml",
    "joint/librispeech-100h/wav2vec2-large-lv60/controls/shuffled-label-supcon/*/full-transformer/evaluation.yaml",
)
TEST_OTHER_BLOCK = (
    '    librispeech_test_other:\n'
    '      split: "test"\n'
    '      parquet: "data/processed/librispeech_test_other/corpus.parquet"\n'
    '      raw_dir: "data/raw/librispeech/test/LibriSpeech/test-other"\n'
)


def add_after_test_clean(text: str) -> str:
    if "librispeech_test_other:" in text:
        return text
    lines = text.splitlines(keepends=True)
    start = next(
        (index for index, line in enumerate(lines) if line.strip() == "librispeech_test_clean:"),
        None,
    )
    if start is None:
        raise ValueError("Evaluation config has no librispeech_test_clean dataset.")
    end = start + 1
    while end < len(lines):
        line = lines[end]
        if line.startswith("    ") and not line.startswith("      ") and line.strip():
            break
        end += 1
    lines.insert(end, TEST_OTHER_BLOCK)
    return "".join(lines)


def main() -> None:
    paths = sorted({path for pattern in PATTERNS for path in EXPERIMENTS.glob(pattern)})
    if not paths:
        raise RuntimeError("No evaluation configs matched the requested experiments.")
    for path in paths:
        text = path.read_text(encoding="utf-8")
        path.write_text(add_after_test_clean(text), encoding="utf-8")
    print(f"Updated {len(paths)} evaluation configs.")


if __name__ == "__main__":
    main()
