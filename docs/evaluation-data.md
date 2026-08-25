# Evaluation data preparation

## Canonical pipeline

All evaluation corpora are prepared by one entry point:

```bash
python -m accented_asr.data.prepare_evaluation_data --help
```

It replaces the evaluation-related legacy importers in `corpus/` without
deleting them. The supported dataset keys are:

| Dataset key | Raw default used by experiment configs | Processed parquet | Evaluated split |
|---|---|---|---|
| `librispeech_test_clean` | `data/raw/librispeech/test/LibriSpeech/test-clean` | `data/processed/evaluation/librispeech_test_clean/corpus.parquet` | official `test-clean` |
| `aesrc` | `data/raw/aesrc` | `data/processed/evaluation/aesrc/corpus.parquet` | project-held-out Canadian and Spanish accents |
| `speech_accent_archive` | `data/raw/speech_accents` | `data/processed/evaluation/speech_accent_archive/corpus.parquet` | all usable speakers |
| `edacc` | `data/raw/edacc` | `data/processed/evaluation/edacc/corpus.parquet` | official test split |

The output schema always includes `dataset`, `speaker_id`, `split`,
`utterance_id`, `transcript`, and a repository-relative `audio_path`. Prepared
audio is mono 16 kHz. Speaker/utterance pairs and audio paths must be unique;
L2-ARCTIC may legitimately reuse a prompt-like utterance ID across speakers.

One Snakemake evaluation config declares a mapping of dataset names to their
`raw_dir`, `parquet`, and split. A single launch expands all mapping entries.
If the parquet is missing, Snakemake runs `prepare_evaluation_dataset`. Existing
parquets are never silently overwritten merely because code changed. It then always requires
`validate_evaluation_data`, which checks the schema, requested split, empty
transcripts, duplicate IDs, duplicate audio paths, and every audio file before
GPU inference starts.

## Legacy importer audit

Audit performed on 2026-08-25:

| Legacy script | Finding | Migration decision |
|---|---|---|
| `corpus/import_librispeech.py` | Generic and mostly correct, but depended on root `utils.py` and did not record dataset identity | migrated with strict missing-transcript/audio failures and portable paths; its otherwise valid old parquet remains validator-compatible |
| `corpus/import_speech_accent.py` | Correctly converts the common passage, but lacked canonical split, speaker, utterance, and duration fields | migrated as a fixed test corpus with canonical identifiers |
| `corpus/import_edacc.py` | Correctly used the HF test split, but emitted absolute paths and depended on an undeclared `datasets` package | migrated with portable paths; `datasets>=4,<5` is pinned |
| `corpus/import_aesrc.py` | Its speaker split and transcript-overlap guard are useful, but destination WAV names used only the source utterance stem | migrated with country/speaker-qualified IDs to prevent collisions |

The pre-existing LLF snapshot showed:

- LibriSpeech test-clean: 2,620 rows, all audio present;
- Speech Accent Archive: 2,138 rows, all audio present, but non-canonical schema;
- AESRC: 162,974 rows, all stored paths stale/absolute and 5,833 duplicated
  audio paths, so this parquet must not be evaluated;
- EDACC: no processed parquet available.

These observations describe that server snapshot, not an accepted paper result.
The new validator is the authoritative acceptance test on every server.

## Direct rebuild commands

Normally these commands are invoked automatically by Snakemake. They remain
available for diagnosis:

```bash
PYTHONPATH=src python -m accented_asr.data.prepare_evaluation_data \
  --dataset librispeech_test_clean \
  --raw-dir data/raw/librispeech/test/LibriSpeech/test-clean \
  --output-parquet data/processed/evaluation/librispeech_test_clean/corpus.parquet

PYTHONPATH=src python -m accented_asr.data.prepare_evaluation_data \
  --dataset aesrc --raw-dir data/raw/aesrc \
  --output-parquet data/processed/evaluation/aesrc/corpus.parquet

PYTHONPATH=src python -m accented_asr.data.prepare_evaluation_data \
  --dataset speech_accent_archive --raw-dir data/raw/speech_accents \
  --output-parquet data/processed/evaluation/speech_accent_archive/corpus.parquet

PYTHONPATH=src python -m accented_asr.data.prepare_evaluation_data \
  --dataset edacc --raw-dir data/raw/edacc \
  --output-parquet data/processed/evaluation/edacc/corpus.parquet
```

`corpus/import_afrispeech.py` also exists in the legacy tree. AfriSpeech is not
part of the currently frozen evaluation set; adding it requires an explicit
protocol decision and a new audited config rather than silently including it.
