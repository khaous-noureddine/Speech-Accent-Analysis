# Evaluation datasets

Evaluation datasets follow the same repository layout as the training data:

```text
data/
├── raw/
│   ├── librispeech/
│   │   └── test/LibriSpeech/test-clean/
│   ├── aesrc/
│   │   └── data/
│   ├── speech_accent_archive/
│   └── edacc/
└── processed/
    ├── librispeech_test_clean/
    │   ├── corpus.parquet
    │   └── wavs/
    ├── aesrc/
    │   ├── corpus.parquet
    │   └── wavs/
    ├── speech_accent_archive/
    │   ├── corpus.parquet
    │   └── wavs/
    └── edacc/
        ├── corpus.parquet
        └── wavs/
```

L2-ARCTIC keeps its existing leave-one-accent-out structure under
`data/raw/l2_arctic/` and
`data/processed/l2_arctic_leave_one_accent_out/`.

The four external datasets share one preparation entry point:

```bash
PYTHONPATH=src python -m accented_asr.data.prepare_evaluation_data --help
```

Normally it is not called manually. The evaluation config records the raw and
processed locations for every dataset. If a declared `corpus.parquet` is
missing, Snakemake prepares it before evaluation. Existing processed data is
reused unless the preparation code is newer, following normal Snakemake
dependency tracking.

Before inference, the evaluation loader checks the required columns, selects
the configured test split, and verifies that every referenced audio file
exists. Preparation also rejects empty corpora and duplicate utterance or
audio identifiers. No separate validation directory or validation rule is
used.

The preparation logic was migrated from:

- `corpus/import_librispeech.py`;
- `corpus/import_aesrc.py`;
- `corpus/import_speech_accent.py`;
- `corpus/import_edacc.py`.

The old scripts remain available for historical reference. New evaluation
runs use `src/accented_asr/data/prepare_evaluation_data.py`.
