# L2-ARCTIC split protocols

The resubmission experiments use one immutable data-split seed, independently
from the random seeds used to train models. All compared conditions must consume
the exact same manifest.

## Input inventory

The canonical inventory is stored at
`data/processed/l2_arctic_cv/resubmission/inventory.parquet`. It is built from
the union of the eight legacy processed Parquets, which are used only as
per-audio metadata sources. Their former train/dev/test assignments are ignored.

The inventory has one row per `(speaker_id, prompt_id)` pair and includes:

- `speaker_id`;
- `native_language`;
- `prompt_id`.

Other columns, such as `audio_path`, `transcript`, `gender`, and `duration_s`,
are preserved in generated corpora. The inventory must contain the six expected
L1 groups and exactly four speakers per L1.

Only prompts with both audio and per-speaker transcript metadata for all 24
speakers are eligible. This conservative policy ensures identical prompt
availability across accents. The canonical inventory contains 953 prompts and
22,872 examples. All audio paths reference the single existing directory
`data/processed/l2_arctic_cv/wavs`; no audio is duplicated.

## Main protocol

For every L1, two speakers are assigned to train, one to development, and one to
test. The assignment is deterministic for a given split seed. Prompts are
partitioned globally using an 80/10/10 ratio, so neither prompts nor speakers
overlap across train, development, and test.

This protocol evaluates unseen speakers and unseen prompts, but not unseen L1
groups: all six L1 groups occur in every split.

## Leave-one-L1-out protocol

Six folds are generated. In each fold:

- all four speakers from one L1 are reserved for test;
- for each of the five seen L1 groups, three speakers are used for train and one
  for development;
- the global 80/10/10 prompt partition remains disjoint.

The strict test set therefore combines an unseen L1, unseen speakers, and unseen
prompts. This must be stated explicitly when reporting zero-shot results.

## Generated artifacts

Split Parquets are generated under the ignored data directory:

```text
data/processed/l2_arctic_cv/resubmission/splits/
├── main/corpus.parquet
└── leave_one_l1_out/<l1>/corpus.parquet
```

Small, versioned metadata is stored under `manifests/l2_arctic/`. Each
protocol/fold directory contains:

- `corpus.parquet`;
- `manifest.json`;
- `manifest.content.sha256` (hash of the canonical manifest payload, excluding
  its self-referential `sha256` field) ;
- `validation_report.json` ;
- `split_stats.csv`.

The manifest records the source-inventory fingerprint, split seed, prompt lists,
speaker roles, held-out L1, and its own deterministic SHA-256. Every training run
must copy the manifest hash into its metadata and checkpoint.

## Usage

All Python CLIs live in the package rather than in the cluster-launcher folder.
Run them from the repository root with `PYTHONPATH=src`.

Build the canonical inventory:

```bash
PYTHONPATH=src python -m accented_asr.cli.build_l2_arctic_inventory \
  --reference-parquet data/processed/l2_arctic_cv/fold_*/corpus.parquet \
  --wav-dir data/processed/l2_arctic_cv/wavs \
  --output-parquet data/processed/l2_arctic_cv/resubmission/inventory.parquet \
  --report manifests/l2_arctic/inventory_report.json \
  --repository-root .
```

Generate the canonical splits:

```bash
PYTHONPATH=src python -m accented_asr.cli.build_l2_arctic_splits \
  --input-parquet data/processed/l2_arctic_cv/resubmission/inventory.parquet \
  --output-dir data/processed/l2_arctic_cv/resubmission/splits \
  --manifest-dir manifests/l2_arctic \
  --split-seed 20260817
```

Perform a full reproducibility audit, including opening all 22,872 audio
headers:

```bash
PYTHONPATH=src python -m accented_asr.cli.verify_l2_arctic_artifacts \
  --inventory data/processed/l2_arctic_cv/resubmission/inventory.parquet \
  --inventory-report manifests/l2_arctic/inventory_report.json \
  --split-data-dir data/processed/l2_arctic_cv/resubmission/splits \
  --manifest-dir manifests/l2_arctic \
  --repository-root . \
  --check-audio
```

This command exits with an error if an audio is missing/unreadable, a hash does
not match, a split Parquet differs from its manifest, or any prompt, speaker, or
held-out-L1 invariant is violated.
