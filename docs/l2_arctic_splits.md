# L2-ARCTIC split protocols

The resubmission experiments use one immutable data-split seed, independently
from the random seeds used to train models. All compared conditions must consume
the exact same manifest.

## Input inventory

`scripts/build_l2_arctic_splits.py` expects an unfiltered Parquet inventory with
one row per `(speaker_id, prompt_id)` pair and these columns:

- `speaker_id`;
- `native_language`;
- `prompt_id`.

Other columns, such as `audio_path`, `transcript`, `gender`, and `duration_s`,
are preserved in generated corpora. The inventory must contain the six expected
L1 groups and exactly four speakers per L1.

Only prompts recorded by all 24 speakers are eligible. This conservative policy
ensures identical prompt availability across accents.

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

Each protocol/fold directory contains:

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

```bash
python scripts/build_l2_arctic_splits.py \
  --input-parquet data/processed/l2_arctic_inventory/corpus.parquet \
  --output-dir data/processed/l2_arctic_splits \
  --split-seed 20260817
```

The generated Parquet files belong under `data/` and are not committed. The
small JSON manifests should later be copied to `manifests/l2_arctic/` and
versioned once the protocol is frozen.
