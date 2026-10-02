# Experiment layout

All active experiments follow the same high-level ordering:

```text
experiments/<methodology>/<LibriSpeech size>/<model>/<contrastive unit>/...
```

The active methodologies are:

- `baselines/internal`: CTC-only checkpoints trained locally from the common
  pretrained backbone, grouped by their LibriSpeech supervision budget.
- `separated`: Stage 2 accent adaptation followed by a distinct Stage 3 ASR
  fine-tuning. Stage-specific configuration and outputs remain separate.
- `joint`: CTC and SupCon are optimized in the same training run. Training
  checkpoints and both greedy and beam-search evaluations live below the same
  `outputs/seed=<seed>/` directory.
- `external-baselines`: published checkpoints loaded directly from their model
  repository; only evaluation outputs are produced locally.

## Internal baselines

```text
baselines/internal/librispeech-<100h|960h>/<model>/ctc-only/full-transformer/
├── config.yaml
├── evaluation.yaml
└── outputs/seed=<seed>/
```

The 100-hour configuration retains the historical `no-stage2` objective name
for compatibility. Both configurations are local CTC-only baselines; neither
uses supervised contrastive learning.

## Separated pipeline

```text
separated/librispeech-100h/<model>/utterance-supcon/<objective>/<accent>/
├── stage2/freeze-18/
│   ├── config.yaml
│   └── outputs/
└── stage3/full-transformer/
    ├── config.yaml
    ├── evaluation.yaml
    └── outputs/seed=<seed>/
        ├── checkpoint_best.pt
        ├── greedy/
        └── beam_4gram/
```

## Joint pipeline

Utterance-level experiments use the held-out L2-ARCTIC accent as their dataset
level:

```text
joint/librispeech-<100h|960h>/<model>/utterance-supcon/<accent>/<freeze mode>/
```

Word-level experiments use the accented word-corpus name and size:

```text
joint/librispeech-<100h|960h>/<model>/word-supcon/
└── mswc-common-voice-<50h|500h>/<freeze mode>/
```

Every final run directory contains `config.yaml`, optionally `evaluation.yaml`,
and `outputs/`. Evaluation artifacts are written below the same seed directory
as the checkpoints.

## External baselines

```text
external-baselines/librispeech-<100h|960h>/<published-model>/
├── evaluation.yaml
└── outputs/
```

`experiments/old/` is archival and is deliberately excluded from the active
layout.

After pulling the commit that introduced this hierarchy on a machine holding
existing checkpoints, migrate the ignored output directories once:

```bash
scripts/migrate_experiment_layout.sh
```

The migration refuses to overwrite an existing destination.

If training or evaluation jobs are still writing to the legacy hierarchy, use
the non-destructive two-phase migration instead:

```bash
scripts/migrate_experiment_layout.sh --live-copy
# Wait for every legacy-path job to finish.
scripts/migrate_experiment_layout.sh --finalize-copy
```

The live pass retains the legacy directories and marks copied output trees with
`.LIVE_MIGRATION_INCOMPLETE`. The final pass refreshes the copies and removes
the markers. Legacy directories remain available until they have been manually
verified and removed.

Verify every migrated best/final checkpoint against its legacy source before
removing any old directory:

```bash
scripts/migrate_experiment_layout.sh --verify
```

The command exits successfully only when every discovered legacy checkpoint is
present and byte-identical at its expected destination.

After all jobs have stopped and verification succeeds, remove the duplicated
legacy hierarchy in one guarded operation:

```bash
scripts/cleanup_legacy_experiment_layout.sh --delete
```

The cleanup refuses to run while Slurm jobs exist for the current user. It
finalizes and verifies migration before deleting only the known legacy paths.
The active experiment roots remain `baselines`, `joint`, `separated`,
`external-baselines`, and `old`.
