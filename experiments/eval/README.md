# Evaluation experiments

Evaluation is organized first by backbone, then by Stage 2 objective, fold,
seed, decoder, and dataset:

```text
experiments/eval/<model>/<objective>/<fold>/
├── config.yaml
└── outputs/seed=<seed>/<decoder>/<dataset>/
```

`<fold>` is omitted for `no-stage2`, which has no accent-dependent adaptation.
Generated prediction and score artifacts all live below the fold-local
`outputs/` directory, which is ignored by Git. The neighboring `config.yaml`
remains versioned. Resolved outputs record the checkpoint path and SHA-256 for
provenance.

A complete launch also creates one checkpoint-level summary:

```text
outputs/seed=<seed>/<decoder>/metrics_summary.json
```

Smoke runs use `metrics_summary.smoke.json`. The summary contains one entry per
dataset with WER both as a ratio and percentage, utterance/reference counts,
and substitution, deletion, and insertion counts. Detailed predictions remain
inside each dataset directory.

The evaluation launchers use Snakemake `--nolock` so an Arabic evaluation can
run while an unrelated Stage 3 fold is training in the same worktree. This is
safe only because their declared outputs are disjoint. Never submit two jobs
that write the same objective/fold/seed evaluation outputs concurrently.

For the complete six-fold SupCon-only campaign, the Slurm array launcher maps
indices 0-5 to Arabic, Chinese, Hindi, Korean, Spanish, and Vietnamese. Its
`%1` limit evaluates only one fold at a time:

```bash
sbatch scripts/slurm/run_evaluation.sbatch supcon-only 13
```

The complete scoring, aggregation, decoder, and output contracts are specified
in [`docs/evaluation-doc.md`](../../docs/evaluation-doc.md).

The first integration config evaluates the Arabic-held-out SupCon-only model
with greedy CTC decoding. On a Slurm host, validate eight utterances first:

```bash
sbatch --array=0 scripts/slurm/run_evaluation.sbatch supcon-only 13 --smoke
```

Remove `--smoke` to evaluate all 380 matching test utterances. Smoke metrics
only validate plumbing and must never be copied into result tables.

One YAML config represents one checkpoint and lists all evaluation datasets.
The evaluation DAG expands it into one job per dataset. For L2-ARCTIC, the
listed parquet is necessarily the matching held-out fold; the Arabic model
therefore uses only the Arabic test subset. For every dataset, the DAG prepares
a missing processed parquet and the evaluation loader checks its schema and
audio paths before inference. See
[`docs/evaluation-data.md`](../../docs/evaluation-data.md) for the audited raw
layouts and rebuild commands.
