# Evaluation experiments

Evaluation is organized first by backbone, then by Stage 2 objective, fold,
seed, decoder, and dataset:

```text
experiments/eval/<model>/<objective>/<fold>/seed=<seed>/<decoder>/<dataset>/
```

`<fold>` is omitted for `no-stage2`, which has no accent-dependent adaptation.
Generated prediction and score artifacts remain separate from Stage 3
checkpoints; their resolved configuration records the checkpoint path and
SHA-256 for provenance.

The complete scoring, aggregation, decoder, and output contracts are specified
in [`docs/evaluation-doc.md`](../../docs/evaluation-doc.md).

The first integration config evaluates the Arabic-held-out SupCon-only model
with greedy CTC decoding. On a Slurm host, validate eight utterances first:

```bash
sbatch scripts/slurm/run_evaluation.sbatch \
  experiments/eval/wav2vec2-large-lv60/supcon-only/arabic/config.yaml --smoke
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
