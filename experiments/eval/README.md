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
