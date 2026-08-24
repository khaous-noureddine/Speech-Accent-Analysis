# Stage 3 experiments

Stage 3 initializes a fresh CTC ASR model from the selected Stage 2 encoder.
Only `backbone.*` tensors are transferred: the Stage 2 contrastive projection
and auxiliary CTC head are discarded, and the downstream CTC head is newly
initialized for every run.

The main protocol uses LibriSpeech `train-clean-100` for training and
`dev-clean` for checkpoint selection by WER. Every Stage 2 objective, held-out
accent, and training seed retains a separate Stage 3 run directory.

```text
experiments/stage3/wav2vec2-large-lv60/
└── <objective>/<held-out-accent>/
    ├── config.yaml
    └── outputs/seed=<seed>/
```

Run the Arabic SupCon-only smoke test from the repository root:

```bash
pixi run snakemake -s Snakefile stage3_asr_finetuning \
  --configfile experiments/stage3/wav2vec2-large-lv60/supcon-only/arabic/config.yaml \
  --config run_seed=13 run_smoke=true --cores 1
```

Remove `run_smoke=true` for the configured full run. The rule requires the
matching Stage 2 `checkpoint_best.pt`; it will never silently fall back to the
base encoder or to `checkpoint_final.pt`.
