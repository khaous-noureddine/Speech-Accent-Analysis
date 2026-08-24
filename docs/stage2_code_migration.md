# Stage 2 code migration

## Design choice

`stage2` is a useful experiment-workflow name, but it does not describe what
the code does. The migrated Python package is therefore named `adaptation`:

```text
src/accented_asr/
├── data/                  # corpus preparation and deterministic splits
└── adaptation/            # L2-ARCTIC representation adaptation
    ├── data.py            # dataset, audio loading, prompt batches
    ├── model.py           # encoder, projection/CTC heads, losses
    └── train.py           # one condition × fold × seed run

experiments/stage2/        # configs and outputs grouped by experiment/accent
scripts/local/             # direct-GPU launchers
scripts/slurm/             # Slurm launchers only
```

The legacy `stage2/` directory is intentionally preserved during the
migration. New experiments should use `src/accented_asr/adaptation`; legacy
files can be removed only after Stage 2 and Stage 3 have been validated end to
end.

## Experimental modes

| Condition | Training mode | Stage 2 objective |
|---|---|---|
| A | `supcon_ctc` | averaged SupCon + 0.1 × averaged CTC |
| E | `supcon_only` | averaged SupCon only |
| F | `ctc_only` | averaged CTC only |

All modes use the fold's `train` rows for optimization and `dev` rows for
checkpoint selection. The held-out `test` rows are rejected by the adaptation
dataset. Every run records the split-manifest SHA-256 in its resolved
configuration and checkpoints.

Each checkpoint stores the complete adaptation model once. Stage 3 will load
the `backbone.*` entries from `model_state_dict`; the encoder is not duplicated
under a second key, which keeps checkpoints substantially smaller.

The local tokenizer contains the fixed 32-character LibriSpeech vocabulary.
It initializes a new random CTC head and does not load weights from an
ASR-fine-tuned checkpoint.

## Commands

Run the first bounded smoke test on a directly accessible GPU:

```bash
scripts/local/run_adaptation.sh \
  experiments/stage2/wav2vec2-base_supcon-only/arabic/config.yaml \
  13 --smoke
```

Submit the same run to Slurm:

```bash
sbatch scripts/slurm/run_adaptation.sbatch \
  experiments/stage2/wav2vec2-base_supcon-only/arabic/config.yaml \
  13 --smoke
```

Remove `--smoke` for a full configured run. Each configuration declares its
condition, loss mode, held-out accent, seeds, output directory, data paths,
model, batching parameters, and optimization hyperparameters. Outputs are
stored beside that config under `outputs/seed=<seed>/`.

The current configuration uses `facebook/wav2vec2-base` to validate the
pipeline cheaply. Before the main paper experiments, freeze one backbone in
the configuration. If retaining the paper's LARGE-capacity comparison, use
the un-fine-tuned `facebook/wav2vec2-large-lv60` checkpoint and adjust the
number of frozen transformer layers accordingly.
