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

experiments/stage2/        # configs grouped by model/objective/accent
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
| F | `ctc_only` | 0.1 × averaged CTC only |

The CTC coefficient is deliberately identical in Conditions A and F. This is
a component-removal ablation: A vs. F adds SupCon while holding the auxiliary
CTC term fixed, and A vs. E adds the same weighted CTC term. The logged
`ctc_loss` remains the unweighted mean CTC value for interpretability, whereas
the logged total `loss` includes the configured coefficient (`ctc_weight=0.1`).

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
  experiments/stage2/wav2vec2-large-lv60/supcon-only/arabic/config.yaml \
  13 --smoke
```

Submit the same run to Slurm:

```bash
sbatch scripts/slurm/run_adaptation.sbatch \
  experiments/stage2/wav2vec2-large-lv60/supcon-only/arabic/config.yaml \
  13 --smoke
```

Remove `--smoke` for a full configured run. Each hierarchical configuration
contains `experiment` and `stage2_adaptation` sections. Slurm resources remain
centralized in `scripts/slurm/run_adaptation.sbatch`, so an infrastructure
change is made only once. Stage 3 and final evaluation are intentionally absent
because they will have independent experiment folders and configurations.
Outputs are stored beside the Stage 2 config under `outputs/seed=<seed>/`.

The frozen backbone for these experiments is
`facebook/wav2vec2-large-lv60`: an un-fine-tuned Wav2Vec2 LARGE checkpoint
with 24 transformer layers. The first 18 transformer layers and the feature
extractor are frozen during Stage 2.
