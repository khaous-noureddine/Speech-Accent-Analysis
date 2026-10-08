# SupCon loss-weight ablation

## Question

This pilot measures the sensitivity of joint utterance-level training to the
coefficient applied to the supervised contrastive objective. The optimized
loss is

```text
L = L_CTC + gamma * L_SupCon
```

The implementation names this coefficient `supcon_weight`; the paper may call
it `gamma` after the notation has been standardized globally.

## Controlled protocol

Only `gamma` changes. Every run uses:

- Wav2Vec2-Large-LV60 initialized from `facebook/wav2vec2-large-lv60`;
- LibriSpeech train-clean-100 for CTC;
- the L2-ARCTIC Arabic-held-out fold for prompt-level SupCon;
- full Transformer fine-tuning with the feature encoder frozen;
- seed 13 and 50,000 optimizer updates;
- LibriSpeech dev-clean WER for checkpoint selection.

The grid is `0`, `0.01`, `0.05`, `0.1`, `0.5`, and `1.0`. Gamma zero is an
in-loop CTC control: the data loader and optimization schedule remain those of
the joint implementation, but the SupCon loss contributes no gradient.

Configurations are stored below:

```text
experiments/ablations/supcon-weight/librispeech-100h/
└── wav2vec2-large-lv60/utterance-supcon/arabic/full-transformer/
    └── gamma-<value>/
        ├── config.yaml
        ├── evaluation.yaml
        └── outputs/seed=13/
```

## Launch

Smoke-test gamma `0.05` first (array index 2):

```bash
sbatch --array=2 scripts/slurm/run_gamma_ablation_training.sbatch --smoke
```

Submit the six training runs with at most two H200 GPUs in use:

```bash
train_job=$(sbatch --parsable --array=0-5%2 \
  scripts/slurm/run_gamma_ablation_training.sbatch)
```

Make all greedy and fixed 4-gram evaluations wait for the complete successful
training array:

```bash
sbatch --dependency="afterok:${train_job}" --array=0-11%2 \
  scripts/slurm/run_gamma_ablation_evaluation.sbatch
```

Training array indices follow the grid order above. Evaluation indices pair
greedy then 4-gram decoding for each gamma value.
