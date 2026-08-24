# Stage 2 experiments

Each experiment directory combines one backbone and one Stage 2 objective.
Its six accent directories contain complete, standalone configurations. Run
artifacts are written next to the corresponding configuration under
`outputs/seed=<seed>/` and are ignored by Git.

Each YAML file is intentionally limited to Stage 2 and follows the project's
hierarchical configuration style: experiment metadata, adaptation data,
sampler, model, training, and development selection. Infrastructure settings
are centralized in the local and Slurm launchers. Stage 3 ASR fine-tuning and
final evaluation will have separate experiment folders.

| Directory | Condition | Objective |
|---|---|---|
| `wav2vec2-base_supcon-ctc` | A | SupCon + auxiliary CTC |
| `wav2vec2-base_supcon-only` | E | SupCon only |
| `wav2vec2-base_ctc-only` | F | auxiliary CTC only |

Example direct-GPU run:

```bash
scripts/local/run_adaptation.sh \
  experiments/stage2/wav2vec2-base_supcon-only/arabic/config.yaml \
  13 --smoke
```

The equivalent Slurm command is:

```bash
sbatch scripts/slurm/run_adaptation.sbatch \
  experiments/stage2/wav2vec2-base_supcon-only/arabic/config.yaml \
  13 --smoke
```

Remove `--smoke` for the configured full run. The seed passed to the
launcher must be declared in the configuration's `seeds` list.
