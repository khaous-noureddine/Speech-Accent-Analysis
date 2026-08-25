# Stage 2 experiments

The first directory level identifies the backbone. Each model directory then
contains one folder per Stage 2 objective, followed by the six held-out accent
folders. Run artifacts are written next to the corresponding configuration
under `outputs/seed=<seed>/` and are ignored by Git.

```text
experiments/stage2/
└── wav2vec2-large-lv60/
    ├── supcon-ctc/
    ├── supcon-only/
    └── ctc-only/
        ├── arabic/
        ├── chinese/
        ├── hindi/
        ├── korean/
        ├── spanish/
        └── vietnamese/
```

Each YAML file is intentionally limited to Stage 2 and follows the project's
hierarchical configuration style: experiment metadata, adaptation data,
sampler, model, training, and development selection. Infrastructure settings
are centralized in the local and Slurm launchers. Stage 3 ASR fine-tuning and
final evaluation will have separate experiment folders.

| Objective directory | Condition | Objective |
|---|---|---|
| `supcon-ctc` | A | SupCon + 0.1 × auxiliary CTC |
| `supcon-only` | E | SupCon only |
| `ctc-only` | F | 0.1 × auxiliary CTC only |

Condition F retains the same CTC coefficient as Condition A. Consequently,
the A-vs-F comparison removes only SupCon, while A-vs-E removes only the
weighted auxiliary CTC component. Metrics report raw `ctc_loss`; the optimized
total `loss` applies the coefficient.

The current model is the un-fine-tuned `facebook/wav2vec2-large-lv60`
checkpoint. It has 24 transformer layers; the feature extractor and first 18
transformer layers are frozen during Stage 2.

Example direct-GPU run:

```bash
scripts/local/run_adaptation.sh \
  experiments/stage2/wav2vec2-large-lv60/supcon-only/arabic/config.yaml \
  13 --smoke
```

On Slurm, one submission runs all six accents sequentially (`%1` limits the
array to one active task). Select the objective and seed as arguments:

```bash
sbatch scripts/slurm/run_adaptation.sbatch supcon-only 13 --smoke
```

Valid objectives are `supcon-only`, `supcon-ctc`, and `ctc-only`. Remove
`--smoke` for full runs. The seed passed to the launcher must be declared in
the configurations' `seeds` lists.

## Snakemake rule

The `Snakefile` contains one generic `stage2_adaptation` rule. It never lists
experiments, accents, or seeds. The shell or Slurm launcher chooses one YAML
configuration and one declared seed.

Dry-run one experiment:

```bash
pixi run snakemake -s Snakefile stage2_adaptation \
  --configfile experiments/stage2/wav2vec2-large-lv60/supcon-only/arabic/config.yaml \
  --config run_seed=13 --cores 1 --dry-run
```

Remove `--dry-run` to execute it. Add `run_smoke=true` after `--config` for a
bounded smoke run. Any loop over conditions, accents, or seeds belongs in the
external local/Slurm launcher, not in this Snakefile.

## TensorBoard

Every new run writes TensorBoard events under its own
`outputs/seed=<seed>/tensorboard/` directory. The dashboard contains total,
SupCon, and CTC losses for train/dev, plus the learning rate.

Start the dashboard from the repository root:

```bash
pixi run tensorboard \
  --logdir experiments/stage2/wav2vec2-large-lv60 \
  --host 127.0.0.1 \
  --port 6006
```

When accessing the server through SSH, forward port 6006 to the local machine
and open `http://localhost:6006` in a browser.
