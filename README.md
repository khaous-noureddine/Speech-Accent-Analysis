# Accent-robust speech recognition

This repository studies whether prompt-level supervised contrastive adaptation
can make self-supervised speech encoders more robust to unseen English accents.
Utterances that share a scripted prompt are aligned across speakers and accents
before a common CTC ASR fine-tuning stage.

The current resubmission experiments use the unfine-tuned
`facebook/wav2vec2-large-lv60` checkpoint and explicitly separate supervised
contrastive learning from auxiliary CTC training on accented speech.

## Experimental pipeline

| Stage | Data | Operation |
|---|---|---|
| 1. SSL initialization | Libri-Light 60k | Load the same unfine-tuned Wav2Vec2 LARGE checkpoint |
| 2. Accent adaptation | L2-ARCTIC | Run SupCon+CTC, SupCon-only, or CTC-only adaptation |
| 3. ASR fine-tuning | LibriSpeech train-clean-100 | Apply the same downstream CTC recipe to every adapted encoder |
| Evaluation | Held-out L2-ARCTIC accent and external corpora | Measure unseen-accent WER after development-set selection |

Stage 2 uses six deterministic leave-one-accent-out folds: Arabic, Chinese,
Hindi, Korean, Spanish, and Vietnamese. Each experiment is repeated with seeds
`13`, `42`, and `77`.

The primary Stage 2 ablations are:

| Objective | SupCon | Accented CTC | Purpose |
|---|:---:|:---:|---|
| `supcon-ctc` | Yes | Yes | Full adaptation method |
| `supcon-only` | Yes | No | Isolate prompt-level contrastive adaptation |
| `ctc-only` | No | Yes | Isolate supervised ASR exposure to L2-ARCTIC |

## Repository layout

```text
.
├── src/accented_asr/
│   ├── data/          # reproducible dataset preparation and L2-ARCTIC folds
│   ├── adaptation/    # Stage 2 data, model, losses, and training
│   └── asr/           # Stage 3 LibriSpeech CTC fine-tuning
├── experiments/
│   ├── stage2/        # one config and output directory per objective/accent
│   ├── stage3/        # matching downstream ASR experiments
│   └── eval/          # model/objective/fold/seed evaluation artifacts
├── scripts/
│   ├── local/         # direct-GPU launchers
│   └── slurm/         # shared-cluster launchers
├── configs/tokenizers/ # fixed character CTC vocabulary
├── corpus/             # legacy dataset importers retained during migration
├── docs/               # protocols, implementation details, and tracker
├── tests/              # data, loss, transfer, and configuration controls
└── Snakefile           # config-driven orchestration for one selected run
```

The legacy `stage2/`, `stage3/`, and older Snakefiles are retained as migration
references. New training code belongs under `src/accented_asr/`.

## Environment

The reproducible environment is managed with [Pixi](https://pixi.sh/):

```bash
pixi install
```

Verify the installation from the repository root:

```bash
PYTHONPATH=src pixi run pytest -q
```

GPU training requires a CUDA-capable host compatible with the PyTorch version
resolved in `pixi.lock`.

## Data preparation

Large datasets and generated manifests are not stored in Git. Expected raw
locations are:

```text
data/raw/l2_arctic/speakers
data/raw/librispeech/train/LibriSpeech/train-clean-100
data/raw/librispeech/eval/LibriSpeech/dev-clean
```

Stage 2 automatically builds all six L2-ARCTIC folds when their processed
artifacts are missing. Stage 3 similarly builds the LibriSpeech train and
development parquets and converts FLAC audio to the project WAV layout when
needed. Existing processed artifacts are reused.

Generated data is written under:

```text
data/processed/l2_arctic_leave_one_accent_out
data/processed/librispeech_train
data/processed/librispeech_eval
```

See [the L2-ARCTIC split protocol](docs/l2_arctic_splits.md) for exact counts,
speaker/prompt separation, hashes, and regeneration commands.

## Running Stage 2

One direct-GPU run takes an experiment config and a seed:

```bash
scripts/local/run_adaptation.sh \
  experiments/stage2/wav2vec2-large-lv60/supcon-only/spanish/config.yaml \
  13
```

On Slurm, one submission launches the six accents sequentially (`0-5%1`), so
the campaign uses at most one GPU at a time:

```bash
sbatch scripts/slurm/run_adaptation.sbatch supcon-only 13
```

Valid objectives are `supcon-only`, `supcon-ctc`, and `ctc-only`.

## Running Stage 3

Adapted Stage 3 conditions require the matching Stage 2 `checkpoint_best.pt`;
they never silently fall back to a base or final checkpoint. The explicit
`no-stage2` condition is the only configuration initialized directly from the
base encoder.

Run one full direct-GPU experiment:

```bash
scripts/local/run_asr_finetuning.sh supcon-only spanish 13
```

Run only Spanish (array index 4) on Slurm:

```bash
sbatch --array=4 scripts/slurm/run_asr_finetuning.sbatch supcon-only 13
```

Run all six accents sequentially:

```bash
sbatch scripts/slurm/run_asr_finetuning.sbatch supcon-only 13
```

Append `--smoke` to a launcher only for bounded pipeline validation. Full Stage
3 training uses 50,000 successful updates and selects `checkpoint_best.pt` by
greedy WER on LibriSpeech `dev-clean`.

## Monitoring and outputs

Each experiment writes into its own objective/accent/seed directory. For
example:

```text
experiments/stage3/wav2vec2-large-lv60/supcon-only/spanish/outputs/seed=13/
├── config.resolved.json
├── metrics.jsonl
├── training.log
├── tensorboard/
├── checkpoint_latest.pt
├── checkpoint_best.pt
└── checkpoint_final.pt
```

Follow the JSON training log:

```bash
tail -f experiments/stage3/wav2vec2-large-lv60/supcon-only/spanish/outputs/seed=13/training.log
```

Or start TensorBoard:

```bash
pixi run tensorboard --logdir experiments --port 6006
```

Checkpoints and generated audio are ignored by Git. The resolved configuration,
manifest hashes, checkpoint metadata, and metric records provide the audit trail
for each run.

## Documentation

- [Stage 3 ASR fine-tuning protocol](docs/asr-ft.md)
- [Evaluation and statistical protocol](docs/evaluation-doc.md)
- [Stage 2 implementation and migration notes](docs/stage2_code_migration.md)
- [L2-ARCTIC split design](docs/l2_arctic_splits.md)
- [Short resubmission tracker](docs/RESUBMISSION_TRACKER.md)
- [Full reviewer-response plan](docs/RESUBMISSION_PLAN.md)
