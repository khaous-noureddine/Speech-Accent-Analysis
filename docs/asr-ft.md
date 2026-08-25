# Stage 3: ASR fine-tuning

## Purpose

Stage 3 measures how well each Stage 2 representation transfers to ASR. A new
CTC head is attached to the adapted encoder and the complete set of conditions
is fine-tuned on the same non-accented ASR data. The Stage 2 projection head and
auxiliary CTC head are never transferred.

The training recipe follows the LibriSpeech 100-hour fine-tuning protocol from
the original [wav2vec 2.0 paper](https://arxiv.org/pdf/2006.11477), especially
Section 4.3 and Appendix B, Table 6. Parameters that differ for this project's
controlled ablations are identified below rather than being presented as an
exact reproduction of the original multi-GPU fairseq setup.

### Reproduction boundary

| Component | wav2vec 2.0 LARGE 100 h | This project |
|---|---|---|
| Update budget | 50,000 | 50,000 |
| Encoder freeze | CTC head only for 10,000 updates; convolutional feature encoder always frozen | Same |
| LR schedule | Tri-stage 10% / 40% / 50%, final scale 0.05 | Same |
| Fine-tuning masks | Time `0.05 × 10`; channel `0.008 × 64`; LayerDrop `0.1` | Same |
| Effective batch | 1,920 seconds over 24 GPUs | 16 utterances on one GPU |
| Learning rate search | Three seeds at both `2e-5` and `3e-5` | Three seeds; fixed backbone/head rates `1e-5` / `1e-4` |
| Development selection | `dev-other`, decoded with the official 4-gram LM | `dev-clean`, greedy CTC WER |

The first four rows reproduce the optimization phases that determine update
semantics. The remaining rows are deliberate project-level differences. An
exact fairseq replication would require duration-budgeted batches with gradient
accumulation, two learning-rate runs per seed, and LM decoding during model
selection. Those changes would substantially expand the experiment matrix and
are not silently approximated here.

## Data and checkpoint selection

- Training data: LibriSpeech `train-clean-100` (28,539 utterances).
- Development data: LibriSpeech `dev-clean` (2,703 utterances).
- `dev-clean` is used only for checkpoint selection; it is not used for
  gradient updates.
- The final standard-speech result will be reported on `test-clean`, which is
  not consulted during training or model selection.
- Audio longer than 20 seconds is rejected by the configured data contract.
- A character CTC tokenizer with 32 entries is loaded from
  `configs/tokenizers/librispeech_char` and is shared by every condition.

The Snakefile treats both processed parquets as generated prerequisites. When
one is absent, it imports the corresponding raw subset, converts its FLAC files
to WAV, and writes the parquet automatically. The configured raw locations are:

```text
data/raw/librispeech/train/LibriSpeech/train-clean-100
data/raw/librispeech/eval/LibriSpeech/dev-clean
```

Existing processed parquets are reused without rebuilding. Raw LibriSpeech must
be extracted at the configured locations before a missing parquet can be built.

Each run loads the matching Stage 2 `checkpoint_best.pt` for the same objective,
held-out accent, and seed. Exactly the `backbone.*` tensors are transferred to
`wav2vec2.*`. A SHA-256 digest and Stage 2 metadata are stored in the resolved
Stage 3 configuration. Any seed, fold, or backbone mismatch stops the run.

## Optimization protocol

| Parameter | Stage 3 value | Rationale |
|---|---:|---|
| Maximum successful optimizer updates | 50,000 | wav2vec 2.0 100-hour fine-tuning budget |
| Batch size | 16 utterances | Fixed project-wide batch for comparable ablations |
| Gradient accumulation | None | Effective project batch is 16 utterances |
| Steps per epoch | 1,783 | `floor(28,539 / 16)` with `drop_last=true` |
| Approximate training epochs | 28.04 | `50,000 / 1,783` |
| CTC-head-only updates | First 10,000 | wav2vec 2.0 freeze period |
| Convolutional feature encoder | Always frozen | wav2vec 2.0 fine-tuning protocol |
| Transformer | Frozen through update 10,000, then trainable | wav2vec 2.0 fine-tuning protocol |
| Backbone learning rate | `1e-5` | Project choice, identical across conditions |
| New CTC-head learning rate | `1e-4` | Project choice, identical across conditions |
| Optimizer | AdamW, betas `(0.9, 0.98)` | Existing controlled Stage 3 recipe |
| Weight decay | `0.01` | Existing controlled Stage 3 recipe |
| Gradient clipping | `1.0` | Existing controlled Stage 3 recipe |
| Mixed precision | Enabled | Compute/memory optimization; skipped overflow updates do not count |

The learning rate uses the wav2vec 2.0 tri-stage schedule:

1. linear warm-up during the first 10% of updates (5,000 updates);
2. constant peak learning rate during the next 40% (20,000 updates);
3. linear decay during the final 50% (25,000 updates) to 5% of the peak rate.

The same multiplicative schedule is applied to the separate backbone and CTC
head learning rates. The first 10,000 *successful* optimizer updates train only
the CTC head. At update 10,000 the Transformer is unfrozen for the next batch;
the convolutional feature encoder remains frozen.

## Fine-tuning regularization

The wav2vec 2.0 100-hour regularization settings are explicit in every config:

- time masking probability `0.05`, span length `10`;
- channel masking probability `0.008`, span length `64`;
- Transformer LayerDrop `0.1` for the LARGE backbone;
- activation dropout `0.1`;
- gradient checkpointing enabled.

These values are passed explicitly when constructing `Wav2Vec2ForCTC`, so a
future Transformers default change cannot silently alter the experiment.

## Evaluation, logs, and checkpoints

- A lightweight JSON training record is flushed every 100 successful updates.
  It contains the mean CTC loss over that interval and both learning rates.
- Full `dev-clean` loss and greedy WER are computed every 5,000 successful
  updates and at update 50,000.
- `checkpoint_latest.pt` is overwritten after each development evaluation.
- `checkpoint_best.pt` is overwritten only when `dev-clean` WER improves.
- `checkpoint_final.pt` records the model at update 50,000 and is not assumed
  to be the best model.
- `metrics.jsonl` stores development evaluations, while TensorBoard stores
  per-update training loss and development loss/WER.
- A mixed-precision overflow skips that optimizer update, scheduler update, and
  global-step increment. The affected mini-batch is not retried.

## Experiment matrix

Stage 3 is run for all three Stage 2 objectives (`supcon-only`, `supcon-ctc`,
and `ctc-only`), all six held-out accents, and seeds 13, 42, and 77. All runs
use the same Stage 3 data, tokenizer, optimization budget, evaluation schedule,
and checkpoint-selection rule. This isolates the Stage 2 objective as the
experimental variable.

## Launch and monitoring

Example smoke test from the repository root:

```bash
pixi run snakemake -s Snakefile stage3_asr_finetuning \
  --configfile experiments/stage3/wav2vec2-large-lv60/supcon-only/arabic/config.yaml \
  --config run_seed=13 run_smoke=true \
  --cores 1 --forcerun stage3_asr_finetuning
```

For a full run, set `run_smoke=false`. The matching Stage 2 best checkpoint must
already exist; the rule does not silently fall back to the base Hugging Face
checkpoint or to a Stage 2 final checkpoint.

For direct-GPU execution, the local launcher accepts one objective, accent, and
seed:

```bash
scripts/local/run_asr_finetuning.sh supcon-only spanish 13
```

On Slurm, submit only the Spanish array element for a single full training test:

```bash
sbatch --array=4 scripts/slurm/run_asr_finetuning.sbatch supcon-only 13
```

Omitting the array override submits all six accents. The `%1` concurrency limit
keeps them serial, so the campaign occupies at most one GPU at a time:

```bash
sbatch scripts/slurm/run_asr_finetuning.sbatch supcon-only 13
```

Both launchers use Snakemake's `--rerun-incomplete` mode. If Slurm terminates a
run after creating only part of its declared outputs, submitting the same
objective/accent/seed again automatically removes and regenerates the artifacts
that Snakemake marked incomplete.
