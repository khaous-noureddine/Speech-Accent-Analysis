# Resubmission experiment tracker

This is the short day-to-day checklist. See `RESUBMISSION_PLAN.md` for the full
review analysis and long-term repository plan.

## Current objective

Show that prompt-level supervised contrastive adaptation improves accented ASR,
and separate its effect from auxiliary CTC training on accented speech.

## Fixed three-stage pipeline

| Stage | Data | Action |
|---|---|---|
| Stage 1 | Unlabeled pretraining corpus | Start from the same SSL base checkpoint with no ASR fine-tuning |
| Stage 2 | L2-ARCTIC train/dev from one accent fold | Apply the condition-specific adaptation |
| Stage 3 | LibriSpeech | Apply the same supervised CTC fine-tuning to every condition |
| Evaluation | Held-out L2-ARCTIC accent and external test sets | Evaluate only after model selection |

## Conditions

| ID | Stage 2 SupCon | Stage 2 CTC | Stage 3 LibriSpeech CTC | Purpose |
|---|:---:|:---:|:---:|---|
| C | No | No | Yes | No-adaptation baseline |
| A | Yes | Yes | Yes | Full method |
| E | Yes | No | Yes | SupCon-only ablation |
| F | No | Yes | Yes | CTC-only ablation |

Required comparisons:

- `A vs C`: overall effect of Stage 2;
- `A vs E`: contribution of auxiliary CTC;
- `A vs F`: contribution of SupCon;
- `E vs C`: effect of SupCon alone;
- `F vs C`: effect of accented CTC adaptation alone.

Condition G, direct ASR training on LibriSpeech mixed with accented data, is a
later experiment and is not part of the first implementation milestone.

## L2-ARCTIC folds

| Fold | Stage 2 unseen/test accent | Stage 2 train/dev accents |
|---|---|---|
| Arabic | Arabic | Chinese, Hindi, Korean, Spanish, Vietnamese |
| Chinese | Chinese | Arabic, Hindi, Korean, Spanish, Vietnamese |
| Hindi | Hindi | Arabic, Chinese, Korean, Spanish, Vietnamese |
| Korean | Korean | Arabic, Chinese, Hindi, Spanish, Vietnamese |
| Spanish | Spanish | Arabic, Chinese, Hindi, Korean, Vietnamese |
| Vietnamese | Vietnamese | Arabic, Chinese, Hindi, Korean, Spanish |

Each fold contains:

- train: 11,445 examples, 763 prompts, 15 speakers, 5 accents;
- development: 480 examples, 96 prompts, 5 speakers, 5 accents;
- test: 380 examples, 95 prompts, 4 speakers, 1 unseen accent.

The split seed is fixed to `20260817`. Planned paired model seeds are `13`,
`42`, and `77`.

## Decisions still required

- [ ] Freeze the exact Wav2Vec2 SSL base checkpoint.
- [ ] Confirm the LibriSpeech subset and Stage 3 training budget.
- [ ] Define equal Stage 2 budgets across A, E, and F.
- [ ] Decide whether the essential experiments will later be replicated with HuBERT.

## Execution checklist

### 1. Data protocol

- [x] Rebuild L2-ARCTIC directly from raw WAV files and transcripts.
- [x] Create six deterministic leave-one-accent-out folds.
- [x] Enforce disjoint prompts and speakers across train/dev/test.
- [x] Generate Parquet files, statistics, manifests, hashes, and validation reports.
- [x] Document the dataset and split implementation.

### 2. Prepare Stage 2 — current milestone

- [ ] Audit the current Stage 2 training entry point.
- [ ] Load `data/processed/l2_arctic_leave_one_accent_out/<fold>/corpus.parquet`.
- [ ] Enforce train-only optimization and dev-only checkpoint selection.
- [ ] Resample 44.1 kHz audio to the model input rate in the data loader.
- [ ] Build valid prompt-level contrastive batches.
- [ ] Add explicit modes: `supcon_ctc`, `supcon_only`, and `ctc_only`.
- [ ] Record the fold manifest SHA-256 in every run and checkpoint.
- [ ] Initialize Stage 2 CTC heads consistently and document whether they are discarded.
- [ ] Add tests for loss modes, batching, masking, and deterministic seeds.

### 3. Smoke tests

- [ ] Run `E / Arabic / seed 13` for a few hundred Stage 2 steps.
- [ ] Confirm finite SupCon loss and valid positive pairs.
- [ ] Check GPU memory and effective contrastive batch size.
- [ ] Save and reload the Stage 2 encoder checkpoint.
- [ ] Run the common Stage 3 LibriSpeech fine-tuning.
- [ ] Evaluate the final model on the held-out Arabic test set.
- [ ] Repeat the smoke test for A and F.

### 4. Main Wav2Vec2 experiments

- [ ] Run A on 6 folds with seed 13.
- [ ] Run E on 6 folds with seed 13.
- [ ] Run F on 6 folds with seed 13.
- [ ] Run C with seed 13 and evaluate it on all 6 accent test sets.
- [ ] Inspect results before expanding the seed budget.
- [ ] Repeat paired runs with seeds 42 and 77.

For each model seed, A/E/F require `3 conditions × 6 folds = 18` Stage 2
adaptations. Condition C has no fold-dependent Stage 2; one Stage 3 model per
seed can be evaluated on all six held-out-accent test sets.

### 5. Evaluation and statistics

- [ ] Report WER separately for every held-out accent.
- [ ] Report the macro-average across the six accents.
- [ ] Report mean and standard deviation across model seeds.
- [ ] Compute paired bootstrap confidence intervals on utterance-level errors.
- [ ] Evaluate LibriSpeech test-clean as the standard-speech control.
- [ ] Evaluate the selected models on AESRC and Speech Accent Archive.

### 6. Paper updates

- [ ] Describe the evaluation as strict leave-one-accent-out cross-validation.
- [ ] State that test jointly contains unseen accents, speakers, and prompts.
- [ ] Report the exact base checkpoint, seeds, budgets, and manifest hashes.
- [ ] Explain the SupCon-only and CTC-only ablations.
- [ ] Acknowledge reliance on scripted parallel prompts.
- [ ] Acknowledge transcript dependence when auxiliary CTC is used.
- [ ] Add comparisons with relevant accent-adaptation and contrastive ASR work.

## Immediate next action

Audit the existing Stage 2 data loader and training code, then implement the
`E / Arabic / seed 13` SupCon-only smoke test without accessing the test split
during training or checkpoint selection.
