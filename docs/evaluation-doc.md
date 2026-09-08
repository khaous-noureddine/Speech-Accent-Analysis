# Evaluation protocol

## Scope

This document is the authoritative specification for evaluating the
resubmission experiments. It records the evaluation datasets, checkpoint and
decoder contracts, WER implementation, fold/seed aggregation, statistical
comparisons, and output layout. Decisions should be updated here before the
paper is rewritten.

The evaluation must answer four questions:

1. Does prompt-level SupCon improve ASR on an accent excluded from Stage 2?
2. How much of the gain comes from accented CTC supervision rather than SupCon?
3. Does adaptation transfer to external datasets never used in Stage 2 or 3?
4. Does accented robustness preserve performance on standard clean English?

## Conditions

| Evaluation label | Stage 2 | Stage 3 | Role |
|---|---|---|---|
| `no-stage2` | None | LibriSpeech CTC | Condition C, SSL-to-ASR baseline |
| `supcon-only` | SupCon | LibriSpeech CTC | Contrastive contribution without accented CTC |
| `ctc-only` | L2-ARCTIC CTC | LibriSpeech CTC | Effect of supervised accented ASR exposure |
| `supcon-ctc` | SupCon + L2-ARCTIC CTC | LibriSpeech CTC | Full method |

All conditions start from `facebook/wav2vec2-large-lv60`, use the same Stage 3
data and optimization budget, and select `checkpoint_best.pt` using LibriSpeech
`dev-clean` greedy WER. Evaluation never uses `checkpoint_final.pt` merely
because it is newer.

The `no-stage2` condition has one Stage 3 run per seed. Adapted conditions have
one Stage 3 run per held-out accent and seed.

## WER implementation

### Definition

Corpus WER is computed as:

```text
WER = (substitutions + deletions + insertions) / reference words
```

Counts are summed over utterances before division. The primary score is
therefore corpus-level micro WER, not the arithmetic mean of utterance WERs.
Empty hypotheses are valid and contribute one deletion for every reference
word. Empty references are a data-contract violation. Audio-loading and model
inference failures abort evaluation; they are never replaced with silence or an
`ERROR` hypothesis.

### Library decision

The canonical scorer is **JiWER 4.x**, called directly through
`jiwer.process_words`. JiWER exposes alignments plus substitution, deletion,
insertion, hit, and reference-word counts. Those utterance-level counts are
required for paired bootstrap comparisons and error analysis.

Hugging Face `evaluate.load("wer")` is not used as the canonical layer because
its WER metric already delegates the edit-distance calculation to JiWER and
primarily returns the final scalar. Adding Evaluate would introduce an extra
dependency and possible metric-download/cache behavior without adding the
counts needed here. TorchMetrics is also unnecessary because scoring is an
offline, CPU-side operation rather than a distributed training metric.

JiWER is pinned to `>=4,<5` in the Pixi environment. A scorer unit test must
cover perfect output, substitutions, insertions, deletions, empty hypotheses,
and corpus aggregation before evaluation results are accepted.

### Text normalization

The same deterministic normalization is applied once to references and
hypotheses before JiWER:

1. Unicode NFKC normalization;
2. uppercase;
3. normalize typographic apostrophes to ASCII `'`;
4. retain `A-Z`, apostrophes, and word boundaries;
5. replace other punctuation with spaces;
6. collapse repeated whitespace and trim.

The tokenizer contains `A-Z`, apostrophe, and the word delimiter. Numeric
tokens in an evaluation reference must be expanded during dataset preparation
or rejected by validation; digits must not be silently deleted at scoring time.
The raw and normalized strings are both retained in prediction artifacts.

Statistical comparisons will additionally use SCTK's Matched Pair Sentence
Segment word-error test (MAPSSWE). It will be computed from paired utterances
already stored in `predictions.parquet`, so adding the test does not require
decoding the evaluation corpora again.

## Decoders

Every checkpoint is evaluated in two separately named modes.

### Greedy CTC

- argmax over the CTC logits;
- collapse repeated tokens and remove blank/padding tokens through the fixed
  project tokenizer;
- no lexicon and no external language model;
- primary result for isolating encoder and acoustic-model differences.

### CTC + language model

- beam-search decoding with one fixed English n-gram/KenLM artifact;
- the same tokenizer, lexicon, beam width, LM weight, and word insertion score
  for every condition, fold, seed, and evaluation dataset;
- decoder hyperparameters selected only on LibriSpeech `dev-clean`, then frozen;
- LM path, SHA-256, vocabulary, beam width, LM weight, and insertion score saved
  in `metrics.json` and the resolved evaluation config;
- reported as a secondary result, never mixed with greedy numbers.

The LM artifact and decoder implementation still need to be selected and
validated. No test-set-specific tuning is permitted.

## L2-ARCTIC leave-one-accent-out evaluation

“Held-out accent” means the accent was absent from both Stage 2 train and Stage
2 development. For each adapted model, evaluation uses only the test split of
the matching fold:

| Stage 2 fold/model | Evaluation subset |
|---|---|
| Arabic held out | Arabic test: 4 unseen speakers, 95 unseen prompts, 380 utterances |
| Chinese held out | Chinese test |
| Hindi held out | Hindi test |
| Korean held out | Korean test |
| Spanish held out | Spanish test |
| Vietnamese held out | Vietnamese test |

It is incorrect to evaluate an Arabic-held-out model as the primary result on a
different L2-ARCTIC fold. Across the six matching folds, report:

- corpus WER for every accent and seed;
- macro mean of the six accent-level WERs for each seed (primary fold summary);
- median, standard deviation, minimum, and maximum across accents (descriptive);
- pooled micro WER over the 2,280 held-out utterances (secondary);
- mean and standard deviation of seed-level macro WER across seeds 13, 42, 77.

Median is useful as a robustness description but does not replace the macro
mean or seed standard deviation requested by the reviewers.

The `no-stage2` baseline is independent of an L2-ARCTIC fold. Each of its three
seed models is evaluated on all six held-out test subsets. The same baseline
prediction for a seed is paired with each adapted condition of that seed.

## External datasets

External evaluation datasets are never used for Stage 2, Stage 3, checkpoint
selection, decoder tuning, or normalization tuning.

For each adapted objective and seed, all six fold-specific models are evaluated
on the complete official external test split. These are six different training
variants, not six partitions of the external test data. Report:

- one corpus WER for every objective/fold/seed;
- mean, median, and standard deviation across the six folds within each seed;
- mean and standard deviation across seed-level fold means;
- paired objective comparisons matched by fold and seed.

The `no-stage2` baseline has only one model per seed and is evaluated once per
external dataset and decoder. Its seed-matched predictions can be reused in
each comparison with the six adapted variants; the baseline must not be
duplicated and treated as six independent runs.

Planned datasets, in priority order:

1. L2-ARCTIC matching held-out tests: primary unseen-accent evaluation;
2. LibriSpeech `test-clean`: standard-speech control;
3. AESRC official test split: external accented speech, with exact seen/unseen
   accent composition documented before use;
4. Speech Accent Archive: broad external accent coverage;
5. EDACC test: spontaneous/conversational accented speech and the strongest
   empirical response to the “in-the-wild” scalability concern.

No external dataset is added to the paper until its official split, license,
reference normalization, sample count, and manifest hash have been audited.
The preparation implementations, raw/processed locations, legacy-script audit,
and direct rebuild commands are recorded in [`evaluation-data.md`](evaluation-data.md).

## Statistical reporting

All comparisons retain utterance-level edit counts and are paired wherever the
same audio is decoded by two systems.

Required comparisons are:

- `supcon-ctc` versus `no-stage2`: total Stage 2 effect;
- `supcon-only` versus `no-stage2`: SupCon without accented CTC;
- `ctc-only` versus `no-stage2`: accented CTC exposure alone;
- `supcon-ctc` versus `supcon-only`: auxiliary CTC contribution;
- `supcon-ctc` versus `ctc-only`: SupCon contribution in the full method.

Report absolute WER difference, relative WER reduction, and a 95% paired
bootstrap confidence interval. The planned default is 10,000 deterministic
replicates. For L2-ARCTIC, resampling must be stratified by held-out accent and
clustered by prompt so the 24 renditions of a shared prompt are not treated as
independent observations. Seed variability is reported separately as mean ±
standard deviation; bootstrap samples must not manufacture additional seeds.

The exact bootstrap algorithm and seed will be frozen in code and covered by a
null comparison test whose confidence interval contains zero.

## Output organization

Evaluation outputs live in a model-first tree separate from Stage 3 training:

```text
experiments/eval/
└── wav2vec2-large-lv60/
    ├── no-stage2/
    │   └── seed=13/
    │       ├── greedy/<dataset>/
    │       └── lm-4gram/<dataset>/
    ├── supcon-only/
    │   └── spanish/
    │       ├── config.yaml
    │       └── outputs/seed=13/
    │           ├── greedy/metrics_summary.json
    │           ├── greedy/l2_arctic/
    │           │   ├── predictions.parquet
    │           │   ├── metrics.json
    │           │   ├── config.resolved.json
    │           │   └── evaluation.log
    │           └── lm-4gram/l2_arctic/
    ├── ctc-only/
    ├── supcon-ctc/
    └── aggregate/
        ├── fold_seed_results.csv
        ├── aggregate_results.csv
        ├── paired_comparisons.csv
        ├── bootstrap_results.json
        └── tables/
```

The evaluation tree never copies Stage 3 checkpoints. Each resolved config and
metrics file records the source checkpoint path and SHA-256, tokenizer SHA-256,
dataset manifest SHA-256, decoder settings, code commit, condition, fold, seed,
and normalization version.

`predictions.parquet` contains at least:

- dataset, split, utterance, prompt, speaker, and accent identifiers when
  available;
- raw and normalized reference;
- raw and normalized hypothesis;
- substitutions, deletions, insertions, hits, and reference-word count;
- model, objective, fold, seed, decoder, checkpoint path, and checkpoint hash.

## Pipeline integration test

The initial greedy integration config uses the in-progress SupCon-only,
Arabic-held-out, seed-13 Stage 3 checkpoint. Its single `datasets` mapping
launches L2-ARCTIC, LibriSpeech test-clean, AESRC, Speech Accent Archive, and
EDACC. For L2-ARCTIC it points only to the matching Arabic-held-out test fold.
The smoke run decodes the first eight rows of every dataset and writes
artifacts under separate dataset-specific `smoke/` directories:

```bash
sbatch --array=0 scripts/slurm/run_evaluation.sbatch supcon-only 13 --smoke
```

This test verifies checkpoint identity and integrity, all 380 audio paths,
tokenizer/model reconstruction, greedy decoding, text normalization, JiWER
counts, Parquet output, hashes, logging, and Snakemake provenance. Because the
source `checkpoint_best.pt` may still be replaced during training, evaluation
aborts if the file changes while it is read. Smoke WER covers eight utterances
and is not an experimental result. Removing `--smoke` evaluates the complete
380-utterance matching test subset.

## Reviewer coverage

| Reviewer concern | Evaluation response |
|---|---|
| Fixed single L2-ARCTIC split | Six strict held-out-accent folds |
| Single seed and narrow margins | Three paired seeds, standard deviations, paired bootstrap CIs |
| Zero-shot accent claim unclear | Matching fold test contains unseen accent, speakers, and prompts |
| Clean-speech degradation | LibriSpeech `test-clean` control |
| Controlled repeated-prompt scalability | External AESRC/SAA and spontaneous EDACC evaluation; limitation remains explicit |
| CTC confound | `supcon-only` and `ctc-only` comparisons against `no-stage2` |
| Fine-tuned baseline initialization | All conditions start from unfine-tuned `wav2vec2-large-lv60` |
| Ambiguous decoding and WER | Versioned normalization, JiWER edit counts, separate greedy and LM results |

## Implementation order

1. finish the `no-stage2` Stage 3 baseline for seeds 13, 42, and 77;
2. implement checkpoint-safe greedy transcription under
   `src/accented_asr/evaluation/`;
3. implement JiWER utterance/corpus scoring and tests;
4. evaluate one Spanish checkpoint end to end;
5. add one config-driven Snakemake evaluation rule and local/Slurm launchers;
6. add the fixed LM decoder as a separate mode;
7. implement fold/seed aggregation and paired bootstrap statistics;
8. audit and connect LibriSpeech test-clean, AESRC, SAA, and EDACC manifests.
