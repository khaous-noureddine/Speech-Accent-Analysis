# Word-level contrastive data

## Objective

This pipeline builds contrastive classes from repeated **words**, not repeated
sentences. A class is identified by `(language, normalized_word)`. Its positive
examples must come from different speakers and accent groups. Consequently, the
same preparation code can be applied to non-parallel corpora and independently
to several languages.

The evaluation separates two questions:

1. **granularity control:** L2-ARCTIC word SupCon versus L2-ARCTIC sentence
   SupCon with the same held-out accent;
2. **non-parallel transfer:** Common Voice word SupCon versus no Stage 2 and
   L2-ARCTIC sentence SupCon, followed by the same LibriSpeech Stage 3.

A downstream WER improvement is required before claiming improved accent
robustness. A decreasing contrastive development loss alone is not evidence for
that claim.

The current implementation supports:

- **L2-ARCTIC**, using the corpus-provided `words` TextGrid tier. This is the
  fully reproducible development corpus and requires no automatic alignment;
- **Mozilla Common Voice**, using genuinely isolated single-word clips with a
  non-empty accent and speaker identifier. Multi-word clips are deliberately
  excluded until word timestamps have been produced by a forced aligner.

AfriSpeech and ordinary Common Voice sentences remain target corpora for the
non-parallel experiment. They require transcript-conditioned forced alignment;
they must not be segmented by uniformly dividing an utterance duration.

## Output

The default L2-ARCTIC build creates:

```text
data/processed/l2_arctic_words/
├── inventory.parquet          # every valid TextGrid word occurrence
├── inventory_report.json      # filters and retained counts
├── vocabulary.csv             # global coverage audit (never used for fitting)
├── accent_stats.csv
├── fold_summary.csv
├── _SUCCESS
└── <heldout-accent>/
    ├── corpus.parquet
    ├── vocabulary.csv         # vocabulary selected from this fold's train rows
    ├── split_stats.csv
    └── validation_report.json
```

Audio is not duplicated. Each Parquet row points to its source WAV and records
`start_s` and `end_s`; the training dataset must slice that interval dynamically.
The canonical columns include dataset, language, accent, speaker, utterance,
word, normalized word, audio path, interval and alignment provenance.

## Leakage policy

For each L2-ARCTIC fold:

- the held-out accent appears only in `test`;
- speakers are disjoint across train, development and test;
- complete prompts are disjoint across all three splits;
- lexical overlap is intentional, because the word is the supervised
  contrastive class;
- development and test are never used to decide vocabulary eligibility. The
  root vocabulary is only a global data audit; each fold vocabulary is fitted
  again using that fold's train rows only.

For a non-parallel corpus without a shared `prompt_id`, speaker-disjoint splitting
is used. Every source adapter must preserve a stable speaker identifier.

## Current verified L2-ARCTIC build

Using all 24 speakers, six accent groups, a minimum of three speakers per accent,
and word durations from 0.12 to 2.0 seconds produced on 30 August 2026:

- 238,702 raw aligned word occurrences;
- 225,193 occurrences before fold-specific prompt selection;
- 2,758 eligible normalized words;
- zero speaker and prompt overlap in every fold.

After fitting its vocabulary from train only, the Arabic-held-out fold retains
2,150 training words and 109,567 train, 3,599 development and 2,935 test
occurrences. Counts differ across held-out accents and are always recorded in
the fold's `vocabulary.csv` and `split_stats.csv`.

## Run locally or on a direct-GPU server

Data preparation uses CPU only:

```bash
pixi install
pixi run snakemake \
  --snakefile Snakefile \
  --configfile configs/data/word_contrastive_l2_arctic.yaml \
  --cores 1
```

Dry-run the DAG first when changing a config:

```bash
pixi run snakemake \
  --snakefile Snakefile \
  --configfile configs/data/word_contrastive_l2_arctic.yaml \
  --cores 1 --dry-run
```

To rebuild after changing filters:

```bash
pixi run snakemake \
  --snakefile Snakefile \
  --configfile configs/data/word_contrastive_l2_arctic.yaml \
  --cores 1 --rerun-incomplete
```

## Prepare Magi

The expected raw directory on Magi is:

```text
/home/dist/noureddine.khaous/napster/accented-speech-recognition/
└── data/raw/l2_arctic/speakers/<speaker>/<speaker>/{wav,textgrid,...}
```

If Magi already has the raw L2-ARCTIC tree, only pull the code and run the same
Snakemake command from the repository root:

```bash
cd /home/dist/noureddine.khaous/napster/accented-speech-recognition
git pull
pixi install
pixi run snakemake \
  --snakefile Snakefile \
  --configfile configs/data/word_contrastive_l2_arctic.yaml \
  --cores 1
```

If the TextGrid directories are missing on Magi, copy the raw corpus from the
machine that already holds the complete release. Run this from that source
machine and replace `MAGI_LOGIN` with the configured SSH host:

```bash
rsync -av --info=progress2 \
  data/raw/l2_arctic/ \
  MAGI_LOGIN:/home/dist/noureddine.khaous/napster/accented-speech-recognition/data/raw/l2_arctic/
```

Do not copy `data/processed/l2_arctic_words`: rebuilding it from raw data is
deterministic and avoids transferring six repeated Parquet folds.

## First training pilot

The first pilot uses the already aligned L2-ARCTIC words. This validates the
training path and isolates word versus sentence granularity before introducing
Common Voice alignment quality as another variable. Each batch contains 16
words and one occurrence from each of five distinct seen accents per word (80
segments). Arabic remains absent from adaptation training and development.

```bash
pixi run snakemake -s Snakefile \
  --configfile configs/data/word_contrastive_l2_arctic.yaml --cores 1

sbatch scripts/slurm/run_word_adaptation.sbatch \
  l2-arctic arabic 13 --smoke
```

If both losses are finite and both checkpoints are created, launch the pilot:

```bash
sbatch scripts/slurm/run_word_adaptation.sbatch l2-arctic arabic 13
```

Outputs are stored under
`experiments/word-contrastive/wav2vec2-large-lv60/l2-arctic/arabic/outputs/seed=13/`.
The best checkpoint must subsequently undergo the same LibriSpeech-100h CTC
fine-tuning and five-dataset evaluation as the sentence-level conditions.

```bash
sbatch scripts/slurm/run_word_asr_finetuning.sbatch \
  l2-arctic arabic 13 --smoke

sbatch scripts/slurm/run_word_asr_finetuning.sbatch l2-arctic arabic 13
```

## Common Voice

Mozilla Common Voice downloads require selecting a release and accepting its
terms through [Mozilla Data Collective](https://commonvoice.mozilla.org/data).
No token or account credential is stored in this repository.

Download the desired language or the multilingual Single Word Target Segment,
then extract it so that one release contains `validated.tsv` and `clips/`:

```text
data/raw/common_voice/en/
└── <release>/
    ├── validated.tsv
    └── clips/
```

Inspect accent coverage before building:

```bash
pixi run python - <<'PY'
import pandas as pd
from pathlib import Path
p = next(Path('data/raw/common_voice/en').rglob('validated.tsv'))
d = pd.read_csv(p, sep='\t', keep_default_na=False)
one = d[d.sentence.str.count(r"[^\W\d_]+") == 1]
print(one[one.accent != ''].groupby('accent').client_id.nunique().sort_values(ascending=False))
PY
```

Copy and edit the example config only after seeing those counts:

```bash
cp configs/data/word_contrastive_common_voice.example.yaml \
   configs/data/word_contrastive_common_voice_en.yaml
pixi run snakemake \
  --snakefile Snakefile \
  --configfile configs/data/word_contrastive_common_voice_en.yaml \
  --cores 1
```

If no words satisfy the requested coverage, the command fails explicitly. It
does not silently weaken the accent or speaker thresholds.

The current Common Voice adapter accepts only genuinely single-word clips. It
is useful for validating the metadata path but is not yet the final paper
experiment. Ordinary non-parallel sentences must first be force-aligned and
exported with the same occurrence schema. The release must be inspected before
fixing accent thresholds or committing a training configuration; those values
must not be guessed.

## MSWC English enriched with Common Voice accents

MSWC was generated from the validated portion of Common Voice v3. Its English
archive already contains one-second word clips and its split files retain the
original Common Voice filename and hashed speaker identifier. The preparation
workflow joins that filename to Common Voice `validated.tsv` metadata to recover
the self-reported accent and source sentence. The historical CV3 endpoint is no
longer publicly accessible, so the rule uses a pinned metadata-only mirror of
Common Voice 21.0. It records filename coverage and verifies that matching
speaker hashes still agree.

The first Magi run downloads the MSWC English audio and split archives plus the
pinned Common Voice metadata file. Common Voice audio is not downloaded or
duplicated. These are large downloads and support resume through
`curl --continue-at -`.

```bash
pixi run snakemake -s Snakefile prepare_mswc_word_contrastive_dataset \
  --configfile configs/data/word_contrastive_mswc_en.yaml \
  --cores 1 --rerun-incomplete
```

The rule produces:

```text
data/processed/mswc_common_voice_words/en/
├── corpus.parquet
├── vocabulary.csv
├── split_stats.csv
├── validation_report.json
└── _SUCCESS
```

It rejects missing or ambiguous filename joins. Because MSWC was built from
Common Voice v3 while the pinned accent metadata comes from CV21, anonymised
speaker hashes can differ between releases. The MSWC `SPEAKER` value is the
canonical speaker identity; hash agreement with CV21 is recorded as an audit
statistic rather than used as a rejection criterion. A new global speaker-disjoint 90/10
train/development split is created because the published MSWC splits guarantee
speaker separation per keyword, not necessarily across the complete lexical
inventory. Word eligibility is fitted on train only; defaults require six
accents and three distinct speakers per accent. Accent values remain
self-reported Common Voice metadata and are never converted into inferred
countries.

## Parameters to report in the paper

The experiment section must state:

- exact dataset release and license;
- selected languages and accent-label source;
- word normalizer and tokenizer;
- alignment source and quality threshold;
- minimum accents and unique speakers per word;
- word-duration limits and context padding used during audio loading;
- cap on occurrences per speaker and word used by the sampler;
- speaker/prompt split seed and held-out accent;
- number of words, occurrences, speakers and hours in every split.

The generated reports provide these counts, but they are not committed because
all processed data and experiment outputs are ignored by Git.

## Controlled 50-hour MSWC held-out-accent corpus

The official MSWC train/development/test files use an 80/10/10 split for each
keyword. They are not an unseen-accent benchmark. Our accent-generalization
experiment therefore derives a new split from the accent-enriched English
manifest above.

The first controlled corpus contains exactly 180,000 one-second word segments
(50 hours):

- 144,000 train examples (40 hours) from six seen accent groups;
- 18,000 development examples (5 hours) from the same seen accents and
  speaker-disjoint from train;
- 18,000 test examples (5 hours) from one completely held-out accent;
- at least three distinct train speakers per retained word and seen accent;
- lexical overlap across partitions is intentional because the word is the
  contrastive class;
- speakers with inconsistent accent labels are excluded.

With `heldout_accent: auto`, the builder evaluates accent coverage and chooses
one sufficiently resourced held-out accent deterministically. It greedily
selects six other accent groups with the largest shared train vocabulary. The
resolved labels, selection seed, exact counts, and leakage checks are written
to `validation_report.json`; they must be inspected before training and
reported in the paper.

Build the subset on Magi after the full joined corpus is ready:

```bash
pixi run snakemake -s Snakefile prepare_mswc_word_heldout_subset \
  --configfile configs/data/word_contrastive_mswc_en_50h.yaml \
  --cores 1 --rerun-incomplete
```

Outputs are stored in:

```text
data/processed/mswc_common_voice_words/en_50h_heldout_accent/
├── corpus.parquet
├── vocabulary.csv
├── split_stats.csv
├── accent_stats.csv
├── validation_report.json
└── _SUCCESS
```

Inspect the selected accents and split sizes with:

```bash
cat data/processed/mswc_common_voice_words/en_50h_heldout_accent/validation_report.json
column -s, -t data/processed/mswc_common_voice_words/en_50h_heldout_accent/split_stats.csv
column -s, -t data/processed/mswc_common_voice_words/en_50h_heldout_accent/accent_stats.csv
```

### Scale the controlled corpus to 500 hours

The 500-hour variant reuses the held-out accent and the six seen accents saved
in the 50-hour `validation_report.json`. It keeps development and held-out test
at 5 hours each and increases only the training partition from 40 to 490 hours.
This makes the comparison primarily a test of accented training-data scale.

Build it on Magi with:

```bash
pixi run snakemake -s Snakefile prepare_mswc_word_heldout_subset \
  --configfile configs/data/word_contrastive_mswc_en_500h.yaml \
  --cores 1 --rerun-incomplete --nolock --printshellcmds
```

The command writes `corpus.parquet`, statistics, and validation artifacts to
`data/processed/mswc_common_voice_words/en_500h_heldout_accent/`. If the six
fixed accents do not contain 490 hours of eligible training clips, preparation
stops explicitly instead of silently changing the accent composition.

After preparation succeeds, submit the 500-hour Word-SupCon training and make
the greedy and 4-gram evaluations wait for its successful completion:

```bash
train_job=$(sbatch --parsable \
  scripts/slurm/run_word_joint_training.sbatch full-transformer 13 100h 500h) && \
echo "Training job: ${train_job}" && \
sbatch --dependency="afterok:${train_job}" --array=0-1%2 \
  scripts/slurm/run_word_joint_evaluation.sbatch 500h
```

`afterok` means that evaluation starts only if training exits successfully.
Array task 0 runs greedy decoding and task 1 runs fixed 4-gram LM decoding.

## Word-level joint training

The word experiment jointly optimizes LibriSpeech CTC and word-level SupCon.
Words are the contrastive classes: recordings of the same word are positives,
while recordings of different words are negatives. The MSWC test split is not
used for optimization; it contains the accent selected as held out by the
50-hour corpus builder.

Before submitting a GPU job, verify that preparation completed:

```bash
test -f data/processed/mswc_common_voice_words/en_50h_heldout_accent/_SUCCESS \
  && echo "Word corpus ready" \
  || echo "Prepare the word corpus first"
```

Run a short end-to-end smoke test on Magi:

```bash
sbatch scripts/slurm/run_word_joint_training.sbatch full-transformer 13 --smoke
```

After the smoke test succeeds, launch the complete experiment:

```bash
sbatch scripts/slurm/run_word_joint_training.sbatch full-transformer 13
```

The matched ablation that freezes the first 18 Transformer blocks is launched
with:

```bash
sbatch scripts/slurm/run_word_joint_training.sbatch freeze-18 13
```

The feature encoder remains frozen in both variants. During the first epoch,
only the CTC head is updated; afterward, the configured Transformer blocks and
both task heads are optimized. The SupCon projection head is discarded for ASR
inference.
