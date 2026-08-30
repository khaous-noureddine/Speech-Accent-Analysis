# Word-level contrastive data

## Objective

This pipeline builds contrastive classes from repeated **words**, not repeated
sentences. A class is identified by `(language, normalized_word)`. Its positive
examples must come from different speakers and accent groups. Consequently, the
same preparation code can be applied to non-parallel corpora and independently
to several languages.

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
