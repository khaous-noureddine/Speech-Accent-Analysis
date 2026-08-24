# L2-ARCTIC leave-one-accent-out splits

## Dataset information

### L2-ARCTIC

L2-ARCTIC is a non-native English speech corpus designed for accented speech
research. It uses prompts from the CMU ARCTIC prompt set and contains 24
official speakers from six first-language (L1) groups, with four speakers per
group.

| L1 group | Speakers |
|---|---|
| Arabic | ABA, SKA, YBAA, ZHAA |
| Chinese/Mandarin | BWC, LXC, NCC, TXHC |
| Hindi | ASI, RRBI, SVBI, TNI |
| Korean | HJK, HKK, YDCK, YKWK |
| Spanish | EBVS, ERMS, MBMPS, NJS |
| Vietnamese | HQTV, PNV, THV, TLV |

The following counts were verified directly from the raw WAV directories used
by this project.

| Speaker | L1 group | Gender | Raw utterances |
|---|---|---:|---:|
| ABA | Arabic | M | 1,129 |
| SKA | Arabic | M | 974 |
| YBAA | Arabic | M | 1,130 |
| ZHAA | Arabic | F | 1,132 |
| BWC | Chinese/Mandarin | M | 1,130 |
| LXC | Chinese/Mandarin | F | 1,131 |
| NCC | Chinese/Mandarin | F | 1,131 |
| TXHC | Chinese/Mandarin | M | 1,132 |
| ASI | Hindi | M | 1,131 |
| RRBI | Hindi | M | 1,130 |
| SVBI | Hindi | F | 1,132 |
| TNI | Hindi | F | 1,131 |
| HJK | Korean | F | 1,131 |
| HKK | Korean | M | 1,131 |
| YDCK | Korean | F | 1,131 |
| YKWK | Korean | M | 1,131 |
| EBVS | Spanish | M | 1,007 |
| ERMS | Spanish | M | 1,132 |
| MBMPS | Spanish | F | 1,132 |
| NJS | Spanish | F | 1,131 |
| HQTV | Vietnamese | M | 1,132 |
| PNV | Vietnamese | F | 1,132 |
| THV | Vietnamese | F | 1,132 |
| TLV | Vietnamese | M | 1,132 |

| Raw-audit statistic | Value |
|---|---:|
| Official speakers | 24 |
| L1 groups | 6 |
| Raw WAV/transcript pairs | 26,867 |
| Missing WAV/transcript pairs | 0 |
| Highest speaker count | 1,132 |
| Lowest speaker count | 974 (SKA) |
| Exact raw-audio duration | 27.074 hours |
| Extra non-speaker folder | `suitcase_corpus` (excluded) |

The earlier local dataset note reported 26,978 utterances as an estimate. The
reproducible raw-directory audit used by the preparation script finds 26,867;
the latter is the authoritative value for the experiments and paper.

### CMU ARCTIC reference corpus

CMU ARCTIC is a native English corpus built from the same prompt set. It is not
part of the six L2-ARCTIC folds described below, but it can be used separately
as a native-speaker reference or prompt-aligned clean baseline.

The existing local audit notes report six native English speakers and 6,779
valid WAV/transcript pairs:

| Speaker | Gender | Valid pairs |
|---|---:|---:|
| `cmu_us_awb_arctic` | M | 1,138 |
| `cmu_us_clb_arctic` | F | 1,132 |
| `cmu_us_rms_arctic` | M | 1,132 |
| `cmu_us_slt_arctic` | F | 1,132 |
| `cmu_us_bdl_arctic` | M | 1,131 |
| `cmu_us_jmk_arctic` | M | 1,114 |

The same notes identify 19 missing transcript files: one for BDL and 18 for
JMK, with no missing WAV files. These CMU ARCTIC figures are contextual and are
not consumed by the current L2-ARCTIC preparation command.

## Experimental objective

For the resubmission experiments, L2-ARCTIC is reconstructed directly from the
raw data under `data/raw/l2_arctic/speakers`. The old processed Parquet files in
`l2_arctic_cv` are not used.

The preparation pipeline pairs raw WAV files with their transcripts, retains
the 954 prompts available for all 24 speakers, copies each WAV only once, and
generates six folds: Arabic, Chinese, Hindi, Korean, Spanish, and Vietnamese.

In each fold:

- the accent named by the fold is completely absent from train and development;
- all four speakers of that accent form the test set;
- the other five accents provide the train and development sets;
- speakers and prompts are disjoint across train, development, and test.

The protocol therefore measures joint generalization to an unseen accent,
unseen speakers, and unseen prompts.

## Implementation details

### Inventory construction

The script scans the raw directories of the 24 official speakers. For each WAV,
it looks for a transcript with the same prompt identifier. An item is rejected
if its transcript is missing or empty, its audio is empty, or its
`(speaker_id, prompt_id)` pair is duplicated.

Transcripts are normalized by trimming surrounding whitespace, lowercasing,
removing punctuation except apostrophes, and collapsing consecutive whitespace
into one space.

Audio is neither resampled nor transformed during preparation. All 26,867 raw
WAV files are mono, 16-bit PCM, at 44.1 kHz. They are copied once into the shared
`wavs/` directory; all folds store relative paths to those files. Resampling to
the model input rate must be performed by the training data loader.

To obtain a balanced parallel corpus, a prompt is eligible only when audio and
a transcript are available for all 24 speakers. The global intersection
contains 954 prompts and 22,896 examples (`954 × 24`). Of the 26,867 raw pairs,
3,971 are excluded because their prompt is not available for every speaker.

Each `inventory.parquet` row stores the corpus, speaker, gender, L1/accent,
utterance and prompt identifiers, normalized transcript, relative audio path,
duration, sample rate, and channel count.

### Deterministic fold construction

The 954 prompts are deterministically shuffled with split seed `20260817` and
globally partitioned using a target ratio of 80/10/10. Largest-remainder
allocation produces exactly 763 train prompts, 96 development prompts, and 95
test prompts. The same prompt partition is used in all six folds.

For each held-out accent:

- its four speakers are assigned to test;
- for each of the five seen accents, three speakers are assigned to train and
  one speaker is assigned to development;
- an inventory item is retained only when its speaker role and prompt role
  designate the same split.

Consequently, 80/10/10 applies to prompts rather than directly to audio
examples. Speaker-prompt crossings assigned to different splits are
deliberately unused in that fold.

The split seed is independent of model-training seeds. Changing a model seed
must never change the prompt, speaker, or accent assignments.

## Distribution in each fold

| Split | Examples | Prompts | Speakers | Accents | Per-accent allocation |
|---|---:|---:|---:|---:|---|
| Train | 11,445 | 763 | 15 | 5 | 3 speakers and 2,289 examples per seen accent |
| Development | 480 | 96 | 5 | 5 | 1 speaker and 96 examples per seen accent |
| Test | 380 | 95 | 4 | 1 | 4 speakers and 380 examples from the held-out accent |

Train and development contain the same five seen accents. Test contains only
the accent named by the fold: Arabic, Chinese, Hindi, Korean, Spanish, or
Vietnamese.

## Validation and reproducibility

Generation automatically verifies:

- the presence of six expected accents and four speakers per accent;
- uniqueness of speaker-prompt pairs;
- zero prompt overlap across train, development, and test;
- zero speaker overlap across train, development, and test;
- complete absence of the held-out accent from train and development;
- exclusive presence of the held-out accent in test;
- non-empty train, development, and test sets.

Each manifest records the split seed, held-out accent, prompt lists, speaker
roles, source-inventory fingerprint, and its own SHA-256. Its
`validation_report.json` must have status `passed` and report zero prompt and
speaker overlap.

Manifests remain next to the generated data under `data/processed/`; they are
not versioned separately. They can be reproduced with the same raw data, code,
and split seed. Every experiment must record `split_manifest_sha256` to identify
the exact fold it consumed.

## Reporting in the paper

The protocol should be described as strict six-fold leave-one-accent-out
cross-validation. For every fold, Stage 2 training and checkpoint selection use
only train and development; test is evaluated only after model selection.

Because test combines an unseen accent, unseen speakers, and unseen prompts,
the results demonstrate joint generalization across all three factors. They
must not be described as isolating only the effect of accent. The paper should
report WER for every held-out accent, the macro-average across the six accents,
and variability across the planned model seeds.

## Generated files

```text
data/processed/l2_arctic_leave_one_accent_out/
├── wavs/                    # one shared copy of the WAV files
├── inventory.parquet       # canonical inventory: 22,896 examples
├── inventory_report.json
├── fold_summary.csv
├── arabic/
├── chinese/
├── hindi/
├── korean/
├── spanish/
└── vietnamese/
```

Each accent directory contains:

- `corpus.parquet`: examples and their `split` column;
- `split_stats.csv`: example, speaker, prompt, accent, and duration statistics;
- `manifest.json`: exact fold definition;
- `manifest.content.sha256`: manifest fingerprint;
- `validation_report.json`: automatic overlap checks.

## Generation command

Run from the repository root:

```bash
PYTHONPATH=src python -m accented_asr.data.prepare_l2_arctic \
  --corpus-dir data/raw/l2_arctic/speakers \
  --output-dir data/processed/l2_arctic_leave_one_accent_out \
  --repository-root . \
  --split-seed 20260817
```

The split seed must remain fixed across all experimental conditions and must not
be confused with the random seeds used to train models.

### Automatic preparation with Snakemake

Stage 2 declares the selected fold's Parquet file and manifest as inputs. If
the processed artifacts are absent, Snakemake automatically runs
`prepare_l2_arctic_splits` first and builds all six folds from
`data/raw/l2_arctic/speakers`. No separate preparation command is required.

The raw corpus must therefore be copied to the same repository-relative path
on every machine. For example, from the machine that already stores L2-ARCTIC:

```bash
rsync -av --info=progress2 \
  data/raw/l2_arctic/ <magi-user>@<magi-host>:<repository>/data/raw/l2_arctic/
```

Run the usual Stage 2 command on Magi afterward. Snakemake will prepare the
data only when one or more declared processed artifacts are missing:

```bash
pixi run snakemake -s Snakefile stage2_adaptation \
  --configfile experiments/stage2/wav2vec2-large-lv60/supcon-only/arabic/config.yaml \
  --config run_seed=13 --cores 1
```

To inspect the dependency chain without executing it, add `--dry-run`. The raw
root can also be overridden by the launcher if a machine uses a different mount
point:

```bash
--config l2_arctic_raw_dir=/path/to/l2_arctic/speakers
```
