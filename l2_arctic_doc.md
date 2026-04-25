# L2-ARCTIC Dataset Summary

## Overview
Non-native English speech corpus for accented ASR research.  
24 official speakers from 6 L1 language groups using CMU ARCTIC prompts.

## Languages / Speakers

| L1 Language | Speakers |
|---|---|
| Arabic | ABA, YBAA, ZHAA, SKA |
| Mandarin | BWC, LXC, NCC, TXHC |
| Hindi | ASI, RRBI, SVBI, TNI |
| Korean | HJK, HKK, YDCK, YKWK |
| Spanish | EBVS, ERMS, MBMPS, NJS |
| Vietnamese | PNV, THV, TLV, HQTV |

## Corpus Statistics (raw audit)

| Metric | Value |
|---|---|
| Official speakers | 24 |
| L1 groups | 6 |
| Typical utterances / speaker | 1130–1132 |
| Highest count | 1132 |
| Lowest official count | 974 (SKA) |
| Extra folder detected | suitcase_corpus (22 files) |
| Missing wav/txt pairs | 0 |

## Speaker Counts

| Speaker | L1 | Gender | Utterances |
|---|---|---|---|
| PNV | Vietnamese | F | 1132 |
| TXHC | Mandarin | M | 1132 |
| ZHAA | Arabic | F | 1132 |
| ERMS | Spanish | M | 1132 |
| TLV | Vietnamese | M | 1132 |
| HQTV | Vietnamese | M | 1132 |
| THV | Vietnamese | F | 1132 |
| MBMPS | Spanish | F | 1132 |
| SVBI | Hindi | F | 1132 |
| ASI | Hindi | M | 1131 |
| TNI | Hindi | F | 1131 |
| YKWK | Korean | M | 1131 |
| NCC | Mandarin | F | 1131 |
| NJS | Spanish | F | 1131 |
| YDCK | Korean | F | 1131 |
| LXC | Mandarin | F | 1131 |
| HKK | Korean | M | 1131 |
| HJK | Korean | F | 1131 |
| RRBI | Hindi | M | 1130 |
| BWC | Mandarin | M | 1130 |
| YBAA | Arabic | M | 1130 |
| ABA | Arabic | M | 1129 |
| EBVS | Spanish | M | 1007 |
| SKA | Arabic | M | 974 |

## Estimated Duration

Average ARCTIC sentence ≈ 3.5–4.5 sec.

| Estimate | Value |
|---|---|
| Total utterances (24 speakers only) | 26,978 |
| Approx total hours | 27 h |
| Approx per speaker | 67 min |



# CMU ARCTIC Dataset Summary

## Overview

CMU ARCTIC is a native English speech corpus built from the same ARCTIC prompt set used by L2-ARCTIC.  
In this project, it can be used as a native-speaker reference corpus for comparison against non-native accented speech.

Unlike L2-ARCTIC, which contains 24 non-native speakers across 6 L1 backgrounds, this raw CMU ARCTIC audit contains 6 native English speakers.

## Speakers

| Speaker ID | Code | Gender | Matched wav/txt pairs |
|---|---:|---|---:|
| cmu_us_awb_arctic | awb | male | 1138 |
| cmu_us_clb_arctic | clb | female | 1132 |
| cmu_us_rms_arctic | rms | male | 1132 |
| cmu_us_slt_arctic | slt | female | 1132 |
| cmu_us_bdl_arctic | bdl | male | 1131 |
| cmu_us_jmk_arctic | jmk | male | 1114 |

## Corpus Statistics Raw Audit

| Metric | Value |
|---|---:|
| Total speakers | 6 |
| Total valid examples wav + transcript | 6,779 |
| Highest matched count | 1,138 awb |
| Lowest matched count | 1,114 jmk |
| Speakers with missing transcripts | 2 |
| Total missing txt files | 19 |
| Total missing wav files | 0 |

## Data Quality Issues

| Speaker | Issue |
|---|---|
| cmu_us_bdl_arctic | 1 wav file has no matching transcript |
| cmu_us_jmk_arctic | 18 wav files have no matching transcript |

All missing-pair issues are transcript-side issues. No missing wav files were detected.

## Comparison With L2-ARCTIC

| Dataset | Speaker type | Speakers | Valid examples | Main role |
|---|---:|---:|---:|---|
| L2-ARCTIC | Non-native English | 24 | 26,978 | Accent-invariance contrastive training |
| CMU ARCTIC | Native English | 6 | 6,779 | Native reference / clean prompt-aligned baseline |

## Estimated Duration

Using the same average ARCTIC sentence duration estimate of approximately 3.5–4.5 seconds:

| Estimate | Value |
|---|---:|
| Total utterances | 6,779 |
| Approx total duration | 6.6–8.5 h |
| Approx average per speaker | 66–85 min |