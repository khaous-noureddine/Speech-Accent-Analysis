#!/bin/bash

cd /home/nkhaous/myLLF/speech_accent/representations_distances
source .venv/bin/activate
cd /home/nkhaous/myLLF/speech_accent/representations_distances/perf-distance_correlation

WER="/home/nkhaous/myLLF/speech_accent/asr_analysis/data/speakers_with_metrics.csv"
PHOIBLE="data/phoible.csv"   # sera téléchargé automatiquement si absent


# ── 1. wav2vec2, toutes les langues ───────────────────────────────────────
python phonological_distance_wer.py \
    --wer_data $WER \
    --wer_column "wer__wav2vec2-large-960h" \
    --phoible_csv $PHOIBLE \
    --model_name "wav2vec2-large-960h" \
    --output_plot  results/phoible/phon_dist/wav2vec2_all_langs.png \
    --output_stats results/phoible/phon_dist/wav2vec2_all_langs.json