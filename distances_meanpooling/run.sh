#!/usr/bin/env bash
set -euo pipefail
set -x

cd /home/nkhaous/myLLF/speech_accent/representations_distances
source .venv/bin/activate
cd /home/nkhaous/myLLF/speech_accent/representations_distances/meanpooling_distances

normalizarion=center_l2
metadata=/home/nkhaous/myLLF/speech_accent/representations_distances/data/metadata_balanced.csv

for model in xlsr53 whisper; do

  if [[ "$model" == "xlsr53" ]]; then
    min_layer=18
    max_layer=24
  elif [[ "$model" == "whisper" ]]; then
    min_layer=0
    max_layer=32
  else
    echo "Unknown model: $model" >&2
    exit 1
  fi

  mkdir -p "results/${model}_${normalizarion}"

  python compute_distance_to_native_meanpooling.py \
    --input_dir "/home/nkhaous/myLLF/speech_accent/representations_distances/representations/${model}_aligned" \
    --metadata "${metadata}" \
    --output_results "results/${model}_${normalizarion}/results_${model}.pkl" \
    --output_figure "results/${model}_${normalizarion}/figure_${model}.png" \
    --normalization "${normalizarion}" \
    --compute_per_l1 \
    --show_individual_l1 \
    --languages_to_plot spanish french mandarin arabic korean dutch german italian turkish vietnamese \
    --min_layer "${min_layer}" \
    --max_layer "${max_layer}"

done

python plot_fixed_colors.py






# python plot.py \
#     --wav2vec_results results/${model}/k-${k}/dtw_results.pkl \
#     --k ${k} \
#     --output_dir results/${model}/ \
#     --plot_single \
#     --show_ci_overall






























# wav2vec
# model=xlsr53

# python compute_distances_L1.py \
#     --input_dir /home/nkhaous/myLLF/speech_accent/representations_distances/representations/${model}_aligned \
#     --metadata /home/nkhaous/myLLF/speech_accent/representations_distances/metadata_balanced.csv \
#     --output_results results/${model}/distance_${model}.pkl \
#     --output_figure results/${model}/figure1_${model}.png \
#     --normalize \
#     --min_layer 0 \
#     --max_layer 24

# python compute_distance_to_native_meanpooling.py \
#   --input_dir /home/nkhaous/myLLF/speech_accent/representations_distances/representations/${model}_aligned \
#   --metadata /home/nkhaous/myLLF/speech_accent/representations_distances/metadata_balanced.csv \
#   --output_results results/${model}/results_${model}.pkl \
#   --output_figure results/${model}/figure_${model}.png \
#   --normalize \
#   --compute_per_l1 \
#   --show_individual_l1 \
#   --languages_to_plot spanish french mandarin arabic korean dutch german italian turkish vietnamese \
#   --min_layer 0 \
#   --max_layer 24



# Compute per L1 distance to native (this script was used to complete the figure with individual languages)
# ---------------------------------
# python per_l1_distance_to_native.py \
#   --input_dir /home/nkhaous/myLLF/speech_accent/representations_distances/representations/aligned \
#   --metadata /home/nkhaous/myLLF/speech_accent/representations_distances/metadata_balanced.csv \
#   --output_pkl results/whisper/results_per_L1.pkl \
#   --normalize \
#   --min_layer 0 \
#   --max_layer 32 \
#   --n_bootstrap 1000 \
#   --ci 95




















# python compute_distances_L1.py \
#     --input_dir /home/nkhaous/myLLF/speech_accent/representations_distances/representations/aligned \
#     --metadata /home/nkhaous/myLLF/speech_accent/representations_distances/metadata_balanced.csv \
#     --output_results results/distance_to_native_full.pkl \
#     --output_figure results/figure1_full.png \
#     --normalize \
#     --min_layer 0 \
#     --max_layer 32