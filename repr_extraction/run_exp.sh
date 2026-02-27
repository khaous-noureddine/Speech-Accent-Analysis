set -euo pipefail
set -x

PYTHON_BIN=/home/nkhaous/myLLF/speech_accent/representations_distances/.venv/bin/python

# # model=whisper
# # ------------------
# # model=xlsr53
# model=w2v-large
# # ------------------
# # model=hubert 


corpus_dir="/home/nkhaous/myLLF/speech_accent/data/speech-accent/recordings/recordings"
data_dir="/home/nkhaous/myLLF/speech_accent/data/speech-accent"
working_dir="/home/nkhaous/myLLF/speech_accent/representations_distances"

if [ ! -d "${working_dir}" ]; then
  mkdir -p $working_dir
fi


model=mfcc
# Extract representation
# ----------------------
python extract_representations.py \
    --corpus_dir $corpus_dir \
    --textgrid_dir $corpus_dir \
    --metadata_path "$data_dir/mfa_input.tsv" \
    --output ${working_dir}/representations/${model}_aligned/ \
    --model $model\
    --device auto \
    --gpu_id 3 \
    # --max_n_files 3 \




# # Compute distances (mean pooling)
# # ----------------------

# # whisper
# model=whisper

# # python compute_distances_L1.py \
# #     --input_dir /home/nkhaous/myLLF/speech_accent/representations_distances/representations/aligned \
# #     --metadata /home/nkhaous/myLLF/speech_accent/representations_distances/metadata_balanced.csv \
# #     --output_results results/distance_to_native_full.pkl \
# #     --output_figure results/figure1_full.png \
# #     --normalize \
# #     --min_layer 0 \
# #     --max_layer 32

# python compute_distance_to_native.py \
#   --input_dir /home/nkhaous/myLLF/speech_accent/representations_distances/representations/${model}_aligned \
#   --metadata /home/nkhaous/myLLF/speech_accent/representations_distances/metadata_balanced.csv \
#   --output_results results/${model}_v2/results_${model}.pkl \
#   --output_figure results/${model}_v2/figure_${model}.png \
#   --normalize \
#   --compute_per_l1 \
#   --show_individual_l1 \
#   --languages_to_plot spanish french mandarin arabic korean dutch german italian turkish vietnamese \
#   --min_layer 18 \
#   --max_layer 32


# # wav2vec
# # model=xlsr53

# # python compute_distances_L1.py \
# #     --input_dir /home/nkhaous/myLLF/speech_accent/representations_distances/representations/${model}_aligned \
# #     --metadata /home/nkhaous/myLLF/speech_accent/representations_distances/metadata_balanced.csv \
# #     --output_results results/${model}/distance_${model}.pkl \
# #     --output_figure results/${model}/figure1_${model}.png \
# #     --normalize \
# #     --min_layer 0 \
# #     --max_layer 24

# # python compute_distance_to_native.py \
# #   --input_dir /home/nkhaous/myLLF/speech_accent/representations_distances/representations/${model}_aligned \
# #   --metadata /home/nkhaous/myLLF/speech_accent/representations_distances/metadata_balanced.csv \
# #   --output_results results/${model}/results_${model}.pkl \
# #   --output_figure results/${model}/figure_${model}.png \
# #   --normalize \
# #   --compute_per_l1 \
# #   --show_individual_l1 \
# #   --languages_to_plot spanish french mandarin arabic korean dutch german italian turkish vietnamese \
# #   --min_layer 0 \
# #   --max_layer 24



# # Compute per L1 distance to native (this script was used to complete the figure with individual languages)
# # ---------------------------------
# # python per_l1_distance_to_native.py \
# #   --input_dir /home/nkhaous/myLLF/speech_accent/representations_distances/representations/aligned \
# #   --metadata /home/nkhaous/myLLF/speech_accent/representations_distances/metadata_balanced.csv \
# #   --output_pkl results/whisper/results_per_L1.pkl \
# #   --normalize \
# #   --min_layer 0 \
# #   --max_layer 32 \
# #   --n_bootstrap 1000 \
# #   --ci 95





# # Compute distances (dtw)
# # ----------------------

# # wav2vec
# model=wav2vec


# python compute_distance_to_native.py \
#   --input_dir /home/nkhaous/myLLF/speech_accent/representations_distances/representations/${model}_aligned \
#   --metadata /home/nkhaous/myLLF/speech_accent/representations_distances/metadata_balanced.csv \
#   --output_results results/${model}_v2/results_${model}.pkl \
#   --output_figure results/${model}_v2/figure_${model}.png \
#   --normalize \
#   --compute_per_l1 \
#   --show_individual_l1 \
#   --languages_to_plot spanish french mandarin arabic korean dutch german italian turkish vietnamese \
#   --min_layer 18 \
#   --max_layer 32



# # plot
# cd /home/nkhaous/myLLF/speech_accent/representations_distances
# conda deactivate
# source .venv/bin/activate
# python plot_fixed_colors.py