set -euo pipefail
set -x

cd /home/nkhaous/myLLF/speech_accent/representations_distances
source .venv/bin/activate
cd /home/nkhaous/myLLF/speech_accent/representations_distances/dtw_distances

metadata=/home/nkhaous/myLLF/speech_accent/representations_distances/data/metadata_balanced.csv
# normalizarion=l2
k=5

for normalization in l2 center center_l2 center_std; do
    for model in whisper xlsr53; do

        if [[ "$model" == "xlsr53" ]]; then
            min_layer=0
            max_layer=24
        elif [[ "$model" == "whisper" ]]; then
            min_layer=0
            max_layer=32
        else
            echo "Unknown model: $model" >&2
            exit 1
        fi


        # centralization + l2 normalization ..
        python distances_to_native_dtw.py \
            --input_dir "/home/nkhaous/myLLF/speech_accent/representations_distances/representations/${model}_aligned" \
            --metadata ${metadata} \
            --output_pkl results/${model}/k-${k}_${normalization}/dtw_results.pkl \
            --output_figure results/${model}/k-${k}_${normalization}/dtw_figure.png \
            --languages spanish french mandarin arabic korean german vietnamese \
            --k ${k} \
            --band_ratio 0.1 \
            --min_layer ${min_layer} \
            --max_layer ${max_layer} \
            --seed 42 \
            --normalization ${normalization}

        python plot.py \
            --wav2vec_results results/${model}/k-${k}_${normalization}/dtw_results.pkl \
            --k ${k} \
            --output_dir results/${model}/ \
            --plot_single \
            --show_ci_overall
    done
done





























# cd /home/nkhaous/myLLF/speech_accent/representations_distances
# source .venv/bin/activate
# cd /home/nkhaous/myLLF/speech_accent/representations_distances/dtw_distances

# k=5
# metadata=/home/nkhaous/myLLF/speech_accent/representations_distances/data/metadata_balanced.csv

# # model=wav2vec
# # min_layer=0
# # max_layer=24
# # input_dir=/home/nkhaous/myLLF/speech_accent/representations_distances/representations/xlsr53_aligned

# model=whisper
# min_layer=16
# max_layer=32
# input_dir=/home/nkhaous/myLLF/speech_accent/representations_distances/representations/whisper_aligned


# # centralization + l2 normalization ..
# python distances_to_native_dtw.py \
#     --input_dir ${input_dir} \
#     --metadata ${metadata} \
#     --output_pkl results/${model}/k-${k}/dtw_results.pkl \
#     --output_figure results/${model}/k-${k}/dtw_figure.png \
#     --languages spanish french mandarin arabic korean german vietnamese \
#     --k ${k} \
#     --band_ratio 0.1 \
#     --min_layer ${min_layer} \
#     --max_layer ${max_layer} \
#     --seed 42 \
#     --normalize

# python plot.py \
#     --wav2vec_results results/${model}/k-${k}/dtw_results.pkl \
#     --k ${k} \
#     --output_dir results/${model}/ \
#     --plot_single \
#     --show_ci_overall