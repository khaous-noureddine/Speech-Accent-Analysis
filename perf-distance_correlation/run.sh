cd /home/nkhaous/myLLF/speech_accent/representations_distances
source .venv/bin/activate
cd /home/nkhaous/myLLF/speech_accent/representations_distances/perf-distance_correlation

speakers_with_metrics=/home/nkhaous/myLLF/speech_accent/asr_analysis/data/speakers_with_metrics.csv







# # mean-pooling
# distance_type=meanpooling

# # for model in whisper xlsr53; do
# for model in xlsr53; do

#   if [ "$model" = "whisper" ]; then
#     target_layer=32
#     distance_results="/home/nkhaous/myLLF/speech_accent/representations_distances/meanpooling_distances/results/${model}/results_${model}.pkl"
#     normalization_type="l2"
#   else
#     target_layer=highest

#     distance_results="/home/nkhaous/myLLF/speech_accent/representations_distances/meanpooling_distances/results/xlsr53_center_l2/results_xlsr53.pkl"
#     normalization_type="center_l2"

#     # distance_results="/home/nkhaous/myLLF/speech_accent/representations_distances/meanpooling_distances/results/xlsr53_l2/results_xlsr53.pkl"
#     # normalization_type="l2"
#   fi

#   python correlation_distance_wer.py \
#     --distance_results "$distance_results" \
#     --wer_data "$speakers_with_metrics" \
#     --wer_column "wer__whisper-large-v3" \
#     --model_name "$(echo $model | tr '[:lower:]' '[:upper:]') (Mean-pooling${normalization_type})" \
#     --output_plot "results/${distance_type}/${model}_${target_layer}_${normalization_type}/correlation_distance_wer.png" \
#     --output_stats "results/${distance_type}/${model}_${target_layer}_${normalization_type}/correlation_stats.json"
#      # --target_layer "$target_layer" \
# done 










# dtw
distance_type=dtw
k=10

for model in wav2vec; do
# for model in whisper wav2vec; do
  if [ "$model" = "whisper" ]; then
    model_name="Whisper large-v3 (DTW)"
    target_layer=32
    # distance_results="/home/nkhaous/myLLF/speech_accent/representations_distances/dtw_distances/results/${model}/k-5/dtw_results.pkl"
    distances_results="/home/nkhaous/myLLF/speech_accent/representations_distances/dtw_distances/results/${model}/k-${k}/dtw_results.pkl"
  else
    model_name="Wav2Vec (DTW)"
    target_layer=21
    distances_results="/home/nkhaous/myLLF/speech_accent/representations_distances/dtw_distances/results/${model}/k-${k}/dtw_results.pkl"

  fi

  python correlation_distance_wer.py \
    --distance_results  "$distances_results" \
    --wer_data "$speakers_with_metrics" \
    --wer_column "wer__whisper-large-v3" \
    --model_name "$model_name" \
    --output_plot "results/${distance_type}/${model}_${target_layer}_${k}/correlation_distance_wer.png" \
    --output_stats "results/${distance_type}/${model}_${target_layer}_${k}/correlation_stats.json" \
    --target_layer "$target_layer"
done