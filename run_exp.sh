set -euo pipefail
set -x


model=whisper
model_name=whisper-large-v3

corpus_dir="/home/nkhaous/myLLF/speech_accent/data/speech-accent/recordings/recordings"
data_dir="/home/nkhaous/myLLF/speech_accent/data/speech-accent"

working_dir="/home/nkhaous/myLLF/speech_accent/representations_distances"
representations_dir=${working_dir}/representations/repr_${model_name}


if [ ! -d "${working_dir}" ]; then
  mkdir -p $working_dir
  mkdir -p $working_dir/distances
fi


# Extract representation
# ----------------------
if [ -d "$representations_dir" ]; then
    echo "$representations_dir exists... do not extract representations"
else
     python extract_representations.py\
           --corpus_dir  $corpus_dir\
           --output $representations_dir\
           --model $model\
           --device auto \
           --gpu_id 1
        #    --max_n_files 10
fi


# 




































# #----------------------------- Wav2Vec2 -----------------------------
# model=whisper
# model_name=whisper-large-v3

# corpus_dir="/home/nkhaous/myLLF/speech_accent/data/speech-accent/recordings/recordings"
# data_dir="/home/nkhaous/myLLF/speech_accent/data/speech-accent"

# working_dir="/home/nkhaous/myLLF/speech_accent/representations_distances"
# representations_dir=${working_dir}/representations/repr_${model_name}


# if [ ! -d "${working_dir}" ]; then
#   mkdir -p $working_dir
#   mkdir -p $working_dir/distances
# fi


# # Extract representation
# # ----------------------
# if [ -d "$representations_dir" ]; then
#     echo "$representations_dir exists... do not extract representations"
# else
#      python extract_representations.py\
#            --corpus_dir  $corpus_dir\
#            --output $representations_dir\
#            --model $model\
#            --device auto \
#            --gpu_id 1
#         #    --max_n_files 10
# fi