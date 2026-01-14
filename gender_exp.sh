set -o xtrace
set -e

PYTHON_BIN=/home/gwisniewski/myLLF/Speech-Representations/.venv/bin/python
language=arabic
model=xlsr53

exp_name=gender_exp_${language}

corpus_dir=/home/gwisniewski/myLLF/Speech-Representations/speech-accent-archive/recordings/recordings
metadata_file=/home/gwisniewski/myLLF/Speech-Representations/speech-accent-archive/speakers_all.tsv
working_dir=/home/gwisniewski/myLLF/Speech-Representations/nour/experiments/${exp_name}/

if [ ! -d "${working_dir}" ]; then
  mkdir -p $working_dir
fi


selected_recordings_dir=${working_dir}/selected_recordings
selected_repr_dir=${working_dir}/selected_repr_${model}

if [ ! -d "${working_dir}" ]; then
  mkdir -p $selected_recordings_dir
  mkdir -p $selected_repr_dir
fi


selected_repr_metadata_dir=${working_dir}/selected_repr_${model}_with_metadata.pkl

aligned_repr_dir=${working_dir}/align_${model}


# # 1. Select recordings -> With respect to constraints we give, select the recordings alongside with their textgrids and lib files

# $PYTHON_BIN select_recordings.py\
#     --recordings_dir $corpus_dir\
#     --metadata_file $metadata_file\
#     --output_dir $working_dir/selected_recordings\
#     --native_language $language 


# # 2. Extract Representations -> Extract the representations for the selected recordings using the specified model

# $PYTHON_BIN extract_representations.py\
#     --corpus_dir $selected_recordings_dir\
#     --output_dir $selected_repr_dir\
#     --model $model\
#     --device cuda\


# # 3. Add Metadata -> this gives us a pickle file containing the representations along with metadata of all the recordings ..

# $PYTHON_BIN add_metadata.py\
#     --audio_representation $selected_repr_dir\
#     --output $selected_repr_metadata_dir\
#     --metadata_path $metadata_file\
#     --textgrid_dir $selected_recordings_dir


# # 4. Align annotation with embeddings 

# $PYTHON_BIN align_sentences.py\
#         --input $selected_repr_metadata_dir\
#         --output_dir $aligned_repr_dir\
#         --model $model

# 5. Compute distances

for fn in `find ${aligned_repr_dir} -name "*layer*"`
do
    layer=`basename $fn`
    echo ${layer}

    if [ ! -f "${working_dir}/distances/word_distances_${model}_${layer}.pkl" ]; then
      $PYTHON_BIN compute_distances_between_words.py\
          --input_dir $fn\
          --repr_column repr\
          --output ${working_dir}/distances/word_distances_${model}_${layer}.pkl
      $PYTHON_BIN compute_distances_between_words.py\
          --input_dir $fn\
          --repr_column repr\
          --normalize\
          --output ${working_dir}/distances/word_distances_${model}_${layer}_normalize.pkl
    fi
done