set -euo pipefail
set -x


corpus_dir="/home/nkhaous/myLLF/speech_accent/data/speech-accent/recordings/recordings"
data_dir="/home/nkhaous/myLLF/speech_accent/data/speech-accent"
working_dir="/home/nkhaous/myLLF/speech_accent/representations_distances"

mfa_dir="/home/nkhaous/myLLF/speech_accent/mfa"


#-------------------------

cd "$working_dir"
source .venv/bin/activate
python get_mfa_input.py \
    --corpus_dir "$corpus_dir" \
    --data_dir "$data_dir" \
    # --lab_dir "$working_dir/data/lab_files"




#-------------------------

cd "$mfa_dir"
pixi run python align_from_tsv.py \
    --corpus "$data_dir/mfa_input.tsv" \
    --corpus_dir "$corpus_dir" \
    --dictionary ./mfa_models/pretrained_models/dictionary/english_mfa.dict \
    --acoustic ./mfa_models/pretrained_models/acoustic/english_mfa.zip \
    --mfa mfa



#-------------------------
