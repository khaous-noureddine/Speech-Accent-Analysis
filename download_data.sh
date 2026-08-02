#!/bin/bash

set -e

DATASET=$1

# ./download_data.sh arctic
download_arctic() {
    echo "Downloading CMU ARCTIC..."

    mkdir -p data/arctic
    cd data/arctic

    for spk in awb bdl clb jmk rms slt; do
        wget -c http://festvox.org/cmu_arctic/packed/cmu_us_${spk}_arctic.tar.bz2
    done

    for file in *.tar.bz2; do
        tar -xvjf "$file"
    done

    rm -f *.tar.bz2
    echo "CMU ARCTIC downloaded successfully."
}

# ./download_data.sh l2_arctic
download_l2arctic() {
    echo "Downloading L2-ARCTIC...
    This dataset is downloaded from the drive then copied to the server directly, one it's prepared we'll put it in next cloud and update the donwload link here
    "
}

# ./download_data.sh speech_accents
download_speechaccents(){
    mkdir -p data/speech_accents
    cd data/speech_accents
    curl -L -o speech-accent-archive.zip https://www.kaggle.com/api/v1/datasets/download/rtatman/speech-accent-archive
    unzip speech-accent-archive.zip
    rm speech-accent-archive.zip
}

# ./download_data.sh librispeech_train
download_librispeech_train(){
    mkdir -p data/raw/librispeech/train
    cd data/raw/librispeech/train
    wget https://www.openslr.org/resources/12/train-clean-100.tar.gz
    tar -xvzf train-clean-100.tar.gz
}

# ./download_data.sh librispeech_dev
download_librispeech_dev(){
    mkdir -p data/raw/librispeech/eval
    cd data/raw/librispeech/eval
    wget https://www.openslr.org/resources/12/dev-clean.tar.gz
    tar -xvzf dev-clean.tar.gz
}


# ./download_data.sh librispeech_test
download_librispeech_test(){
    mkdir -p data/raw/librispeech/test
    cd data/raw/librispeech/test
    wget https://www.openslr.org/resources/12/test-clean.tar.gz 
    # wget https://openslr.trmal.net/resources/12/test-clean.tar.gz   
    tar -xvzf test-clean.tar.gz
}


case "$DATASET" in
    librispeech_train)
        download_librispeech_train
        ;;
    librispeech_dev)
        download_librispeech_dev
        ;;
    librispeech_test)
        download_librispeech_test
        ;;
    speech_accents)
        download_speechaccents
        ;;
    arctic)
        download_arctic
        ;;
    l2_arctic)
        download_l2arctic
        ;;
    all)
        download_arctic
        cd ../../
        download_l2arctic
        ;;
    *)
        echo "Usage:"
        echo "./download.sh arctic"
        echo "./download.sh l2_arctic"
        echo "./download.sh all"
        exit 1
        ;;
esac