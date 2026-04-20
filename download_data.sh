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



case "$DATASET" in
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