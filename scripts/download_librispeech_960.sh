#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
destination="${repository_root}/data/raw/librispeech/train"
mkdir -p "${destination}"

for subset in train-clean-100 train-clean-360 train-other-500; do
  extracted="${destination}/LibriSpeech/${subset}"
  archive="${destination}/${subset}.tar.gz"
  if [[ -d "${extracted}" ]]; then
    echo "Already extracted: ${extracted}"
    continue
  fi
  curl --fail --location --retry 5 --continue-at - \
    "https://www.openslr.org/resources/12/${subset}.tar.gz" \
    --output "${archive}"
  tar -xzf "${archive}" -C "${destination}"
done

echo "LibriSpeech 960 h subsets are ready in ${destination}/LibriSpeech"
