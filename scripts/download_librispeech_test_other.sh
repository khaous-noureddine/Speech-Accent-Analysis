#!/usr/bin/env bash
set -euo pipefail

repository_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
destination="${repository_root}/data/raw/librispeech/test"
archive="${destination}/test-other.tar.gz"
extracted="${destination}/LibriSpeech/test-other"
mkdir -p "${destination}"

if [[ ! -d "${extracted}" ]]; then
  curl --fail --location --retry 5 --continue-at - \
    https://www.openslr.org/resources/12/test-other.tar.gz \
    --output "${archive}"
  tar -xzf "${archive}" -C "${destination}"
fi
echo "LibriSpeech test-other is ready in ${extracted}"
