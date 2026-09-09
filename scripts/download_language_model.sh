#!/usr/bin/env bash

set -euo pipefail

repository="patrickvonplaten/wav2vec2-base-100h-with-lm"
revision="main"
destination="${1:-language_models/wav2vec2-base-100h-with-lm}"
base_url="https://huggingface.co/${repository}/resolve/${revision}"

files=(
  "alphabet.json"
  "language_model/4-gram.bin"
  "language_model/attrs.json"
  "language_model/unigrams.txt"
)

for relative_path in "${files[@]}"; do
  target="${destination}/${relative_path}"
  mkdir -p "$(dirname "${target}")"
  if [[ -s "${target}" ]]; then
    echo "Already present: ${target}"
    continue
  fi
  echo "Downloading ${repository}/${relative_path}"
  curl --fail --location --retry 5 --retry-all-errors \
    --continue-at - --output "${target}.part" \
    "${base_url}/${relative_path}"
  mv "${target}.part" "${target}"
done

echo "Language-model assets ready in ${destination}"
