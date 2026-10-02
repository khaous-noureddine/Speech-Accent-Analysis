#!/usr/bin/env bash
set -euo pipefail

destination="${1:-data/raw/mswc_common_voice_en}"
mswc_audio_url="${MSWC_AUDIO_URL:-https://mswc.mlcommons-storage.org/audio/en.tar.gz}"
mswc_splits_url="${MSWC_SPLITS_URL:-https://mswc.mlcommons-storage.org/splits/en.tar.gz}"
cv_metadata_url="${COMMON_VOICE_METADATA_URL:-https://huggingface.co/datasets/mteb/common_voice_21_0/resolve/ff3d45c434d8dc5408def55080a6171f0e2278dc/transcripts/en/validated.tsv?download=true}"

mkdir -p "${destination}/archives" "${destination}/mswc" "${destination}/common_voice_21"

download() {
  local url="$1"
  local target="$2"
  if [[ ! -s "${target}" ]]; then
    curl --fail --location --retry 5 --retry-all-errors --continue-at - \
      --output "${target}" "${url}"
  fi
}

download "${mswc_splits_url}" "${destination}/archives/mswc-en-splits.tar.gz"
if [[ ! -f "${destination}/mswc/en_train.csv" ]]; then
  tar -xzf "${destination}/archives/mswc-en-splits.tar.gz" \
    -C "${destination}/mswc"
fi

download "${mswc_audio_url}" "${destination}/archives/mswc-en-audio.tar.gz"
if [[ ! -f "${destination}/mswc/.audio_extracted" ]]; then
  tar -xzf "${destination}/archives/mswc-en-audio.tar.gz" \
    -C "${destination}/mswc"
  touch "${destination}/mswc/.audio_extracted"
fi

# CV3's historical endpoint is no longer public. Filenames and hashed client
# IDs are stable, so use a pinned CV21 metadata-only mirror and audit coverage.
download "${cv_metadata_url}" "${destination}/common_voice_21/validated.tsv"

touch "${destination}/_DOWNLOAD_SUCCESS"
