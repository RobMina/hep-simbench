#!/usr/bin/env bash
# Fetch the public CaloChallenge 2022 Dataset 1 photon files (CC-BY-4.0) and check every md5 against Zenodo's record.
#   bash validation/calochallenge/fetch.sh DATA_DIR
# About 2.5 GB: Zenodo 10.5281/zenodo.8099322 (Geant4, two photon files) and 10.5281/zenodo.15961728 (15 submitted
# samples), plus the binning file from github.com/CaloChallenge/homepage (sha256 printed for the run record).
set -euo pipefail
out=${1:?DATA_DIR}
mkdir -p "$out"
fetch_record() {  # record id, name filter (python regex)
  curl -fsSL "https://zenodo.org/api/records/$1" | python3 -c '
import json, re, sys
rec = json.load(sys.stdin)
for f in rec["files"]:
    if re.fullmatch(sys.argv[1], f["key"]):
        print(f["key"], f["checksum"].split(":", 1)[1], f["links"]["self"])' "$2" |
  while read -r key md5 url; do
    if [[ -s "$out/$key" ]] && [[ $(md5sum "$out/$key" | cut -d" " -f1) == "$md5" ]]; then echo "ok   $key"; continue; fi
    curl -fsSL -o "$out/$key.part" "$url"
    got=$(md5sum "$out/$key.part" | cut -d" " -f1)
    [[ "$got" == "$md5" ]] || { echo "md5 mismatch for $key: $got != $md5" >&2; exit 1; }
    mv "$out/$key.part" "$out/$key"; echo "got  $key"
  done
}
fetch_record 8099322 'dataset_1_photons_[12]\.hdf5'
fetch_record 15961728 'ds1-photons_[0-9]+\.hdf5'
curl -fsSL -o "$out/binning_dataset_1_photons.xml" \
  https://raw.githubusercontent.com/CaloChallenge/homepage/main/code/binning_dataset_1_photons.xml
sha256sum "$out/binning_dataset_1_photons.xml"
