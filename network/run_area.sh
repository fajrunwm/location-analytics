#!/usr/bin/env bash
# Full per-area pipeline on Azure ML: fetch -> centrality (drive, walk) -> features.
# Works on the node's local disk, then copies results to the mounted output folder.
set -euo pipefail
AREA=$1; OUT=$2; OUTLETS=$3
WORK=./work
nproc; free -g | head -2
python fetch_data.py --area "$AREA" --out "$WORK" --cache "$WORK/_cache"
python centrality.py --area "$AREA" --net drive --out "$WORK"
python centrality.py --area "$AREA" --net walk --out "$WORK"
python build_features.py --area "$AREA" --out "$WORK" --outlets "$OUTLETS"
mkdir -p "$OUT/$AREA"
cp -r "$WORK/$AREA/." "$OUT/$AREA/"
ls -la "$OUT/$AREA"
