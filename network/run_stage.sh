#!/usr/bin/env bash
# One stage of the per-area pipeline, so big areas can run drive and walk on separate nodes.
#   run_stage.sh fetch|drive|walk|features AREA OUT OUTLETS
# Heavy work happens on local disk; inputs/outputs live in the mounted OUT folder.
set -euo pipefail
STAGE=$1; AREA=$2; OUT=$3; OUTLETS=$4
WORK=./work
nproc; free -g | head -2
mkdir -p "$WORK/$AREA/raw" "$OUT/$AREA/raw"
case "$STAGE" in
  fetch)
    python fetch_data.py --area "$AREA" --out "$WORK" --cache "$WORK/_cache"
    cp -r "$WORK/$AREA/raw/." "$OUT/$AREA/raw/"
    ;;
  drive|walk)
    cp "$OUT/$AREA/raw/$STAGE.graphml" "$OUT/$AREA/raw/"{places,population,transit}.parquet "$WORK/$AREA/raw/"
    python centrality.py --area "$AREA" --net "$STAGE" --out "$WORK"
    cp "$WORK/$AREA/nodes_$STAGE.parquet" "$OUT/$AREA/"
    [ "$STAGE" = walk ] && cp "$WORK/$AREA/segments_walk_angular.parquet" "$OUT/$AREA/"
    ;;
  features)
    python build_features.py --area "$AREA" --out "$OUT" --outlets "$OUTLETS"
    ;;
esac
ls -la "$OUT/$AREA"
