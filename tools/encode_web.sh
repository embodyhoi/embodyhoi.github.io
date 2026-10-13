#!/usr/bin/env bash
# Transcode render masters to web formats: mp4 (h264) + webm (vp9) + poster jpg.
# Usage: tools/encode_web.sh SRC_DIR DST_DIR [crf=26] [extra ffmpeg -vf filter]
set -euo pipefail
SRC=$1; DST=$2; CRF=${3:-26}; EXTRA_VF=${4:-}
mkdir -p "$DST"
for f in "$SRC"/*.mp4; do
  [ "$(stat -L -c%s "$f")" -gt 1000 ] || continue
  n=$(basename "${f%.mp4}")
  VF="scale=trunc(iw/2)*2:trunc(ih/2)*2${EXTRA_VF:+,$EXTRA_VF}"
  ffmpeg -v error -y -i "$f" -vf "$VF" -c:v libx264 -preset slow -crf "$CRF" -pix_fmt yuv420p -movflags +faststart -an "$DST/$n.mp4"
  ffmpeg -v error -y -i "$DST/$n.mp4" -c:v libvpx-vp9 -crf $((CRF+8)) -b:v 0 -row-mt 1 -pix_fmt yuv420p -an "$DST/$n.webm"
  ffmpeg -v error -y -i "$DST/$n.mp4" -frames:v 1 -q:v 3 "$DST/$n.jpg"
  echo "$n $(stat -c%s "$DST/$n.mp4")B"
done
