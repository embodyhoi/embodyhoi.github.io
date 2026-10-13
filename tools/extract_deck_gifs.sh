#!/usr/bin/env bash
# Extract the curated GIF clips from the ICRA slide deck and convert them to web video.
# Usage: tools/extract_deck_gifs.sh  (writes public/videos/deck/*.mp4|webm|jpg)
set -euo pipefail
SITE=/scr/kartiksh/embodyhoi-website
DECK=/scr/kartiksh/EmbodyHOI-ICRA/ICRA_presentation_v2.pptx
RAW=$SITE/media/deck_gifs
OUT=$SITE/public/videos/deck
mkdir -p "$RAW" "$OUT"
unzip -qo -j "$DECK" 'ppt/media/*.gif' -d "$RAW"

# content-based names (see ICRA_video_script.md for the slide each clip sits on)
declare -A NAME=(
  [image1]=g1_box_unguided_fail        # slide 2/9: unguided reference, arm cannot reach the contact
  [image20]=g1_box_guided_success      # slide 9: guided reference, same task
  [image14]=human_box_mano             # slide 2: human-space HOI (MANO/SMPL) opening box
  [image13]=hoi_overlay_mixer          # slide 5: guided vs unguided overlay (mixer)
  [image3]=hoi_overlay_a               # slide 5
  [image8]=hoi_overlay_b               # slide 3/5
  [image5]=g1_waffleiron_rollout       # slide 3: G1 waffle iron
  [image15]=g1_mixer_rollout           # slide 8
  [image19]=g1_box_reach_rollout       # slide 8
  [image21]=g1_box_lift_rollout        # slide 8
  [image4]=g1_box_open_rollout         # slide 8
  [image7]=g1_box_tilt_rollout_long    # slide 12 (10 s)
  [image16]=h12_mixer_rollout          # slide 12: H1-2
  [image18]=h12_base_placement_side    # slide 11: H1-2 steps back (dark bg)
  [image23]=g1_base_placement_side     # slide 11: G1 closes in (dark bg)
  [image22]=g1_mixer_slip_failure      # slide 13: failure case
)
for k in "${!NAME[@]}"; do
  src=$RAW/$k.gif; dst=$OUT/${NAME[$k]}
  # the GIFs are 10 fps; motion-interpolate to 30 fps (checked: no visible artifacts), even dims for yuv420p
  VF="minterpolate=fps=30:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1,scale=trunc(iw/2)*2:trunc(ih/2)*2"
  ffmpeg -v error -y -i "$src" -vf "$VF" -c:v libx264 -crf 22 -pix_fmt yuv420p -movflags +faststart "$dst.mp4"
  ffmpeg -v error -y -i "$dst.mp4" -c:v libvpx-vp9 -crf 34 -b:v 0 -row-mt 1 -pix_fmt yuv420p "$dst.webm"
  ffmpeg -v error -y -i "$dst.mp4" -frames:v 1 -q:v 3 "$dst.jpg"
done
ls -la "$OUT"
