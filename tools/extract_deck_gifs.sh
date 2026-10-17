#!/usr/bin/env bash
# Extract the curated GIF clips from the ICRA slide deck and convert them to web video.
# Usage: tools/extract_deck_gifs.sh  (writes public/videos/deck/*.mp4|webm|jpg)
set -euo pipefail
SITE=/scr/kartiksh/embodyhoi-website
# GIFs are byte-identical in ICRA_presentation_v2.pptx and "ICRA presentation improvement plan-2.pptx";
# the image numbers below are those of ICRA_presentation_v2.pptx. Labels follow the slides of
# "ICRA presentation improvement plan-2.pptx" (slide numbers in parentheses).
DECK=/scr/kartiksh/EmbodyHOI-ICRA/ICRA_presentation_v2.pptx
RAW=$SITE/media/deck_gifs
OUT=$SITE/public/videos/deck
mkdir -p "$RAW" "$OUT"
unzip -qo -j "$DECK" 'ppt/media/*.gif' -d "$RAW"

declare -A NAME=(
  [image14]=human_demo_box               # (2) human demonstration, MANO/SMPL
  [image1]=g1_box_unguided_ref           # (2,9) retarget+RL from the unguided reference: arm cannot reach
  [image20]=g1_box_guided_ref            # (9) same task, guided reference
  [image8]=hoi_overlay_waffleiron        # (3,5) "Close the waffle iron with the right hand", guided over unguided
  [image3]=hoi_overlay_box               # (5)
  [image13]=hoi_overlay_mixer            # (5)
  [image5]=g1_waffleiron_close_right     # (3) RL policy for the waffle-iron prompt
  [image4]=g1_close_box_both             # (8) "Close box with both hands"
  [image19]=g1_microwave_close_open_left # (8) "Close and open microwave with left hand"
  [image15]=g1_mixer_open_close_left     # (8) "Open and close mixer with left hand"
  [image21]=g1_microwave_pick_place_both # (8) "Pick and place microwave with both hands"
  [image7]=g1_box_long                   # (12) "G1 - BOX", 10 s
  [image16]=h12_mixer                    # (12) "H1-2 - MIXER"
  [image18]=h12_base_moves_back          # (11) H1-2: base moves back
  [image23]=g1_base_moves_closer         # (11) G1: base moves closer
  [image22]=g1_mixer_slip_failure        # (13) failure case
)
for k in "${!NAME[@]}"; do
  src=$RAW/$k.gif; dst=$OUT/${NAME[$k]}
  # the GIFs are 10 fps; motion-interpolate to 30 fps (checked: no visible artifacts), even dims for yuv420p
  VF="minterpolate=fps=30:mi_mode=mci:mc_mode=aobmc:me_mode=bidir:vsbmc=1,scale=trunc(iw/2)*2:trunc(ih/2)*2"
  ffmpeg -v error -y -i "$src" -vf "$VF" -c:v libx264 -crf 22 -pix_fmt yuv420p -movflags +faststart "$dst.mp4"
  ffmpeg -v error -y -i "$dst.mp4" -c:v libvpx-vp9 -crf 34 -b:v 0 -row-mt 1 -pix_fmt yuv420p "$dst.webm"
  ffmpeg -v error -y -i "$dst.mp4" -frames:v 1 -q:v 3 "$dst.jpg"
done
ls "$OUT"
