#!/usr/bin/env python3
"""Tile a mid-frame of every mp4 in the given dirs into one labelled PNG for picking clips.
Usage: make_contact_sheet.py OUT.png DIR [DIR ...]  (uses ffmpeg/ffprobe + PIL)"""
import glob, os, subprocess, sys, tempfile
from PIL import Image, ImageDraw, ImageFont

out, dirs = sys.argv[1], sys.argv[2:]
files = sorted(f for d in dirs for f in glob.glob(os.path.join(d, "*.mp4")) if os.path.getsize(f) > 1000)
tile, cols = 360, 4
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 16)
rows = (len(files) + cols - 1) // cols
sheet = Image.new("RGB", (cols * tile, rows * (tile + 24)), "white")
d = ImageDraw.Draw(sheet)
with tempfile.TemporaryDirectory() as td:
    for i, f in enumerate(files):
        dur = float(subprocess.check_output(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", f]).decode().strip() or 0)
        png = os.path.join(td, f"{i}.png")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{dur*0.5:.2f}", "-i", f, "-frames:v", "1", png], check=True)
        im = Image.open(png).convert("RGB"); im.thumbnail((tile, tile))
        x, y = (i % cols) * tile, (i // cols) * (tile + 24)
        sheet.paste(im, (x + (tile - im.width) // 2, y))
        d.text((x + 4, y + tile + 3), f"{i:02d} {os.path.basename(f)[:40]} {dur:.1f}s", fill="black", font=font)
sheet.save(out); print(out, sheet.size, len(files), "clips")
