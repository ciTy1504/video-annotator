#!/usr/bin/env python3
"""
cut_video.py — Cut annotated video into segments.

Requires: pip install imageio imageio-ffmpeg
  (imageio-ffmpeg bundles its own ffmpeg binary — no manual install needed!)

Usage:
    python cut_video.py <annotation.json> [--out_dir PATH]

Example:
    python cut_video.py videos/video_001.json
    python cut_video.py videos/video_001.json --out_dir clips/

Output: <video_basename>_001.mp4, _002.mp4, ...
Audio is stripped (sign language videos don't need it).
Already-existing clips are skipped (idempotent).
"""

import argparse
import json
import os
import subprocess
import sys


def get_ffmpeg():
    """Return bundled ffmpeg from imageio-ffmpeg (no system install needed)."""
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except ImportError:
        print("ERROR: imageio-ffmpeg not installed.")
        print("Please run:  pip install imageio imageio-ffmpeg")
        sys.exit(1)


def get_duration(ffmpeg, video_path):
    """Get video duration in seconds using ffprobe bundled with imageio-ffmpeg."""
    ffprobe = ffmpeg.replace('ffmpeg', 'ffprobe')
    # imageio-ffmpeg ships ffmpeg only, so fall back to ffmpeg -i trick
    result = subprocess.run(
        [ffmpeg, '-i', video_path],
        capture_output=True, text=True
    )
    import re
    m = re.search(r'Duration: (\d+):(\d+):(\d+\.\d+)', result.stderr)
    if m:
        h, mi, s = int(m.group(1)), int(m.group(2)), float(m.group(3))
        return h * 3600 + mi * 60 + s
    return None


def cut_segment(ffmpeg, video_path, start, end, out_path):
    """Cut one segment — no audio, stream copy video track."""
    cmd = [
        ffmpeg,
        '-y',                          # overwrite output
        '-ss', str(start),             # fast seek BEFORE -i
        '-to', str(end),
        '-i', video_path,
        '-an',                         # strip audio
        '-c:v', 'copy',                # copy video stream (no re-encode, very fast)
        '-avoid_negative_ts', '1',
        out_path,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"\n  Error:\n{result.stderr[-600:]}")
        return False
    return True


def main():
    parser = argparse.ArgumentParser(description='Cut annotated video into segments.')
    parser.add_argument('annotation', help='Path to annotation JSON file')
    parser.add_argument('--out_dir', default='', help='Output directory (default: <video_dir>/clips/)')
    args = parser.parse_args()

    # ── Load annotation ──────────────────────────────────────────────────────
    json_path = os.path.abspath(args.annotation)
    if not os.path.exists(json_path):
        print(f"ERROR: Annotation file not found: {json_path}")
        sys.exit(1)

    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    video_name = data.get('video', '')
    splits     = sorted(data.get('splits', []))
    duration   = data.get('duration', None)

    # ── Resolve video path ───────────────────────────────────────────────────
    json_dir   = os.path.dirname(json_path)
    video_path = os.path.join(json_dir, video_name)
    if not os.path.exists(video_path):
        print(f"ERROR: Video not found: {video_path}")
        sys.exit(1)

    # ── Get ffmpeg ───────────────────────────────────────────────────────────
    ffmpeg = get_ffmpeg()

    # ── Get duration if missing ──────────────────────────────────────────────
    if not duration:
        duration = get_duration(ffmpeg, video_path)
    if not duration:
        print("ERROR: Could not determine video duration.")
        sys.exit(1)

    # ── Build segments ───────────────────────────────────────────────────────
    boundaries = [0.0] + splits + [duration]
    segments   = [(boundaries[i], boundaries[i+1]) for i in range(len(boundaries)-1)]

    # ── Output dir ───────────────────────────────────────────────────────────
    if args.out_dir:
        out_dir = os.path.abspath(args.out_dir)
    else:
        out_dir = os.path.join(json_dir, 'clips')
    os.makedirs(out_dir, exist_ok=True)

    base = os.path.splitext(video_name)[0]
    ext  = os.path.splitext(video_name)[1] or '.mp4'

    # ── Print summary ────────────────────────────────────────────────────────
    print(f"\nVideo Annotator - Cut Video")
    print(f"Source     : {os.path.basename(video_path)}")
    print(f"Duration   : {duration:.2f}s")
    print(f"Splits     : {len(splits)}  →  {len(segments)} segments")
    print(f"Output dir : {out_dir}\n")

    ok = skip = fail = 0
    for i, (start, end) in enumerate(segments):
        out_name = f"{base}_{i+1:03d}{ext}"
        out_path = os.path.join(out_dir, out_name)

        if os.path.exists(out_path):
            print(f"  [{i+1:03d}/{len(segments)}] SKIP  {out_name}")
            skip += 1
            continue

        label = f"  [{i+1:03d}/{len(segments)}] CUT   {start:.2f}s -> {end:.2f}s  -> {out_name}"
        print(label, end='', flush=True)
        if cut_segment(ffmpeg, video_path, start, end, out_path):
            print('  OK')
            ok += 1
        else:
            print('  FAIL')
            fail += 1

    print(f"\nDone: {ok} cut, {skip} skipped, {fail} failed.")
    print(f"Output: {out_dir}\n")


if __name__ == '__main__':
    main()
