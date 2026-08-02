#!/usr/bin/env python3
"""Decode 20 frames per clip at 1 Hz, fast enough to do it in the dataloader.

The naive route is unworkable. `fps=1` makes ffmpeg decode all 605 HEVC frames of
a 1080p clip to keep 20 of them: 1.05 s per clip, and only 5.9 clip/s across 48
processes, against the ~80 clip/s a 5-GPU step consumes. That would have meant
pre-extracting every clip to disk -- 8.2 hours and 39 GB.

None of that is necessary. The clips are encoded with a one-second GOP: probing
one gives exactly 21 I-frames at t = 0, 1, 2 ... 20 s. The keyframes *are* the
1 Hz sampling we want, so `-skip_frame nokey` skips every inter-frame and decodes
only what we keep. That is 0.17 s per clip, 151.9 clip/s across 48 processes --
a 26x speedup that turns a required preprocessing stage into a dataloader call.

Frames come out at 384x216, chosen against the Qwen3-VL vision tower: patch 16
with spatial_merge 2 gives (384/16 x 216/16) / 4 = 78 tokens per frame, so 20
frames cost 1,560 tokens. Downscale in the processor for a tighter budget --
320x180 gives 55 tokens per frame and 320x180x20 = 1,100 total.

    python -m training.video_frames --check 20
"""

import argparse
import glob
import io
import os
import shutil
import subprocess
import tempfile
import time
import zipfile

import numpy as np
import pandas as pd
from PIL import Image

from datatools import paths

FRAME_WIDTH = 384
FRAME_HEIGHT = 216
N_FRAMES = 20
JPEG_QUALITY = 4          # ffmpeg -q:v, lower is better; 4 is visually clean


def decode_keyframes(mp4_bytes, n_frames=N_FRAMES, width=FRAME_WIDTH,
                     height=FRAME_HEIGHT, quality=JPEG_QUALITY):
    """Keyframes only, as a list of PIL images.

    ffmpeg is given the file on disk rather than a pipe: seeking a byte stream
    forces it back to sequential decoding, which is precisely the cost this
    function exists to avoid.
    """
    workdir = tempfile.mkdtemp(prefix="radarvlm_")
    try:
        source = os.path.join(workdir, "clip.mp4")
        with open(source, "wb") as fh:
            fh.write(mp4_bytes)
        pattern = os.path.join(workdir, "f_%03d.jpg")
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-threads", "1",
             "-skip_frame", "nokey", "-i", source,
             "-vf", f"scale={width}:{height}",
             "-frames:v", str(n_frames), "-q:v", str(quality), pattern],
            capture_output=True, check=False)
        frames = []
        for path in sorted(glob.glob(os.path.join(workdir, "f_*.jpg"))):
            with Image.open(path) as image:
                frames.append(image.convert("RGB").copy())
        return frames
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def clip_frames(nvidia_root, row, **kwargs):
    """Decode one clip's camera video straight out of its chunk archive."""
    with zipfile.ZipFile(os.path.join(nvidia_root, row["camera_zip"])) as zf:
        data = zf.read(row["camera_video"])
    return decode_keyframes(data, **kwargs)


def pad_frames(frames, n_frames=N_FRAMES):
    """Repeat the last frame if a clip yielded fewer than expected."""
    if not frames:
        blank = Image.new("RGB", (FRAME_WIDTH, FRAME_HEIGHT))
        return [blank] * n_frames
    while len(frames) < n_frames:
        frames.append(frames[-1])
    return frames[:n_frames]


def token_count(width=FRAME_WIDTH, height=FRAME_HEIGHT, patch=16, merge=2):
    """Vision tokens per frame after the tower's spatial merge."""
    return (width // patch) * (height // patch) // (merge * merge)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", type=int, default=8, help="clips to decode")
    ap.add_argument("--nvidia-root", default=paths.NVIDIA_ROOT)
    args = ap.parse_args(argv)

    clips = pd.read_parquet(os.path.join(paths.COMMON_DIR, "nvidia_clips.parquet"))
    sample = clips.head(args.check)

    print(f"{FRAME_WIDTH}x{FRAME_HEIGHT} -> {token_count()} vision tokens/frame, "
          f"{token_count() * N_FRAMES} for {N_FRAMES} frames")
    started = time.monotonic()
    counts, sizes = [], []
    for clip_id, row in sample.iterrows():
        frames = pad_frames(clip_frames(args.nvidia_root, row))
        counts.append(len(frames))
        buffer = io.BytesIO()
        frames[0].save(buffer, format="JPEG", quality=90)
        sizes.append(buffer.tell())
    elapsed = time.monotonic() - started
    print(f"{len(sample)} clips in {elapsed:.1f}s -> {len(sample)/elapsed:.1f} clip/s "
          f"single process")
    print(f"frames per clip: {min(counts)}..{max(counts)}, "
          f"first frame {np.mean(sizes)/1024:.0f} KB as JPEG q90")
    print(f"48 processes would give roughly {len(sample)/elapsed*48:.0f} clip/s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
