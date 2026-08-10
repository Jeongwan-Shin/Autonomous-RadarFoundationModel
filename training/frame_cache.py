#!/usr/bin/env python3
"""클립마다 20프레임을 한 번만 디코딩해 두고, 필요한 장만 꺼내 쓴다.

지금은 아이템 하나를 만들 때마다 mp4 를 열어 20장을 전부 풀고 그중 1~5장만
쓴다. `det_objects` 는 1장, `plan_ego`·`agent_traj`·`motion_seg` 는 2장,
`track_step` 은 5장을 쓰고, 20장을 다 쓰는 것은 혼합의 7.7% 인 `qa` 뿐이다.
디코딩의 약 90% 가 버려진다.

캐시는 클립당 npz 하나이고 안에 JPEG 바이트가 프레임별로 따로 들어간다.
`np.load` 는 npz 안의 배열을 요청할 때 읽으므로, 1장만 필요하면 1장만 읽힌다.

    python -m training.frame_cache --workers 60          # 만들기
    python -m training.frame_cache --check 200           # 원본과 대조

측정해 둘 것: 이것으로 벽시계가 줄지는 않는다. 클립 하나 디코딩이 0.155초이고
한 스텝(9.13초)에 아이템 160개면 24.8 코어초인데, 워커 30개에 흩어지면 0.83초로
GPU 연산과 겹쳐 사라진다. 실제로 워커를 30에서 120으로 올려도 스텝 시간은
9.13초에서 9.16초로 그대로였다. 줄어드는 것은 CPU 부하이고, 스텝이 더 빨라진
뒤에 — checkpointing 을 끄거나 LoRA 로 바꾸면 — 비중이 커진다.
"""

import argparse
import glob
import io
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from PIL import Image

from datatools import paths
from training.video_frames import (FRAME_HEIGHT, FRAME_WIDTH, JPEG_QUALITY,
                                   N_FRAMES, clip_frames, pad_frames)

# JPEG_QUALITY is ffmpeg's -q:v, where *lower* is better and 4 is near-lossless.
# PIL's `quality=` runs the other way, 0 worst and 95 the usual ceiling. Handing
# the one to the other silently wrote the whole cache at PIL quality 4: it built
# and loaded fine, and every frame was off by a mean of 10.4/255. Nothing here
# re-encodes any more, so the two scales no longer meet.

CACHE_DIR = os.environ.get("RAVL_FRAME_CACHE",
                           os.path.join(paths.SPLIT_ROOT, "common",
                                        "frame_cache"))


def cache_path(clip_id):
    # 클립이 16만 개다. 한 디렉터리에 다 넣으면 조회가 느려지므로 두 글자로 쪼갠다.
    return os.path.join(CACHE_DIR, clip_id[:2], f"{clip_id}.npz")


def keyframe_bytes(mp4_bytes, n_frames=N_FRAMES, width=FRAME_WIDTH,
                   height=FRAME_HEIGHT, quality=JPEG_QUALITY):
    """`decode_keyframes` 와 같은 ffmpeg 호출인데, 디코딩한 이미지 대신 ffmpeg 이
    써낸 JPEG 파일의 바이트를 그대로 돌려준다.

    캐시가 무손실이 되는 유일한 방법이다. PIL 로 다시 인코딩하면 이미 압축된
    그림을 한 번 더 압축하는 것이라, 품질을 95 로 올려도 픽셀이 평균 0.35/255
    움직인다. 여기서는 파일을 옮겨 담기만 하므로 학습이 보는 그림이 mp4 를
    직접 푼 것과 바이트 단위로 같다.
    """
    workdir = tempfile.mkdtemp(prefix="ravlcache_")
    try:
        source = os.path.join(workdir, "clip.mp4")
        with open(source, "wb") as fh:
            fh.write(mp4_bytes)
        pattern = os.path.join(workdir, "f_%03d.jpg")
        subprocess.run(
            ["ffmpeg", "-y", "-loglevel", "error", "-threads", "1",
             "-skip_frame", "nokey", "-i", source,
             # 없으면 같은 키프레임이 스무 번 복제된다 -- video_frames.py 의
             # 같은 자리에 이유를 적어 두었다.
             "-fps_mode", "passthrough",
             "-vf", f"scale={width}:{height}",
             "-frames:v", str(n_frames), "-q:v", str(quality), pattern],
            capture_output=True, check=False)
        out = []
        for path in sorted(glob.glob(os.path.join(workdir, "f_*.jpg"))):
            with open(path, "rb") as fh:
                out.append(fh.read())
        # 20 장이 전부 같은 그림인 채로 캐시가 통째로 만들어진 적이 있다.
        # 장수도 크기도 형식도 옳아서 아무 데서도 걸리지 않았다. 프레임끼리
        # 비교하는 것이 그 실패를 잡는 유일한 검사라, 여기서 한다.
        if len(out) > 1 and len(set(out)) == 1:
            raise RuntimeError("모든 키프레임이 동일합니다 -- "
                               "-fps_mode passthrough 가 빠졌는지 확인하세요")
        return out
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def write_clip(args_tuple):
    clip_id, row, nvidia_root, force = args_tuple
    path = cache_path(clip_id)
    if os.path.exists(path) and not force:
        return clip_id, 0
    try:
        with zipfile.ZipFile(os.path.join(nvidia_root, row["camera_zip"])) as zf:
            raw = keyframe_bytes(zf.read(row["camera_video"]))
    except Exception:
        return clip_id, -1
    if not raw:
        return clip_id, -1
    # 키프레임이 스무 장에 못 미치는 클립이 있다. 로더의 `pad_frames` 와 같은
    # 규칙으로 마지막 장을 반복해 길이를 맞춘다.
    while len(raw) < N_FRAMES:
        raw.append(raw[-1])
    blobs = {f"f{i:02d}": np.frombuffer(b, dtype=np.uint8)
             for i, b in enumerate(raw[:N_FRAMES])}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    # 같은 캐시를 여러 학습 프로세스가 동시에 읽는다. 반쯤 쓰인 파일이 보이면
    # 그 랭크만 죽으므로 다 쓴 뒤에 제자리로 옮긴다.
    #
    # 파일 이름이 아니라 핸들을 넘긴다. `np.savez` 는 이름을 받으면 `.npz` 로
    # 끝나지 않을 때 확장자를 붙이므로, 이름을 넘기면 `.tmp` 가 `.tmp.npz` 가
    # 되고 뒤따르는 rename 이 있지도 않은 파일을 찾는다.
    with open(tmp, "wb") as fh:
        np.savez(fh, **blobs)
    os.replace(tmp, path)
    return clip_id, os.path.getsize(path)


def load(clip_id, indices=None):
    """캐시에서 프레임을 읽는다. 없으면 None -- 부르는 쪽이 mp4 로 되돌아간다.

    `indices` 를 주면 그 장만 읽는다. npz 는 멤버 단위로 읽으므로 20장 중
    1장만 필요할 때 1장 값만 디코딩된다.
    """
    path = cache_path(clip_id)
    if not os.path.exists(path):
        return None
    try:
        with np.load(path) as z:
            want = range(N_FRAMES) if indices is None else indices
            return [Image.open(io.BytesIO(z[f"f{i:02d}"].tobytes())).convert("RGB")
                    for i in want]
    except Exception:
        # 잘린 파일 하나가 학습을 멈추게 두지 않는다.
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--nvidia-root", default=paths.NVIDIA_ROOT)
    ap.add_argument("--workers", type=int, default=60)
    # 캐시 전체가 잘못 만들어진 적이 있고, 그때 이미 있는 파일을 건너뛰는
    # 규칙이 고친 코드를 무력화했다. 다시 만들 길을 열어 둔다.
    ap.add_argument("--force", action="store_true",
                    help="이미 있는 캐시도 다시 만든다")
    ap.add_argument("--check", type=int, default=0,
                    help="만들지 않고, 캐시와 mp4 디코딩 결과를 이만큼 대조")
    args = ap.parse_args(argv)

    clips = pd.read_parquet(os.path.join(paths.COMMON_DIR,
                                         "nvidia_clips.parquet"))
    # 전방 레이더가 없는 클립은 학습에 쓰이지 않으므로 캐시도 만들지 않는다.
    carries = (clips["has_radar_extrinsics"].fillna(False)
               & (clips["has_lrr1"].fillna(False)
                  | clips["has_mrr2"].fillna(False)
                  | clips["has_srr0"].fillna(False)))
    usable = clips[carries & clips["camera_zip"].notna()]

    if args.check:
        rng = np.random.default_rng(0)
        pick = usable.iloc[rng.choice(len(usable), args.check, replace=False)]
        same = missing = 0
        for cid, row in pick.iterrows():
            cached = load(cid)
            if cached is None:
                missing += 1
                continue
            truth = pad_frames(clip_frames(args.nvidia_root, row), N_FRAMES)
            if all(np.array_equal(np.asarray(a), np.asarray(b))
                   for a, b in zip(cached, truth)):
                same += 1
        print(f"대조 {len(pick)}클립: 일치 {same}, 캐시 없음 {missing}, "
              f"불일치 {len(pick) - same - missing}")
        return 0 if same + missing == len(pick) else 1

    print(f"{len(usable):,} 클립 → {CACHE_DIR}", flush=True)
    tasks = [(cid, usable.loc[cid].to_dict(), args.nvidia_root, args.force)
             for cid in usable.index]
    done = failed = 0
    total_bytes = 0
    started = time.monotonic()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(write_clip, t) for t in tasks]
        for i, f in enumerate(as_completed(futures), 1):
            _, size = f.result()
            if size < 0:
                failed += 1
            else:
                done += 1
                total_bytes += size
            if i % 5000 == 0 or i == len(tasks):
                rate = i / (time.monotonic() - started)
                print(f"  {i:,}/{len(tasks):,}  {rate:.0f} clip/s  "
                      f"{total_bytes/1e9:.1f} GB  실패 {failed}", flush=True)
    print(f"\n{done:,} 클립, {total_bytes/1e9:.1f} GB, 실패 {failed}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
