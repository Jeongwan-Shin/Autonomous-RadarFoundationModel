#!/usr/bin/env python3
"""Fetch the non-offline sensor_extrinsics, which is the only source of radar poses.

`calibration/sensor_extrinsics.offline/` carries 7 cameras and the lidar but no
radar at all, so nothing can lift a radar return into the rig frame from what
was originally downloaded. The non-offline `calibration/sensor_extrinsics/`
variant adds radar entries, including `radar_front_center_imaging_lrr_1` and
`radar_front_center_mrr_2`. Applying it puts box centres a median 1.31 m from
their nearest radar return; without it the best guessed convention lands at
10.6 m, which is useless for associating labels with returns.

Downloads go through the `resolve/main/...` file endpoint rather than
`hf_hub_download`. The hub client makes an `/api/...` call per file (revision
lookup, then an xet read token), and 1,838 of those exhaust the API quota of
10,000 requests per 5 minutes: a first attempt using the client managed one file
in eight minutes. The resolve endpoint is served as a plain file transfer and is
not subject to that quota.

    python -m datatools.fetch_radar_extrinsics
    python -m datatools.fetch_radar_extrinsics --verify
"""

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd
import requests

from . import paths

REPO_ID = "nvidia/PhysicalAI-Autonomous-Vehicles"
FEATURE = "calibration/sensor_extrinsics"
RESOLVE = f"https://huggingface.co/datasets/{REPO_ID}/resolve/main/{FEATURE}"

_lock = threading.Lock()


def log(msg):
    with _lock:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def read_token():
    candidates = [os.environ.get("HF_TOKEN")]
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        candidates.append(os.path.join(hf_home, "token"))
    candidates.append(os.path.expanduser("~/.cache/huggingface/token"))
    for candidate in candidates:
        if not candidate:
            continue
        if candidate.startswith("hf_"):
            return candidate
        if os.path.exists(candidate):
            token = open(candidate).read().strip()
            if token:
                return token
    raise SystemExit("No HF token found. Set HF_TOKEN or run `hf auth login`.")


def local_path(nvidia_root, chunk):
    return os.path.join(nvidia_root, FEATURE,
                        f"sensor_extrinsics.chunk_{chunk:04d}.parquet")


def fetch_one(chunk, nvidia_root, session, tries, timeout):
    path = local_path(nvidia_root, chunk)
    if os.path.exists(path) and os.path.getsize(path) > 1000:
        return chunk, "cached"

    url = f"{RESOLVE}/sensor_extrinsics.chunk_{chunk:04d}.parquet"
    tmp = f"{path}.part"
    for attempt in range(1, tries + 1):
        try:
            response = session.get(url, timeout=timeout)
            if response.status_code == 200 and len(response.content) > 1000:
                with open(tmp, "wb") as fh:
                    fh.write(response.content)
                os.replace(tmp, path)
                return chunk, "ok"
            if response.status_code == 429 and attempt < tries:
                time.sleep(float(response.headers.get("Retry-After", 10)))
                continue
            if attempt == tries:
                return chunk, f"failed: HTTP {response.status_code}"
        except requests.RequestException as exc:
            if attempt == tries:
                return chunk, f"failed: {type(exc).__name__}"
            time.sleep(2 * attempt)
    return chunk, "failed"


def verify(nvidia_root, chunks, sample=25):
    present = [c for c in chunks if os.path.exists(local_path(nvidia_root, c))]
    if not present:
        log("verify: nothing downloaded")
        return
    step = max(1, len(present) // sample)
    picked = present[::step][:sample]
    radar_clips = total_clips = 0
    sensors = set()
    for chunk in picked:
        frame = pd.read_parquet(local_path(nvidia_root, chunk))
        names = frame.index.get_level_values("sensor_name")
        sensors.update(n for n in names.unique() if "radar" in n)
        has_radar = (frame.reset_index()
                     .groupby("clip_id")["sensor_name"]
                     .apply(lambda s: any("radar" in x for x in s)))
        radar_clips += int(has_radar.sum())
        total_clips += int(len(has_radar))
    log(f"verify: {len(picked)} chunks sampled, {radar_clips}/{total_clips} clips "
        f"carry radar extrinsics ({radar_clips/max(total_clips,1)*100:.1f}%)")
    for name in sorted(sensors):
        log(f"  {name}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--nvidia-root", default=paths.NVIDIA_ROOT)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tries", type=int, default=4)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--verify", action="store_true")
    args = ap.parse_args(argv)

    manifest = json.load(open(os.path.join(args.nvidia_root, "manifest.json")))
    chunks = sorted(manifest["chunks_complete"])
    os.makedirs(os.path.join(args.nvidia_root, FEATURE), exist_ok=True)

    todo = [c for c in chunks
            if not (os.path.exists(local_path(args.nvidia_root, c))
                    and os.path.getsize(local_path(args.nvidia_root, c)) > 1000)]
    log(f"{len(chunks)} chunks total, {len(todo)} to fetch")

    if todo:
        session = requests.Session()
        session.headers["Authorization"] = f"Bearer {read_token()}"
        counts = {"ok": 0, "cached": 0, "failed": 0}
        failures = []
        started = time.monotonic()
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(fetch_one, c, args.nvidia_root, session,
                                   args.tries, args.timeout) for c in todo]
            for i, future in enumerate(as_completed(futures), 1):
                chunk, status = future.result()
                key = "failed" if status.startswith("failed") else status
                counts[key] += 1
                if key == "failed":
                    failures.append((chunk, status))
                if i % 200 == 0 or i == len(todo):
                    rate = i / (time.monotonic() - started)
                    log(f"  {i}/{len(todo)}  ok={counts['ok']} "
                        f"failed={counts['failed']}  {rate:.1f} file/s")
        if failures:
            log(f"{len(failures)} failed; re-run to retry")
            for chunk, status in failures[:10]:
                log(f"  chunk_{chunk:04d}: {status}")

    if args.verify:
        verify(args.nvidia_root, chunks)
    log("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
