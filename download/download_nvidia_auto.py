#!/usr/bin/env python3
"""Download a camera+radar subset of nvidia/PhysicalAI-Autonomous-Vehicles.

The dataset is 133 TB in full, so this pulls a chunk-wise slice: only the
chunks where BOTH the selected camera view and the selected radar exist, and
within those chunks only the components listed in COMPONENTS below.

Every run appends to a manifest that records which chunks are eligible, which
are on disk, and which are still missing - so a later run can tell exactly
what has not been fetched yet.

The repo is gated. Accept the licence at
https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles
then either `hf auth login` or export HF_TOKEN.

    python download_nvidia_auto.py --chunks 10        # first 10 eligible chunks
    python download_nvidia_auto.py --chunks 50        # extend to 50
    python download_nvidia_auto.py --status           # report only, no download
"""

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

from huggingface_hub import HfApi, get_token, hf_hub_download
from huggingface_hub.utils import GatedRepoError, HfHubHTTPError

REPO_ID = "nvidia/PhysicalAI-Autonomous-Vehicles"
DEFAULT_DEST = "/NHNHOME/workspace/dataset/Auto_data/Nvidia_AUTO"

CAMERA = "camera_front_wide_120fov"

# The fleet is split: srr_0 sits on one vehicle generation (chunks from 0000),
# lrr_1 and mrr_2 on another (chunks from 0095), overlapping in only 122
# chunks. A chunk qualifies if it carries at least one of these, and whichever
# ones it has get downloaded.
RADARS = [
    "radar_front_center_srr_0",            # short range
    "radar_front_center_mrr_2",            # mid range
    "radar_front_center_imaging_lrr_1",    # long range, imaging
]

# (repo directory, file extension). One file per chunk, required in every chunk.
COMPONENTS = [
    (f"camera/{CAMERA}", "zip"),
    ("labels/obstacle.offline", "zip"),
    ("labels/egomotion.offline", "zip"),
    ("calibration/camera_intrinsics.offline", "parquet"),
    ("calibration/sensor_extrinsics.offline", "parquet"),
]

# Repo-wide files, not chunked. Fetched once.
GLOBAL_FILES = [
    "metadata/data_collection.parquet",
    "metadata/feature_presence.parquet",
    "clip_index.parquet",
    "features.csv",
    "README.md",
    # Human-verified Chain-of-Causation labels for a curated OOD subset -- the
    # only hand-checked ground truth in the release.
    "reasoning/ood_reasoning.parquet",
]

MANIFEST = "manifest.json"
LOGFILE = "download.log"


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Fetch a camera+radar slice of PhysicalAI-Autonomous-Vehicles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--dest", default=DEFAULT_DEST, help="destination directory")
    p.add_argument("--chunks", type=int, default=10,
                   help="how many eligible chunks to have on disk when done")
    p.add_argument("--token", default=None,
                   help="HF token (else HF_TOKEN env or a cached login)")
    p.add_argument("--status", action="store_true",
                   help="report what is present and missing, download nothing")
    p.add_argument("--retries", type=int, default=3,
                   help="download attempts per file")
    p.add_argument("--workers", type=int, default=6,
                   help="chunks downloaded concurrently")
    return p.parse_args(argv)


def utc_now():
    return datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


class Log:
    """Writes to stdout and appends to the on-disk log at the same time."""

    def __init__(self, path):
        self.path = path
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.fh = open(path, "a", encoding="utf-8")
        self.lock = threading.Lock()

    def __call__(self, msg=""):
        with self.lock:  # worker threads log concurrently
            print(msg, flush=True)
            self.fh.write(msg + "\n")
            self.fh.flush()

    def close(self):
        self.fh.close()


def chunk_id(path):
    m = re.search(r"chunk_(\d+)", path)
    return int(m.group(1)) if m else None


def present_chunks(files, directory):
    found = {chunk_id(f) for f in files if f.startswith(directory + "/")}
    found.discard(None)
    return found


def radar_availability(files):
    """Which chunks carry each radar."""
    return {r: present_chunks(files, f"radar/{r}") for r in RADARS}


def eligible_chunks(files, radars):
    """Chunks with every required component and at least one front radar."""
    required = [present_chunks(files, d) for d, _ext in COMPONENTS]
    any_radar = set().union(*radars.values())
    return sorted(set.intersection(*required) & any_radar)


def chunk_files(chunk, radars):
    """Repo paths making up one chunk: the required set plus the radars it has."""
    out = []
    for directory, ext in COMPONENTS:
        name = directory.split("/")[-1]
        out.append(f"{directory}/{name}.chunk_{chunk:04d}.{ext}")
    for radar, available in radars.items():
        if chunk in available:
            out.append(f"radar/{radar}/{radar}.chunk_{chunk:04d}.zip")
    return out


def file_sizes(api):
    """repo path -> byte size, for every component and radar directory.

    Sizes come from the repo listing, which is readable without the gate, and
    drive the throughput/ETA line during the run.
    """
    sizes = {}
    dirs = [d for d, _ext in COMPONENTS] + [f"radar/{r}" for r in RADARS]
    for directory in dirs:
        for item in api.list_repo_tree(REPO_ID, path_in_repo=directory,
                                       repo_type="dataset", recursive=False):
            if getattr(item, "size", None) is not None:
                sizes[item.path] = item.size
    return sizes


def load_manifest(path):
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    return {"repo": REPO_ID, "camera": CAMERA, "radars": RADARS,
            "runs": [], "files": {}, "chunks_complete": [], "chunk_radars": {}}


def save_manifest(path, manifest):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    os.replace(tmp, path)


def fetch(repo_path, dest, token, retries, log):
    """Download one file into dest, returning its size or None on failure."""
    for attempt in range(1, retries + 1):
        try:
            local = hf_hub_download(
                REPO_ID, repo_path, repo_type="dataset",
                local_dir=dest, token=token,
            )
            return os.path.getsize(local)
        except GatedRepoError:
            raise
        except (HfHubHTTPError, OSError) as exc:
            log(f"      attempt {attempt}/{retries} failed: "
                f"{type(exc).__name__}: {str(exc)[:120]}")
            if attempt < retries:
                time.sleep(5 * attempt)
    return None


def main(argv=None):
    args = parse_args(argv)
    dest = os.path.abspath(args.dest)
    os.makedirs(dest, exist_ok=True)

    log = Log(os.path.join(dest, LOGFILE))
    manifest_path = os.path.join(dest, MANIFEST)
    manifest = load_manifest(manifest_path)

    # get_token() picks up a cached `hf auth login`, which the env var misses.
    token = args.token or os.environ.get("HF_TOKEN") or get_token()

    log("=" * 78)
    log(f"run started {utc_now()}")
    log(f"  repo   {REPO_ID}")
    log(f"  camera {CAMERA}")
    for r in RADARS:
        log(f"  radar  {r}")
    log(f"  dest   {dest}")
    log(f"  target {args.chunks} chunks")
    log("=" * 78)

    api = HfApi(token=token)
    try:
        files = api.list_repo_files(REPO_ID, repo_type="dataset")
    except Exception as exc:
        log(f"!! cannot list repo: {type(exc).__name__}: {exc}")
        log.close()
        return 1

    radars = radar_availability(files)
    eligible = eligible_chunks(files, radars)
    done = set(manifest["chunks_complete"])
    log(f"eligible chunks (camera + at least one front radar): {len(eligible)}")
    log(f"  range {eligible[0]}..{eligible[-1]}" if eligible else "  none")
    for r, available in radars.items():
        log(f"  {r:34s} on {len(set(eligible) & available)} eligible chunks")
    log(f"already complete on disk: {len(done)}")

    wanted = [c for c in eligible if c in done][:args.chunks]
    if len(wanted) < args.chunks:
        for c in eligible:
            if c not in done and len(wanted) < args.chunks:
                wanted.append(c)
    wanted = sorted(wanted)
    todo = [c for c in wanted if c not in done]

    head = ", ".join(f"{c:04d}" for c in todo[:20])
    log(f"this run will fetch: {len(todo)} chunks -> {head}"
        f"{' ...' if len(todo) > 20 else ''}")
    remaining = [c for c in eligible if c not in done and c not in todo]
    log(f"still not downloaded after this run: {len(remaining)} chunks")

    if args.status:
        log("--status: nothing downloaded.")
        log.close()
        return 0

    if not token:
        log("!! No token. Accept the licence on the dataset page, then run")
        log("!!   hf auth login          (or export HF_TOKEN=hf_...)")
        log.close()
        return 1

    log("indexing file sizes...")
    planned_bytes = file_sizes(api)

    total_bytes = 0
    failures = []

    # Repo-wide files first: cheap, and the metadata is useful immediately.
    log("\n-- global files --")
    for repo_path in GLOBAL_FILES:
        if manifest["files"].get(repo_path, {}).get("ok"):
            log(f"  skip (have)  {repo_path}")
            continue
        log(f"  get          {repo_path}")
        try:
            size = fetch(repo_path, dest, token, args.retries, log)
        except GatedRepoError:
            log("!! Gated repo: this token has not been granted access.")
            log("!! Accept the licence at "
                f"https://huggingface.co/datasets/{REPO_ID}")
            save_manifest(manifest_path, manifest)
            log.close()
            return 1
        if size is None:
            failures.append(repo_path)
            manifest["files"][repo_path] = {"ok": False, "at": utc_now()}
        else:
            total_bytes += size
            manifest["files"][repo_path] = {"ok": True, "bytes": size,
                                            "at": utc_now()}
        save_manifest(manifest_path, manifest)

    log(f"\n-- chunks (workers={args.workers}) --")
    started_at = time.monotonic()
    lock = threading.Lock()
    finished = 0
    total_todo_bytes = sum(
        planned_bytes.get(p, 0)
        for c in todo for p in chunk_files(c, radars)
        if not manifest["files"].get(p, {}).get("ok")
    )

    def do_chunk(chunk):
        """Fetch one chunk's files. Runs on a worker thread."""
        results = []
        for repo_path in chunk_files(chunk, radars):
            if manifest["files"].get(repo_path, {}).get("ok"):
                continue
            results.append((repo_path, fetch(repo_path, dest, token,
                                             args.retries, log)))
        return chunk, results

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(do_chunk, c): c for c in todo}
        for future in as_completed(futures):
            chunk, results = future.result()
            with lock:
                finished += 1
                chunk_ok = True
                chunk_bytes = 0
                has = [r for r, avail in radars.items() if chunk in avail]
                manifest["chunk_radars"][str(chunk)] = has
                for repo_path, size in results:
                    if size is None:
                        failures.append(repo_path)
                        manifest["files"][repo_path] = {"ok": False,
                                                        "at": utc_now()}
                        chunk_ok = False
                    else:
                        chunk_bytes += size
                        total_bytes += size
                        manifest["files"][repo_path] = {"ok": True,
                                                        "bytes": size,
                                                        "at": utc_now()}
                if chunk_ok and chunk not in manifest["chunks_complete"]:
                    manifest["chunks_complete"].append(chunk)
                    manifest["chunks_complete"].sort()

                elapsed = time.monotonic() - started_at
                rate = total_bytes / elapsed if elapsed > 0 else 0
                left = total_todo_bytes - total_bytes
                eta = left / rate / 3600 if rate > 0 else float("nan")
                log(f"[{finished}/{len(todo)}] chunk_{chunk:04d} "
                    f"{chunk_bytes/1e9:5.2f} GB "
                    f"{'ok' if chunk_ok else 'INCOMPLETE'}  "
                    f"| {', '.join(r.replace('radar_front_center_', '') for r in has)} "
                    f"| {total_bytes/1e9:7.1f}/{total_todo_bytes/1e9:.0f} GB "
                    f"{rate/1e6:5.0f} MB/s  ETA {eta:.1f}h")
                save_manifest(manifest_path, manifest)

    done = set(manifest["chunks_complete"])
    missing = [c for c in eligible if c not in done]
    manifest["runs"].append({
        "at": utc_now(),
        "requested_chunks": args.chunks,
        "fetched_chunks": todo,
        "bytes": total_bytes,
        "failures": failures,
    })
    manifest["eligible_total"] = len(eligible)
    manifest["chunks_missing_count"] = len(missing)
    save_manifest(manifest_path, manifest)

    log("\n" + "=" * 78)
    log(f"run finished {utc_now()}")
    log(f"  downloaded this run : {total_bytes/1e9:.2f} GB")
    log(f"  chunks complete     : {len(done)} / {len(eligible)} eligible")
    log(f"  chunks NOT fetched  : {len(missing)}")
    if missing:
        preview = ", ".join(f"{c:04d}" for c in missing[:20])
        log(f"    next up: {preview}{' ...' if len(missing) > 20 else ''}")
    if failures:
        log(f"  FAILURES ({len(failures)}) - re-run to retry:")
        for f in failures:
            log(f"    {f}")
    log(f"  manifest: {manifest_path}")
    log("=" * 78)
    log.close()
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
