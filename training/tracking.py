#!/usr/bin/env python3
"""Weights & Biases, wrapped so that a run never dies for want of a dashboard.

A training run here is thirty hours on five B200s. If the logger raises -- no
credentials, network gone, project renamed -- the run has to keep going and say
so once, not crash at hour twenty. Every call here is a no-op on failure and
reports the reason a single time.

Rank 0 only. Five processes writing the same scalars would triple-count the
throughput and race on the run id.

    wandb login                       # once, interactively
    RAVL_WANDB=off  ...               # opt out for a throwaway run
    WANDB_MODE=offline ...            # record locally, sync later
"""

import os
import time

ENTITY = os.environ.get("RAVL_WANDB_ENTITY", "DGIST_IRS")
PROJECT = os.environ.get("RAVL_WANDB_PROJECT", "RadarAutonomous_FM")

_run = None
_warned = False


def _off():
    return os.environ.get("RAVL_WANDB", "").lower() in ("off", "0", "false", "no")


def _warn(msg):
    global _warned
    if not _warned:
        print(f"[wandb] {msg} -- 기록 없이 계속합니다", flush=True)
        _warned = True


def start(name, config, rank=0, job="train", tags=()):
    """Open a run. Returns the run, or None when tracking is unavailable."""
    global _run
    if rank != 0 or _off():
        return None
    try:
        import wandb
    except ImportError:
        _warn("wandb 가 설치되어 있지 않습니다")
        return None
    try:
        _run = wandb.init(entity=ENTITY, project=PROJECT, name=name,
                          job_type=job, tags=list(tags), config=dict(config),
                          settings=wandb.Settings(init_timeout=120))
        print(f"[wandb] {_run.url}", flush=True)
        return _run
    except Exception as exc:                       # 인증 없음, 네트워크 없음 등
        _warn(f"시작 실패: {type(exc).__name__}: {exc}")
        _run = None
        return None


def log(metrics, step=None):
    if _run is None:
        return
    try:
        _run.log({k: v for k, v in metrics.items() if v is not None}, step=step)
    except Exception as exc:
        _warn(f"기록 실패: {type(exc).__name__}: {exc}")


def summary(metrics):
    """Values that describe the whole run rather than one step."""
    if _run is None:
        return
    try:
        for k, v in metrics.items():
            if v is not None:
                _run.summary[k] = v
    except Exception as exc:
        _warn(f"요약 실패: {type(exc).__name__}: {exc}")


def table(name, columns, rows):
    """A table -- per-task scores, or a sample of generations to read."""
    if _run is None or not rows:
        return
    try:
        import wandb
        _run.log({name: wandb.Table(columns=list(columns),
                                    data=[list(r) for r in rows])})
    except Exception as exc:
        _warn(f"표 기록 실패: {type(exc).__name__}: {exc}")


def finish():
    global _run
    if _run is None:
        return
    try:
        _run.finish()
    except Exception:
        pass
    _run = None


def run_name(prefix, args):
    """A name that says what the run was, not when it happened.

    Two runs that differ only in seed should sort together; two that differ in
    batch or objective should be distinguishable at a glance in the run list.
    """
    bits = [prefix, f"{getattr(args, 'model', '?')}",
            f"{getattr(args, 'stage', '?')}",
            f"b{getattr(args, 'micro_batch', 0) * getattr(args, 'accum', 0)}",
            f"lr{getattr(args, 'lr', 0):g}"]
    if getattr(args, "digit_weight", 0):
        bits.append(f"num{args.digit_weight:g}")
    if getattr(args, "resume", None):
        bits.append("resume")
    bits.append(f"s{getattr(args, 'seed', 0)}")
    bits.append(time.strftime("%m%d-%H%M"))
    return "-".join(str(b) for b in bits)
