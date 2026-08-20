#!/usr/bin/env python3
"""학습이 저장할 때마다 그 체크포인트를 재고, 추세를 한 줄씩 쌓는다.

한 번의 점수는 좋아지고 있는지 말해 주지 않는다. 200 스텝마다 같은 항목을 같은
방식으로 재서 나란히 놓아야, 방위각이 실제로 벌어지고 있는지 아니면 손실만
내려가고 있는지가 보인다.

학습 프로세스 안이 아니라 옆에서 돈다. 학습 루프에 생성을 끼워 넣으면 거기서
난 버그가 학습 전체를 죽이는데, 이쪽은 죽어도 학습이 계속된다. 대신 GPU 를
나눠 써야 하므로 학습의 micro-batch 를 낮춰 자리를 비워 두어야 한다.

    python -m training.watch_eval --checkpoint <학습 --out 경로> --device 4

무엇을 보는가:

  det_objects_azdeg   F1, 그리고 생성된 |방위각| 의 평균. 이 모델의 알려진
                      실패가 방위각을 정면으로 몰아넣는 것이라, 정답의 평균과
                      나란히 찍는다. 둘이 가까워지는 것이 이 학습이 답해야 할
                      질문이다.
  det_objects_3dbbox  F1 과 크기·요각 오차
  plan_ego_xy         변위 오차와, 예측 경로가 라벨된 물체와 겹치는 비율
"""

import argparse
import json
import math
import os
import re
import shutil
import sys
import time

import numpy as np

WATCHED = ("det_objects_azdeg", "det_objects_3dbbox", "plan_ego_xy")
AZ = re.compile(r"az\s*([+-]?\d+)\s*deg")
# 단위가 붙은 뒤로 좌표는 `+2.0m` 로 나온다. `m?` 가 없으면 이 정규식은
# 아무것도 못 읽고, L2 평균은 나오는데 1s/2s/3s 칸만 비는 형태로 조용히
# 틀린다 -- 채점기 쪽 WAYPOINT 는 이미 이 접미사를 허용하고 있었다.
XY = re.compile(r"\+(\d)s\s*\(\s*([+-]?[\d.]+)m?\s*,\s*([+-]?[\d.]+)m?\s*\)")

# The ego footprint the collision test sweeps along the predicted path. The
# release does not ship vehicle dimensions per clip, so this is the usual
# passenger-car box; the number is a rate under a fixed assumption, not a
# certified safety figure, and it is comparable across checkpoints because the
# assumption never changes.
EGO_LENGTH, EGO_WIDTH = 4.6, 1.9


def log(m):
    print(f"[{time.strftime('%H:%M:%S')}] {m}", flush=True)


def saved_step(path):
    try:
        h = json.load(open(os.path.join(path, "latest", "history.json")))
        return int(h[-1]["step"]) if h else None
    except Exception:
        return None


def mean_abs_azimuth(text):
    v = [abs(float(x)) for x in AZ.findall(text or "")]
    return float(np.mean(v)) if v else None


def _traj_wp(text):
    """궤적 토큰이면 되돌려 지점으로. 텍스트면 빈 값."""
    if "<|traj_" not in (text or ""):
        return {}
    from training.traj_tokens import to_waypoints
    return {int(h): xy for h, xy in to_waypoints(text).items()}


def waypoints(text):
    return {int(h): (float(x), float(y)) for h, x, y in XY.findall(text or "")}


def collides(path, obstacles):
    """예측 경로 위 자차 사각형이 라벨된 물체 상자와 겹치는가.

    축 정렬 근사다. 자차의 진행 방향으로 회전시키지 않으므로 회전 구간에서는
    조금 넉넉하게 잡힌다 -- 체크포인트끼리 비교하는 데는 같은 근사가 양쪽에
    걸리므로 문제가 되지 않지만, 절대값으로 인용하면 안 된다.
    """
    hit = 0
    for h, (x, y) in path.items():
        for ox, oy, ol, ow in obstacles.get(h, ()):
            if (abs(x - ox) < (EGO_LENGTH + ol) / 2
                    and abs(y - oy) < (EGO_WIDTH + ow) / 2):
                hit += 1
                break
    return hit



# 학습 카메라의 초점거리. 120 도 화각을 폭 1920 으로 본 것이므로
# f = (1920/2) / tan(60 deg) = 554 px. 모델은 이 축척으로 "겉보기 크기 -> 거리",
# "픽셀 위치 -> 각도" 를 배웠다.
TRAIN_FOCAL_PX = (1920 / 2) / math.tan(math.radians(60.0))

OBJ_XYZ = re.compile(r"\(([+-]?[\d.]+)m, ([+-]?[\d.]+)m, ([+-]?[\d.]+)m\)")
OBJ_POLAR = re.compile(r"(\d+)m az ([+-]\d+)deg")


def refocus(text, focal_px):
    """다른 초점거리의 카메라로 본 답을, 학습 카메라 축척으로 되돌린다.

    라벨을 쓰지 않는다 -- 카메라 사양에서만 나온다. 물체의 겉보기 크기는 f/거리
    에 비례하므로, 학습보다 f 가 큰 카메라(좁은 화각)에서는 같은 물체가 크게
    보이고 모델은 더 가깝다고 답한다. 각도도 같다: 픽셀 오프셋 u 를 학습의 f 로
    나눠 각을 만들므로 tan(답) = tan(참) x f_nu/f_학습 이 된다.

    측정으로 확인한 값 -- nuScenes 에서 예측/정답 거리비의 이론값 0.608 대
    실측 0.75(가까이)~0.99(멀리), tan 비 이론값 0.608 대 실측 0.545. NVIDIA 는
    둘 다 1.00 이라 이 보정이 원래 맞던 것을 건드리지 않는다.
    """
    k = TRAIN_FOCAL_PX / focal_px          # 0.608
    if not text:
        return text

    def xyz(m):
        x, y, z = (float(v) for v in m.groups())
        r, a = math.hypot(x, y), math.atan2(y, x)
        r, a = r / k, math.atan(math.tan(a) * k)
        return f"({r * math.cos(a):+.1f}m, {r * math.sin(a):+.1f}m, {z:+.1f}m)"

    def polar(m):
        r, a = float(m.group(1)), math.radians(float(m.group(2)))
        return (f"{r / k:.0f}m az "
                f"{math.degrees(math.atan(math.tan(a) * k)):+.0f}deg")

    return OBJ_POLAR.sub(polar, OBJ_XYZ.sub(xyz, text))

def bundle_rows(root, task, limit):
    """번들에서 `limit` 건. 클립을 가로질러 고르게 뽑는다.

    앞에서부터 자르면 안 된다. 파일이 클립 순서로 쌓여 있어 클립당 20 건이
    붙어 있고, 40 건을 앞에서 자르면 장면 두 개만 스무 번씩 보게 된다.
    실제로 그렇게 재고 있었고, 생성이 40 건 내내 거의 같았다 -- 모델이 굳은
    것이 아니라 입력이 같았던 것인데, 점수만 보면 둘이 구별되지 않는다.
    """
    path = os.path.join(root, "by_task", f"{task}.jsonl")
    if not os.path.exists(path):
        return []
    rows = [json.loads(l) for l in open(path)]
    by_clip = {}
    for r in rows:
        by_clip.setdefault(r["clip_id"], []).append(r)
    out, k = [], 0
    while len(out) < min(limit, len(rows)):
        added = False
        for clip in sorted(by_clip):
            if k < len(by_clip[clip]):
                out.append(by_clip[clip][k])
                added = True
                if len(out) >= limit:
                    break
        if not added:
            break
        k += 1
    return out


def encoder_channels():
    from training.radar_data import CHANNELS
    return len(CHANNELS)


def run_bundle(root, task, rows, loaded, budget):
    """번들 형식(nuScenes)에 대해 생성한다.

    학습 데이터와 같은 모델·같은 주입 방식을 쓴다. 다른 것은 어디서 프레임과
    레이더를 읽느냐뿐이다 -- 그래서 여기서 나오는 점수와 NVIDIA 쪽 점수는
    같은 체크포인트의 같은 능력을 두 센서 rig 에서 잰 것이다.
    """
    import torch
    from PIL import Image
    from transformers.video_utils import VideoMetadata
    from training.train_vlm import RadarInjector
    tokenizer, processor, llm, encoder, connector, pad_id, trained, device = loaded
    from training.instruct_data import SYSTEM

    injector = RadarInjector(llm.get_input_embeddings(), pad_id)
    out = []
    for rec in rows:
        d = os.path.join(root, "clips", rec["clip_id"])
        frames = [Image.open(os.path.join(d, "frames", f"f{i:02d}.jpg")).convert("RGB")
                  for i in rec["frames"]]
        z = np.load(os.path.join(d, "radar", f"{rec['radar']}.npz"))
        nf, mp, nc = [int(v) for v in z["shape"]]
        # 번들은 자기가 만들어질 때의 채널 배치로 굳어 있다. 인코더가 그 뒤에
        # sin_az/cos_az 를 얻으면 여기서 20480x54 대 56x384 라는 행렬 곱 오류로
        # 터지는데, 그 메시지만 보고는 원인이 데이터 변환기라는 것을 알 수 없다.
        if nc != encoder_channels():
            raise RuntimeError(
                f"번들 채널 {nc} != 인코더 {encoder_channels()} -- "
                f"{root} 를 지금 채널로 다시 만들어야 한다")
        kept, scan = z["points"].astype(np.float32), z["scan"]
        # 번들은 만들어질 때의 프레임 수로 굳어 있다. 인코더가 그보다 적은
        # 슬롯을 쓰면 고르게 솎아 맞춘다 -- 번들을 다시 만들지 않고도 스캔
        # 수가 바뀐 인코더를 시험할 수 있다.
        want = encoder.n_frames if hasattr(encoder, "n_frames") else nf
        if want != nf:
            take = np.linspace(0, nf - 1, want).round().astype(int)
            remap = {int(o): i for i, o in enumerate(take)}
            sel = np.isin(scan, take)
            kept, scan = kept[sel], np.array([remap[int(s)] for s in scan[sel]])
            nf = want
        pts = np.zeros((nf, mp, nc), dtype=np.float32)
        msk = np.zeros((nf, mp), dtype=bool)
        for f in range(nf):
            sel = kept[scan == f]
            n = min(len(sel), mp)
            if n:
                pts[f, :n], msk[f, :n] = sel[:n], True
        messages = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
                    {"role": "user", "content": [{"type": "video"},
                                                 {"type": "text", "text": rec["user"]}]}]
        text = processor.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
        meta = [VideoMetadata(total_num_frames=len(frames), fps=1.0,
                              width=frames[0].width, height=frames[0].height,
                              duration=float(len(frames)), video_backend="manual",
                              frames_indices=list(range(len(frames))))]
        batch = processor(text=[text], videos=[frames], video_metadata=meta,
                          return_tensors="pt").to(device)
        with torch.no_grad():
            r = encoder(torch.from_numpy(pts).unsqueeze(0).to(device, torch.bfloat16),
                        torch.from_numpy(msk).unsqueeze(0).to(device),
                        torch.tensor([rec["sensor"]], device=device))
            injector.pending = connector(r["tokens"])
            cut = batch["input_ids"].shape[1]
            got = llm.generate(**batch, max_new_tokens=budget, do_sample=False,
                               pad_token_id=tokenizer.pad_token_id
                               or tokenizer.eos_token_id)
        out.append({"task": task, "clip_id": rec["clip_id"],
                    "generated": tokenizer.decode(got[0, cut:],
                                                  skip_special_tokens=True).strip(),
                    "reference": rec["target"]})
    injector.remove()
    return out


def detection_scores(gens, prefix):
    """mAP 와, 매칭 성공 여부와 무관한 위치 오차.

    F1 과 거리 MAE 는 둘 다 짝지어진 물체만 센다. 짝이 12.5% 뿐인 구간에서는
    가장 잘 맞은 여덟 중 하나만 보고 있는 셈이라, 실제보다 좋게 보인다.
    Chamfer 는 정답마다 가장 가까운 생성까지의 거리라 짝이 없어도 값이 나온다.
    """
    from training.task_scorers import detection_map, parse_objects
    items, chamfer = [], []
    for g in gens:
        p = parse_objects(g.get("generated"))
        t = parse_objects(g.get("reference"))
        items.append((p, t, g.get("confidence") or []))
        if p and t:
            for o in t:
                chamfer.append(min(
                    float(np.hypot(o.get("x", 0) - q.get("x", 0),
                                   o.get("y", 0) - q.get("y", 0)))
                    if o.get("x") is not None and q.get("x") is not None
                    else float(abs((o.get("rng") or 0) - (q.get("rng") or 0)))
                    for q in p))
    m = detection_map(items)
    out = {f"{prefix}_map": m["mAP"]}
    if chamfer:
        c = np.array(chamfer)
        out[f"{prefix}_chamfer"] = float(np.median(c))
        out[f"{prefix}_within2m"] = float(np.mean(c <= 2.0))
    return out


def worker_main(argv):
    """한 GPU 에서 자기 몫만 생성하고 JSONL 로 뱉는다.

    점수를 내지 않고 생성만 남기는 이유는, 채점기가 지금까지 네 번 틀렸고 그때
    마다 추론을 다시 돌려야 했기 때문이다. 텍스트는 GPU 가 실제로 내놓은 것이고
    지표는 그것에 대한 의견이다 -- 의견은 나중에 고칠 수 있어야 한다.
    """
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--shards", type=int, default=1)
    ap.add_argument("--items", type=int, default=60)
    ap.add_argument("--nus-items", type=int, default=40)
    ap.add_argument("--nuscenes", default=None)
    ap.add_argument("--model", default="8B")
    ap.add_argument("--split", default="test")
    ap.add_argument("--plan-samples", type=int, default=6)
    a = ap.parse_args(argv)
    from training.eval_all_tasks import MAX_NEW, load_model, run_task

    class A:
        pass
    cfg = A()
    cfg.checkpoint, cfg.model, cfg.split = a.snapshot, a.model, a.split
    cfg.items, cfg.workers, cfg.show = a.items, 2, 0
    cfg.max_new_floor, cfg.all_profiles, cfg.seed = 0, True, 0
    cfg.radar_dropout, cfg.out = 0.0, None
    cfg.shard, cfg.shards = a.shard, a.shards
    cfg.plan_samples, cfg.plan_temperature = a.plan_samples, 1.0
    loaded = load_model(cfg)

    # 단계가 끝날 때마다 바로 쓴다. 처음에는 끝에서 한 번에 썼는데, nuScenes
    # 단계가 죽자 그 앞에서 이미 생성해 둔 NVIDIA 결과까지 같이 사라졌다 --
    # 한 시간짜리 추론을, 그 추론과 상관없는 곳에서 난 오류로 버린 셈이다.
    fh = open(a.out, "w")

    def keep(rows):
        for x in rows or []:
            fh.write(json.dumps(x, ensure_ascii=False) + "\n")
        fh.flush()

    for task in WATCHED:
        try:
            r = run_task(task, cfg, loaded)
        except Exception as exc:
            print(f"{task}: {type(exc).__name__} {exc}", file=sys.stderr)
            continue
        if r:
            keep(r.get("generations"))
    if a.nuscenes and os.path.isdir(a.nuscenes):
        for task in WATCHED:
            got = bundle_rows(a.nuscenes, task, a.nus_items)[a.shard::a.shards]
            if not got:
                continue
            try:
                keep({**g, "dataset": "nuscenes", "mode": "full"}
                     for g in run_bundle(a.nuscenes, task, got, loaded,
                                         MAX_NEW.get(task, 48)))
            except Exception as exc:
                print(f"nuscenes/{task}: {type(exc).__name__} {exc}",
                      file=sys.stderr)
    fh.close()
    return 0


def keep_best(args, snap, step, row):
    """성적이 좋은 스냅샷을 K 개까지 남긴다.

    학습은 200 스텝마다 `latest` 를 덮어쓰므로, 남기지 않으면 중간 지점의
    가중치는 다음 저장 때 사라진다 -- 실제로 이 실행의 최고점(step 5,400)이
    그렇게 없어졌고, 남은 것은 마지막 하나뿐이었다.

    K 개를 남기는 이유는 60건 표본의 흔들림 때문이다. 최근 여덟 점의 3D F1
    표준편차가 0.014 라, 한 점의 최고값은 실력이 아니라 그날의 표본일 수
    있다. 후보를 몇 개 남겨 두었다가 끝에서 항목을 늘려 다시 재는 편이,
    잡음의 최댓값을 골라 놓고 그것을 최고 성능이라 부르는 것보다 낫다.
    """
    metric = args.keep_metric
    value = row.get(metric)
    if not args.keep or value is None:
        shutil.rmtree(snap, ignore_errors=True)
        return
    root = args.checkpoint
    kept = []
    for name in os.listdir(root):
        m = re.match(r"best_(\d+)_([\d.]+)$", name)
        if m:
            kept.append((float(m.group(2)), int(m.group(1)), name))
    if len(kept) >= args.keep and value <= min(kept)[0]:
        shutil.rmtree(snap, ignore_errors=True)
        return
    target = os.path.join(root, f"best_{step}_{value:.3f}")
    shutil.rmtree(target, ignore_errors=True)
    os.rename(snap, target)
    log(f"  스냅샷 보관 {os.path.basename(target)} ({metric} {value:.3f})")
    kept.append((value, step, os.path.basename(target)))
    for _, _, name in sorted(kept, reverse=True)[args.keep:]:
        shutil.rmtree(os.path.join(root, name), ignore_errors=True)
        log(f"  스냅샷 정리 {name}")


def freest_device(devices):
    """지금 가장 여유가 많은 GPU. 학습이 랭크마다 다른 양을 잡고 있다."""
    import subprocess
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free",
                            "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=30)
        free = {}
        for line in q.stdout.strip().splitlines():
            idx, mb = [s.strip() for s in line.split(",")]
            free[idx] = int(mb)
        got = [d for d in devices if d in free]
        if got:
            return max(got, key=lambda d: free[d])
    except Exception:
        pass
    return devices[0]


def evaluate_sharded(snapshot, args):
    """GPU 마다 프로세스 하나씩 띄워 생성만 모으고, 채점은 여기서 한 번에.

    배치는 1 그대로다. 배치를 키우려면 좌측 패딩과 샘플별 자르기와 신뢰도
    분리를 다시 써야 하는데, 이 평가기는 이미 네 번 틀렸던 코드다. 쪼개는
    쪽은 아이템별 논리를 건드리지 않는다.
    """
    import subprocess, tempfile
    devices = [d.strip() for d in args.devices.split(",") if d.strip()]
    tmp = tempfile.mkdtemp(prefix="shardeval_")
    procs = []
    for i, dev in enumerate(devices):
        out = os.path.join(tmp, f"{i}.jsonl")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=dev,
                   TOKENIZERS_PARALLELISM="false",
                   # 학습이 GPU 하나에 159 GiB 를 잡아 둔 옆자리라 남는 19 GiB
                   # 안에서 18.3 GiB 를 써야 한다. 조각이 나면 2.3 GiB 하나를
                   # 못 잡고 죽는다 -- 실제로 한 워커가 그렇게 죽었다.
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        cmd = [sys.executable, "-m", "training.watch_eval", "--worker",
               "--snapshot", snapshot, "--out", out,
               "--shard", str(i), "--shards", str(len(devices)),
               "--items", str(args.items), "--nus-items", str(args.nus_items),
               "--model", args.model, "--split", args.split,
               "--plan-samples", str(args.plan_samples)]
        if args.nuscenes:
            cmd += ["--nuscenes", args.nuscenes]
        procs.append((subprocess.Popen(cmd, env=env,
                                       stdout=subprocess.DEVNULL,
                                       stderr=subprocess.PIPE), out))
    gens, retry = [], []
    for i, (proc, out) in enumerate(procs):
        _, err = proc.communicate()
        if proc.returncode != 0:
            tail = err.decode(errors="replace").strip().splitlines()[-1:]
            log(f"  워커 {i} 실패 ({proc.returncode}): {tail}")
            retry.append(i)
        # 실패했더라도 그때까지 쓴 줄은 쓸모가 있다.
        if os.path.exists(out):
            gens.extend(json.loads(l) for l in open(out))

    # 학습이 GPU 마다 다른 양을 잡고 있어서, 어떤 자리에서는 평가가 들어가고
    # 어떤 자리에서는 2 GiB 가 모자란다. 조각을 잃으면 그만큼 항목이 비므로,
    # 실패한 조각만 지금 제일 빈 GPU 에서 다시 돌린다 -- 1 파의 워커들이
    # 이미 나갔으니 그 자리는 비어 있다.
    for i in retry:
        dev = freest_device(devices)
        out = os.path.join(tmp, f"{i}_retry.jsonl")
        log(f"  조각 {i} 를 GPU {dev} 에서 다시")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=dev,
                   TOKENIZERS_PARALLELISM="false",
                   PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
        cmd = [sys.executable, "-m", "training.watch_eval", "--worker",
               "--snapshot", snapshot, "--out", out,
               "--shard", str(i), "--shards", str(len(devices)),
               "--items", str(args.items), "--nus-items", str(args.nus_items),
               "--model", args.model, "--split", args.split,
               "--plan-samples", str(args.plan_samples)]
        if args.nuscenes:
            cmd += ["--nuscenes", args.nuscenes]
        r = subprocess.run(cmd, env=env, stdout=subprocess.DEVNULL,
                           stderr=subprocess.PIPE)
        if os.path.exists(out):
            gens.extend(json.loads(l) for l in open(out))
        if r.returncode != 0:
            log(f"    다시도 실패: "
                f"{r.stderr.decode(errors='replace').strip().splitlines()[-1:]}")

    shutil.rmtree(tmp, ignore_errors=True)
    log(f"  생성 {len(gens):,}건 모음")
    return (score_generations(gens, bundle_focal(args.nuscenes))
            if gens else {}), gens


def bundle_focal(root):
    """번들이 기록해 둔 카메라 초점거리(px). 없으면 보정을 건너뛴다."""
    try:
        with open(os.path.join(root or "", "manifest.json")) as fh:
            return float(json.load(fh)["camera_focal_px"])
    except Exception:
        return None


def score_generations(gens, focal_px=None):
    """모아 온 생성에서 표의 모든 열을 계산한다.

    nuScenes 는 두 번 잰다. `nus_` 는 손대지 않은 값이고 `nusf_` 는 카메라
    초점거리로 축척만 되돌린 값이다. 둘을 나란히 두어야 무엇이 센서 차이이고
    무엇이 모델의 한계인지 갈린다 -- 하나만 남기면 그 구별이 사라진다.
    """
    from training.task_scorers import scorer_for, summarise
    row = {}
    passes = [(None, ""), ("nuscenes", "nus_")]
    if focal_px:
        passes.append(("nuscenes", "nusf_"))
    for ds, pre in passes:
        for task in WATCHED:
            got = [g for g in gens if g.get("task") == task
                   and g.get("dataset") == ds and g.get("mode", "full") == "full"]
            if not got:
                continue
            if pre == "nusf_":
                got = [{**g, "generated": refocus(g["generated"], focal_px)}
                       for g in got]
            fn = scorer_for(task)
            recs = []
            for g in got:
                try:
                    recs.append(fn(g["generated"], g["reference"], ""))
                except TypeError:
                    recs.append(fn(g["generated"], g["reference"]))
            s = summarise(task, recs)
            if task == "det_objects_azdeg":
                row[f"{pre}det_f1"] = s.get("f1")
                # nuScenes 쪽 열 이름은 예전 행과 맞춰 둔다. 한 파일 안에서
                # 같은 지표가 두 이름으로 갈리면 추세가 끊겨 보인다.
                row[f"{pre}range_mae" if pre else "det_range_mae"] = \
                    s.get("range_mae")
                row[f"{pre}det_az_mae"] = s.get("az_mae")
                a = [mean_abs_azimuth(g["generated"]) for g in got]
                b = [mean_abs_azimuth(g["reference"]) for g in got]
                row[f"{pre}az_gen"] = float(np.mean([v for v in a if v] or [0]))
                row[f"{pre}az_truth"] = float(np.mean([v for v in b if v] or [0]))
                row.update({f"{pre}{k}": v for k, v in
                            detection_scores(got, "det").items()})
                if ds is None:
                    other = [g for g in gens if g.get("task") == task
                             and g.get("dataset") is None
                             and g.get("mode") == "shuffled"]
                    if other:
                        sh = summarise(task, [fn(g["generated"], g["reference"])
                                              for g in other])
                        row["det_f1_shuffled"] = sh.get("f1")
            elif task == "det_objects_3dbbox":
                row[f"{pre}box_f1"] = s.get("f1")
                row[f"{pre}box_size_mae"] = s.get("size_mae")
                row[f"{pre}box_yaw_mae"] = s.get("yaw_mae")
                row[f"{pre}vel_mae"] = s.get("vel_mae")
                row[f"{pre}attr_acc"] = s.get("attr_acc")
                row.update({f"{pre}{k}": v for k, v in
                            detection_scores(got, "box").items()})
            elif task == "plan_ego_xy":
                row[f"{pre}l2"] = s.get("displacement_mae_m")
                # minADE_k -- 여섯 갈래 중 정답에 가장 가까운 것. 탐욕 L2 와
                # 나란히 둔다. 둘의 간격이 "옳은 미래를 후보에는 담고 있으나
                # 첫 번째로 고르지 못한다" 는 상태를 드러낸다.
                best, cover = [], 0
                for g in got:
                    cand = g.get("samples") or []
                    truth = waypoints(g["reference"]) or _traj_wp(g["reference"])
                    if not cand or not truth:
                        continue
                    errs = []
                    for c in cand:
                        p_ = waypoints(c) or _traj_wp(c)
                        shared = set(p_) & set(truth)
                        if shared:
                            errs.append(float(np.mean([
                                np.hypot(p_[h][0] - truth[h][0],
                                         p_[h][1] - truth[h][1])
                                for h in shared])))
                    if errs:
                        best.append(min(errs)); cover += 1
                if best:
                    row[f"{pre}l2_min6"] = float(np.mean(best))
                    row[f"{pre}l2_min_cover"] = cover / len(got)
                per = {1: [], 2: [], 3: []}
                for g in got:
                    p, q = waypoints(g["generated"]), waypoints(g["reference"])
                    for h in per:
                        if h in p and h in q:
                            per[h].append(float(np.hypot(p[h][0] - q[h][0],
                                                         p[h][1] - q[h][1])))
                for h, v in per.items():
                    row[f"{pre}l2_{h}s"] = float(np.mean(v)) if v else None
    return row


def evaluate(snapshot, args):
    from training.eval_all_tasks import load_model, run_task

    class A:
        pass
    a = A()
    a.checkpoint, a.model, a.split = snapshot, args.model, args.split
    a.items, a.workers, a.show = args.items, 2, 0
    a.max_new_floor, a.all_profiles, a.seed = 0, True, 0
    a.radar_dropout, a.out = 0.0, None
    loaded = load_model(a)

    row, kept = {}, []
    for task in WATCHED:
        r = run_task(task, a, loaded)
        if not r:
            continue
        kept.extend(r.get("generations") or [])
        # `run_task` scores twice, once on the real radar and once on a
        # shuffled one. `full` is the model; `shuffled` is the control.
        s = r.get("full") or {}
        gens = r.get("generations") or []
        if task == "det_objects_azdeg":
            g = [mean_abs_azimuth(x.get("generated")) for x in gens]
            t = [mean_abs_azimuth(x.get("reference") or x.get("target"))
                 for x in gens]
            row["det_f1_shuffled"] = (r.get("shuffled") or {}).get("f1")
            row["az_gen"] = float(np.mean([v for v in g if v is not None] or [0]))
            row["az_truth"] = float(np.mean([v for v in t if v is not None] or [0]))
            row["det_f1"] = s.get("f1")
            row["det_range_mae"] = s.get("range_mae")
            row["det_az_mae"] = s.get("az_mae")
            row.update(detection_scores(gens, "det"))
        elif task == "det_objects_3dbbox":
            row["box_f1"] = s.get("f1")
            row["box_size_mae"] = s.get("size_mae")
            row["box_yaw_mae"] = s.get("yaw_mae")
            row.update(detection_scores(gens, "box"))
        elif task == "plan_ego_xy":
            row["l2"] = s.get("displacement_mae_m")
            per_h = {1: [], 2: [], 3: []}
            for x in gens:
                p, t = waypoints(x["generated"]), waypoints(x["reference"])
                for h in per_h:
                    if h in p and h in t:
                        per_h[h].append(float(np.hypot(p[h][0] - t[h][0],
                                                       p[h][1] - t[h][1])))
            for h, v in per_h.items():
                row[f"l2_{h}s"] = float(np.mean(v)) if v else None

    if args.nuscenes and os.path.isdir(args.nuscenes):
        from training.eval_all_tasks import MAX_NEW
        from training.task_scorers import scorer_for, summarise
        for task in WATCHED:
            rows = bundle_rows(args.nuscenes, task, args.nus_items)
            if not rows:
                continue
            gens = run_bundle(args.nuscenes, task, rows, loaded,
                              MAX_NEW.get(task, 48))
            kept.extend({**g, "dataset": "nuscenes"} for g in gens)
            recs = []
            fn = scorer_for(task)
            for g in gens:
                try:
                    recs.append(fn(g["generated"], g["reference"], ""))
                except TypeError:
                    recs.append(fn(g["generated"], g["reference"]))
            s = summarise(task, recs)
            if task == "det_objects_azdeg":
                row["nus_det_f1"] = s.get("f1")
                row["nus_range_mae"] = s.get("range_mae")
                g = [mean_abs_azimuth(x["generated"]) for x in gens]
                t2 = [mean_abs_azimuth(x["reference"]) for x in gens]
                row["nus_az_gen"] = float(np.mean([v for v in g if v] or [0]))
                row["nus_az_truth"] = float(np.mean([v for v in t2 if v] or [0]))
            elif task == "det_objects_3dbbox":
                row["nus_box_f1"] = s.get("f1")
            elif task == "plan_ego_xy":
                row["nus_l2"] = s.get("displacement_mae_m")
    return row, kept


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if "--worker" in argv:
        return worker_main([a for a in argv if a != "--worker"])
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--checkpoint", required=True, help="학습의 --out 경로")
    ap.add_argument("--device", default="4", help="(구) 단일 GPU")
    ap.add_argument("--devices", default="0,1,2,3,4",
                    help="쉼표로 구분한 GPU 목록. 하나씩 프로세스를 띄워 "
                         "항목을 나눠 생성하고, 채점은 부모가 모아서 한다")
    ap.add_argument("--items", type=int, default=60,
                    help="태스크당 항목 수. 추세를 보는 것이므로 작아도 되지만 "
                         "체크포인트마다 같아야 한다")
    ap.add_argument("--model", default="8B")
    ap.add_argument("--split", default="test")
    ap.add_argument("--stride", type=int, default=400,
                    help="이 배수의 스텝만 잰다. 세는 대신 스텝 번호로 정하므로 "
                         "감시기를 다시 띄워도 같은 지점이 찍힌다")
    ap.add_argument("--nuscenes", default=None,
                    help="nuScenes 번들 폴더. 주면 같은 체크포인트를 두 rig 에서 "
                         "잰다 -- 재학습 없는 전이 시험")
    ap.add_argument("--nus-items", type=int, default=40)
    ap.add_argument("--plan-samples", type=int, default=6,
                    help="계획에서 뽑을 갈래 수. Alpamayo 와 같은 6")
    ap.add_argument("--offset", type=int, default=0,
                    help="재개한 실행은 스텝을 0 부터 다시 센다. 재개 지점을 "
                         "여기 주면 추세가 한 축 위에 놓이고, 이미 잰 스텝과 "
                         "번호가 겹쳐 건너뛰는 일도 없어진다")
    ap.add_argument("--poll", type=int, default=120)
    ap.add_argument("--keep", type=int, default=0,
                    help="성적 상위 몇 개의 스냅샷을 남길지. 0 이면 남기지 "
                         "않는다 (예전 동작). 하나가 17 GB 다")
    ap.add_argument("--keep-metric", default="box_f1",
                    help="무엇을 기준으로 남길지. 3D 상자가 이 연구의 주력이라 "
                         "기본값은 box_f1")
    args = ap.parse_args(argv)

    trend = os.path.join(args.checkpoint, "trend.jsonl")
    done = set()
    if os.path.exists(trend):
        done = {json.loads(l)["step"] for l in open(trend)}
    log(f"감시 시작 {args.checkpoint} · GPU {args.devices} · "
        f"태스크당 {args.items}건 · 이미 잰 것 {len(done)}개")

    while True:
        step = saved_step(args.checkpoint)
        # `done` 은 누적 스텝으로 들고 있어야 한다. 재개한 실행은 0 부터 다시
        # 세므로, 실행 기준 번호로 비교하면 재개 전에 이미 잰 400, 800 ... 과
        # 부딪혀 전부 건너뛴다 -- 실제로 세 시간을 그렇게 놀았다.
        absolute = None if step is None else step + args.offset
        if absolute is None or absolute in done:
            time.sleep(args.poll)
            continue
        if step % args.stride:
            done.add(absolute)
            continue
        snap = os.path.join(args.checkpoint, f"_eval_{step}")
        try:
            shutil.rmtree(snap, ignore_errors=True)
            shutil.copytree(os.path.join(args.checkpoint, "latest"), snap)
            if saved_step(args.checkpoint) != step:
                log(f"step {step} 복사 중 저장이 끼어들었습니다 -- 건너뜁니다")
                shutil.rmtree(snap, ignore_errors=True)
                done.add(absolute)
                continue
            started = time.monotonic()
            row, gens = evaluate_sharded(snap, args)
            # A score cannot tell a wrong answer from an unparsed one. Twice
            # now a 0.00 turned out to be the scorer failing to read a perfectly
            # good generation, and both times the text was what settled it.
            outdir = os.path.join(args.checkpoint, "generations")
            os.makedirs(outdir, exist_ok=True)
            with open(os.path.join(outdir, f"step{step:06d}.jsonl"), "w") as fh:
                for g in gens:
                    fh.write(json.dumps(g, ensure_ascii=False) + "\n")
            row["step"] = absolute
            row["run_step"] = step
            row["seconds"] = round(time.monotonic() - started, 1)
            keep_best(args, snap, step, row)
            with open(trend, "a") as fh:
                fh.write(json.dumps(row) + "\n")
            log(f"step {absolute:>6}  det F1 {row.get('det_f1'):.3f}  "
                f"|az| 생성 {row.get('az_gen'):.1f} / 정답 "
                f"{row.get('az_truth'):.1f}  L2 {row.get('l2'):.3f} m  "
                f"({row['seconds']:.0f}s)")
        except Exception as exc:
            log(f"step {step} 평가 실패: {type(exc).__name__} {exc}")
        finally:
            shutil.rmtree(snap, ignore_errors=True)
            done.add(absolute)


if __name__ == "__main__":
    raise SystemExit(main())
