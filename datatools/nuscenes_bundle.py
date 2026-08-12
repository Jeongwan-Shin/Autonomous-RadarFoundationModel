#!/usr/bin/env python3
"""nuScenes 를 우리 평가 번들 형식으로 옮긴다 -- 재학습 없는 전이 시험용.

레이더 인코더를 6 채널로 줄이고 밀도 증강을 넣은 이유가 이것이다. 우리가
버린 두 채널(z, snr)은 nuScenes 의 ARS408 이 낼 수 없는 것들이고, 남긴 여섯은
양쪽에서 똑같이 계산된다. 그래서 학습된 가중치를 그대로 얹을 수 있다 -- 입력
표현이 글자 그대로 같으므로, 여기서 나오는 점수는 정직한 언씬 시험이다.

나오는 것은 `20260808_test/data` 와 같은 배치라 그 폴더의 `run_eval.py` 가
바꿀 것 없이 읽는다.

    python -m datatools.nuscenes_bundle --scenes 40 --out 20260812_nuscenes/data

## 두 rig 이 다른 곳

카메라가 좁다. nuScenes CAM_FRONT 는 수평 65 도(±32.5)이고 우리 rig 은 120
도다. 우리 태스크가 ±60 도 섹터를 묻는데 그 절반은 카메라에 아예 안 잡히므로,
정답을 `--sector` 로 좁힌다. 좁힌 만큼 우리 데이터의 점수와 직접 비교되지
않지만, nuScenes 안에서는 공정하다.

레이더 신원이 없다. 우리 인코더는 lrr1/mrr2/srr0/none 중 하나를 라우팅
사전값으로 받는데 ARS408 은 그중 무엇도 아니다. 스캔당 125 점으로 가장 성긴
`srr0` 를 준다 -- 전문가 라우팅이 꺼져 있어 지금은 임베딩 하나의 차이지만,
어느 쪽을 골랐는지는 적어 둘 값어치가 있다.

시간 격자가 두 배다. nuScenes 키프레임은 2 Hz 이고 우리는 1 Hz 이므로 격
프레임만 쓴다. 20 초 장면에서 정확히 20 장이 나온다.
"""

import argparse
import collections
import json
import math
import os
import sys

import numpy as np
from PIL import Image

ROOT = "/NHNHOME/workspace/dataset/raw_Auto_datasets/nuScenes/trainval"
META = f"{ROOT}/v1.0-trainval"

# ARS408 의 PCD. 한 점 43 바이트, 정렬 없음.
RADAR_DT = np.dtype([
    ("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("dyn_prop", "i1"),
    ("id", "<i2"), ("rcs", "<f4"), ("vx", "<f4"), ("vy", "<f4"),
    ("vx_comp", "<f4"), ("vy_comp", "<f4"), ("is_quality_valid", "i1"),
    ("ambig_state", "i1"), ("x_rms", "i1"), ("y_rms", "i1"),
    ("invalid_state", "i1"), ("pdh0", "i1"), ("vx_rms", "i1"), ("vy_rms", "i1")])

# nuScenes 의 클래스 이름을 우리 어휘로. 우리 모델은 이 낱말들만 써 봤으므로
# 정답이 다른 낱말을 쓰면 채점기가 클래스를 틀린 것으로 셀 뿐 아니라 모델이
# 맞힐 수 없는 답을 요구하게 된다.
CLASS_MAP = {
    "vehicle.car": "automobile", "vehicle.truck": "heavy_truck",
    "vehicle.bus.rigid": "bus", "vehicle.bus.bendy": "bus",
    "vehicle.trailer": "trailer", "vehicle.construction": "other_vehicle",
    "vehicle.emergency.police": "automobile",
    "vehicle.emergency.ambulance": "other_vehicle",
    "vehicle.motorcycle": "rider", "vehicle.bicycle": "rider",
    "human.pedestrian.adult": "person", "human.pedestrian.child": "person",
    "human.pedestrian.construction_worker": "person",
    "human.pedestrian.police_officer": "person",
    "animal": "animal",
}

# barrier 와 trafficcone 은 뺀다. 우리 어휘의 `protruding_object` 로 옮길 수는
# 있지만 그 낱말은 학습 데이터의 0.83% 뿐이라 모델이 거의 써 본 적이 없고,
# nuScenes 에서는 21.4% 라 정답의 최다 클래스가 된다. 그대로 두면 이 시험은
# 전이를 재는 것이 아니라 모델이 안 배운 낱말을 아는지 묻는 것이 된다.
DROPPED = ("movable_object.barrier", "movable_object.trafficcone",
           "movable_object.pushable_pullable", "movable_object.debris",
           "static_object.bicycle_rack")

HORIZONS = (1.0, 2.0, 3.0)
MAX_LISTED = 8


def log(m):
    print(m, flush=True)


def load(name):
    return json.load(open(f"{META}/{name}.json"))


def quat_yaw(q):
    w, x, y, z = q
    return math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def quat_matrix(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)]])


def read_radar(path):
    raw = open(path, "rb").read()
    o = raw.index(b"DATA binary\n") + len(b"DATA binary\n")
    n = int([l.split()[1] for l in raw[:o].decode().splitlines()
             if l.startswith("POINTS")][0])
    return np.frombuffer(raw[o:o + n * RADAR_DT.itemsize], dtype=RADAR_DT)


def radar_channels(d, calib, normalise):
    """ARS408 을 우리 여섯 채널로. 센서 좌표에서 ego 좌표로 옮긴 뒤 계산한다.

    `radial_velocity` 는 (vx, vy) 를 시선 방향에 투영한 것이고,
    `doppler_residual` 은 자차 운동이 보상된 (vx_comp, vy_comp) 를 같은 방향에
    투영한 것이다. 우리 rig 에서 그 둘의 관계와 같은 것을 실제로 확인했다:
    |residual| 의 중앙값 0.10 m/s 대 |radial| 5.68 m/s.
    """
    R = quat_matrix(calib["rotation"])
    t = np.asarray(calib["translation"])
    p = np.stack([d["x"], d["y"], d["z"]], axis=1).astype(np.float64)
    p = p @ R.T + t
    v = np.stack([d["vx"], d["vy"], np.zeros(len(d))], axis=1).astype(np.float64) @ R.T
    vc = np.stack([d["vx_comp"], d["vy_comp"], np.zeros(len(d))],
                  axis=1).astype(np.float64) @ R.T
    x, y = p[:, 0], p[:, 1]
    rng = np.hypot(x, y)
    safe = np.maximum(rng, 1e-6)
    radial = (v[:, 0] * x + v[:, 1] * y) / safe
    resid = (vc[:, 0] * x + vc[:, 1] * y) / safe
    stack = np.stack([x, y, rng, radial, resid, d["rcs"].astype(np.float64)],
                     axis=1)
    return (stack / normalise).astype(np.float32)


def ego_series(poses, tokens):
    """키프레임마다 속도, 가속도, 요레이트. 자세 차분에서 나온다."""
    xy = np.array([poses[t]["translation"][:2] for t in tokens])
    yaw = np.unwrap([quat_yaw(poses[t]["rotation"]) for t in tokens])
    ts = np.array([poses[t]["timestamp"] for t in tokens]) / 1e6
    dt = np.gradient(ts)
    v = np.gradient(xy, axis=0) / dt[:, None]
    speed = np.hypot(v[:, 0], v[:, 1])
    return np.stack([speed, np.gradient(speed) / dt,
                     np.gradient(yaw) / dt], axis=1).astype(np.float32), xy, yaw, ts


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scenes", type=int, default=40)
    ap.add_argument("--sector", type=float, default=30.0,
                    help="정답에 넣을 방위각 반각. CAM_FRONT 가 ±32.5 도다")
    ap.add_argument("--max-range", type=float, default=40.0)
    ap.add_argument("--out", default="20260812_nuscenes/data")
    ap.add_argument("--sensor", default="srr0",
                    help="우리 인코더에 줄 센서 신원. ARS408 은 스캔당 125 점으로 "
                         "srr0 에 가장 가깝다")
    args = ap.parse_args(argv)

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from training.connector import radar_prompt_block
    from training.instruct_data import SENSOR_LABEL, ego_text
    from training.radar_data import CHANNELS, NORMALISE
    from training.radar_encoder import SENSOR_IDS

    log(f"채널 {CHANNELS}")
    log("메타데이터 읽는 중 (sample_data 1.3 GB)…")
    scenes, samples = load("scene"), load("sample")
    poses = {p["token"]: p for p in load("ego_pose")}
    calib = {c["token"]: c for c in load("calibrated_sensor")}
    cats = {c["token"]: c["name"] for c in load("category")}
    inst = {i["token"]: cats[i["category_token"]] for i in load("instance")}
    ann = collections.defaultdict(list)
    for a in load("sample_annotation"):
        ann[a["sample_token"]].append(a)
    sd = collections.defaultdict(dict)
    for d in load("sample_data"):
        if d["is_key_frame"]:
            sd[d["sample_token"]][d["filename"].split("/")[1]] = d
    by_scene = collections.defaultdict(list)
    for s in samples:
        by_scene[s["scene_token"]].append(s)
    for k in by_scene:
        by_scene[k].sort(key=lambda s: s["timestamp"])
    log(f"scene {len(scenes)} · sample {len(samples):,}")

    os.makedirs(args.out, exist_ok=True)
    by_task = collections.defaultdict(list)
    clip_rows, n_items = [], 0
    block = radar_prompt_block(256)
    label = SENSOR_LABEL.get(args.sensor, "radar")

    for sc in scenes[: args.scenes]:
        chain = by_scene[sc["token"]][::2][:20]        # 2 Hz -> 1 Hz
        if len(chain) < 12:
            continue
        cid = sc["name"]
        d = os.path.join(args.out, "clips", cid)
        os.makedirs(os.path.join(d, "frames"), exist_ok=True)
        os.makedirs(os.path.join(d, "radar"), exist_ok=True)

        tokens = [sd[s["token"]]["CAM_FRONT"]["ego_pose_token"] for s in chain]
        state, xy, yaw, ts = ego_series(poses, tokens)
        np.save(os.path.join(d, "ego.npy"), state)

        for j, s in enumerate(chain):
            src = os.path.join(ROOT, sd[s["token"]]["CAM_FRONT"]["filename"])
            with Image.open(src) as im:
                im.convert("RGB").resize((1152, 648), Image.BILINEAR).save(
                    os.path.join(d, "frames", f"f{j:02d}.jpg"), quality=88)

        # 레이더는 20 장 전부를 한 창으로 -- 우리 clip-level 창과 같은 모양
        pts = np.zeros((20, 1024, len(CHANNELS)), dtype=np.float32)
        mask = np.zeros((20, 1024), dtype=bool)
        for j, s in enumerate(chain):
            rd = sd[s["token"]].get("RADAR_FRONT")
            if rd is None:
                continue
            raw = read_radar(os.path.join(ROOT, rd["filename"]))
            ch = radar_channels(raw, calib[rd["calibrated_sensor_token"]],
                                NORMALISE)
            n = min(len(ch), 1024)
            pts[j, :n], mask[j, :n] = ch[:n], True
        np.savez_compressed(
            os.path.join(d, "radar", "clip.npz"),
            points=pts[mask].astype(np.float16),
            scan=np.repeat(np.arange(20, dtype=np.uint8), mask.sum(axis=1)),
            shape=np.array(pts.shape, dtype=np.int32))

        rows = []
        for j, s in enumerate(chain):
            pose = poses[tokens[j]]
            o = np.asarray(pose["translation"][:2])
            c, sn = math.cos(-yaw[j]), math.sin(-yaw[j])
            seen = []
            for a in ann[s["token"]]:
                name = CLASS_MAP.get(inst[a["instance_token"]])
                if name is None:
                    continue
                delta = np.asarray(a["translation"][:2]) - o
                ex = c * delta[0] - sn * delta[1]
                ey = sn * delta[0] + c * delta[1]
                rng = math.hypot(ex, ey)
                az = math.degrees(math.atan2(ey, ex))
                if rng > args.max_range or abs(az) > args.sector:
                    continue
                w, l, h = a["size"]                     # nuScenes: width, length, height
                seen.append((rng, az, name, ex, ey, a["translation"][2], l, w, h,
                             math.degrees(quat_yaw(a["rotation"]) - yaw[j])))
            seen.sort()
            seen = seen[:MAX_LISTED]

            ego = ego_text(state, j)
            head = (f"{block}\nSensors present: camera, ego motion, {label}.\n"
                    f"Ego motion (binned, 1 Hz, offsets in seconds from t=0s): "
                    f"{ego}\nAt frame {j + 1}. ")

            q = (f"List the road users within {args.max_range:.0f} m ahead, "
                 f"nearest first, up to {MAX_LISTED}, with class, range and "
                 f"azimuth.")
            a_txt = ("No road users in the forward sector." if not seen else
                     "; ".join(f"{n} {r:.0f} m az {z:+.0f} deg"
                               for r, z, n, *_ in seen))
            rows.append({"task": "det_objects_azdeg", "frame": j,
                         "user": head + q, "target": a_txt})

            q3 = (f"List the road users within {args.max_range:.0f} m ahead, "
                  f"nearest first, up to {MAX_LISTED}, as a 3D box: class, "
                  f"centre (x, y, z) in metres with x forward, y left and z up, "
                  f"then size as length x width x height in metres, then yaw in "
                  f"degrees, positive to the left of the ego heading.")
            a3 = ("No road users in the forward sector." if not seen else
                  "; ".join(f"{n} ({ex:.1f}, {ey:.1f}, {ez:.1f}) size "
                            f"{l:.1f}x{w:.1f}x{h:.1f} m yaw {yw:+.0f} deg"
                            for _, _, n, ex, ey, ez, l, w, h, yw in seen))
            rows.append({"task": "det_objects_3dbbox", "frame": j,
                         "user": head + q3, "target": a3})

            # 자차 경로. 1 Hz 격자이므로 +1/+2/+3 초는 j+1, j+2, j+3 이다.
            if j + 3 < len(chain):
                way = []
                for k in (1, 2, 3):
                    delta = xy[j + k] - xy[j]
                    way.append((c * delta[0] - sn * delta[1],
                                sn * delta[0] + c * delta[1]))
                rows.append({
                    "task": "plan_ego_xy", "frame": j,
                    "user": head + ("Navigation command: STRAIGHT. Predict the "
                                    "ego vehicle's path over the next 3 seconds "
                                    "as (x, y) offsets in metres."),
                    "target": "; ".join(f"+{k}s ({x:+.1f}, {y:+.1f})"
                                        for k, (x, y) in enumerate(way, 1))})

        # 태스크마다 보는 프레임이 다르다. 우리 모델은 `det_objects` 를 한
        # 장으로, `plan_ego` 를 두 장으로 배웠고, 스무 장을 주면 1152 해상도에서
        # 7,200 비전 토큰이 되어 3,584 예산을 넘는다. 여기서 창을 맞추지 않으면
        # 전이를 재는 것이 아니라 처음 보는 입력 모양을 주는 것이 된다.
        window = {"det_objects_azdeg": 1, "det_objects_3dbbox": 1,
                  "plan_ego_xy": 2}
        for r in rows:
            n = window.get(r["task"], 2)
            end = r["frame"]
            uses = list(range(max(0, end - n + 1), end + 1))
            r.update({"id": f"{cid}/{r['task']}/{r['frame']:02d}",
                      "clip_id": cid, "sensor": SENSOR_IDS[args.sensor],
                      "radar": "clip", "frames": uses,
                      "radar_points": int(mask.sum())})
            by_task[r["task"]].append(r)
        with open(os.path.join(d, "tasks.jsonl"), "w") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        clip_rows.append({"clip_id": cid, "n_frames": len(chain),
                          "n_items": len(rows),
                          "tasks": sorted({r["task"] for r in rows}),
                          "radar_windows": 1})
        n_items += len(rows)
        log(f"  {cid}  프레임 {len(chain)}  아이템 {len(rows)}  "
            f"레이더 {int(mask.sum())}점")

    os.makedirs(os.path.join(args.out, "by_task"), exist_ok=True)
    for task, rows in sorted(by_task.items()):
        with open(os.path.join(args.out, "by_task", f"{task}.jsonl"), "w") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(args.out, "clips.jsonl"), "w") as fh:
        for r in clip_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    json.dump({"split": "nuscenes-trainval", "n_clips": len(clip_rows),
               "n_items": n_items, "channels": list(CHANNELS),
               "tasks": {t: len(r) for t, r in sorted(by_task.items())},
               "layout": "clip-centric", "sector_deg": args.sector,
               "max_range_m": args.max_range, "sensor_id": args.sensor},
              open(os.path.join(args.out, "manifest.json"), "w"),
              indent=1, ensure_ascii=False)
    log(f"\n클립 {len(clip_rows)} · 아이템 {n_items:,} → {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
