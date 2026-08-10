#!/usr/bin/env python3
"""번들이 스스로 완결되어 있는지 확인한다 -- 모델 없이, 저장소 없이.

옮긴 서버에서 `run_eval.py` 를 돌리기 전에 이것부터 돌리면, 20 GiB 짜리 모델을
올린 뒤에야 파일이 하나 빠진 것을 알게 되는 일을 피할 수 있다.

    python verify_bundle.py
"""

import argparse
import json
import os
import sys

import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    args = ap.parse_args(argv)
    bad = []
    data = args.data

    # 1. 저장소를 참조하지 않는가
    sys.path.insert(0, HERE)
    try:
        from ravl.task_scorers import scorer_for, summarise
        from ravl.number_tokens import number_tokens
        from ravl.connector import RadarConnector          # noqa: F401
        from ravl.radar_encoder import RadarEncoder        # noqa: F401
    except Exception as exc:
        print(f"!! 모듈 적재 실패: {exc}")
        return 1
    lits = number_tokens()
    print(f"모듈 정상 · 숫자 어휘 {len(lits):,}")
    if len(lits) != 2503:
        bad.append(f"숫자 어휘가 {len(lits)} (학습 때는 2,503)")

    # 2. 아이템이 다 있고, 파일이 실제로 열리는가
    manifest = json.load(open(os.path.join(data, "manifest.json")))
    print(f"매니페스트: {manifest['split']} 분할, 클립 {manifest.get('n_clips','?')}개, "
          f"태스크 {len(manifest['tasks'])}종")

    # 클립이 자기 프레임과 ego 를 갖고 있는가. 아이템은 프레임을 번호로만
    # 가리키므로, 클립 폴더가 비면 모든 태스크가 한꺼번에 죽는다.
    clips = [json.loads(l) for l in open(os.path.join(data, "clips.jsonl"))]
    for c in clips:
        d = os.path.join(data, "clips", c["clip_id"])
        got = len([f for f in os.listdir(os.path.join(d, "frames"))
                   if f.endswith(".jpg")]) if os.path.isdir(
                       os.path.join(d, "frames")) else 0
        if got != c["n_frames"]:
            bad.append(f"{c['clip_id']}: 프레임 {got} / {c['n_frames']}")
        if not os.path.exists(os.path.join(d, "ego.npy")):
            bad.append(f"{c['clip_id']}: ego.npy 없음")
    print(f"클립 {len(clips)}개 · 프레임과 ego 확인")

    total = 0
    for task, n in sorted(manifest["tasks"].items()):
        path = os.path.join(data, "by_task", f"{task}.jsonl")
        if not os.path.exists(path):
            bad.append(f"{task}.jsonl 없음")
            continue
        rows = [json.loads(l) for l in open(path)]
        total += len(rows)
        if len(rows) != n:
            bad.append(f"{task}: 매니페스트 {n} vs 실제 {len(rows)}")
        if scorer_for(task).__name__ == "score_text" and not task.startswith("desc_"):
            bad.append(f"{task}: 채점기가 일반 텍스트로 떨어짐")
        # 첫 건과 끝 건을 실제로 열어 본다
        for rec in (rows[0], rows[-1]):
            d = os.path.join(data, "clips", rec["clip_id"])
            for idx in rec["frames"]:
                p = os.path.join(d, "frames", f"f{idx:02d}.jpg")
                if not os.path.exists(p):
                    bad.append(f"{rec['id']}: f{idx:02d}.jpg 없음"); break
            else:
                Image.open(os.path.join(
                    d, "frames", f"f{rec['frames'][0]:02d}.jpg")).convert("RGB")
            rp = os.path.join(d, "radar", f"{rec['radar']}.npz")
            if not os.path.exists(rp):
                bad.append(f"{rec['id']}: {rec['radar']}.npz 없음"); continue
            z = np.load(rp)
            if "points" not in z or "scan" not in z or "shape" not in z:
                bad.append(f"{rec['id']}: radar.npz 내용 부족")
            if not rec.get("user") or rec.get("target") is None:
                bad.append(f"{rec['id']}: 프롬프트 또는 정답 없음")
            if "<|radar_pad|>" not in rec["user"]:
                bad.append(f"{rec['id']}: 프롬프트에 레이더 자리표시자 없음")
    print(f"아이템 {total:,}건 · 파일 열기 확인")

    # 3. 채점기가 정답을 만점으로 매기는가
    for task in sorted(manifest["tasks"]):
        rows = [json.loads(l) for l in
                open(os.path.join(data, "by_task", f"{task}.jsonl"))]
        r = rows[0]
        fn = scorer_for(task)
        try:
            got = fn(r["target"], r["target"], r["user"])
        except TypeError:
            got = fn(r["target"], r["target"])
        s = summarise(task, [got])
        for key in ("f1", "accuracy"):
            if key in s and s[key] < 0.999:
                bad.append(f"{task}: 정답을 넣었는데 {key}={s[key]:.3f}")

    size = sum(os.path.getsize(os.path.join(p, f))
               for p, _, fs in os.walk(HERE) for f in fs)
    print(f"번들 크기 {size/1e9:.2f} GB")
    if bad:
        print(f"\n!! 문제 {len(bad)}건")
        for b in bad[:20]:
            print(f"   {b}")
        return 1
    print("\n이상 없음 — run_eval.py 를 돌려도 됩니다")
    return 0


if __name__ == "__main__":
    sys.exit(main())
