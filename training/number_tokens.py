#!/usr/bin/env python3
"""숫자 하나를 토큰 하나로. 값 기반 초기화까지.

정답의 44~88% 가 숫자다. `-13.0` 이 `'-','1','3','.','0'` 다섯 토큰으로 쪼개지고,
`[301, 495, 371, 569]` 는 24 토큰 중 20 개가 숫자와 구분자다. 측정하면 이 태스크
정답이 절반으로 줄어든다 -- `motion_seg_bbox` 143 → 65, `det_objects_3dbbox`
263 → 134.

토큰을 더하는 것만으로는 "값이 가까우면 비슷하다" 가 생기지 않는다. 자릿수
임베딩의 평균으로 초기화해 재보면 100 과 200 의 유사도가 0.933 인데 100 과 110
은 0.916 이다 -- 더 먼 쪽이 더 비슷하다. 자릿수를 공유하는지만 반영하고 크기는
반영하지 않기 때문이다.

그래서 초기화를 값의 함수로 한다. 위치 인코딩과 같은 방식으로 값을 여러 주파수의
sin/cos 로 펼치면, 두 숫자의 내적이 값 차이에 따라 단조롭게 줄어든다. 시작부터
거리 구조가 들어가 있고, 학습이 그것을 무너뜨리지 않도록 `--digit-weight` 가
값 차이에 비례하는 벌점을 계속 준다.

    python -m training.number_tokens --scan      # 필요한 값 목록을 데이터에서
    python -m training.number_tokens --check     # 토큰화와 거리 구조 확인
"""

import argparse
import os
import re
import sys

import numpy as np
import torch

# 단위가 숫자에 붙어 있다. `37m`, `+12deg`, `7.3m/s`, `-1.6deg/s`, `378px`.
# 미터는 기본 단위라 접미가 없고, 나머지는 붙는다 -- 그러면 같은 `37` 이
# 미터와 도와 화소를 동시에 뜻하는 일이 없어지고, 손실이 같은 자로 잰
# 값끼리만 거리를 잴 수 있다.
NUMBER = re.compile(r"-?\d+(?:\.\d+)?(?:deg/s|m/s|deg|px|m)?")
UNITS = ("deg/s", "m/s", "deg", "px", "m")


def split_unit(literal):
    """('37m') -> (37.0, 'm').  단위가 없으면 '' 로 둔다."""
    for u in UNITS:
        if literal.endswith(u):
            return float(literal[: -len(u)]), u
    return float(literal), ""
VOCAB_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "training", "number_vocab.txt")


def number_tokens():
    """The literals that get one token each, read from the file built by --scan."""
    if not os.path.exists(VOCAB_PATH):
        return []
    with open(VOCAB_PATH) as fh:
        return [l.strip() for l in fh if l.strip()]


def add_number_tokens(tokenizer, model=None, embeddings=None):
    """Register the number literals and initialise their embeddings by value.

    `added_tokens` are matched before the BPE merges, so "24" inside
    "automobile 24 m az" becomes the single token rather than '2','4'. The
    minus sign and the decimal point are part of the literal, so "-13.0" is one
    token and not four.
    """
    literals = number_tokens()
    if not literals:
        return 0
    added = tokenizer.add_tokens(literals)
    if model is None or not added:
        return added
    model.resize_token_embeddings(len(tokenizer))
    ids = tokenizer.convert_tokens_to_ids(literals)
    # 값만 뽑아 초기화한다. 단위가 다른 토큰끼리는 어차피 손실에서 서로
    # 비교되지 않으므로, 여기서는 같은 값이면 같은 자리에서 출발해도 된다.
    values = np.array([split_unit(x)[0] for x in literals], dtype=np.float64)
    table = model.get_input_embeddings().weight
    init = value_embeddings(values, table.shape[1],
                            scale=float(table.detach().float().std()))
    with torch.no_grad():
        table[torch.tensor(ids)] = init.to(table.dtype).to(table.device)
        out = model.get_output_embeddings()
        if out is not None and out.weight is not table:
            out.weight[torch.tensor(ids)] = init.to(out.weight.dtype).to(out.weight.device)
    return added


def unit_of_each(literals):
    """각 리터럴의 단위. 손실이 같은 단위끼리만 거리를 재게 하는 데 쓴다."""
    return [split_unit(x)[1] for x in literals]


def value_embeddings(values, dim, scale=0.02, n_bands=64):
    """Sinusoids of the value, so nearby numbers start out nearby.

    A single frequency would make 0 and 2*pi/w identical; a geometric ladder of
    frequencies makes the similarity fall off monotonically over the range that
    matters and keeps distant values apart. The remaining dimensions are left
    small and random so the model still has room to encode whatever else a
    number means in context.
    """
    values = np.asarray(values, dtype=np.float64)
    span = max(np.abs(values).max(), 1.0)
    # Wavelengths from a tenth of a unit up to four times the full span: the
    # short ones separate 1.2 from 1.3, the long ones keep 12 away from 900.
    bands = np.geomspace(0.1, 4.0 * span, n_bands)
    phase = 2.0 * np.pi * values[:, None] / bands[None, :]
    feat = np.concatenate([np.sin(phase), np.cos(phase)], axis=1)
    feat = feat / np.sqrt(feat.shape[1])

    rng = np.random.default_rng(0)
    out = rng.normal(0.0, 0.02, size=(len(values), dim))
    take = min(dim, feat.shape[1])
    out[:, :take] = feat[:, :take]
    return torch.tensor(out * scale / max(out.std(), 1e-8), dtype=torch.float32)


def scan(paths_module, limit_per_task=20000, coverage=0.999):
    """Which literals actually occur, and how many cover `coverage` of them."""
    import collections
    import pandas as pd
    counts = collections.Counter()
    path = os.path.join(paths_module.COMMON_DIR,
                        "instruct_items_tasks01_06.parquet")
    d = pd.read_parquet(path, columns=["task", "target", "prompt"])
    for task, g in d.groupby("task"):
        g = g.head(limit_per_task)
        for col in ("target", "prompt"):
            for s in g[col]:
                counts.update(NUMBER.findall(s))
    total = sum(counts.values())
    kept, seen = [], 0
    for lit, n in counts.most_common():
        kept.append(lit)
        seen += n
        if seen >= coverage * total:
            break
    return kept, counts, total


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--scan", action="store_true",
                    help="rebuild number_vocab.txt from the built data")
    ap.add_argument("--coverage", type=float, default=0.999)
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)

    if args.scan:
        from datatools import paths
        kept, counts, total = scan(paths, coverage=args.coverage)
        with open(VOCAB_PATH, "w") as fh:
            fh.write("\n".join(kept) + "\n")
        dec = sum(1 for k in kept if "." in k)
        print(f"고유 숫자 {len(counts):,} 중 {len(kept):,} 개로 "
              f"{100*args.coverage:.1f}% 를 덮음")
        print(f"  소수 {dec:,} · 정수 {len(kept)-dec:,}")
        print(f"wrote {VOCAB_PATH}")
        return 0

    if args.check:
        from transformers import AutoTokenizer
        from training.train_vlm import MODEL_DIR
        tok = AutoTokenizer.from_pretrained(MODEL_DIR["8B"])
        before = len(tok)
        n = add_number_tokens(tok)
        print(f"어휘 {before:,} → {len(tok):,}  (+{n:,})")
        for s in ("automobile 24 m az +13 deg",
                  "automobile (9.7, -13.0, 0.9) size 5.3x2.2x1.9 m yaw -179 deg",
                  "#12 automobile [301, 495, 371, 569]",
                  "+1s (+8.8, +0.0)"):
            ids = tok(s, add_special_tokens=False)["input_ids"]
            print(f"\n  {len(ids):>3} 토큰  {s}")
            print("      " + " | ".join(repr(tok.decode([i])) for i in ids))

        lits = number_tokens()
        vals = np.array([float(x) for x in lits])
        E = value_embeddings(vals, 4096)
        E = E / E.norm(dim=1, keepdim=True)
        pick = [i for i, v in enumerate(vals) if v in (100.0, 110.0, 200.0, 900.0)]
        got = {vals[i]: i for i in pick}
        if len(got) == 4:
            a = got[100.0]
            print("\n값 기반 초기화의 거리 구조")
            for b in (110.0, 200.0, 900.0):
                print(f"  100 ↔ {b:.0f}: 유사도 {float(E[a] @ E[got[b]]):+.3f}")
        sub = np.argsort(np.abs(vals))[:400]
        S = (E[sub] @ E[sub].T).numpy()
        D = np.abs(np.subtract.outer(vals[sub], vals[sub]))
        iu = np.triu_indices(len(sub), 1)
        print(f"  값 차이 vs 유사도 상관 {np.corrcoef(D[iu], S[iu])[0,1]:+.3f} "
              f"(음수일수록 좋음)")
        return 0

    ap.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
