#!/usr/bin/env python3
"""자차 이력 궤적을 어휘 안의 토큰으로.

텍스트도 아니고 별도 인코더도 아니다. 프롬프트에 자리표시자를 깔아 두고 그
자리의 **토큰 id** 를 궤적 토큰 id 로 바꿔치기한다 -- 궤적이 어휘의 일부가
된다. 레이더는 자리표시자의 *임베딩* 을 후크로 덮어쓰는데, 이쪽은 id 자체를
바꾼다는 점만 다르다.

Alpamayo (NVlabs, arXiv 2511.00088) 의 `DeltaTrajectoryTokenizer` 를 따른다.
학습된 VQ 가 아니라 고정된 균등 양자화다:

  ① 위치가 아니라 직전 대비 변화량        xyz[1:] - xyz[:-1]
  ② 축마다 정해진 범위로 정규화 후 양자화   (v - lo) / (hi - lo) * (bins - 1)
  ③ 축마다 토큰 하나                      17점 -> 델타 16개 x 3축 = 48 토큰

축별로 코드북을 따로 둔다. 우리 데이터에서 0.25초 델타를 재 보면 x 는 -0.08
에서 +9.05 m, y 는 +-4.3 m, z 는 +-0.44 m 다. 하나의 범위를 공유하면 y 와 z 는
구간의 1% 만 쓰게 되고, 그것은 숫자 토큰에서 이미 겪은 실패다 -- `37` 이
미터도 도도 되던 문제와 같은 형태다.

    python -m training.traj_tokens --check
"""

import argparse
import re

import numpy as np
import torch

# NVIDIA egomotion 파일이 내주는 그대로. 150 클립에서 재 보면 표본 간격
# 중앙값이 100 ms, 즉 10 Hz 다 -- 리샘플링하지 않고 원본 표본을 쓴다.
# 17점이면 델타 16개 x 3축 = 48 토큰으로 Alpamayo 의 tokens_per_history_traj
# 와 같고, 이력은 1.6초가 된다.
HIST_HZ = 10.0
HIST_POINTS = 17
HIST_SECONDS = (HIST_POINTS - 1) / HIST_HZ           # 1.6
N_AXES = 3
TOKENS_PER_HISTORY = (HIST_POINTS - 1) * N_AXES      # 48
BINS = 1000

# Alpamayo 의 xyz_min=(-4,-4,-10), xyz_max=(4,4,10) 을 따르되 z 만 좁힌다.
#
# 처음에는 150 클립에서 재서 x 를 (-0.5, 3.5) 로 잡았다. 그것은 좁았다 --
# 독일 400 개를 섞은 1,000 클립에서 다시 재니 x 가 3.5 m 를 넘는 클립이
# 2.0%, 최대가 4.136 m 였다. 0.1초에 4 m 면 144 km/h 이고, 아우토반이 있는
# 데이터셋에서 그 위는 조용히 잘린다. Alpamayo 의 +-4 는 임의로 넉넉히 잡은
# 값이 아니라 이 데이터의 실제 상한이었다.
#
# z 만 다르다. 실측 최대가 0.299 m 인데 +-10 을 쓰면 1,000 구간 중 30 개만
# 쓰게 된다 -- Alpamayo 는 같은 토크나이저를 6.4초 미래 궤적에도 쓰므로 그
# 범위가 필요했겠지만, 우리는 1.6초 이력에만 쓴다.
AXIS_RANGE = {
    "x": (-4.0, 4.0),       # Alpamayo 와 같음. 실측 클립별 최대의 최대 4.136
    "y": (-4.0, 4.0),       # Alpamayo 와 같음. 실측 최대 2.423
    "z": (-0.5, 0.5),       # 실측 최대 0.299. Alpamayo 의 +-10 은 미래용
}
AXES = ("x", "y", "z")

# 자리표시자. 이 토큰이 프롬프트에 TOKENS_PER_HISTORY 개 깔리고, 붙이는 쪽이
# 그 자리의 id 를 아래 궤적 토큰으로 바꾼다.
PAD_TOKEN = "<|traj_pad|>"


def traj_tokens():
    """어휘에 추가할 궤적 토큰 이름. 축마다 BINS 개씩."""
    return [f"<|traj_{a}_{i}|>" for a in AXES for i in range(BINS)]


# 미래 궤적도 같은 코드북으로 낸다. 3초를 10 Hz 로 31점 -> 델타 30 x 3축 = 90.
#
# 입력에만 두면 3,000 개 임베딩이 프롬프트 쪽 기울기로만 배우는데, 그 기울기는
# "궤적 토큰을 무시하고 카메라로 답해도 손실이 비슷하다" 는 상태에서 거의 0 이다
# -- 레이더가 안 읽히던 것과 같은 구조다. 출력에도 두면 매 스텝 90 개 토큰에
# 직접 교차엔트로피가 걸리고, 같은 코드북이므로 거기서 배운 의미가 입력 해석에
# 그대로 쓰인다. Alpamayo 는 6.4초 64 지점을 이렇게 낸다.
FUT_SECONDS = 3.0
FUT_POINTS = 31
TOKENS_PER_FUTURE = (FUT_POINTS - 1) * N_AXES        # 90


def encode(xyz_points, expect=None):
    """(N, 3) 자차 좌표계 궤적 -> ((N-1)*3,) 구간 인덱스.

    이력이든 미래든 같은 함수다 -- Alpamayo 도 이력을 `fut_xyz` 자리에 넘겨
    같은 인코딩을 태운다. 반환값은 축별 코드북 안의 위치이고, 어휘 id 로 바꾸는
    것은 부르는 쪽의 몫이다. 그래야 이 함수가 토크나이저를 몰라도 된다.
    """
    xyz = np.asarray(xyz_points, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != N_AXES:
        raise ValueError(f"(N, {N_AXES}) 여야 한다: {xyz.shape}")
    if expect is not None and xyz.shape[0] != expect:
        raise ValueError(f"{expect} 점이어야 한다: {xyz.shape[0]}")
    delta = np.diff(xyz, axis=0)
    out = np.zeros(delta.shape, dtype=np.int64)
    for k, axis in enumerate(AXES):
        lo, hi = AXIS_RANGE[axis]
        v = (delta[:, k] - lo) / (hi - lo)
        # 범위 밖은 자른다. 버리면 토큰 수가 달라져 자리표시자와 어긋난다.
        out[:, k] = np.clip(np.round(v * (BINS - 1)), 0, BINS - 1)
        out[:, k] += k * BINS                        # 축별 코드북으로 옮긴다
    return out.reshape(-1)                           # 점 순서, 점마다 x y z


def decode(indices):
    """구간 인덱스 -> 자차 좌표계 궤적. 길이는 준 것에서 정한다."""
    idx = np.asarray(indices, dtype=np.int64).reshape(-1, N_AXES)
    delta = np.zeros(idx.shape, dtype=np.float64)
    for k, axis in enumerate(AXES):
        lo, hi = AXIS_RANGE[axis]
        centre = (idx[:, k] - k * BINS) / (BINS - 1)
        delta[:, k] = centre * (hi - lo) + lo
    return np.vstack([np.zeros((1, N_AXES)), np.cumsum(delta, axis=0)])


def quantisation_error():
    """구간 하나가 몇 미터인가. 이 값보다 정밀한 것은 표현되지 않는다."""
    return {a: (hi - lo) / (BINS - 1) for a, (lo, hi) in AXIS_RANGE.items()}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args(argv)

    print(f"궤적 토큰 {len(traj_tokens()):,}개 "
          f"({N_AXES}축 x {BINS}구간), 이력 하나에 {TOKENS_PER_HISTORY}토큰")
    for a, e in quantisation_error().items():
        print(f"  {a}: 구간 하나 {e*100:.2f} cm")
    if args.check:
        rng = np.random.default_rng(0)
        speed = 12.0
        step = speed * HIST_SECONDS / (HIST_POINTS - 1)
        truth = np.cumsum(
            np.stack([np.full(HIST_POINTS, step),
                      rng.normal(0, 0.3, HIST_POINTS),
                      rng.normal(0, 0.02, HIST_POINTS)], axis=1), axis=0)
        truth -= truth[0]
        ids = encode(truth)
        back = decode(ids)
        err = np.abs(back - truth)
        print(f"\n왕복 시험: 토큰 {len(ids)}개 · 최대 오차 "
              f"x {err[:,0].max()*100:.1f} cm · y {err[:,1].max()*100:.1f} cm "
              f"· z {err[:,2].max()*100:.1f} cm")
        print(f"  id 범위 {ids.min()} ~ {ids.max()} (축별 코드북 "
              f"0~{BINS-1}, {BINS}~{2*BINS-1}, {2*BINS}~{3*BINS-1})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


def add_traj_tokens(tokenizer, model=None):
    """자리표시자와 궤적 토큰을 어휘에 등록한다.

    `add_radar_tokens` 와 같은 모양이다. 다른 점은 이쪽 토큰이 임베딩을 덮어쓰는
    자리가 아니라 실제로 생성·소비되는 id 라는 것 -- 자리표시자는 붙이는 쪽에서
    id 째로 바뀐다.
    """
    added = tokenizer.add_special_tokens(
        {"additional_special_tokens": [PAD_TOKEN] + traj_tokens()})
    if model is not None and added:
        model.resize_token_embeddings(len(tokenizer))
    first = tokenizer.convert_tokens_to_ids(f"<|traj_{AXES[0]}_0|>")
    return tokenizer.convert_tokens_to_ids(PAD_TOKEN), first


def init_embeddings(tokenizer, model):
    """궤적 토큰을 구간의 *값* 모양으로 초기화한다.

    무작위로 두면 `<|traj_x_500|>` 과 `<|traj_x_501|>` 이 서로 무관한 벡터가
    된다. 4 mm 차이인 두 구간을 처음부터 남남으로 시작시키는 셈이고, 숫자
    토큰에서 이미 같은 문제를 겪었다 -- 값 모양으로 심으면 인접한 구간이
    가까운 데서 출발한다.
    """
    from training.number_tokens import value_embeddings
    emb = model.get_input_embeddings()
    dim = emb.weight.shape[1]
    with torch.no_grad():
        for axis in AXES:
            lo, hi = AXIS_RANGE[axis]
            centres = lo + (hi - lo) * np.arange(BINS) / (BINS - 1)
            ids = [tokenizer.convert_tokens_to_ids(f"<|traj_{axis}_{i}|>")
                   for i in range(BINS)]
            vecs = value_embeddings(centres, dim).to(emb.weight.dtype)
            emb.weight[torch.tensor(ids)] = vecs.to(emb.weight.device)
    return BINS * len(AXES)


def traj_prompt_block():
    """프롬프트에 깔 자리표시자 문자열."""
    return PAD_TOKEN * TOKENS_PER_HISTORY


def token_string(indices):
    """구간 인덱스 -> 어휘 토큰 문자열. 정답에 그대로 쓴다."""
    idx = np.asarray(indices, dtype=np.int64).reshape(-1, N_AXES)
    out = []
    for row in idx:
        for k, axis in enumerate(AXES):
            out.append(f"<|traj_{axis}_{int(row[k]) - k * BINS}|>")
    return "".join(out)


TOKEN_RE = re.compile(r"<\|traj_([xyz])_(\d+)\|>")


def parse_tokens(text):
    """생성된 문자열에서 구간 인덱스를 뽑는다. 축이 어긋난 것은 버린다.

    모델은 초반에 아무 토큰이나 낸다. 축 순서(x, y, z)가 맞는 삼중항만 취해서,
    한 축이 빠졌을 때 뒤가 통째로 밀리는 일을 막는다.
    """
    got = TOKEN_RE.findall(text or "")
    idx, buf = [], []
    for axis, num in got:
        want = AXES[len(buf)]
        if axis != want:
            buf = []
            if axis != AXES[0]:
                continue
        buf.append(AXES.index(axis) * BINS + int(num))
        if len(buf) == N_AXES:
            idx.append(buf); buf = []
    return np.array(idx, dtype=np.int64).reshape(-1) if idx else np.zeros(0, np.int64)


def to_waypoints(text, horizons=(1.0, 2.0, 3.0)):
    """생성 문자열 -> {지평선: (x, y)}. 채점기가 쓰는 형태."""
    idx = parse_tokens(text)
    if len(idx) < N_AXES:
        return {}
    path = decode(idx)
    step = FUT_SECONDS / (FUT_POINTS - 1)
    out = {}
    for h in horizons:
        j = int(round(h / step))
        if j < len(path):
            out[h] = (float(path[j, 0]), float(path[j, 1]))
    return out


def render(text):
    """사람이 읽을 수 있게. 생성물 파일에 이 형태로도 남긴다."""
    w = to_waypoints(text)
    if not w:
        return "(궤적 토큰 없음)"
    return "; ".join(f"+{h:.0f}s ({x:+.1f}m, {y:+.1f}m)" for h, (x, y) in w.items())
