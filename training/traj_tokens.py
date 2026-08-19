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

# 축마다 다른 범위. 150 클립 12,000 개 델타(0.1초)에서 잰 값에 여유를 둔 것이다.
# x 는 앞으로만 가므로 음수 쪽이 거의 없고, y 는 대칭이며, z 는 노면 굴곡뿐이다.
AXIS_RANGE = {
    "x": (-0.5, 3.5),       # 실측 -0.052 ~ +3.088 (0.1초에 최대 31 m/s)
    "y": (-1.0, 1.0),       # 실측 -0.724 ~ +0.583
    "z": (-0.2, 0.2),       # 실측 -0.095 ~ +0.073, 노면 굴곡뿐이다
}
AXES = ("x", "y", "z")

# 자리표시자. 이 토큰이 프롬프트에 TOKENS_PER_HISTORY 개 깔리고, 붙이는 쪽이
# 그 자리의 id 를 아래 궤적 토큰으로 바꾼다.
PAD_TOKEN = "<|traj_pad|>"


def traj_tokens():
    """어휘에 추가할 궤적 토큰 이름. 축마다 BINS 개씩."""
    return [f"<|traj_{a}_{i}|>" for a in AXES for i in range(BINS)]


def encode(history_xyz):
    """(HIST_POINTS, 3) 자차 좌표계 이력 -> (TOKENS_PER_HISTORY,) 구간 인덱스.

    반환값은 축별 코드북 안의 위치다. 어휘 id 로 바꾸는 것은 부르는 쪽의 몫이고,
    그래야 이 함수가 토크나이저를 몰라도 된다.
    """
    xyz = np.asarray(history_xyz, dtype=np.float64)
    if xyz.shape != (HIST_POINTS, N_AXES):
        raise ValueError(f"이력은 ({HIST_POINTS}, {N_AXES}) 여야 한다: {xyz.shape}")
    delta = np.diff(xyz, axis=0)                     # (16, 3)
    out = np.zeros(delta.shape, dtype=np.int64)
    for k, axis in enumerate(AXES):
        lo, hi = AXIS_RANGE[axis]
        v = (delta[:, k] - lo) / (hi - lo)
        # 범위 밖은 자른다. 버리면 토큰 수가 달라져 자리표시자와 어긋난다.
        out[:, k] = np.clip(np.round(v * (BINS - 1)), 0, BINS - 1)
        out[:, k] += k * BINS                        # 축별 코드북으로 옮긴다
    return out.reshape(-1)                           # 점 순서, 점마다 x y z


def decode(indices):
    """구간 인덱스 -> 자차 좌표계 이력. 되돌려 보고 확인하는 용도."""
    idx = np.asarray(indices, dtype=np.int64).reshape(HIST_POINTS - 1, N_AXES)
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
