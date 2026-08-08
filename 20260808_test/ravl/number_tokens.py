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

이 사본은 읽기 전용이다. 어휘를 다시 만드는 --scan 과 확인용 --check 는 원본
저장소에만 있다 -- 번들 안에서 어휘를 다시 만들면 학습 때와 순서가 달라져
151,672 이후의 모든 토큰 id 가 밀린다.
"""

import os
import re
import sys

import numpy as np
import torch

NUMBER = re.compile(r"-?\d+(?:\.\d+)?")
VOCAB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "number_vocab.txt")


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
    values = np.array([float(x) for x in literals], dtype=np.float64)
    table = model.get_input_embeddings().weight
    init = value_embeddings(values, table.shape[1],
                            scale=float(table.detach().float().std()))
    with torch.no_grad():
        table[torch.tensor(ids)] = init.to(table.dtype).to(table.device)
        out = model.get_output_embeddings()
        if out is not None and out.weight is not table:
            out.weight[torch.tensor(ids)] = init.to(out.weight.dtype).to(out.weight.device)
    return added


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
