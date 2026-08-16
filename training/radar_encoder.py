#!/usr/bin/env python3
"""Lightweight transformer encoder for radar point clouds.

Deliberately not PointTransformerV3. PTv3's serialisation machinery exists to
make neighbourhood attention tractable on the 100k+ point clouds a lidar
produces. This radar returns a median of 733 points per scan, where full
attention is 733^2 = 537k pairs -- free on a B200. The serialisation buys
nothing and costs complexity, so this is plain attention over the points.

Shape of the stack:

  per-point embedding    8 channels -> d, with Fourier features on the geometry
                         because raw metres are badly conditioned next to
                         normalised RCS and Doppler
  spatial encoder        L layers of self-attention within each frame, optionally
                         with routed experts (see MoEMLP)
  frame resampler        K learned queries cross-attend to the points, giving a
                         fixed K tokens per frame regardless of point count,
                         plus a summed and a maxed token that attention cannot
                         produce
  temporal encoder       self-attention across the 20 frames
  readout                frame-aligned by default; see RadarEncoder

Resampling is the point of the design. 20 frames x 733 points is 14,655 points; a
language model cannot take that, and the token-budget analysis put the workable
radar allowance at roughly 256. Reduction has to happen inside the encoder, and
learned queries do it without the non-differentiable top-k that hard point
selection would need.

Pretraining heads, all supervised by signals that cost nothing to compute:

  moving/static     from the ego-compensated Doppler residual
  box class         which labelled object a point falls inside, if any. Sparse by
                    nature: 3.9% of points, and only vehicle classes carry enough
                    of them to learn from
  ego motion        speed, acceleration and yaw rate regressed from the points
                    alone, which forces the encoder to represent scene geometry
                    and its own motion rather than memorising clutter
  statistics        per-frame detection count, moving count and RCS extremes,
                    read off the tokens that leave the encoder

That last head is not an extra: without it no loss reaches the output at all.
`moving` and `box class` stop at the per-point features and `ego motion` at the
frame tokens, so the temporal stack and the readout took no gradient during
pretraining and shipped at their initialisation. Measured on the checkpoint that
resulted, a linear probe recovered a frame's detection count from the emitted
tokens at R^2 0.31, against 0.97 from a plain masked sum of the same points.

    python -m training.radar_encoder --self-test
"""

import argparse
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Taken from radar_data rather than restated, because the two drifted apart
# once already: the geometry channels below were written as the fixed indices
# [0, 1, 2, 7] of the old eight-channel layout, and dropping to six turned
# index 7 into an out-of-bounds read. Naming the channels makes a rename or a
# reorder a loud failure instead of a silent one.
# The channels both this rig and nuScenes can produce, so an encoder trained
# here can be tested there without the input representation changing.
#
# Dropped from the original eight:
#   z    nuScenes' ARS408 is a 2D radar -- measured over 49,599 returns, z is
#        0.000 for every one of them. Ours carries a real height (median
#        1.57 m), so keeping it would be a channel that says which dataset a
#        sample came from.
#   snr  ARS408 does not report it. It is not redundant here -- rcs and range
#        explain only R^2 0.471 of it, rcs alone 0.145 -- so this is a real
#        loss, taken deliberately in exchange for a transferable encoder.
#
# `range` is kept although it is exactly hypot(x, y) -- verified to 0.000 m
# over 630,693 returns. It costs one channel and saves a point-wise encoder
# from having to build a non-linear function of two inputs in its first layer,
# and both datasets compute it identically.
CHANNELS = ("x", "y", "range", "sin_az", "cos_az",
            "radial_velocity", "doppler_residual", "rcs")

_CHANNELS = CHANNELS
N_CHANNELS = len(_CHANNELS)

# What the sinusoidal expansion is applied to: the channels that are positions
# in metres. Doppler and rcs are not, and expanding them at eight octaves would
# spend the same capacity on quantities that need none.
# `sin_az`/`cos_az` are already bounded and periodic, so the sinusoidal
# expansion has nothing to add to them; only the metric channels get it.
GEOMETRY_CHANNELS = tuple(c for c in ("x", "y", "z", "range") if c in _CHANNELS)
GEOMETRY_INDEX = [_CHANNELS.index(c) for c in GEOMETRY_CHANNELS]
N_CLASSES = 12          # 11 named classes plus an "other" bucket
GEOMETRY_DIMS = len(GEOMETRY_CHANNELS)

# `nn.MultiheadAttention` has a fused inference path that reads out of bounds
# here. It is size-dependent, not data-dependent: the same sixteen validation
# clips pass at batch 8 and abort the CUDA context at batch 16, with no NaN, no
# infinity and no fully-padded row in the input. Measured on torch
# 2.10.0a0+b4e4ee81d3.nv25.12 with 20 frames of 1,024 points, dim 384, 8 heads,
# bfloat16, a bool `key_padding_mask`, and `eval()` -- the fast path only runs
# in eval, which is why training reached the first validation before dying.
#
# The failure surfaces as CUBLAS_STATUS_EXECUTION_FAILED or an illegal memory
# access, and it poisons the context, so every later call fails too and the
# traceback points wherever the process happened to be. Disabling the fast path
# fixes it; the slow path is the one training already uses.
torch.backends.mha.set_fastpath_enabled(False)

# Per-frame quantities the output tokens are asked to retain. Every one is
# computed from the input batch, so this costs no labels, and every one is
# something the instruction set asks for in words: `radar_probe` wants the
# detection count, the moving count and the strongest return; `desc_radar` wants
# the same three in a sentence.
STATISTICS = ("log_n_points", "log_n_moving", "max_rcs", "mean_rcs", "max_range")
LOG_SCALE = 7.0         # log1p(1024) = 6.93, so this maps a full scan to ~1.0

# Which front radar a clip carries. `none` exists because 17,130 clips have no
# front radar at all and still supply camera and ego motion.
SENSOR_IDS = {"lrr1": 0, "mrr2": 1, "srr0": 2, "none": 3}


# Heads used only during pretraining. A checkpoint written before one of them
# existed is still a valid encoder, so these may be missing on load; anything
# else missing means the architecture genuinely disagrees.
PRETRAIN_HEADS = ("head_moving", "head_class", "head_ego", "head_stats",
                  "head_bearing")

# The bird's-eye occupancy grid the emitted tokens are asked to fill in:
# twelve azimuth cells of 10 degrees by four range rings of 10 m. Defined in
# `radar_data` alongside the target that fills it.
OCC_CELLS = 12 * 4

# The polar readout's grid. 32 azimuth cells of 3.75 degrees by 8 range rings
# of 5 m, which is 256 tokens -- the same count the global readout emitted.
# 프레임마다 이 격자 하나씩. 10 슬롯 x 16 x 4 = 640 토큰.
#
# 예전에는 32 x 8 = 256 개를 스무 프레임 전체에 하나만 두었다. 공간은 촘촘했지만
# 시간이 통째로 사라졌다 -- `_polar_pool` 이 (B, F*P) 로 펼쳐서 모으는 바람에,
# 한 셀이 "지난 20초 동안 이 구역에 있던 모든 반환" 이 되었다. 프레임 위치
# 부호와 시간축 어텐션은 계산되고도 `polar` 경로에서 버려지고 있었다.
#
# 레이더가 카메라보다 유리한 것은 거리, 시선속도, 그리고 시간에 걸친 추적인데
# 셋째가 표현에서 제거되어 있었으니, 언어모델이 레이더 토큰을 무시한 것은
# 합리적인 선택이었다 -- 39개 측정점에서 섞은 레이더 대조군과의 비가 평균
# 1.10 이었다.
POLAR_AZ, POLAR_RANGE = 16, 4
POLAR_LIMIT = 60.0         # 전방 섹터의 반각
POLAR_MAX_RANGE = 40.0
RANGE_SCALE = 100.0        # `range` is normalised by this in radar_data


def encoder_kwargs(saved):
    """Constructor kwargs recovered from a checkpoint's saved `args`.

    The flag names and the constructor names differ (`--experts` against
    `n_experts`), and forgetting the mapping silently builds a dense encoder for
    a routed checkpoint -- which fails on load rather than quietly, only because
    `load_encoder_state` is strict.
    """
    mapping = {"readout": "readout", "frame_queries": "frame_queries",
               "experts": "n_experts", "dim": "dim", "frames": "n_frames"}
    return {arg: saved[key] for key, arg in mapping.items() if key in saved}


def load_encoder_state(encoder, state_dict):
    """Load weights, tolerating absent pretraining heads and nothing else."""
    missing, unexpected = encoder.load_state_dict(state_dict, strict=False)
    stray = [k for k in missing if not k.startswith(PRETRAIN_HEADS)]
    if stray or unexpected:
        raise RuntimeError(
            f"radar encoder checkpoint does not match the model: "
            f"missing {stray[:6]}, unexpected {list(unexpected)[:6]}")
    return [k for k in missing if k.startswith(PRETRAIN_HEADS)]


class FourierFeatures(nn.Module):
    """Sinusoidal expansion of the geometric channels.

    Positions arrive in normalised metres spanning a couple of orders of
    magnitude. A single linear layer resolves fine spatial structure poorly from
    that; sinusoids at several frequencies give the attention something to key on
    at every scale.
    """

    def __init__(self, n_freq=8):
        super().__init__()
        self.register_buffer("freqs", 2.0 ** torch.arange(n_freq) * math.pi)

    @property
    def out_dim(self):
        return GEOMETRY_DIMS * len(self.freqs) * 2

    def forward(self, geometry):
        scaled = geometry.unsqueeze(-1) * self.freqs
        return torch.cat([scaled.sin(), scaled.cos()], dim=-1).flatten(-2)


class MoEMLP(nn.Module):
    """Top-1 routed experts, with the sensor identity as a routing prior.

    The three front radars are not variations on one sensor: measured per scan,
    the SRR returns 205 points, the MRR 338 and the imaging LRR 638, over
    different unambiguous ranges and at different angular resolutions. One set of
    MLP weights has to average over all three, and since a rig carries either the
    SRR or the MRR+LRR pair and never both, the two populations are disjoint.

    Routing is learned rather than switched on the sensor id, so the experts can
    also split on something other than the sensor if the data prefers it -- but
    the id enters as a bias on the router logits, because making the router
    rediscover a fact already known for free wastes both capacity and training
    signal.
    """

    def __init__(self, dim, hidden, n_experts, n_sensors, dropout=0.0):
        super().__init__()
        self.n_experts = n_experts
        self.router = nn.Linear(dim, n_experts, bias=False)
        self.sensor_bias = nn.Embedding(n_sensors, n_experts)
        nn.init.zeros_(self.sensor_bias.weight)
        self.w_in = nn.Parameter(torch.empty(n_experts, dim, hidden))
        self.w_out = nn.Parameter(torch.empty(n_experts, hidden, dim))
        self.b_in = nn.Parameter(torch.zeros(n_experts, hidden))
        self.b_out = nn.Parameter(torch.zeros(n_experts, dim))
        for e in range(n_experts):
            nn.init.xavier_uniform_(self.w_in[e])
            nn.init.xavier_uniform_(self.w_out[e])
        self.dropout = nn.Dropout(dropout)
        self.aux_loss = None

    def forward(self, x, sensor=None):
        shape = x.shape
        flat = x.reshape(-1, shape[-1])
        logits = self.router(flat)
        if sensor is not None:
            # `sensor` is per sample; broadcast it over this sample's tokens.
            per_token = sensor.reshape(-1, 1).expand(
                -1, flat.shape[0] // sensor.numel()).reshape(-1)
            logits = logits + self.sensor_bias(per_token)
        probs = torch.softmax(logits.float(), dim=-1)
        chosen = probs.argmax(dim=-1)
        weight = probs.gather(1, chosen.unsqueeze(1)).squeeze(1).to(x.dtype)

        out = torch.zeros_like(flat)
        for e in range(self.n_experts):
            picked = chosen == e
            if not picked.any():
                continue
            taken = flat[picked]
            hidden = F.gelu(taken @ self.w_in[e] + self.b_in[e])
            out[picked] = hidden @ self.w_out[e] + self.b_out[e]
        # Scaling by the router probability is what gives the router a gradient
        # at all: the argmax itself is not differentiable.
        out = out * weight.unsqueeze(-1)

        # Switch-Transformer load balancing: without it one expert takes every
        # token and the others are dead weight.
        share = torch.zeros(self.n_experts, device=x.device, dtype=torch.float32)
        share.scatter_add_(0, chosen, torch.ones_like(chosen, dtype=torch.float32))
        share = share / max(chosen.numel(), 1)
        self.aux_loss = self.n_experts * (share * probs.mean(dim=0)).sum()
        return self.dropout(out.reshape(shape))


class Block(nn.Module):
    """Pre-norm transformer block. Self-attention when `memory` is None."""

    def __init__(self, dim, heads, mlp_ratio=4.0, dropout=0.0, n_experts=0,
                 n_sensors=len(SENSOR_IDS)):
        super().__init__()
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, dropout=dropout,
                                          batch_first=True)
        self.norm_mlp = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.n_experts = n_experts
        if n_experts:
            self.mlp = MoEMLP(dim, hidden, n_experts, n_sensors, dropout)
        else:
            self.mlp = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(),
                                     nn.Linear(hidden, dim))

    def forward(self, x, memory=None, key_padding_mask=None, sensor=None,
                attn_bias=None):
        q = self.norm_q(x)
        kv = q if memory is None else self.norm_kv(memory)
        # `attn_bias` is (B, queries, keys) and is repeated per head, which is
        # the shape `attn_mask` takes when it is float rather than boolean.
        mask = None
        if attn_bias is not None:
            heads = self.attn.num_heads
            # The bias has to carry the query's dtype -- the model runs in
            # bfloat16 and a float32 mask is rejected outright.
            mask = attn_bias.repeat_interleave(heads, dim=0).to(q.dtype)
        attended, _ = self.attn(q, kv, kv, need_weights=False,
                                key_padding_mask=key_padding_mask,
                                attn_mask=mask)
        x = x + attended
        normed = self.norm_mlp(x)
        return x + (self.mlp(normed, sensor) if self.n_experts
                    else self.mlp(normed))


class RadarEncoder(nn.Module):
    """`readout` decides what the language model receives.

    `global` was the original: 256 learned queries cross-attend to every frame's
    tokens at once. Measured with `training.probe_radar_tokens`, a linear readout
    of those 256 tokens recovers a frame's detection count at R^2 0.31 and its
    moving count at 0.26, against 0.68 and 0.78 one stage earlier and 0.97 from a
    plain masked sum of the point features. The queries are not tied to frames,
    so nothing in them marks which frame a number belongs to -- and every radar
    question in the instruction set names a frame.

    `frame` keeps the frame axis instead of pooling it away. Each frame emits
    `frame_queries` resampled tokens plus two computed ones, and they are handed
    over in frame order, so token f * (frame_queries + 2) + k belongs to frame f
    by construction rather than by whatever the queries learn.

    The two computed tokens exist because attention cannot produce them.
    Attention weights are a softmax and therefore sum to one, so its output is an
    average and is near-invariant to how many points were averaged; `sum` carries
    the cardinality, alongside log(1 + n) so the count survives even if the
    summed features saturate. `max` carries extremes: the strongest return in a
    scan is a max, not a mean, and it was unpredictable from every pooled
    representation measured (R^2 -0.29 from the global tokens, -0.01 even from
    the masked sum) while `radar_probe` asks for it outright.
    """

    def __init__(self, dim=384, heads=8, spatial_layers=4, temporal_layers=3,
                 frame_queries=48, global_queries=256, n_frames=20,
                 dropout=0.0, readout="global", n_experts=0, text_dim=0):
        super().__init__()
        if readout not in ("global", "frame", "polar", "polar_time"):
            raise ValueError(f"readout must be 'global', 'frame', 'polar' or "
                         f"'polar_time', "
                         f"got {readout}")
        self.fourier = FourierFeatures()
        self.embed = nn.Sequential(
            nn.Linear(N_CHANNELS + self.fourier.out_dim, dim), nn.GELU(),
            nn.Linear(dim, dim))

        # Experts sit in the per-point path only. That is where a sensor's
        # density and range distribution actually show up; the temporal stack
        # sees an already-resampled fixed-size summary where they do not.
        self.spatial = nn.ModuleList(
            Block(dim, heads, dropout=dropout, n_experts=n_experts)
            for _ in range(spatial_layers))

        self.frame_query = nn.Parameter(torch.randn(frame_queries, dim) * 0.02)
        self.frame_pool = Block(dim, heads, dropout=dropout)

        self.frame_pos = nn.Parameter(torch.randn(n_frames, dim) * 0.02)
        self.temporal = nn.ModuleList(
            Block(dim, heads, dropout=dropout) for _ in range(temporal_layers))

        self.readout = readout
        # 언어모델 임베딩으로 쏘는 투영. 사전학습에서 여기까지 학습해 두면
        # SFT 의 커넥터가 무에서 시작하지 않는다.
        self.text_proj = nn.Linear(dim, text_dim) if text_dim else None
        self.text_temperature = nn.Parameter(torch.tensor(float(np.log(0.07))))
        if readout == "global":
            self.global_query = nn.Parameter(torch.randn(global_queries, dim) * 0.02)
        if readout in ("polar", "polar_time"):
            # One query per (azimuth, range) cell, so "where" lives in the
            # token index instead of having to be encoded inside the vector.
            # 32 x 8 is 3.75 degrees by 5 m, and measured over 3,969 objects
            # only 3.2% of cells hold more than one -- against 12.0% at the
            # 12 x 4 grid, and the same 256 tokens either way.
            self.polar_query = nn.Parameter(
                torch.randn(POLAR_AZ * POLAR_RANGE, dim) * 0.02)
            # Attention is biased toward a query's own cell rather than
            # restricted to it. A hard mask puts an object at +9.9 degrees and
            # one at +10.1 into unrelated tokens; a bias lets the boundary
            # blur, which is what a real detector's anchors do.
            self.polar_bias = nn.Parameter(torch.tensor([2.0, 0.5]))
            self.polar_pool_block = Block(dim, heads, dropout=dropout)
            self.global_pool = Block(dim, heads, dropout=dropout)
        else:
            # +1 input for log(1 + n): the summed features alone confound "many
            # weak returns" with "few strong ones".
            self.sum_proj = nn.Linear(dim + 1, dim)
            self.max_proj = nn.Linear(dim, dim)
        self.norm_out = nn.LayerNorm(dim)

        # Pretraining heads, discarded once the encoder is attached to an LLM.
        self.head_moving = nn.Linear(dim, 2)
        self.head_class = nn.Linear(dim, N_CLASSES + 1)     # +1 for "no box"
        self.head_ego = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, 3))
        # Supervises the tokens the language model is actually handed. Without
        # it the temporal stack and the readout take no gradient during
        # pretraining at all -- `losses` reached only `per_point` and
        # `frame_tokens`, so the encoder shipped a randomly initialised temporal
        # mixer and, under `global` readout, a random projection on top of it.
        self.head_stats = nn.Sequential(nn.LayerNorm(dim),
                                        nn.Linear(dim, len(STATISTICS)))
        # Where the returns are, not just how many. Nothing in the other four
        # objectives asks for that: `moving` is decided by a Doppler residual,
        # `box_class` by rcs and local shape, `ego` and `stats` are per-frame
        # scalars. Measured with a linear probe on the trained encoder,
        # recovering the azimuth histogram of the returns -- a quantity
        # computed directly from the input, so a perfect encoder scores 1.0 --
        # gave R^2 0.717 from the point features and 0.517 from the tokens the
        # language model receives. The bearing survives, but nothing had asked
        # it to, and detection is exactly the task that needs it.
        #
        # The label costs nothing: it is a histogram of atan2(y, x) over the
        # returns already in the batch.
        self.head_bearing = nn.Sequential(nn.LayerNorm(dim),
                                          nn.Linear(dim, OCC_CELLS))

        self.dim = dim
        self.frame_queries = frame_queries
        self.global_queries = global_queries
        self.n_frames = n_frames
        self.n_experts = n_experts

    @property
    def n_tokens(self):
        """How many tokens reach the language model. The prompt has to reserve
        exactly this many radar placeholders."""
        if self.readout == "polar":
            return POLAR_AZ * POLAR_RANGE
        if self.readout == "polar_time":
            return self.n_frames * POLAR_AZ * POLAR_RANGE
        if self.readout == "global":
            return self.global_queries
        return self.n_frames * (self.frame_queries + 2)

    def _statistic_tokens(self, per_point, mask):
        """Masked sum (with count) and max over the points of each frame."""
        weights = mask.unsqueeze(-1).to(per_point.dtype)
        summed = (per_point * weights).sum(dim=2)
        count = mask.sum(dim=2, keepdim=False).to(per_point.dtype)
        # /64 keeps the summed features near unit scale at the median of ~640
        # returns without clipping the tail at 1,024.
        sum_token = self.sum_proj(
            torch.cat([summed / 64.0, torch.log1p(count).unsqueeze(-1)], dim=-1))

        filled = per_point.masked_fill(~mask.unsqueeze(-1), torch.finfo(
            per_point.dtype).min)
        maxed = filled.max(dim=2).values
        # A frame with no returns would otherwise carry dtype-min everywhere.
        maxed = torch.where(mask.any(dim=2, keepdim=True),
                            maxed, torch.zeros_like(maxed))
        return sum_token.unsqueeze(2), self.max_proj(maxed).unsqueeze(2)

    def router_loss(self):
        """Mean load-balancing penalty over the routed blocks, 0 without them."""
        losses = [b.mlp.aux_loss for b in self.spatial
                  if b.n_experts and b.mlp.aux_loss is not None]
        if not losses:
            return None
        return torch.stack(losses).mean()

    def encode_points(self, points, mask, sensor=None):
        """Per-point features after the spatial layers. (B, F, P, dim)"""
        B, Fr, P, _ = points.shape
        geometry = torch.stack([points[..., k] for k in GEOMETRY_INDEX], dim=-1)
        x = self.embed(torch.cat([points, self.fourier(geometry)], dim=-1))

        x = x.reshape(B * Fr, P, self.dim)
        # MultiheadAttention wants True where a key should be ignored.
        pad = ~mask.reshape(B * Fr, P)
        # A frame with no returns would make every key padded and produce NaNs;
        # let such rows attend to themselves and mask their loss instead.
        empty = pad.all(dim=1)
        pad = pad.clone()
        pad[empty] = False
        for block in self.spatial:
            x = block(x, key_padding_mask=pad, sensor=sensor)
        return x.reshape(B, Fr, P, self.dim), pad.reshape(B, Fr, P), empty

    def forward(self, points, mask, sensor=None):
        B, Fr, P, _ = points.shape
        per_point, pad, _ = self.encode_points(points, mask, sensor)

        queries = self.frame_query.expand(B * Fr, -1, -1)
        frame_tokens = self.frame_pool(
            queries, memory=per_point.reshape(B * Fr, P, self.dim),
            key_padding_mask=pad.reshape(B * Fr, P))
        frame_tokens = frame_tokens.reshape(B, Fr, self.frame_queries, self.dim)

        if self.readout == "frame":
            summed, maxed = self._statistic_tokens(per_point, mask)
            frame_tokens = torch.cat([frame_tokens, summed, maxed], dim=2)

        # Flatten frames and queries for temporal attention, adding a per-frame
        # positional code so the model can tell time order.
        tokens = (frame_tokens + self.frame_pos[None, :Fr, None, :])
        tokens = tokens.reshape(B, Fr * frame_tokens.shape[2], self.dim)
        for block in self.temporal:
            tokens = block(tokens)

        if self.readout == "global":
            tokens = self.global_pool(self.global_query.expand(B, -1, -1),
                                      memory=tokens)
        elif self.readout == "polar":
            tokens = self._polar_pool(points, mask, per_point)
        elif self.readout == "polar_time":
            tokens = self._polar_pool_time(points, mask, per_point)
        return {"tokens": self.norm_out(tokens),
                "per_point": per_point,
                "frame_tokens": frame_tokens}

    def _polar_pool(self, points, mask, per_point):
        """Pool the points into a polar grid, softly.

        Each query owns a cell centre; the attention logit over a point is
        lowered by how far that point sits from the centre, in degrees and in
        metres. With the two coefficients learnable the model can widen or
        narrow its cells, and `softplus` keeps them from going negative and
        turning the bias into an attractor for far points.
        """
        B, Fr, P, _ = points.shape
        idx = _CHANNELS.index
        az = torch.rad2deg(torch.atan2(points[..., idx("y")],
                                       points[..., idx("x")]))
        rng = points[..., idx("range")] * RANGE_SCALE
        flat_az = az.reshape(B, Fr * P)
        flat_rng = rng.reshape(B, Fr * P)
        memory = per_point.reshape(B, Fr * P, self.dim)
        keep = mask.reshape(B, Fr * P)

        centres_az = torch.linspace(-POLAR_LIMIT, POLAR_LIMIT,
                                    POLAR_AZ + 1, device=points.device)
        centres_az = (centres_az[:-1] + centres_az[1:]) / 2
        step = POLAR_MAX_RANGE / POLAR_RANGE
        centres_rng = torch.arange(POLAR_RANGE, device=points.device) * step + step / 2
        ca = centres_az.repeat_interleave(POLAR_RANGE)
        cr = centres_rng.repeat(POLAR_AZ)

        alpha, beta = F.softplus(self.polar_bias) + 1e-3
        bias = -(alpha * (flat_az[:, None, :] - ca[None, :, None]).abs()
                 + beta * (flat_rng[:, None, :] - cr[None, :, None]).abs())
        bias = bias.masked_fill(~keep[:, None, :], float("-inf"))
        # A query whose cell is empty would see every key masked; let it fall
        # back to the whole scan rather than producing NaNs.
        dead = torch.isinf(bias).all(dim=-1, keepdim=True)
        bias = torch.where(dead, torch.zeros_like(bias), bias)
        q = self.polar_query.expand(B, -1, -1)
        return self.polar_pool_block(q, memory=memory, attn_bias=bias)

    def _polar_pool_time(self, points, mask, per_point):
        """프레임마다 극좌표 격자 하나. 슬롯에 시간 부호를 붙이고 시간축을 잇는다.

        `_polar_pool` 과 다른 점은 프레임을 펼치지 않는다는 것뿐이다. 질의가
        자기 프레임 안에서만 모으므로 "몇 초 시점, 어느 방향, 어느 거리" 가
        토큰 하나에 들어간다. 어텐션 편향도 (B*F, 격자, P) 라 예전의
        (B, 격자, F*P) 보다 F 배 작다.
        """
        B, Fr, P, _ = points.shape
        idx = _CHANNELS.index
        az = torch.rad2deg(torch.atan2(points[..., idx("y")],
                                       points[..., idx("x")]))
        rng = points[..., idx("range")] * RANGE_SCALE

        flat_az = az.reshape(B * Fr, P)
        flat_rng = rng.reshape(B * Fr, P)
        memory = per_point.reshape(B * Fr, P, self.dim)
        keep = mask.reshape(B * Fr, P)

        centres_az = torch.linspace(-POLAR_LIMIT, POLAR_LIMIT,
                                    POLAR_AZ + 1, device=points.device)
        centres_az = (centres_az[:-1] + centres_az[1:]) / 2
        step = POLAR_MAX_RANGE / POLAR_RANGE
        centres_rng = torch.arange(POLAR_RANGE, device=points.device) * step + step / 2
        ca = centres_az.repeat_interleave(POLAR_RANGE)
        cr = centres_rng.repeat(POLAR_AZ)

        alpha, beta = F.softplus(self.polar_bias) + 1e-3
        bias = -(alpha * (flat_az[:, None, :] - ca[None, :, None]).abs()
                 + beta * (flat_rng[:, None, :] - cr[None, :, None]).abs())
        bias = bias.masked_fill(~keep[:, None, :], float("-inf"))
        dead = torch.isinf(bias).all(dim=-1, keepdim=True)
        bias = torch.where(dead, torch.zeros_like(bias), bias)

        cells = POLAR_AZ * POLAR_RANGE
        q = self.polar_query.expand(B * Fr, -1, -1)
        tokens = self.polar_pool_block(q, memory=memory, attn_bias=bias)
        tokens = tokens.reshape(B, Fr, cells, self.dim)
        # 시간 부호는 여기서 붙는다. 이 한 줄이 없으면 열 개의 슬롯이 서로
        # 구별되지 않아, 격자를 프레임마다 나눈 의미가 없어진다.
        tokens = tokens + self.frame_pos[None, :Fr, None, :]
        tokens = tokens.reshape(B, Fr * cells, self.dim)
        for block in self.temporal:
            tokens = block(tokens)
        return tokens

    def text_anchor_loss(self, tokens, batch, anchors, anchor_of_class):
        """격자 칸의 토큰을, 그 칸에 있는 물체의 *언어모델 임베딩* 쪽으로.

        지금까지의 다섯 목표는 전부 "레이더에서 라벨을 맞혀라" 였다. 그것은
        인코더 *내부*를 정보 있게 만들지만, 토큰이 언어모델에게 읽히는지와는
        무관하다 -- 그 다리는 SFT 가 커넥터로 놓아야 하는데, SFT 손실은 카메라만
        보고도 최소화되므로 다리를 놓을 이유가 없다. 측정: 39개 측정점에서
        섞은 레이더 대조군과의 비가 평균 1.10.

        그래서 목표를 Qwen 의 입력 임베딩으로 직접 잡는다. `automobile` 이라는
        낱말이 실제로 놓인 자리로 칸을 끌어당기면, 커넥터는 SFT 시작 시점에
        이미 절반쯤 놓여 있다.

        mmLIP(NeurIPS 2026 투고) 이 CLIP 텍스트 공간을 거쳐 배치 내 InfoNCE 로
        하는 것과 다른 점이 둘 있다. 하나는 중간 공간을 거치지 않는 것이고,
        (그쪽은 LLM 이 동결이라 사전 정렬이 결정적이지만 우리는 함께 학습한다)
        다른 하나는 짝을 추정하지 않는다는 것이다 -- 점마다 어느 상자에 속하는지
        라벨이 있으므로, 어느 칸이 어느 낱말과 짝인지 우리는 알고 있다. 그쪽의
        신뢰도 재가중(PCL)이 추정하려는 "클러터" 도 여기서는 상자 밖 점이라는
        라벨 그 자체다.
        """
        B, T, _ = tokens.shape
        cells = POLAR_AZ * POLAR_RANGE
        Fr = T // cells
        # 칸마다 그 안의 점들이 속한 상자의 클래스. 없으면 "빈 칸".
        idx = _CHANNELS.index
        points, mask = batch["points"], batch["mask"]
        az = torch.rad2deg(torch.atan2(points[..., idx("y")], points[..., idx("x")]))
        rng = points[..., idx("range")] * RANGE_SCALE
        a_bin = ((az + POLAR_LIMIT) / (2 * POLAR_LIMIT) * POLAR_AZ).long()
        r_bin = (rng / POLAR_MAX_RANGE * POLAR_RANGE).long()
        inside = (mask & (a_bin >= 0) & (a_bin < POLAR_AZ)
                  & (r_bin >= 0) & (r_bin < POLAR_RANGE))
        cell = (a_bin.clamp(0, POLAR_AZ - 1) * POLAR_RANGE
                + r_bin.clamp(0, POLAR_RANGE - 1))

        target = torch.full((B, Fr, cells), len(anchor_of_class),
                            dtype=torch.long, device=tokens.device)
        cls = batch["box_class"]
        has_box = inside & (cls >= 0)
        if has_box.any():
            b_i, f_i, p_i = has_box.nonzero(as_tuple=True)
            f_i = f_i.clamp(max=Fr - 1)
            target[b_i, f_i, cell[b_i, f_i, p_i]] = anchor_of_class[
                cls[b_i, f_i, p_i]]

        z = F.normalize(self.text_proj(tokens), dim=-1)
        a = F.normalize(anchors, dim=-1)
        logits = z @ a.t() / self.text_temperature.exp().clamp(max=100.0)
        return F.cross_entropy(logits.reshape(-1, a.shape[0]),
                               target.reshape(-1))

    def losses(self, batch):
        points, mask = batch["points"], batch["mask"]
        out = self.forward(points, mask, batch.get("sensor"))
        per_point = out["per_point"]
        valid = mask

        losses = {}
        moving_logits = self.head_moving(per_point)
        losses["moving"] = F.cross_entropy(
            moving_logits[valid], batch["is_moving"][valid])

        # Points inside no box are the overwhelming majority, so they are folded
        # into a dedicated class rather than dropped; otherwise 3.9% of returns
        # would carry the entire gradient for this head.
        target = batch["box_class"].clone()
        target[target < 0] = N_CLASSES
        losses["box_class"] = F.cross_entropy(
            self.head_class(per_point)[valid], target[valid])

        # Ego motion from the frame tokens: mean-pool each frame, regress the
        # three channels. Frames with no returns are excluded.
        frame_repr = out["frame_tokens"].mean(dim=2)
        has_points = valid.any(dim=2)
        predicted = self.head_ego(frame_repr)
        if has_points.any():
            losses["ego"] = F.smooth_l1_loss(predicted[has_points],
                                             batch["ego_state"][has_points])
        else:
            losses["ego"] = predicted.sum() * 0.0

        losses["stats"] = self._statistics_loss(out["tokens"], batch, has_points)
        losses["bearing"] = self._bearing_loss(out["tokens"], batch, has_points)
        return out, losses

    def _bearing_loss(self, tokens, batch, has_points):
        """Make the emitted tokens say where the objects are.

        Binary cross-entropy against a coarse bird's-eye occupancy grid -- did
        a labelled road user fall in this azimuth cell at this range ring.

        The first version of this asked for the bearing histogram of the
        *returns*, which needed no label at all. It also taught nothing:
        predicting the dataset-average histogram already costs 2.064 against a
        floor of 2.019, because most returns are road and clutter and their
        distribution barely moves between scenes. The objects do move -- the
        same spread over their bearings is 1.264, twenty-seven times wider --
        so the target is the objects.

        Defined for both readouts, unlike `_statistics_loss`. `global` pools
        the frame axis away, but "where are the road users over this window" is
        still well posed, and it is the question detection asks.
        """
        occ = batch.get("occupancy")
        if occ is None or not has_points.any():
            return tokens.sum() * 0.0
        # Over the window, so it matches what a pooled readout can answer.
        target = (occ.sum(dim=1) > 0).to(torch.float32)
        predicted = self.head_bearing(tokens.mean(dim=1)).float()
        return F.binary_cross_entropy_with_logits(predicted, target)

    def _statistics_loss(self, tokens, batch, has_points):
        """Force the emitted tokens to retain per-frame radar statistics.

        Only defined for the frame-aligned readout: `global` pools the frame axis
        away, so there is no token group to attribute a frame's statistics to --
        which is the reason that readout could not answer "at frame N" questions
        in the first place.
        """
        if self.readout != "frame":
            return tokens.sum() * 0.0

        points, mask = batch["points"], batch["mask"]
        B, Fr, _ = mask.shape
        per_frame = tokens.reshape(B, Fr, -1, self.dim).mean(dim=2)

        weights = mask.to(points.dtype)
        count = weights.sum(dim=2)
        moving = (batch["is_moving"] * mask).sum(dim=2).to(points.dtype)
        rcs = points[..., _CHANNELS.index("rcs")]
        rng = points[..., _CHANNELS.index("range")]
        floor = torch.finfo(points.dtype).min
        target = torch.stack([
            torch.log1p(count) / LOG_SCALE,
            torch.log1p(moving) / LOG_SCALE,
            rcs.masked_fill(~mask, floor).max(dim=2).values,
            (rcs * weights).sum(dim=2) / count.clamp(min=1.0),
            rng.masked_fill(~mask, floor).max(dim=2).values,
        ], dim=-1)

        predicted = self.head_stats(per_frame)
        if not has_points.any():
            return predicted.sum() * 0.0
        return F.smooth_l1_loss(predicted[has_points], target[has_points])


def self_test():
    torch.manual_seed(0)
    B, Fr, P = 2, 20, 700
    model = RadarEncoder()
    points = torch.randn(B, Fr, P, N_CHANNELS)
    mask = torch.rand(B, Fr, P) > 0.3
    mask[0, 3] = False                      # an empty frame, to exercise the guard
    batch = {"points": points, "mask": mask,
             "is_moving": (torch.rand(B, Fr, P) > 0.9).long(),
             "box_class": torch.where(torch.rand(B, Fr, P) > 0.96,
                                      torch.randint(0, 11, (B, Fr, P)),
                                      torch.full((B, Fr, P), -1)),
             "ego_state": torch.randn(B, Fr, 3)}
    out, losses = model.losses(batch)
    params = sum(p.numel() for p in model.parameters())
    print(f"parameters       {params/1e6:.1f} M")
    print(f"tokens out       {tuple(out['tokens'].shape)}")
    print(f"frame tokens     {tuple(out['frame_tokens'].shape)}")
    for name, value in losses.items():
        print(f"loss {name:11s} {value.item():.4f}  finite={torch.isfinite(value).item()}")
    total = sum(losses.values())
    total.backward()
    grads = sum(1 for p in model.parameters() if p.grad is not None)
    print(f"backward OK, {grads} tensors received gradients")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
