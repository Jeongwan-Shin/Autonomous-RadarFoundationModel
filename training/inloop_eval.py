#!/usr/bin/env python3
"""학습 루프 안에서 재는 평가기.

별도 프로세스로 재던 것을 안으로 들인다. 이유는 메모리다 -- 밖에서 재면
17.7 GB 짜리 모델을 한 벌 더 올려야 하고, 그래서 GPU 하나를 통째로 비워
두거나(자원의 20% 유휴) 같이 쓰다가 메모리 부족으로 죽었다. 매 측정점마다
모델 적재에 2~3분을 쓰는 것도 그 구조의 대가였다.

안에서 재면 모델이 이미 올라가 있다. 추가 메모리는 KV 캐시뿐이고, 학습을
잠깐 멈추는 대신 다섯 랭크가 항목을 나눠 생성한다 -- 밖에서 하던 5-샤딩과
같은 병렬성이되 적재도 재시도도 없다.

바깥 평가기(`watch_eval`)는 그대로 둔다. 끝난 체크포인트를 다시 재거나,
이 경로가 같은 숫자를 내는지 대조할 때 필요하다 -- 채점은 지금까지 네 번
틀렸고, 두 경로가 다른 값을 내면 그 자체가 신호다.
"""

import contextlib
import json
import os
import time

import numpy as np
import torch
import torch.distributed as dist

from training.traj_tokens import decode_answer


class InLoopEvaluator:
    """학습 중간에 몇 개 태스크를 재고 추세 파일에 한 줄 더한다.

    한 번 만들어 두고 `maybe_run(step)` 만 부르면 된다. 데이터셋은 처음
    한 번만 짓는다 -- 파케를 다시 읽으면 20분이고, 그것이 매 측정점마다
    반복되면 평가가 학습보다 비싸진다.
    """

    TASKS = ("det_objects_azdeg", "det_objects_3dbbox", "plan_ego_xy")

    def __init__(self, *, every, items, out_dir, processor, tokenizer,
                 n_frames, radar_tokens, radar_frames, max_length,
                 plan_samples=6, split="test", rank=0, world=1, log=print):
        self.every = every
        self.items = items
        self.out_dir = out_dir
        self.processor = processor
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.plan_samples = plan_samples
        self.rank = rank
        self.world = world
        self.log = log
        self._loaders = None
        self._cfg = dict(n_frames=n_frames, radar_tokens=radar_tokens,
                         radar_frames=radar_frames, split=split)

    # ------------------------------------------------------------------ 준비
    def _build(self):
        """태스크마다 이 랭크가 맡을 항목만 담은 데이터셋."""
        from training.instruct_data import InstructDataset, build_collate
        from torch.utils.data import DataLoader

        collate = build_collate(self.processor, self.tokenizer, self.max_length)
        out = {}
        for task in self.TASKS:
            ds = InstructDataset(
                tasks=(task,), split=self._cfg["split"],
                processor=self.processor, tokenizer=self.tokenizer,
                n_frames=self._cfg["n_frames"],
                radar_tokens=self._cfg["radar_tokens"],
                radar_frames=self._cfg["radar_frames"],
                samples=self.items, all_profiles=True,
                radar_dropout=0.0, camera_dropout=0.0)
            # 랭크마다 자기 몫만. 배치는 1 그대로라 항목별 논리가 바뀌지 않는다.
            ds.items = ds.items[self.rank::self.world]
            if not len(ds):
                continue
            out[task] = DataLoader(ds, batch_size=1, shuffle=False,
                                   num_workers=2, collate_fn=collate)
        self._loaders = out
        n = sum(len(d.dataset) for d in out.values())
        self.log(f"루프 안 평가 준비: 태스크 {len(out)}종, 이 랭크가 {n}건")

    # ------------------------------------------------------------------ 실행
    def maybe_run(self, step, llm, encoder, connector, device):
        """`every` 의 배수일 때만 잰다. 아니면 아무 일도 하지 않는다."""
        if not self.every or step % self.every or step == 0:
            return None
        if self._loaders is None:
            self._build()
        started = time.monotonic()
        was_training = llm.training
        llm.eval(); encoder.eval(); connector.eval()
        try:
            with self._whole(llm):
                rows = self._generate(llm, encoder, connector, device)
        finally:
            if was_training:
                llm.train(); encoder.train(); connector.train()
        rows = self._gather(rows)
        if self.rank != 0:
            return None
        row = self._score(rows)
        row.update(step=step, seconds=round(time.monotonic() - started, 1))
        self._write(step, row, rows)
        return row

    @contextlib.contextmanager
    def _whole(self, llm):
        """평가 동안만 파라미터를 모아 랭크마다 완전한 사본을 만든다.

        샤딩된 채로 생성하면 토큰 하나마다 층마다 all-gather 가 일어나는데,
        그것은 다섯 랭크가 함께 참여해야 하는 집합 통신이다. 랭크마다 답
        길이가 다르므로(48 토큰에서 끝나는 것과 240 까지 가는 것) 먼저 끝난
        랭크가 빠지면 나머지가 영원히 기다린다 -- 실제로 10분 타임아웃까지
        갔다. 모아 두면 통신이 아예 없어져 길이가 달라도 상관없다.

        자리는 학습을 멈춘 덕에 난다. 활성값과 기울기가 풀리고, 캐시에 예약만
        되어 있던 블록(측정된 적이 있다: 33.79 GiB)을 돌려주면 랭크당 17.7 GB
        를 놓을 자리가 생긴다.
        """
        from torch.distributed.fsdp import FSDPModule
        mods = [m for m in llm.modules() if isinstance(m, FSDPModule)]
        torch.cuda.empty_cache()
        free_before = torch.cuda.mem_get_info()[0] / 1024 ** 3
        for m in mods:
            m.unshard()
        self.log(f"  파라미터 모음: FSDP 모듈 {len(mods)}개 · "
                 f"여유 {free_before:.1f} -> "
                 f"{torch.cuda.mem_get_info()[0] / 1024 ** 3:.1f} GiB")
        try:
            yield
        finally:
            for m in mods:
                m.reshard()
            torch.cuda.empty_cache()

    def _generate(self, llm, encoder, connector, device):
        from training.eval_all_tasks import MAX_NEW
        from training.train_vlm import RadarInjector

        core = connector.module if hasattr(connector, "module") else connector
        injector = RadarInjector(llm.get_input_embeddings(),
                                 self.tokenizer.convert_tokens_to_ids("<|radar_pad|>"))
        header = self.tokenizer("<|im_start|>assistant\n",
                                add_special_tokens=False)["input_ids"]
        rows = []
        try:
            for task, loader in self._loaders.items():
                budget = MAX_NEW.get(task, 48)
                for batch in loader:
                    rows.append(self._one(task, batch, budget, llm, encoder,
                                          core, injector, header, device))
        finally:
            injector.remove()
        return [r for r in rows if r]

    def _one(self, task, batch, budget, llm, encoder, core, injector, header,
             device):
        points = batch.pop("points").to(device, torch.bfloat16)
        radar_mask = batch.pop("radar_mask").to(device)
        sensor = batch.pop("sensor", None)
        if sensor is not None:
            sensor = sensor.to(device)
        batch.pop("task", None)
        ids = batch["input_ids"][0].tolist()
        cut = None
        for i in range(len(ids) - len(header) + 1):
            if ids[i:i + len(header)] == header:
                cut = i + len(header)
        if cut is None:
            return None
        reference = decode_answer(self.tokenizer, ids[cut:])
        prompt = {k: v[:, :cut].to(device) for k, v in batch.items()
                  if torch.is_tensor(v) and v.dim() == 2 and k != "labels"}
        for k, v in batch.items():
            if torch.is_tensor(v) and k not in prompt and k != "labels":
                prompt[k] = v.to(device)

        with torch.no_grad():
            radar = encoder(points, radar_mask, sensor)
            injector.pending = core(radar["tokens"])
            got = llm.generate(**prompt, max_new_tokens=budget, do_sample=False,
                               pad_token_id=self.tokenizer.pad_token_id
                               or self.tokenizer.eos_token_id)
            text = decode_answer(self.tokenizer, got[0, cut:])

            samples = []
            if self.plan_samples > 1 and task.startswith("plan_ego_xy"):
                injector.pending = core(radar["tokens"])
                many = llm.generate(**prompt, max_new_tokens=budget,
                                    do_sample=True, temperature=1.0, top_p=0.95,
                                    num_return_sequences=self.plan_samples,
                                    pad_token_id=self.tokenizer.pad_token_id
                                    or self.tokenizer.eos_token_id)
                samples = [decode_answer(self.tokenizer, many[i, cut:])
                           for i in range(many.shape[0])]
        return {"task": task, "mode": "full", "generated": text.strip(),
                "reference": reference, "samples": samples or None}

    def _gather(self, rows):
        """랭크마다 만든 것을 0 으로 모은다."""
        if self.world == 1 or not dist.is_initialized():
            return rows
        bucket = [None] * self.world
        dist.all_gather_object(bucket, rows)
        return [r for part in bucket for r in (part or [])]

    def _score(self, rows):
        from training.watch_eval import score_generations
        return score_generations(rows)

    def _write(self, step, row, rows):
        os.makedirs(self.out_dir, exist_ok=True)
        with open(os.path.join(self.out_dir, "trend.jsonl"), "a") as fh:
            fh.write(json.dumps(row) + "\n")
        gen_dir = os.path.join(self.out_dir, "generations")
        os.makedirs(gen_dir, exist_ok=True)
        with open(os.path.join(gen_dir, f"step{step:06d}.jsonl"), "w") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        self.log(f"step {step:>6}  det F1 {_f(row.get('det_f1'))}  "
                 f"3D F1 {_f(row.get('box_f1'))}  L2 {_f(row.get('l2'))}  "
                 f"({row['seconds']:.0f}s)")


def _f(v):
    return "—" if v is None else f"{v:.3f}"
