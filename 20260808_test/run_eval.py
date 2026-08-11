#!/usr/bin/env python3
"""번들만으로 평가한다 -- 원본 아카이브도 파케이도 없이.

`data/` 안에 아이템마다 모델이 실제로 받는 것이 들어 있다. 프레임 JPEG, 레이더
반사점, 프롬프트 전문, 그리고 정답. 이 스크립트는 그것을 읽어 생성하고 태스크에
맞는 채점기로 점수를 낸다.

체크포인트를 풀어 두면 인자 없이 돈다.

    tar xzf model_8b_v9_6ch_step2200.tar.gz
    python run_eval.py

가중치는 체크포인트 것을 쓰고, 토크나이저와 프로세서 설정만 `base/` 에서
읽는다. 그래서 원본 17 GB 를 따로 받을 필요가 없다.

레이더는 학습 때와 같은 방식으로 주입한다. `inputs_embeds` 로 넘기면 Qwen3-VL 의
비디오 스캐터가 꺼지므로, 입력 임베딩 모듈에 forward hook 을 걸어 자리표시자 행만
덮어쓴다. 이 부분이 학습 코드와 달라지면 점수가 조용히 무너진다.
"""

import argparse
import glob
import io
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from ravl.task_scorers import scorer_for, split_rationale, summarise

# 이 두 표는 저장소의 살아 있는 값에서 생성된다 -- `export_items.py` 가
# 빌드할 때마다 `ravl/eval_config.py` 를 다시 쓴다. 번들이 자기 사본을 들고
# 있었더니 어긋났다: `agent_traj_xy` 가 표에 없어 기본값 48 을 받았고, 그러면
# `agent_traj_xy_cot` 도 파생되지 않아 근거까지 써야 하는 답이 48 토큰에서
# 잘린다. 그 실패는 "못 푸는 모델" 과 똑같이 보인다.
from ravl.eval_config import MAX_NEW, SYSTEM

# CoT 는 근거를 먼저 쓰므로 평문의 +640 토큰을 받는다.
for _p, _b in list(MAX_NEW.items()):
    MAX_NEW.setdefault(f"{_p}_cot", _b + 640)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class RadarInjector:
    """Overwrite the radar placeholder rows in the embedding output.

    Same as training. Passing `inputs_embeds` instead would disable the video
    scatter that runs afterwards -- Qwen3VLModel.forward takes one or the other.
    """

    def __init__(self, embed_module, pad_token_id):
        self.pad_token_id = pad_token_id
        self.pending = None
        self.handle = embed_module.register_forward_hook(self._hook)

    def _hook(self, module, args, output):
        if self.pending is None:
            return output
        input_ids, radar = args[0], self.pending
        self.pending = None
        mask = input_ids == self.pad_token_id
        if not mask.any():
            return output
        if radar.shape[0] == 1 and input_ids.shape[0] > 1:
            radar = radar.expand(input_ids.shape[0], -1, -1)
        out = output.clone()
        for b in range(input_ids.shape[0]):
            pos = mask[b].nonzero(as_tuple=True)[0]
            if pos.numel() == 0:
                continue
            n = min(pos.numel(), radar.shape[1])
            out[b, pos[:n]] = radar[b, :n].to(out.dtype)
        return out

    def remove(self):
        self.handle.remove()


def load_model(args):
    from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
    from ravl.connector import RadarConnector, add_radar_tokens, llm_hidden_size
    from ravl.number_tokens import add_number_tokens
    from ravl.radar_encoder import RadarEncoder, encoder_kwargs, load_encoder_state

    device = torch.device("cuda", 0)
    tokenizer = AutoTokenizer.from_pretrained(args.base)
    processor = AutoProcessor.from_pretrained(args.base)
    weights = os.path.join(args.checkpoint, "model")
    source = weights if os.path.isdir(weights) else args.base
    llm = AutoModelForImageTextToText.from_pretrained(
        source, dtype=torch.bfloat16, attn_implementation="sdpa").to(device)

    pad_id = add_radar_tokens(tokenizer, llm)
    # The checkpoint was trained with these in the vocabulary. Adding them in a
    # different order, or not at all, silently shifts every id past 151,672.
    n_num = add_number_tokens(tokenizer, llm)
    processor.tokenizer = tokenizer
    log(f"vocab {len(tokenizer):,} (radar pad {pad_id}, +{n_num:,} numbers)")
    if llm.get_input_embeddings().weight.shape[0] != len(tokenizer):
        raise SystemExit(
            f"어휘 불일치: 모델 {llm.get_input_embeddings().weight.shape[0]:,} "
            f"vs 토크나이저 {len(tokenizer):,}. number_vocab.txt 가 학습 때와 "
            f"같은지 확인하세요.")

    state = torch.load(os.path.join(args.checkpoint, "adapters.pt"),
                       map_location="cpu")
    trained = state["args"]
    encoder = RadarEncoder(**{"dim": trained["radar_dim"],
                              "n_frames": trained["frames"],
                              **{k: v for k, v in encoder_kwargs(trained).items()
                                 if k not in ("dim", "n_frames")}})
    load_encoder_state(encoder, state["encoder"])
    encoder = encoder.to(device).to(torch.bfloat16).eval()
    connector = RadarConnector(trained["radar_dim"], llm_hidden_size(args.base))
    connector.load_state_dict(state["connector"])
    connector = connector.to(device).to(torch.bfloat16).eval()
    llm.eval()
    return tokenizer, processor, llm, encoder, connector, pad_id, trained, device


def load_item(root, rec, n_frames, max_points, n_channels):
    """Rebuild one sample's tensors from the bundle.

    The bundle is clip-centric: a clip holds its twenty frames once and an
    item names the ones it uses by index, so `det_objects` reading one frame
    and `track_step` reading five share the same files. Radar windows are
    shared the same way -- two items at the same instant and sample rate name
    the same npz.
    """
    d = os.path.join(root, "clips", rec["clip_id"])
    frames = [Image.open(os.path.join(d, "frames", f"f{i:02d}.jpg")).convert("RGB")
              for i in rec["frames"]]
    z = np.load(os.path.join(d, "radar", f"{rec['radar']}.npz"))
    pts = np.zeros((n_frames, max_points, n_channels), dtype=np.float32)
    mask = np.zeros((n_frames, max_points), dtype=bool)
    kept, scan = z["points"].astype(np.float32), z["scan"]
    for f in range(n_frames):
        sel = kept[scan == f]
        n = min(len(sel), max_points)
        if n:
            pts[f, :n] = sel[:n]
            mask[f, :n] = True
    return frames, torch.from_numpy(pts), torch.from_numpy(mask)


def run_task(task, records, args, loaded, root):
    from transformers.video_utils import VideoMetadata
    tokenizer, processor, llm, encoder, connector, pad_id, trained, device = loaded
    shape = json.load(open(os.path.join(root, "manifest.json")))
    n_channels = len(shape["channels"])
    injector = RadarInjector(llm.get_input_embeddings(), pad_id)
    out_records, generations = [], []
    budget = MAX_NEW.get(task, 48)
    if args.max_new_floor:
        budget = max(budget, args.max_new_floor)

    for i, rec in enumerate(records):
        z = np.load(os.path.join(root, "clips", rec["clip_id"], "radar",
                                 f"{rec['radar']}.npz"))
        n_frames, max_points, _ = z["shape"]
        frames, points, mask = load_item(root, rec, int(n_frames),
                                         int(max_points), n_channels)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": [{"type": "video"},
                                         {"type": "text", "text": rec["user"]}]},
        ]
        text = processor.apply_chat_template(messages, tokenize=False,
                                             add_generation_prompt=True)
        meta = [VideoMetadata(total_num_frames=len(frames), fps=1.0,
                              width=frames[0].width, height=frames[0].height,
                              duration=float(len(frames)), video_backend="manual",
                              frames_indices=list(range(len(frames))))]
        batch = processor(text=[text], videos=[frames], video_metadata=meta,
                          return_tensors="pt").to(device)
        with torch.no_grad():
            radar_out = encoder(points.unsqueeze(0).to(device, torch.bfloat16),
                                mask.unsqueeze(0).to(device),
                                torch.tensor([rec["sensor"]], device=device))
            injector.pending = connector(radar_out["tokens"])
            cut = batch["input_ids"].shape[1]
            got = llm.generate(**batch, max_new_tokens=budget, do_sample=False,
                               pad_token_id=tokenizer.pad_token_id
                               or tokenizer.eos_token_id)
        gen = tokenizer.decode(got[0, cut:], skip_special_tokens=True).strip()
        fn = scorer_for(task)
        try:
            r = fn(gen, rec["target"], rec["user"])
        except TypeError:
            r = fn(gen, rec["target"])
        out_records.append(r)
        generations.append({"task": task, "id": rec["id"],
                            "clip_id": rec["clip_id"], "generated": gen,
                            "reference": rec["target"]})
        if (i + 1) % 25 == 0:
            log(f"    {task} {i+1}/{len(records)}")
    injector.remove()
    return summarise(task, out_records), generations


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    # 가중치는 체크포인트 것을 쓰므로, 여기서 필요한 것은 토크나이저와 프로세서
    # 설정뿐이다. 그 파일들(4 MB)은 base/ 에 들어 있어 원본 17 GB 를 받을
    # 필요가 없다.
    ap.add_argument("--base", default=os.path.join(HERE, "base"),
                    help="토크나이저·프로세서 설정 폴더 (기본: 번들의 base/)")
    ap.add_argument("--checkpoint",
                    default=os.path.join(HERE, "vlm_8B_v9_6ch_step2200"))
    ap.add_argument("--data", default=os.path.join(HERE, "data"))
    ap.add_argument("--out", default=os.path.join(HERE, "results"))
    ap.add_argument("--tasks", default="all")
    ap.add_argument("--items", type=int, default=0, help="0 = 번들에 있는 전부")
    ap.add_argument("--max-new-floor", type=int, default=0)
    args = ap.parse_args(argv)

    manifest = json.load(open(os.path.join(args.data, "manifest.json")))
    names = (sorted(manifest["tasks"]) if args.tasks == "all"
             else [t.strip() for t in args.tasks.split(",") if t.strip()])
    log(f"클립 {manifest.get('n_clips', '?')}개 · 아이템 "
        f"{manifest.get('n_items', 0):,}건 · 태스크 {len(manifest['tasks'])}종")
    if manifest.get("n_clips", 0) < 50:
        log("  주의: 클립 수가 적어 여기 나오는 점수는 지표가 아니라 참고입니다. "
            "채점용 표본을 늘리려면 export_items.py --clips 를 올리세요.")
    os.makedirs(args.out, exist_ok=True)
    loaded = load_model(args)

    scores, all_gen = {}, []
    for task in names:
        path = os.path.join(args.data, "by_task", f"{task}.jsonl")
        if not os.path.exists(path):
            log(f"--- {task}: 번들에 없음")
            continue
        records = [json.loads(l) for l in open(path)]
        if args.items:
            records = records[: args.items]
        log(f"--- {task}  {len(records)}건")
        s, g = run_task(task, records, args, loaded, args.data)
        scores[task] = s
        all_gen.extend(g)
        print("   ", {k: (round(v, 4) if isinstance(v, float) else v)
                      for k, v in s.items()}, flush=True)

    json.dump(scores, open(os.path.join(args.out, "scores.json"), "w"),
              indent=1, ensure_ascii=False)
    with open(os.path.join(args.out, "generations.jsonl"), "w") as fh:
        for g in all_gen:
            fh.write(json.dumps(g, ensure_ascii=False) + "\n")
    log(f"wrote {args.out}/scores.json 와 generations.jsonl "
        f"({len(all_gen):,} 생성)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
