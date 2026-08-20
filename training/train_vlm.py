#!/usr/bin/env python3
"""Train the radar+vision+text model on top of Qwen3-VL.

Radar is injected with a forward hook on the input-embedding module rather than
by passing `inputs_embeds`. Qwen3VLModel.forward locates its video placeholders
with `get_placeholder_mask(input_ids, ...)` and then `masked_scatter`s the tower
output in, so handing it `inputs_embeds` would disable the native vision path --
`forward` accepts exactly one of the two. The hook sees `input_ids` as its
argument and the embeddings as its output, so it can overwrite the radar
placeholder rows and leave everything else, including the video scatter that runs
afterwards, untouched.

Three stages:

  align   language model and vision tower frozen, radar encoder frozen. Only the
          connector learns, which is the cheapest way to find out whether the
          pretrained radar representation is legible to the language model at
          all.
  joint   LoRA on the language model, connector trainable, and the top of the
          radar encoder unfrozen. Freezing the encoder permanently is the usual
          way this goes wrong: it was pretrained on physics objectives, not on
          whatever the language model turns out to need, and the connector alone
          has to bridge that gap.
  full    no adapters at all -- every weight in the language model, the radar
          encoder and the connector trains, sharded with FSDP2. DDP cannot hold
          it: full AdamW state is 16 bytes per parameter, so 33 B parameters
          want 534 GB against one device's 179 GB.

    torchrun --nproc_per_node=5 -m training.train_vlm --model 8B --stage align
    torchrun --nproc_per_node=5 -m training.train_vlm --model 8B --stage joint \
        --resume checkpoints/vlm_8B_align
    torchrun --nproc_per_node=5 -m training.train_vlm --model 8B --stage full \
        --tasks all --all-profiles --radar-dropout 0.25
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

# cuDNN 의 SDPA 백엔드를 끈다.
#
# v16 이 step 200 에서 죽었다 -- rank 2 의 그래디언트 체크포인팅 재계산 중
# qwen3_vl 의 scaled_dot_product_attention 에서
# "Expected mha_graph.execute(...).is_good() to be true, but got false".
# 데이터가 아니라 백엔드 결함이고, 크기에 따라 터진다. 레이더 인코더에서
# nn.MultiheadAttention 의 fastpath 를 껐던 것과 같은 계열이다.
#
# 끄면 flash 나 mem-efficient 로 내려간다. 둘 다 같은 수를 내고, 이 모델
# 크기에서 속도 차이는 측정되지 않았다.
torch.backends.cuda.enable_cudnn_sdp(False)
MODEL_DIR = {
    "8B": "/NHNHOME/workspace/models/Qwen3-VL-8B-Instruct",
    "32B": "/NHNHOME/workspace/models/Qwen3-VL-32B-Instruct",
    "30B-A3B": "/NHNHOME/workspace/models/Qwen3-VL-30B-A3B-Instruct",
}

# Measured on 5 x B200 at ~2,300 tokens of context; see training/memory_plan.py.
# Global batch is held at 160 everywhere so a size comparison is not confounded
# by batch size.
#
# The stages need different plans. In `align` the language model is frozen, so no
# activations are kept for a backward pass through it and a micro-batch of 8 sits
# at 107 GiB. `joint` attaches LoRA, gradients then flow through all 36 layers,
# and the same batch OOMs -- the logits alone are batch x seq x 151,936, which
# cross_entropy upcasts to fp32: 11 GiB for one tensor at micro-batch 8.
# `full` fine-tunes every weight and needs FSDP2 rather than DDP: at 16 bytes per
# parameter for bf16 weights, bf16 gradients, fp32 master and fp32 Adam moments,
# 33.4 B parameters is 534 GB against 179 GB on a single B200. Sharded over the
# five devices that is 107 GB each, which fits.
BATCH_PLAN = {
    ("8B", "align"): {"micro": 8, "accum": 4},
    ("8B", "joint"): {"micro": 2, "accum": 16, "checkpointing": True},
    # Sharding makes `full` the cheapest stage per device, not the dearest:
    # measured 25.6 GiB peak at micro-batch 2 against the 111 GiB `align` reaches
    # with the model replicated. 8 keeps the global batch at 160 and amortises
    # the per-forward all-gather over four times as many samples.
    ("8B", "full"): {"micro": 8, "accum": 4, "checkpointing": True},
    ("32B", "align"): {"micro": 16, "accum": 2, "checkpointing": True},
    ("32B", "joint"): {"micro": 2, "accum": 16, "checkpointing": True},
    ("32B", "full"): {"micro": 1, "accum": 32, "checkpointing": True},
    ("30B-A3B", "align"): {"micro": 8, "accum": 4},
    ("30B-A3B", "joint"): {"micro": 2, "accum": 16, "checkpointing": True},
    ("30B-A3B", "full"): {"micro": 1, "accum": 32, "checkpointing": True},
}


def log(rank, msg):
    if rank == 0:
        print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


class RadarInjector:
    """Splices projected radar tokens into the embedding output.

    Stateful by necessity: the hook fires inside the model's forward and has no
    other way to receive this step's radar embeddings. `pending` is set just
    before the call and cleared by the hook, so a stale tensor cannot leak into
    the next step.
    """

    def __init__(self, embed_module, pad_token_id):
        self.pad_token_id = pad_token_id
        self.pending = None
        self.handle = embed_module.register_forward_hook(self._hook)

    def _hook(self, module, args, output):
        if self.pending is None:
            return output
        input_ids = args[0]
        radar = self.pending
        self.pending = None
        mask = input_ids == self.pad_token_id
        if not mask.any():
            return output
        # `generate(num_return_sequences=G)` expands the prompt to G rows while
        # the radar stays at one. The scene is the same for every sample of that
        # prompt, so broadcast rather than asking every caller to expand.
        if radar.shape[0] == 1 and input_ids.shape[0] > 1:
            radar = radar.expand(input_ids.shape[0], -1, -1)
        out = output.clone()
        for b in range(input_ids.shape[0]):
            positions = mask[b].nonzero(as_tuple=True)[0]
            if positions.numel() == 0:
                continue
            n = min(positions.numel(), radar.shape[1])
            out[b, positions[:n]] = radar[b, :n].to(out.dtype)
        return out

    def remove(self):
        self.handle.remove()


# Qwen3 tokenises the digits 0-9 as ten consecutive single tokens, so a number is
# emitted one digit at a time and each digit is an independent 151,936-way choice.
DIGIT_TOKEN_IDS = tuple(range(15, 25))


def digit_distance_loss(logits, labels, digit_ids):
    """How far the emitted digit is from the right one, in value rather than in
    probability.

    Cross-entropy is the wrong shape for a quantity. Asked for 638 it punishes
    639 and 100 almost equally, because it only ever asks whether the argmax is
    the label. Measured on this model that costs nearly everything: the hidden
    state that generates the first answer token carries the strongest radar
    return at R^2 0.65, and the number the model writes correlates with the truth
    at 0.14 -- about 3% of what is present.

    This restricts the distribution to the ten digit tokens, takes its
    expectation, and penalises the distance to the true digit. The gradient then
    says "move towards 8", not merely "raise the logit of 8".
    """
    positions = torch.isin(labels, digit_ids)
    if not positions.any():
        return logits.sum() * 0.0
    selected = logits[positions][:, digit_ids]
    expected = (torch.softmax(selected.float(), dim=-1)
                * torch.arange(10, device=logits.device, dtype=torch.float32)).sum(-1)
    truth = (labels[positions].unsqueeze(-1) == digit_ids).float().argmax(-1)
    return torch.nn.functional.smooth_l1_loss(expected, truth.float())


def number_distance_loss(logits, labels, number_ids, number_values,
                         number_units=None):
    """The same idea once a number is one token: penalise the distance in value.

    `digit_distance_loss` works on the ten digit tokens and therefore grades
    each decimal place separately -- writing 900 for 100 is punished as one
    wrong hundreds digit, exactly as writing 200 would be. With a token per
    literal the model chooses among values directly, so the expectation is over
    the values themselves and the error is what a metre or a degree actually
    costs.

    Scaled by the spread of the true values in the batch, so a task answering in
    hundreds of pixels does not swamp one answering in metres.
    """
    positions = torch.isin(labels, number_ids)
    if not positions.any():
        return logits.sum() * 0.0
    lookup = torch.full((int(number_ids.max()) + 1,), float("nan"),
                        device=logits.device)
    lookup[number_ids] = number_values
    truth = lookup[labels[positions]]
    selected = logits[positions][:, number_ids]

    # Only compare like with like. The literals carry their unit -- `37m`,
    # `+12deg`, `7.3m/s`, `378px` -- and a distance between a metre and a
    # degree is not a quantity. Before the units were fused the softmax ran
    # over all 2,503 literals at once, so answering `37` in metres was
    # penalised for every degree and pixel value it did not choose, by an
    # amount that meant nothing.
    if number_units is not None:
        unit_of = torch.full((int(number_ids.max()) + 1,), -1,
                             dtype=torch.long, device=logits.device)
        unit_of[number_ids] = number_units
        want = unit_of[labels[positions]]
        allowed = number_units.unsqueeze(0) == want.unsqueeze(1)
        selected = selected.masked_fill(~allowed, float("-inf"))
    probs = torch.softmax(selected.float(), dim=-1)

    # The expectation of the squared distance, not the distance of the
    # expectation.
    #
    # The first version penalised |E[v] - truth|, which any distribution whose
    # mean lands on the truth drives to zero -- half the mass on 0 and half on
    # 200 scores perfectly for a target of 100, while the argmax that generation
    # actually emits is 0 or 200. Verified on constructed distributions: that
    # one scored 0.0000. This form is linear in the probabilities, so its
    # minimum is the point mass on the true value and spreading is always paid
    # for.
    #
    # Absolute rather than squared. Squaring was chosen because the scorers are
    # thresholded, so a far miss should cost far more -- but it also makes the
    # term explode early, when the mass is still spread over 2,503 literals: the
    # first step measured 30,229 against a language loss near 4, and the run
    # spiked between steps 20 and 40 whenever one far number landed in a batch.
    # More importantly it minimises at the conditional mean, and the azimuth
    # distribution this model keeps collapsing on is symmetric about zero --
    # mean, median and mode all sit at 0, measured over 115,553 answers -- so
    # the squared form pays the model to hedge toward the middle.
    #
    # Normalised per position, not per batch. A single batch scale would be the
    # mean over every number in it, and the tasks differ by 254x: box
    # coordinates run to a median of 492 while `plan_ego` sits at 2. Squaring
    # that mismatch puts the box tasks 64,000x ahead of the rest, which is not
    # a weighting anyone chose. Dividing each position by its own magnitude
    # makes the term a relative error, and the clamp keeps a truth of 0.0 --
    # common for a yaw or a lateral offset -- from dividing by nothing.
    scale = truth.abs().clamp(min=1.0).unsqueeze(1)
    gap = (number_values.unsqueeze(0) - truth.unsqueeze(1)) / scale
    return (probs * gap.abs()).sum(-1).mean()


def setup():
    if int(os.environ.get("WORLD_SIZE", 1)) > 1:
        dist.init_process_group("nccl")
        rank = dist.get_rank()
        torch.cuda.set_device(rank % torch.cuda.device_count())
        return rank, dist.get_world_size()
    torch.cuda.set_device(0)
    return 0, 1


def build(args, rank):
    from transformers import AutoModelForImageTextToText, AutoProcessor, AutoTokenizer
    from training.connector import RadarConnector, add_radar_tokens, llm_hidden_size
    from training.radar_encoder import (RadarEncoder, encoder_kwargs,
                                        load_encoder_state)

    model_dir = MODEL_DIR[args.model]
    log(rank, f"loading {model_dir}")
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    processor = AutoProcessor.from_pretrained(model_dir)
    # AutoModel picks Qwen3VLMoe for the A3B checkpoint. Naming the dense class
    # explicitly loaded MoE weights into a dense architecture, which transformers
    # only warns about -- the run proceeds with a silently wrong model.
    # fp32 storage with bf16 compute is what keeps Adam's moments precise over a
    # full fine-tune; FSDP's MixedPrecisionPolicy casts for the forward. 8.8 B
    # parameters is 35 GB in fp32, which fits on one B200 before sharding.
    weight_dtype = (torch.float32
                    if args.stage == "full" and args.master_dtype == "fp32"
                    else torch.bfloat16)
    # `--resume` used to restore the connector and the encoder and leave the
    # language model at its released weights, which for a `full` run silently
    # threw away everything the run had learnt -- the 17.5 GB it had just
    # written was never read back. Take the saved weights when they are there,
    # the way `train_grpo --init` already does.
    trained_weights = (os.path.join(args.resume, "model") if args.resume
                       else None)
    source = (trained_weights if trained_weights
              and os.path.isdir(trained_weights) else model_dir)
    if source != model_dir:
        log(rank, f"language model resumed from {source}")
    llm = AutoModelForImageTextToText.from_pretrained(
        source, dtype=weight_dtype,
        attn_implementation="sdpa").to(torch.cuda.current_device())

    pad_id = add_radar_tokens(tokenizer, llm)
    # One token per numeric literal, initialised so that nearby values start out
    # nearby. Answers are 44-88% digits, and a token per digit both costs the
    # context and grades each decimal place separately -- 900 for 100 reads as
    # one wrong hundreds digit, the same as 200.
    from training.number_tokens import add_number_tokens
    n_num = add_number_tokens(tokenizer, llm)
    # 자차 이력 궤적 토큰. 숫자 토큰 뒤에 붙어야 한다 -- 순서가 바뀌면 id 가
    # 밀려 예전 체크포인트가 조용히 다른 낱말을 가리킨다.
    from training.traj_tokens import add_traj_tokens, init_embeddings
    traj_pad_id, traj_first = add_traj_tokens(tokenizer, llm)
    # 이어받는 실행은 이미 학습된 임베딩을 갖고 있으므로 덮어쓰면 안 된다.
    if not args.resume:
        init_embeddings(tokenizer, llm)
    processor.tokenizer = tokenizer
    log(rank, f"radar pad token id {pad_id}, +{n_num:,} number tokens, "
              f"traj pad {traj_pad_id} (+3,000 궤적 토큰, 첫 id {traj_first}), "
              f"vocab now {len(tokenizer)}")

    # The encoder's own recorded geometry wins over the flags: the number of
    # tokens it emits is what the prompt has to reserve placeholders for, and a
    # mismatch silently leaves some placeholders holding their text embedding.
    encoder_args = {}
    state = None
    if args.radar_checkpoint and os.path.exists(args.radar_checkpoint):
        state = torch.load(args.radar_checkpoint, map_location="cpu")
        saved = state.get("args", {})
        encoder_args = encoder_kwargs(saved)
        encoder_args.pop("dim", None)
        encoder_args.pop("n_frames", None)
    # 사전학습이 텍스트 정렬을 했다면 그 투영 층도 체크포인트에 들어 있다.
    # 모양을 맞춰 세워야 불러올 수 있고, 그래야 아래에서 커넥터의 첫 층을
    # 그것으로 시작할 수 있다 -- 커넥터가 무에서 시작하지 않는 것이
    # 텍스트 정렬을 한 이유의 절반이다.
    text_dim = 0
    if args.radar_checkpoint and os.path.exists(args.radar_checkpoint):
        _peek = torch.load(args.radar_checkpoint, map_location="cpu",
                           weights_only=False).get("model", {})
        if "text_proj.weight" in _peek:
            text_dim = _peek["text_proj.weight"].shape[0]
    encoder = RadarEncoder(dim=args.radar_dim, n_frames=args.radar_frames,
                           text_dim=text_dim,
                           **encoder_args)
    if state is not None:
        load_encoder_state(encoder, state["model"])
        log(rank, f"radar encoder restored from {args.radar_checkpoint} "
                  f"(readout {encoder.readout}, {encoder.n_tokens} tokens)")
    if args.radar_tokens != encoder.n_tokens:
        log(rank, f"radar tokens {args.radar_tokens} -> {encoder.n_tokens} "
                  f"to match the encoder")
        args.radar_tokens = encoder.n_tokens
    # Recorded so eval_vlm can rebuild the same geometry from the checkpoint
    # alone; `vars(args)` is what gets saved.
    args.readout = encoder.readout
    args.frame_queries = encoder.frame_queries
    encoder = encoder.to(torch.cuda.current_device()).to(weight_dtype)

    connector = RadarConnector(args.radar_dim, llm_hidden_size(model_dir))
    # 사전학습이 이미 384 -> 4096 을 낱말 임베딩 쪽으로 맞춰 두었다. 커넥터의
    # 첫 선형층을 그것으로 시작하면 첫 스텝부터 레이더 토큰이 낱말이 놓인
    # 근처에 떨어진다. 무작위 투영으로 시작하면 그동안 카메라를 쓰는 편이
    # 엄격히 쉬우므로, 모델은 그쪽으로 갔다가 돌아오지 않는다.
    if (text_dim and encoder.text_proj is not None
            and encoder.text_proj.out_features == connector.d_llm):
        with torch.no_grad():
            connector.project[0].weight.copy_(encoder.text_proj.weight)
            connector.project[0].bias.copy_(encoder.text_proj.bias)
        log(rank, "커넥터 첫 층을 사전학습된 텍스트 투영으로 시작")
    # Without this the later stages threw away everything `align` learned. The
    # connector is the only thing that makes the radar tokens legible, and
    # starting it from noise means the language model spends its first steps
    # being handed a random projection -- at which point using the camera
    # instead is strictly easier, and it never comes back. `--resume` was in the
    # module docstring from the start but had never been implemented.
    if args.resume:
        prior = torch.load(os.path.join(args.resume, "adapters.pt"),
                           map_location="cpu")
        connector.load_state_dict(prior["connector"])
        load_encoder_state(encoder, prior["encoder"])
        log(rank, f"connector and encoder resumed from {args.resume}")
    connector = connector.to(torch.cuda.current_device()).to(weight_dtype)
    return tokenizer, processor, llm, encoder, connector, pad_id


def shard_for_full_finetune(llm, encoder, connector, master_dtype, rank, world):
    """Shard everything trainable, so no plain tensor meets a DTensor later."""
    """FSDP2 over the transformer blocks so every weight can be trained.

    Blocks are found by class name rather than by attribute path: the dense and
    MoE Qwen3-VL variants disagree on where the decoder lives, and the vision
    tower is a third naming. Sharding the root as well covers the embeddings and
    the language-model head, which for a 151,936-entry vocabulary is 1.2 GB of
    parameters on its own.
    """
    from torch.distributed.device_mesh import init_device_mesh
    from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy

    # FSDP2 will not fall back to the default process group: without an explicit
    # mesh it fails inside _init_sharded_param with "Invariant encountered:
    # value was None".
    mesh = init_device_mesh("cuda", (world,))

    # Gradients reduce in fp32. bf16 all-reduce over 33 B parameters loses enough
    # in the summation to stall training well before the loss curve shows it.
    policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16,
                                  reduce_dtype=torch.float32)
    candidates = [m for m in llm.modules()
                  if type(m).__name__.endswith(("DecoderLayer", "VisionBlock"))]
    # Never shard a module nested inside another that is about to be sharded.
    # The outer call turns its whole subtree into DTensors, and the inner call
    # then takes FSDP's tensor-parallel branch and dies asking for a TP mesh --
    # which is how `Qwen3VLVisionMLP`, a child of `Qwen3VLVisionBlock`, broke
    # this the first time.
    inner = set()
    for block in candidates:
        for child in block.modules():
            if child is not block:
                inner.add(id(child))
    blocks = [b for b in candidates if id(b) not in inner]
    for block in blocks:
        fully_shard(block, mesh=mesh, mp_policy=policy)
    fully_shard(llm, mesh=mesh, mp_policy=policy)

    # The radar encoder and the connector are sharded too, small as they are.
    # Mixing DTensor parameters with plain ones in a single optimiser breaks on
    # `_foreach_mul_`, and clip_grad_norm_ over the mixture would compute a norm
    # that is not the global one. Uniformity is cheaper than the special cases.
    fully_shard(encoder, mesh=mesh, mp_policy=policy)
    fully_shard(connector, mesh=mesh, mp_policy=policy)

    # Master precision is decided at load time, not here. Rewriting `p.data`
    # after sharding leaves FSDP's own view of the shard pointing at the old
    # storage, and the optimiser then sees a mix of dtypes it cannot group.
    log(rank, f"FSDP2 over {len(blocks)} blocks plus the radar path, "
              f"master weights {master_dtype}")
    return llm


def configure_stage(args, llm, encoder, connector, rank):
    """Set requires_grad per stage and return the parameters to optimise."""
    for p in llm.parameters():
        p.requires_grad = False
    for p in encoder.parameters():
        p.requires_grad = False
    for p in connector.parameters():
        p.requires_grad = True

    trainable = list(connector.parameters())
    if args.stage == "full":
        # No adapters: the whole stack learns, which is what LoRA was standing in
        # for. The radar encoder goes with it -- freezing it here would leave the
        # one module the task depends on as the only frozen thing in the model.
        for p in llm.parameters():
            p.requires_grad = True
        from training.radar_encoder import PRETRAIN_HEADS
        for name, p in encoder.named_parameters():
            # The pretraining heads are discarded once the encoder feeds a
            # language model; leaving them trainable puts parameters in the
            # optimiser that no loss can ever reach.
            p.requires_grad = not name.startswith(PRETRAIN_HEADS)
        if args.world_size > 1:
            llm = shard_for_full_finetune(llm, encoder, connector,
                                          args.master_dtype, rank,
                                          args.world_size)
        else:
            # One device needs no sharding: 8.8 B parameters in bf16 with bf16
            # Adam moments is 65.6 GiB of state against a B200's 179. Skipping
            # FSDP is what lets five independent configurations run at once, one
            # per device, instead of five sequential sharded runs.
            log(rank, "single device: full fine-tune without sharding")
        # Collected only now. `fully_shard` replaces a module's Parameter objects
        # with DTensor ones, so the connector parameters gathered at the top of
        # this function are stale by this point -- the optimiser was holding
        # pre-shard plain tensors alongside sharded fp32 ones and died grouping
        # them by dtype.
        trainable = [p for module in (connector, encoder, llm)
                     for p in module.parameters() if p.requires_grad]
    if args.stage == "joint":
        from peft import LoraConfig, get_peft_model
        # MoE experts hold gate/up/down under a different module path and there
        # are 128 of them; adapting every expert would multiply the adapter count
        # by 128 while each one sees only 1/16 of the tokens. Attention only.
        is_moe = "A3B" in args.model
        targets = ["q_proj", "k_proj", "v_proj", "o_proj"]
        if not is_moe:
            targets += ["gate_proj", "up_proj", "down_proj"]
        lora = LoraConfig(r=args.lora_rank, lora_alpha=2 * args.lora_rank,
                          lora_dropout=0.05, bias="none",
                          target_modules=targets, task_type="CAUSAL_LM")
        llm = get_peft_model(llm, lora)
        trainable += [p for p in llm.parameters() if p.requires_grad]
        # Unfreeze the temporal half of the radar encoder. The spatial layers
        # learned per-scan geometry from physics and are worth keeping; the
        # temporal and pooling layers are what have to adapt to the language
        # model's needs.
        for name, p in encoder.named_parameters():
            if name.startswith(("temporal", "global_pool", "global_query",
                                "norm_out")):
                p.requires_grad = True
                trainable.append(p)

    n = sum(p.numel() for p in trainable)
    log(rank, f"stage {args.stage}: {n/1e6:.1f} M trainable parameters")
    return llm, trainable


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", default="8B", choices=list(MODEL_DIR))
    ap.add_argument("--stage", default="align",
                    choices=("align", "joint", "full"))
    ap.add_argument("--master-dtype", default="bf16", choices=("fp32", "bf16"),
                    help="full stage only. bf16 is the default and the one that "
                         "works: with fp32 storage transformers casts "
                         "pixel_values to the model's reported dtype while FSDP "
                         "casts the weights to bf16, and the vision tower dies "
                         "on 'mat1 and mat2 must have the same dtype'. Gradients "
                         "still all-reduce in fp32 either way.")
    ap.add_argument("--tasks", default="qa,description,ood_reasoning")
    ap.add_argument("--frames", type=int, default=20)
    ap.add_argument("--radar-frames", type=int, default=10,
                    help="레이더 스캔 수. 카메라 프레임 수와 별개다 -- 창의 "
                         "길이는 태스크가 정하고, 이 값은 그 창을 몇 등분해 "
                         "볼지를 정한다")
    ap.add_argument("--radar-dim", type=int, default=384)
    ap.add_argument("--radar-tokens", type=int, default=256)
    ap.add_argument("--radar-checkpoint",
                    default="/NHNHOME/workspace/checkpoints/radar_encoder_all/encoder.pt")
    ap.add_argument("--micro-batch", type=int, default=0, help="0 uses BATCH_PLAN")
    ap.add_argument("--accum", type=int, default=0)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--lora-rank", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--steps", type=int, default=0)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--max-length", type=int, default=3584,
                    help="was 4096. The longest item in the build is 3,404 "
                         "tokens, so 4,096 never bounded anything real -- it "
                         "only let a truncated outlier pad its whole batch to "
                         "4,096 and ask for 37 GiB of fp32 logits at once. "
                         "3,584 clears the real maximum and caps the worst case")
    ap.add_argument("--samples", type=int, default=0,
                    help="items per epoch, drawn with the task mixture; 0 uses all")
    ap.add_argument("--mixture", default=None,
                    help="task:weight pairs, e.g. qa:2,desc_radar:3")
    ap.add_argument("--verified-only", action="store_true")
    ap.add_argument("--camera-dropout", type=float, default=0.15,
                    help="이 비율의 항목에서 카메라를 검게 지운다. 레이더가 "
                         "유일한 출처인 항목이 없으면 레이더를 읽을 이유가 "
                         "없다 -- v12 에서 섞은 레이더 대조군과의 비가 39개 "
                         "측정점 내내 1.10 이었다")
    ap.add_argument("--radar-dropout", type=float, default=0.0,
                    help="blank the radar on this fraction of items and replace "
                         "radar-only targets with a refusal")
    ap.add_argument("--eval-every", type=int, default=0,
                    help="이 배수의 스텝마다 루프 안에서 평가한다. 0 이면 안 한다")
    ap.add_argument("--eval-items", type=int, default=60,
                    help="평가 태스크당 항목 수. 랭크들이 나눠 맡는다")
    ap.add_argument("--eval-samples", type=int, default=6,
                    help="계획에서 뽑을 갈래 수 (minADE_k). 1 이면 안 뽑는다")
    ap.add_argument("--save-every", type=int, default=0,
                    help="also write to <out>/latest every N optimiser steps")
    ap.add_argument("--seed", type=int, default=0,
                    help="draws a different sample of the mixture and different "
                         "dropout, so repeats measure run-to-run noise. Offset "
                         "per rank for the torch stream, shared for the data")
    ap.add_argument("--resume", default=None,
                    help="checkpoint directory to take the trained connector and "
                         "radar encoder from, e.g. an align run. Without it every "
                         "stage restarts the connector from noise")
    ap.add_argument("--digit-weight", type=float, default=0.0,
                    help="weight on a distance-aware penalty over digit tokens, "
                         "so 639 is scored as nearly right for 638 and 100 is "
                         "not. Targets the one stage measured to lose the radar")
    ap.add_argument("--digit-warmup", type=int, default=1500,
                    help="이 스텝까지 숫자 손실의 가중치를 0 에서 선형으로 "
                         "올린다. 형식이 잡히기 전에 전량을 걸면 모든 값의 "
                         "주변 평균 쪽으로 당기는데, 방위각의 주변 평균은 0 이다")
    ap.add_argument("--radar-contrast", type=float, default=0.0,
                    help="weight on a hinge that requires another clip's radar "
                         "to score at least --radar-margin worse. Costs a second "
                         "forward per micro-batch")
    ap.add_argument("--radar-margin", type=float, default=0.15,
                    help="nats the wrong radar must cost; measured, a healthy "
                         "align run reaches 0.10-0.18 on the radar tasks")
    ap.add_argument("--router-weight", type=float, default=0.01,
                    help="load-balancing weight for the radar encoder's experts")
    ap.add_argument("--all-profiles", action="store_true",
                    help="include clips whose rig carries the SRR or no front "
                         "radar, instead of imaging-LRR clips only")
    ap.add_argument("--out", default=None)
    args = ap.parse_args(argv)

    plan = BATCH_PLAN[(args.model, args.stage)]
    micro = args.micro_batch or plan["micro"]
    accum = args.accum or plan["accum"]
    out_dir = args.out or f"/NHNHOME/workspace/checkpoints/vlm_{args.model}_{args.stage}"

    rank, world = setup()
    device = torch.device("cuda", torch.cuda.current_device())
    # Ranks must disagree on dropout but agree on which items the epoch holds,
    # so the data seed is shared and only the torch stream is offset.
    torch.manual_seed(args.seed * 1000 + rank)
    log(rank, f"world {world}, micro-batch {micro}, accum {accum}, "
              f"global batch {micro * world * accum}")

    tokenizer, processor, llm, encoder, connector, pad_id = build(args, rank)
    if plan.get("checkpointing"):
        llm.config.use_cache = False
        llm.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
        llm.enable_input_require_grads()
        log(rank, "gradient checkpointing on")
    args.world_size = world
    llm, trainable = configure_stage(args, llm, encoder, connector, rank)

    from training.instruct_data import (InstructDataset, build_collate,
                                        expand_tasks)
    mixture = None
    if args.mixture:
        mixture = {p.split(":")[0]: float(p.split(":")[1])
                   for p in args.mixture.split(",")}
    dataset = InstructDataset(tasks=expand_tasks(args.tasks), split="train",
                              processor=processor, tokenizer=tokenizer,
                              n_frames=args.frames, radar_tokens=args.radar_tokens,
                              radar_frames=args.radar_frames,
                              camera_dropout=args.camera_dropout,
                              verified_only=args.verified_only,
                              samples=args.samples, mixture=mixture,
                              all_profiles=args.all_profiles,
                              radar_dropout=args.radar_dropout, seed=args.seed)
    log(rank, f"dataset {len(dataset):,} items")
    counts = dict(sorted(dataset.task_counts().items()))
    for task, count in counts.items():
        log(rank, f"  {task:24s} {count:7,}  ({count/len(dataset)*100:5.1f}%)")

    from training import tracking
    tracking.start(
        tracking.run_name("sft", args),
        {**vars(args), "world_size": world, "micro": micro, "accum": accum,
         "global_batch": micro * accum * world, "items": len(dataset),
         "vocab": len(tokenizer),
         "trainable_params": sum(p.numel() for p in trainable),
         "tasks": len(counts),
         **{f"mix/{k}": v / len(dataset) for k, v in counts.items()}},
        rank=rank, job="train",
        tags=(args.stage, args.model, "numtok" if args.digit_weight else "plain"))
    sampler = DistributedSampler(dataset) if world > 1 else None
    loader = DataLoader(dataset, batch_size=micro, sampler=sampler,
                        shuffle=sampler is None, num_workers=args.workers,
                        collate_fn=build_collate(processor, tokenizer,
                                                 args.max_length),
                        drop_last=True, pin_memory=True)

    digit_ids = torch.tensor(DIGIT_TOKEN_IDS, device=device)
    from training.number_tokens import number_tokens as _num_literals
    _lits = _num_literals()
    number_ids = number_values = number_units = None
    if _lits:
        _ids = tokenizer.convert_tokens_to_ids(_lits)
        keep = [(i, l) for i, l in zip(_ids, _lits) if i is not None and i >= 0]
        from training.number_tokens import split_unit
        number_ids = torch.tensor([i for i, _ in keep], device=device)
        pairs = [split_unit(l) for _, l in keep]
        number_values = torch.tensor([v for v, _ in pairs],
                                     device=device, dtype=torch.float32)
        units = sorted({u for _, u in pairs})
        number_units = torch.tensor([units.index(u) for _, u in pairs],
                                    device=device, dtype=torch.long)
        log(rank, f"number-distance loss over {len(keep):,} literals "
                  f"in {len(units)} units ({', '.join(u or 'm' for u in units)})")
    injector = RadarInjector(llm.get_input_embeddings(), pad_id)
    # `full` shards these with FSDP2 instead, inside configure_stage.
    if world > 1 and args.stage != "full":
        connector = DistributedDataParallel(connector, device_ids=[device.index])
        # The encoder is trainable from `joint` onwards but was never wrapped, so
        # each rank was stepping it on its own gradients and the five copies
        # silently diverged; only rank 0's was ever saved.
        if args.stage != "full" and any(p.requires_grad
                                        for p in encoder.parameters()):
            # `norm_kv` exists in every Block but is only read for
            # cross-attention, so the self-attention stacks always leave it
            # without a gradient. The encoder is 15-29 M parameters, so the
            # extra traversal this costs is noise next to the language model.
            encoder = DistributedDataParallel(encoder, device_ids=[device.index],
                                              find_unused_parameters=True)

    optimiser = torch.optim.AdamW(trainable, lr=args.lr, betas=(0.9, 0.95),
                                  weight_decay=0.01)
    total_steps = args.steps or (args.epochs * len(loader) // accum)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=args.lr, total_steps=max(total_steps, 1), pct_start=0.03)

    os.makedirs(out_dir, exist_ok=True)

    def save(tag=None):
        """Write the checkpoint. Collective, so every rank must call it."""
        state, radar_state, conn_state = None, None, None
        core_c = connector.module if hasattr(connector, "module") else connector
        core_e = encoder.module if hasattr(encoder, "module") else encoder
        conn_state, radar_state = core_c.state_dict(), core_e.state_dict()
        if args.stage == "full":
            from torch.distributed.checkpoint.state_dict import (
                StateDictOptions, get_model_state_dict)
            options = StateDictOptions(full_state_dict=True, cpu_offload=True)
            state = get_model_state_dict(llm, options=options)
            radar_state = get_model_state_dict(core_e, options=options)
            conn_state = get_model_state_dict(core_c, options=options)
        if rank != 0:
            return
        target = out_dir if tag is None else os.path.join(out_dir, tag)
        os.makedirs(target, exist_ok=True)
        torch.save({"connector": conn_state, "encoder": radar_state,
                    "args": vars(args), "history": history},
                   os.path.join(target, "adapters.pt"))
        if args.stage == "joint":
            llm.save_pretrained(os.path.join(target, "lora"))
        elif args.stage == "full":
            llm.save_pretrained(
                os.path.join(target, "model"),
                state_dict={k: v.to(torch.bfloat16) for k, v in state.items()},
                safe_serialization=True)
        with open(os.path.join(target, "history.json"), "w") as fh:
            json.dump(history, fh, indent=2)
        log(rank, f"saved to {target}")

    evaluator = None
    if args.eval_every:
        from training.inloop_eval import InLoopEvaluator
        evaluator = InLoopEvaluator(
            every=args.eval_every, items=args.eval_items, out_dir=out_dir,
            processor=processor, tokenizer=tokenizer, n_frames=args.frames,
            radar_tokens=args.radar_tokens, radar_frames=args.radar_frames,
            max_length=args.max_length, plan_samples=args.eval_samples,
            split="test", rank=rank, world=world,
            log=lambda m: log(rank, m))

    step, micro_step = 0, 0
    started = time.monotonic()
    history = []
    llm.train()
    for epoch in range(args.epochs):
        if sampler is not None:
            sampler.set_epoch(epoch)
        for batch in loader:
            points = batch.pop("points").to(device, torch.bfloat16, non_blocking=True)
            radar_mask = batch.pop("radar_mask").to(device, non_blocking=True)
            sensor = batch.pop("sensor", None)
            if sensor is not None:
                sensor = sensor.to(device, non_blocking=True)
            tasks = batch.pop("task")
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()
                     if torch.is_tensor(v)}

            with torch.set_grad_enabled(args.stage in ("joint", "full")):
                radar_out = encoder(points, radar_mask, sensor)
            core = connector.module if hasattr(connector, "module") else connector
            injector.pending = core(radar_out["tokens"])

            out = llm(**batch)
            loss = out.loss
            numeric = None
            if args.digit_weight > 0:
                # The weight ramps in rather than starting at full strength.
                # The number term is only meaningful once the answer has the
                # right shape; before that it is a pull toward the marginal
                # mean of every field, and the marginal mean of azimuth is
                # zero. Measured on the last run, the generated |azimuth| fell
                # from 13.9 to 5.5 degrees over the first 1,200 steps -- the
                # model hedging to the middle -- before it began to open up.
                warm = min(1.0, step / max(args.digit_warmup, 1))
                # Labels are shifted by one against logits, as in any causal LM.
                numeric = (number_distance_loss(out.logits[:, :-1],
                                                batch["labels"][:, 1:],
                                                number_ids, number_values,
                                                number_units)
                           if number_ids is not None else
                           digit_distance_loss(out.logits[:, :-1],
                                               batch["labels"][:, 1:], digit_ids))
                loss = loss + args.digit_weight * warm * numeric
            grounding = None
            if args.radar_contrast > 0 and batch["input_ids"].shape[0] > 1:
                # The metric this project is judged on is how much worse the
                # model gets when the radar is swapped for another clip's. Every
                # recipe tried so far drove that to ~0 while the loss improved,
                # because the camera and the ego stream answer most questions on
                # their own. So make it an objective instead of an observation:
                # penalise the model whenever the wrong radar is not at least
                # `margin` nats worse than the right one.
                injector.pending = core(radar_out["tokens"]).roll(shifts=1, dims=0)
                mismatched = llm(**batch).loss
                grounding = torch.clamp(args.radar_margin - (mismatched - loss),
                                        min=0.0)
                loss = loss + args.radar_contrast * grounding
            # Keeps the routed experts from collapsing onto one. Zero without
            # experts, so this is a no-op for the dense encoder.
            radar_core = encoder.module if hasattr(encoder, "module") else encoder
            balance = radar_core.router_loss()
            if balance is not None:
                loss = loss + args.router_weight * balance
            loss = loss / accum
            loss.backward()
            micro_step += 1

            if micro_step % accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                optimiser.step()
                schedule.step()
                optimiser.zero_grad(set_to_none=True)
                step += 1
                if step % 10 == 0 or step == 1:
                    seen = step * micro * world * accum
                    rate = seen / (time.monotonic() - started)
                    peak = torch.cuda.max_memory_allocated() / 2 ** 30
                    extra = ("" if grounding is None
                             else f"  hinge {grounding.item():.4f}")
                    extra += "" if numeric is None else f"  digit {numeric.item():.3f}"
                    log(rank, f"step {step}/{total_steps}  loss {loss.item()*accum:.4f}"
                              f"{extra}  {rate:.1f} sample/s  peak {peak:.1f} GiB")
                    history.append({"step": step, "loss": loss.item() * accum})
                    tracking.log({
                        "train/loss": loss.item() * accum,
                        "train/number_distance": None if numeric is None else numeric.item(),
                        "train/radar_hinge": None if grounding is None else grounding.item(),
                        "train/lr": schedule.get_last_lr()[0],
                        "perf/sample_per_s": rate,
                        "perf/peak_gib": peak,
                        "perf/samples_seen": seen,
                    }, step=step)
                # A run of a few thousand steps is hours long; without this a
                # crash at the end loses all of it.
                if args.save_every and step % args.save_every == 0:
                    save("latest")
                # 루프 안에서 잰다. 모델이 이미 올라가 있으므로 추가 메모리는
                # KV 캐시뿐이고, 밖에서 재던 17.7 GB 짜리 두 번째 사본도
                # 매번의 적재도 없다.
                if evaluator is not None:
                    evaluator.maybe_run(step, llm, encoder, connector, device)
                if args.steps and step >= args.steps:
                    break
        if args.steps and step >= args.steps:
            break

    save()
    injector.remove()
    tracking.summary({"final/step": step,
                      "final/loss": history[-1]["loss"] if history else None,
                      "final/samples": step * micro * world * accum,
                      "final/hours": (time.monotonic() - started) / 3600.0,
                      "final/out": out_dir})
    tracking.finish()
    if world > 1:
        dist.destroy_process_group()
    return 0


if __name__ == "__main__":
    sys.exit(main())
