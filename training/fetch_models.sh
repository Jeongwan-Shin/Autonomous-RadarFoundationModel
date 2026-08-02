#!/bin/bash
# Pull the three Qwen3-VL tiers. snapshot_download is right here: few large
# shards, so the per-file API overhead that forced the resolve-endpoint trick for
# the 1,838 tiny extrinsics files is irrelevant, and resume plus integrity
# checking matter for 146 GB.
set -u
source /NHNHOME/workspace/venv/av/bin/activate
export HF_HOME=/NHNHOME/workspace/.hf_cache
DEST=/NHNHOME/workspace/models
for repo in Qwen/Qwen3-VL-8B-Instruct Qwen/Qwen3-VL-32B-Instruct Qwen/Qwen3-VL-30B-A3B-Instruct; do
  name=$(basename "$repo")
  echo "=== $repo -> $DEST/$name"
  python - "$repo" "$DEST/$name" <<'PY'
import sys, time
from huggingface_hub import snapshot_download
repo, dest = sys.argv[1], sys.argv[2]
t0 = time.time()
snapshot_download(repo, local_dir=dest, max_workers=8,
                  allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model",
                                  "*.py", "*.jinja"])
print(f"done in {(time.time()-t0)/60:.1f} min")
PY
done
echo ALL_MODELS_DONE
