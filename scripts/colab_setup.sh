#!/usr/bin/env bash
# scripts/colab_setup.sh — one-shot environment setup for Google Colab (or any
# fresh Linux box with an NVIDIA GPU).  Safe to re-run.
#
#   git clone https://github.com/TryingtobeingNikhil/vLLM_Inference_Engine.git
#   cd vLLM_Inference_Engine && bash scripts/colab_setup.sh
set -euo pipefail
cd "$(dirname "$0")/.."

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
  echo "WARNING: no NVIDIA GPU visible. In Colab: Runtime ▸ Change runtime type ▸ T4/L4/A100 GPU."
fi

# Colab already ships a CUDA build of torch that satisfies torch>=2.3, so pip
# leaves it alone; only the missing / out-of-range packages are installed.
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt

python - <<'PY'
import torch, transformers
print(f"torch {torch.__version__} (CUDA {torch.version.cuda}) | transformers {transformers.__version__}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    bf16 = "bf16" if p.major >= 8 else "fp16 (no bf16 on this GPU)"
    print(f"GPU: {p.name}, {p.total_memory / 2**30:.1f} GB, compute {p.major}.{p.minor}, dtype → {bf16}")
from benchmarks.common import pick_model
print("Default benchmark model for this GPU:", pick_model("auto"))
PY
echo "Setup complete."
