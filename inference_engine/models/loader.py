"""
models/loader.py — Model and tokenizer loading with device-aware strategy.

Key design decisions
--------------------
* CUDA: weights are loaded straight onto the GPU (``device_map={"": idx}``).
  Loading to CPU first would need the full fp16 model in host RAM (15 GB for a
  7B model — more than a standard Colab VM has).  We deliberately avoid
  ``device_map="auto"``: when the GPU is tight it silently offloads layers to
  CPU and every forward pass becomes 10-100x slower.
* dtype "auto": bfloat16 on Ampere+ (A100/L4), float16 on Turing (T4, no
  bf16 support) and MPS, float32 on CPU (fp16 matmuls on CPU are emulated).
* MPS (Apple Silicon): load on CPU then ``.to("mps")``.
"""

from __future__ import annotations

import logging
from typing import List, NamedTuple, Optional

import torch
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer, PreTrainedModel, PreTrainedTokenizerBase

from inference_engine.config import Config

logger = logging.getLogger(__name__)

_DTYPES = {
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "float32": torch.float32,
}


class LoadedModel(NamedTuple):
    model: PreTrainedModel
    tokenizer: PreTrainedTokenizerBase
    device: str


def resolve_dtype(dtype: str, device: str) -> torch.dtype:
    """Map the config dtype string to a torch dtype for *device*."""
    if dtype != "auto":
        return _DTYPES[dtype]
    if device == "cuda":
        major, _ = torch.cuda.get_device_capability()
        return torch.bfloat16 if major >= 8 else torch.float16
    if device == "mps":
        return torch.float16
    return torch.float32


def _dtype_kwarg() -> str:
    """transformers renamed ``torch_dtype`` to ``dtype`` in 4.56.

    Older versions forward unknown kwargs into the model config, so passing
    ``dtype`` there would silently load fp32 weights — check the version.
    """
    major, minor = (int(x) for x in transformers.__version__.split(".")[:2])
    return "dtype" if (major, minor) >= (4, 56) else "torch_dtype"


def load_model(model_name: str, device: str, dtype: str = "auto") -> PreTrainedModel:
    """Load a causal LM onto *device* in eval mode."""
    torch_dtype = resolve_dtype(dtype, device)
    kwargs = {
        _dtype_kwarg(): torch_dtype,
        "attn_implementation": "sdpa",
        "low_cpu_mem_usage": True,
    }
    logger.info("Loading model '%s' on %s (%s)", model_name, device, torch_dtype)

    if device == "cuda":
        kwargs["device_map"] = {"": torch.cuda.current_device()}
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
        model = model.to(device)

    model.eval()
    logger.info(
        "Model ready. Parameters: %.1f M | dtype: %s",
        sum(p.numel() for p in model.parameters()) / 1e6,
        next(model.parameters()).dtype,
    )
    return model


def load_tokenizer(model_name: str) -> PreTrainedTokenizerBase:
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    # Ensure a pad token exists (some models omit it).
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def load_model_and_tokenizer(config: Config, model_name: Optional[str] = None) -> LoadedModel:
    """
    Load a causal LM and its tokenizer according to config.device.

    Returns a LoadedModel named tuple so callers can unpack as:
        model, tokenizer, device = load_model_and_tokenizer(config)
    """
    name = model_name or config.model_name
    tokenizer = load_tokenizer(name)
    model = load_model(name, config.device, config.dtype)
    return LoadedModel(model=model, tokenizer=tokenizer, device=config.device)


def get_eos_token_ids(model: PreTrainedModel, tokenizer: PreTrainedTokenizerBase) -> List[int]:
    """All token ids that end generation.

    Instruct models often have several (e.g. Qwen2.5 stops on both
    ``<|im_end|>`` and ``<|endoftext|>``); the complete list lives in
    ``generation_config`` rather than on the tokenizer.
    """
    ids: set[int] = set()
    gen_cfg = getattr(model, "generation_config", None)
    gen_eos = getattr(gen_cfg, "eos_token_id", None) if gen_cfg is not None else None
    if isinstance(gen_eos, int):
        ids.add(gen_eos)
    elif isinstance(gen_eos, (list, tuple)):
        ids.update(int(t) for t in gen_eos)
    if getattr(tokenizer, "eos_token_id", None) is not None:
        ids.add(int(tokenizer.eos_token_id))
    return sorted(ids)
