"""
Tiny randomly-initialised models for fast, offline engine tests.

Real checkpoints are slow to load and need a network; correctness of the
paged-attention engine does not depend on the weights, only on the
architecture.  These helpers build 2-layer Qwen2 / Llama models (GQA,
explicit head_dim) in float64 so engine output can be compared with
HuggingFace ``generate()`` token-for-token.
"""

from __future__ import annotations

import os
from typing import List

import torch
from transformers import LlamaConfig, LlamaForCausalLM, Qwen2Config, Qwen2ForCausalLM

VOCAB = 97


class FakeTokenizer:
    """Minimal tokenizer: one token per character (mod VOCAB)."""

    eos_token_id = None
    pad_token_id = 0
    eos_token = None

    def __call__(self, text: str, add_special_tokens: bool = True):
        return {"input_ids": [ord(c) % VOCAB for c in text]}

    def decode(self, ids: List[int], skip_special_tokens: bool = True) -> str:
        return " ".join(str(i) for i in ids)


def tiny_qwen2(seed: int = 0, layers: int = 2, dtype=torch.float64, vocab: int = VOCAB):
    torch.manual_seed(seed)
    config = Qwen2Config(
        vocab_size=vocab, hidden_size=64, intermediate_size=128,
        num_hidden_layers=layers, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=512,
    )
    return Qwen2ForCausalLM(config).to(dtype).eval()


def tiny_llama(seed: int = 0, dtype=torch.float64):
    """Llama with head_dim != hidden/heads and 4:1 GQA."""
    torch.manual_seed(seed)
    config = LlamaConfig(
        vocab_size=VOCAB, hidden_size=64, intermediate_size=128,
        num_hidden_layers=2, num_attention_heads=8, num_key_value_heads=2,
        head_dim=16, max_position_embeddings=512,
    )
    return LlamaForCausalLM(config).to(dtype).eval()


def hf_greedy(model, prompt: List[int], n: int) -> List[int]:
    """Reference: HuggingFace generate(), exactly *n* greedy tokens."""
    ids = torch.tensor([prompt], device=next(model.parameters()).device)
    out = model.generate(
        ids,
        attention_mask=torch.ones_like(ids),
        max_new_tokens=n,
        min_new_tokens=n,
        do_sample=False,
        pad_token_id=0,
        eos_token_id=None,
    )
    return out[0, len(prompt):].tolist()


def save_tiny_checkpoint(path: str) -> str:
    """Write a tiny Qwen2 model + byte-level BPE tokenizer to *path*.

    Lets the real server code path (``from_pretrained`` by name) run offline.
    """
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    alphabet = sorted(pre_tokenizers.ByteLevel.alphabet())
    vocab = {ch: i for i, ch in enumerate(alphabet)}
    eos_id = len(vocab)
    vocab["<eos>"] = eos_id
    tok = Tokenizer(models.BPE(vocab=vocab, merges=[]))
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    fast = PreTrainedTokenizerFast(tokenizer_object=tok, eos_token="<eos>", pad_token="<eos>")
    fast.save_pretrained(path)

    model = tiny_qwen2(seed=0, dtype=torch.float32, vocab=eos_id + 1)
    model.config.eos_token_id = eos_id
    model.generation_config.eos_token_id = eos_id
    model.save_pretrained(path)
    return os.path.abspath(path)
