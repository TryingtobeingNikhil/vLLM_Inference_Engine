"""
engine/attention_wrapper.py — Phase 8: Attention Layer Integration.

Provides helper functions that reconstruct HuggingFace-compatible
past_key_values from the PagedKVCacheManager tensor pool, enabling the
decode forward pass to read KV state from the paged pool instead of the
live Sequence.past_key_values tensor.

No async.  No custom CUDA kernel.  All torch operations are device-agnostic.
"""

from __future__ import annotations

import logging
from typing import Any

import torch

logger = logging.getLogger(__name__)

# Module-level flag to warn exactly once if DynamicCache is unavailable.
_dynamic_cache_warned: bool = False


# ── Internal compat helper (mirrors scheduler.py's _extract_kv_layer) ─────────

def _extract_kv_layer(
    past_key_values: Any,
    layer_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract (key, value) tensors for one layer from past_key_values.

    Handles three formats returned by different transformers versions:

    1. Legacy tuple-of-tuples (transformers < ~4.36):
       ``past_key_values[layer_idx]`` → ``(key_tensor, value_tensor)``
       shape: [batch, num_kv_heads, seq_len, head_dim]

    2. DynamicCache with key_cache / value_cache lists (transformers ~4.36-4.40):
       ``past_key_values.key_cache[layer_idx]`` → key tensor
       ``past_key_values.value_cache[layer_idx]`` → value tensor

    3. DynamicCache with layers list (transformers >= ~4.41):
       ``past_key_values.layers[layer_idx].keys`` → key tensor
       ``past_key_values.layers[layer_idx].values`` → value tensor
       shape: [batch, num_kv_heads, seq_len, head_dim]

    Returns
    -------
    (key_tensor, value_tensor)
        Both shaped [batch, num_kv_heads, seq_len, head_dim].
    """
    # Format 3: DynamicCache with .layers list holding DynamicLayer objects
    if hasattr(past_key_values, "layers"):
        layer = past_key_values.layers[layer_idx]
        return layer.keys, layer.values

    # Format 2: DynamicCache with .key_cache / .value_cache lists
    if hasattr(past_key_values, "key_cache"):
        return (
            past_key_values.key_cache[layer_idx],
            past_key_values.value_cache[layer_idx],
        )

    # Format 1: Legacy tuple-of-tuples
    layer = past_key_values[layer_idx]
    return layer[0], layer[1]


# ── Public API ─────────────────────────────────────────────────────────────────

def reconstruct_past_key_values(
    seq_id: str,
    paged_kv_cache: "Any",  # PagedKVCacheManager — avoid circular import
    num_layers: int,
    device: str,
) -> tuple:
    """Assemble a legacy tuple-of-tuples past_key_values from the paged pool.

    Reads all filled KV slots for *seq_id* from the paged pool, reshapes from
    ``[total_tokens, num_kv_heads, head_dim]`` to
    ``[1, num_kv_heads, total_tokens, head_dim]`` and returns a tuple of
    ``(key, value)`` pairs, one per layer.

    Parameters
    ----------
    seq_id:
        Sequence whose KV tensors to assemble.
    paged_kv_cache:
        PagedKVCacheManager instance.
    num_layers:
        Number of transformer layers.
    device:
        Target device string (e.g. ``"cpu"``, ``"cuda"``, ``"mps"``).

    Returns
    -------
    tuple of length num_layers
        Each element is ``(key_tensor, value_tensor)`` with shapes
        ``[1, num_kv_heads, total_tokens, head_dim]``.
    """
    layers = []
    for layer_idx in range(num_layers):
        keys, values = paged_kv_cache.read_kv_sequence(seq_id, layer_idx)
        # keys:   [total_tokens, num_kv_heads, head_dim]
        # Reshape to [1, num_kv_heads, total_tokens, head_dim]
        key_tensor = keys.unsqueeze(0).permute(0, 2, 1, 3).to(device)
        value_tensor = values.unsqueeze(0).permute(0, 2, 1, 3).to(device)
        layers.append((key_tensor, value_tensor))
    return tuple(layers)


def reconstruct_dynamic_cache(
    seq_id: str,
    paged_kv_cache: "Any",  # PagedKVCacheManager
    num_layers: int,
    device: str,
) -> Any:
    """Assemble a DynamicCache past_key_values object from the paged pool.

    Same logic as :func:`reconstruct_past_key_values` but returns a
    ``transformers.DynamicCache`` for compatibility with newer transformers.

    Parameters
    ----------
    seq_id, paged_kv_cache, num_layers, device:
        See :func:`reconstruct_past_key_values`.

    Returns
    -------
    DynamicCache
        Populated with all filled KV slots for *seq_id*.
    """
    from transformers import DynamicCache

    cache = DynamicCache()
    for layer_idx in range(num_layers):
        keys, values = paged_kv_cache.read_kv_sequence(seq_id, layer_idx)
        key_tensor = keys.unsqueeze(0).permute(0, 2, 1, 3).to(device)
        value_tensor = values.unsqueeze(0).permute(0, 2, 1, 3).to(device)
        cache.update(key_tensor, value_tensor, layer_idx)
    return cache


def build_past_key_values(
    seq_id: str,
    paged_kv_cache: "Any",  # PagedKVCacheManager
    num_layers: int,
    device: str,
    use_dynamic_cache: bool = True,
) -> Any:
    """Entry point: assemble past_key_values from the paged pool.

    Tries DynamicCache first when *use_dynamic_cache* is True.
    Falls back to the legacy tuple format if the import fails, warning once.

    Parameters
    ----------
    seq_id, paged_kv_cache, num_layers, device:
        Forwarded to the underlying reconstruct function.
    use_dynamic_cache:
        When True, attempt to build a ``DynamicCache`` object.

    Returns
    -------
    DynamicCache | tuple
        Past key-value representation compatible with HuggingFace model.forward().
    """
    global _dynamic_cache_warned

    if use_dynamic_cache:
        try:
            return reconstruct_dynamic_cache(seq_id, paged_kv_cache, num_layers, device)
        except Exception:
            if not _dynamic_cache_warned:
                logger.warning(
                    "DynamicCache unavailable or failed — falling back to "
                    "legacy tuple format for past_key_values."
                )
                _dynamic_cache_warned = True

    return reconstruct_past_key_values(seq_id, paged_kv_cache, num_layers, device)


def extract_new_token_kv(
    past_key_values: Any,
    layer_idx: int,
    token_position: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract KV tensors for a newly generated token from a model output.

    After a single-token decode forward pass the model appends one new token
    to ``past_key_values``.  This function extracts that token's key and value
    slices at ``token_position`` in the sequence dimension.

    Handles all three past_key_values formats (legacy tuple, DynamicCache with
    key_cache, DynamicCache with layers) using the same priority order as
    Phase 7's ``_extract_kv_layer``.

    Parameters
    ----------
    past_key_values:
        Output past_key_values from a HuggingFace model forward pass.
    layer_idx:
        Layer to extract from.
    token_position:
        The global token position of the newly generated token.
        Used to index position ``-1`` in the sequence dimension
        (the newly appended token).

    Returns
    -------
    (key_slice, value_slice)
        Both shaped ``[num_kv_heads, head_dim]``.
    """
    layer_key, layer_val = _extract_kv_layer(past_key_values, layer_idx)
    # layer_key shape: [batch=1, num_kv_heads, seq_len, head_dim]
    # The newly generated token is always the last slot.
    key_slice = layer_key[0, :, -1, :]   # [num_kv_heads, head_dim]
    value_slice = layer_val[0, :, -1, :]
    return key_slice, value_slice
