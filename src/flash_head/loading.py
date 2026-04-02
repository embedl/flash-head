# Copyright (C) 2025, 2026 Embedl AB

"""Loading and saving FlashHead state for vLLM inference."""

import json
import logging
import os
import tempfile
from typing import Optional, Tuple

import torch
from huggingface_hub import hf_hub_download, try_to_load_from_cache
from safetensors import safe_open
from torch import nn
from flash_head.flash_head import FlashHead, get_flash_head_parameters

logger = logging.getLogger(__name__)

METADATA_PATH = os.path.join(tempfile.gettempdir(), "flashhead_metadata.pt")

LM_HEAD_KEYS = [
    "lm_head.weight",
    "model.lm_head.weight",
    "transformer.lm_head.weight",
    "model.embed_tokens.weight",  # tied embedding fallback
]


# ---------------------------------------------------------------------------
# File resolution — prefer HF cache, fall back to download
# ---------------------------------------------------------------------------

def _is_local_dir(path: str) -> bool:
    return os.path.isdir(path)


def _resolve_file(model: str, filename: str) -> Optional[str]:
    """Resolve a model file to a local path.

    For local directories, check directly.  For HF repos, check the local
    HF cache first (no network) and only download if not cached.
    """
    if _is_local_dir(model):
        p = os.path.join(model, filename)
        return p if os.path.exists(p) else None
    try:
        cached = try_to_load_from_cache(model, filename)
        if isinstance(cached, str):
            return cached
        return hf_hub_download(repo_id=model, filename=filename)
    except Exception:
        return None


def _find_weight_key_in_index(index_json: dict) -> Optional[str]:
    weight_map = index_json.get("weight_map", {})
    for k in LM_HEAD_KEYS:
        if k in weight_map:
            return k
    return None


def _load_lm_head_weight(model: str) -> Tuple[torch.Tensor, str]:
    index_path = _resolve_file(model, "model.safetensors.index.json")
    if index_path is not None:
        with open(index_path, encoding="utf-8") as f:
            index = json.load(f)

        chosen = _find_weight_key_in_index(index)
        if chosen is None:
            raise KeyError(
                f"No lm_head/tied embedding key found in index. "
                f"Looked for: {LM_HEAD_KEYS}"
            )

        shard_name = index["weight_map"][chosen]
        shard_path = _resolve_file(model, shard_name)
        if shard_path is None:
            raise FileNotFoundError(f"Could not resolve shard: {shard_name}")

        with safe_open(shard_path, framework="pt", device="cpu") as f:
            if chosen not in f.keys():
                raise KeyError(
                    f"Expected {chosen} in {shard_name}, but not found."
                )
            return f.get_tensor(chosen), chosen

    st_path = _resolve_file(model, "model.safetensors")
    if st_path is not None:
        with safe_open(st_path, framework="pt", device="cpu") as f:
            available_keys = f.keys()
            for k in LM_HEAD_KEYS:
                if k in available_keys:
                    return f.get_tensor(k), k
        raise KeyError(
            f"Could not find lm_head/tied embedding weight in model.safetensors. "
            f"Looked for: {LM_HEAD_KEYS}"
        )

    raise FileNotFoundError(
        f"No supported weight files found for {model}. "
        f"Expected model.safetensors(.index.json)."
    )


# ---------------------------------------------------------------------------
# Config reading
# ---------------------------------------------------------------------------

def _read_config_json(model: str) -> Optional[dict]:
    """Read config.json without AutoConfig."""
    path = _resolve_file(model, "config.json")
    if path is None:
        return None
    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# FlashHead lazy loading for vLLM worker processes
# ---------------------------------------------------------------------------

def get_flash_head() -> Optional[nn.Module]:
    """Load FlashHead lazily when first needed (in worker process).

    Reads metadata from METADATA_PATH and constructs
    FlashHead on GPU only when first accessed during inference.
    """
    if not os.path.exists(METADATA_PATH):
        return None

    metadata = torch.load(METADATA_PATH, map_location="cpu", weights_only=True)

    model_path = metadata["model_path"]
    cache_dir = metadata["cache_dir"]
    vocab_size = metadata["vocab_size"]
    hidden_size = metadata["hidden_size"]
    special_token_ids = metadata["special_token_ids"]
    dtype_str = metadata.get("dtype", "torch.bfloat16")
    dtype = torch.bfloat16 if dtype_str == "torch.bfloat16" else torch.float16

    w, chosen_key = _load_lm_head_weight(model_path)
    if w.shape != (vocab_size, hidden_size):
        if w.shape == (hidden_size, vocab_size):
            w = w.t().contiguous()
        else:
            raise ValueError(
                f"Unexpected lm_head weight shape {tuple(w.shape)}; "
                f"expected {(vocab_size, hidden_size)} or {(hidden_size, vocab_size)}"
            )

    dummy_lm_head = torch.nn.Linear(hidden_size, vocab_size, bias=False)
    dummy_lm_head.weight.data.copy_(w)

    flash_head = FlashHead(
        dummy_lm_head,
        **get_flash_head_parameters(
            dummy_lm_head,
            cache_dir=cache_dir,
            model_or_dir=model_path,
        ),
        special_token_ids=special_token_ids,
    ).to(device="cuda", dtype=dtype)

    logger.info("[FlashHead] Loaded lazily on GPU using '%s'", chosen_key)
    return flash_head


def set_flash_head(flash_head_or_metadata):
    """Save FlashHead metadata for lazy loading in worker process."""
    if flash_head_or_metadata is not None:
        torch.save(flash_head_or_metadata, METADATA_PATH)
    else:
        if os.path.exists(METADATA_PATH):
            os.remove(METADATA_PATH)


def load_flash_head_from_checkpoint(model: str, dtype=torch.bfloat16):
    """Prepare FlashHead metadata without loading to GPU.

    vLLM spawns worker processes, so we save metadata and let the worker
    reconstruct FlashHead on GPU to avoid memory duplication.
    """
    config_dict = _read_config_json(model)
    if config_dict is None or "flash_head_cache_dir" not in config_dict:
        return None

    cache_dir = config_dict["flash_head_cache_dir"]
    if _is_local_dir(model) and not os.path.isabs(cache_dir):
        cache_dir = os.path.join(model, cache_dir)

    # Look for vocab_size/hidden_size in config or nested text_config
    text_config = config_dict.get("text_config", {})
    vocab_size = config_dict.get("vocab_size") or text_config.get("vocab_size")
    hidden_size = config_dict.get("hidden_size") or text_config.get("hidden_size")
    special_token_ids = config_dict.get("flash_head_special_token_ids")

    metadata = {
        "model_path": model,
        "cache_dir": cache_dir,
        "vocab_size": vocab_size,
        "hidden_size": hidden_size,
        "special_token_ids": special_token_ids,
        "dtype": str(dtype),
    }

    logger.info("[FlashHead] Metadata prepared for lazy loading from %s", cache_dir)
    return metadata
