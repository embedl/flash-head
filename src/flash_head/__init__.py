# Copyright (C) 2026 Embedl AB

"""FlashHead: fast approximate language model head as a vLLM plugin.

FlashHead replaces the standard lm_head (final vocabulary projection layer)
with a clustering-based approximation that only evaluates logits for tokens
in the top-k most similar clusters, delivering significant speedups on
edge devices.

This package integrates FlashHead into vLLM via the official plugin system
(vllm.general_plugins entry point), eliminating the need for source patches
or custom Docker images.
"""

import logging
import os

from flash_head._version import __version__

logger = logging.getLogger(__name__)

_patches_applied = False


def register():
    """vLLM plugin entry point. Called in every process before model init."""
    global _patches_applied
    if _patches_applied:
        return
    _patches_applied = True

    _register_architectures()

    if os.environ.get("FLASHHEAD_ENABLED", "1") == "0":
        logger.info("[FlashHead] Disabled via FLASHHEAD_ENABLED=0")
        return

    from flash_head.patches import apply_all

    apply_all()
    logger.info("[FlashHead] Plugin registered")


def _register_architectures():
    """Register FlashHead model architectures so vLLM recognizes them.

    Models published with architectures like 'FlashHeadQwen3VLForConditionalGeneration'
    will fail to load without this plugin -- giving a clear error instead of
    silently falling back to the slow standard lm_head path.
    """
    try:
        from vllm import ModelRegistry

        # Map FlashHead architecture names to their base vLLM model classes.
        # Uses lazy string imports to avoid premature CUDA initialization.
        # The FlashHead interception happens via the LogitsProcessor patch,
        # not via a custom model class, so we just need vLLM to accept the
        # architecture name and load the base model.
        _FLASHHEAD_ARCHITECTURES = {
            "FlashHeadLlamaForCausalLM": "vllm.model_executor.models.llama:LlamaForCausalLM",
            "FlashHeadQwen3ForCausalLM": "vllm.model_executor.models.qwen3:Qwen3ForCausalLM",
            "FlashHeadQwen3VLForConditionalGeneration": "vllm.model_executor.models.qwen3_vl:Qwen3VLForConditionalGeneration",
            "FlashHeadGemma3ForCausalLM": "vllm.model_executor.models.gemma3:Gemma3ForCausalLM",
        }

        supported = ModelRegistry.get_supported_archs()
        for fh_arch, model_cls_path in _FLASHHEAD_ARCHITECTURES.items():
            if fh_arch not in supported:
                ModelRegistry.register_model(fh_arch, model_cls_path)
                logger.info("[FlashHead] Registered architecture %s", fh_arch)
    except Exception as e:
        logger.debug("[FlashHead] Architecture registration skipped: %s", e)
