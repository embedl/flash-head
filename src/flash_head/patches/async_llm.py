# Copyright (C) 2026 Embedl AB

"""Patch AsyncLLM.__init__ to load FlashHead metadata.

We patch AsyncLLM.__init__ in addition to LLMEngine.from_engine_args so the
metadata load also runs under `vllm serve`, which reaches the engine via
AsyncLLM.from_vllm_config and bypasses LLMEngine.from_engine_args.
"""

import logging

logger = logging.getLogger(__name__)

# Sentinel for one-shot metadata load: AsyncLLM.__init__ may be called more
# than once per process by parts of vLLM we don't control, but the metadata
# only needs to be loaded once.
_FLASH_HEAD_NOT_LOADED = object()
_flash_head_meta = _FLASH_HEAD_NOT_LOADED


def _model_id_from_vllm_config(vllm_config) -> str | None:
    """Pull the HF model id / local path out of a VllmConfig."""
    try:
        return vllm_config.model_config.model
    except AttributeError:
        return None


def patch_async_llm():
    try:
        from vllm.v1.engine.async_llm import AsyncLLM
    except Exception as e:
        logger.debug("[FlashHead] AsyncLLM not available, skipping patch: %s", e)
        return

    _original_init = AsyncLLM.__init__

    def _patched_init(self, vllm_config, *args, **kwargs):
        global _flash_head_meta
        if _flash_head_meta is _FLASH_HEAD_NOT_LOADED:
            model = _model_id_from_vllm_config(vllm_config)
            if model is not None:
                try:
                    from flash_head.loading import (
                        load_flash_head_from_checkpoint,
                        set_flash_head,
                    )
                    _flash_head_meta = load_flash_head_from_checkpoint(model)
                    set_flash_head(_flash_head_meta)
                    if _flash_head_meta:
                        logger.info("[FlashHead] Metadata saved for model: %s", model)
                except Exception as e:
                    logger.warning("[FlashHead] Metadata loading skipped: %s", e)

        return _original_init(self, vllm_config, *args, **kwargs)

    AsyncLLM.__init__ = _patched_init
    logger.info("[FlashHead] Patched AsyncLLM.__init__")
