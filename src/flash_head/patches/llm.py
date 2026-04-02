# Copyright (C) 2026 Embedl AB

"""Patch LLMEngine.from_engine_args to load FlashHead metadata.

We patch LLMEngine.from_engine_args instead of LLM.__init__ because the
plugin's register() is called from EngineArgs.__post_init__(), which happens
INSIDE LLM.__init__. By the time register() patches LLM.__init__, we're
already executing the original. But LLMEngine.from_engine_args is called
AFTER EngineArgs construction, so our patch IS in effect.
"""

import logging

logger = logging.getLogger(__name__)


def patch_llm():
    from vllm.v1.engine.llm_engine import LLMEngine

    _original_from_engine_args = LLMEngine.from_engine_args

    @classmethod
    def _patched_from_engine_args(cls, engine_args, *args, **kwargs):
        model = engine_args.model
        try:
            from flash_head.loading import (
                load_flash_head_from_checkpoint,
                set_flash_head,
            )
            flash_head_meta = load_flash_head_from_checkpoint(model)
            set_flash_head(flash_head_meta)
            if flash_head_meta:
                logger.info("[FlashHead] Metadata saved for model: %s", model)
            else:
                _warn_if_flashhead_model(model)
        except Exception as e:
            logger.warning("[FlashHead] Metadata loading skipped: %s", e)

        return _original_from_engine_args.__func__(cls, engine_args, *args, **kwargs)

    LLMEngine.from_engine_args = _patched_from_engine_args
    logger.info("[FlashHead] Patched LLMEngine.from_engine_args")


def _warn_if_flashhead_model(model: str):
    """Check if a model expects FlashHead but it didn't activate."""
    import os
    if not os.path.isdir(model):
        return
    config_path = os.path.join(model, "config.json")
    if not os.path.exists(config_path):
        return
    import json
    try:
        with open(config_path) as f:
            config = json.load(f)
        if config.get("flash_head_cache_dir"):
            logger.warning(
                "\n"
                "================================================================\n"
                " WARNING: Model has flash_head_cache_dir in config but\n"
                " FlashHead did NOT activate. Inference will be slower.\n"
                " Check that FLASHHEAD_ENABLED is not set to 0.\n"
                "================================================================"
            )
    except Exception:
        pass
