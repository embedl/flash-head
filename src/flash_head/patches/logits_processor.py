# Copyright (C) 2026 Embedl AB

"""Patch LogitsProcessor to intercept _get_logits with FlashHead."""

import logging

import torch

logger = logging.getLogger(__name__)

# Sentinel for lazy loading
_FLASH_HEAD_NOT_LOADED = object()
_flash_head = _FLASH_HEAD_NOT_LOADED


def _get_flash_head():
    global _flash_head
    if _flash_head is _FLASH_HEAD_NOT_LOADED:
        from flash_head.loading import get_flash_head
        _flash_head = get_flash_head()
    return _flash_head


def patch_logits_processor():
    from vllm.model_executor.layers.logits_processor import LogitsProcessor

    _original_get_logits = LogitsProcessor._get_logits

    def _patched_get_logits(self, hidden_states, lm_head, embedding_bias):
        flash_head = _get_flash_head()
        # Only use FlashHead for single-token decode; let vLLM handle
        # prefill natively (shape[0] > 1 means multiple tokens).
        if flash_head is not None and hidden_states.shape[0] == 1:
            hs = hidden_states.unsqueeze(0) if hidden_states.dim() == 2 else hidden_states
            token_ids = flash_head.get_next_token(hs)
            if token_ids.dim() == 0:
                token_ids = token_ids.view(1, 1)
            elif token_ids.dim() == 1:
                token_ids = token_ids.unsqueeze(-1)
            return token_ids.to(torch.int32)

        return _original_get_logits(self, hidden_states, lm_head, embedding_bias)

    LogitsProcessor._get_logits = _patched_get_logits
    logger.info("[FlashHead] Patched LogitsProcessor._get_logits")
