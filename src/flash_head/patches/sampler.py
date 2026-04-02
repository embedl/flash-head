# Copyright (C) 2026 Embedl AB

"""Patch Sampler to detect FlashHead token IDs and bypass sampling."""

import logging

logger = logging.getLogger(__name__)


def patch_sampler():
    import torch
    from vllm.v1.sample.sampler import Sampler, SamplerOutput

    _original_forward = Sampler.forward

    def _patched_forward(self, logits, sampling_metadata, predict_bonus_token=False, logprobs_mode_override=None):
        # FlashHead: logits is actually token IDs [batch, 1]
        if logits.ndim == 1 or (logits.ndim == 2 and logits.shape[-1] == 1):
            sampled = logits.view(-1).to(torch.int32)
            return SamplerOutput(
                sampled_token_ids=sampled.unsqueeze(-1),
                logprobs_tensors=None,
            )

        return _original_forward(
            self, logits, sampling_metadata,
            predict_bonus_token=predict_bonus_token,
            logprobs_mode_override=logprobs_mode_override,
        )

    Sampler.forward = _patched_forward
    logger.info("[FlashHead] Patched Sampler.forward")
