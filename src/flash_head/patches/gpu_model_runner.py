# Copyright (C) 2026 Embedl AB

"""Patch gpu_model_runner to handle FlashHead token IDs in warmup."""

import logging

logger = logging.getLogger(__name__)


def patch_gpu_model_runner():
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    except ImportError:
        logger.debug("[FlashHead] GPUModelRunner not available, skipping patch")
        return

    if not hasattr(GPUModelRunner, "_dummy_sampler_run"):
        logger.debug("[FlashHead] GPUModelRunner._dummy_sampler_run not found, skipping patch")
        return

    _original_dummy_sampler_run = GPUModelRunner._dummy_sampler_run

    def _patched_dummy_sampler_run(self, *args, **kwargs):
        try:
            return _original_dummy_sampler_run(self, *args, **kwargs)
        except Exception:
            # FlashHead returns [N, 1] which may break warmup assumptions
            import torch
            hidden_states = args[0] if args else kwargs.get("hidden_states")
            if hidden_states is not None:
                logits = self.model.compute_logits(hidden_states)
                if logits is not None and (logits.dim() < 2 or logits.size(1) == 1):
                    return logits
            raise

    GPUModelRunner._dummy_sampler_run = _patched_dummy_sampler_run
    logger.info("[FlashHead] Patched GPUModelRunner._dummy_sampler_run")
