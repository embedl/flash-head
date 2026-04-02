# Copyright (C) 2026 Embedl AB

"""Patch EagleProposer to handle FlashHead token IDs in draft proposals."""

import logging

logger = logging.getLogger(__name__)


def patch_eagle():
    try:
        import torch
        from vllm.v1.spec_decode.eagle import EagleProposer
    except ImportError:
        logger.debug("[FlashHead] EagleProposer not available, skipping patch")
        return

    # Find the greedy sample method -- name varies across vLLM versions
    method_name = None
    for name in ("_greedy_sample", "greedy_sample", "_sample_greedy"):
        if hasattr(EagleProposer, name):
            method_name = name
            break

    if method_name is None:
        logger.debug("[FlashHead] EagleProposer greedy_sample method not found, skipping patch")
        return

    _original = getattr(EagleProposer, method_name)

    def _patched_greedy_sample(self, hidden_states):
        if hasattr(self, "use_local_argmax_reduction") and self.use_local_argmax_reduction:
            return self.model.get_top_tokens(hidden_states)
        logits = self.model.compute_logits(hidden_states)
        # FlashHead: logits is [N, 1] token IDs
        if logits.ndim == 2 and logits.shape[1] == 1:
            return logits.squeeze(-1).to(torch.int64)
        return logits.argmax(dim=-1)

    setattr(EagleProposer, method_name, _patched_greedy_sample)
    logger.info("[FlashHead] Patched EagleProposer.%s", method_name)
