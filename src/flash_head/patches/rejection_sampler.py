# Copyright (C) 2026 Embedl AB

"""Patch RejectionSampler to handle FlashHead token IDs in spec decode."""

import logging

logger = logging.getLogger(__name__)


def patch_rejection_sampler():
    import torch

    try:
        from vllm.v1.sample.rejection_sampler import (
            RejectionSampler,
            SamplerOutput,
            PLACEHOLDER_TOKEN_ID,
        )
    except ImportError:
        logger.debug("[FlashHead] RejectionSampler not available, skipping patch")
        return

    _original_forward = RejectionSampler.forward

    def _patched_forward(self, logits, sampling_metadata):
        # We need to intercept after target_logits are extracted.
        # The challenge is that RejectionSampler.forward does a lot of work
        # before we can check the shape. We use a different approach:
        # patch the method to check if logits contain FlashHead token IDs.
        #
        # FlashHead token IDs have shape [N, 1]. If ANY logits in the batch
        # have this shape, we know FlashHead is active.
        if logits.ndim == 2 and logits.shape[1] == 1:
            # All logits are FlashHead token IDs -- handle entirely
            return _flashhead_rejection(self, logits, sampling_metadata)

        return _original_forward(self, logits, sampling_metadata)

    def _flashhead_rejection(self, logits, sampling_metadata):
        metadata = sampling_metadata
        # Extract target and bonus token IDs from the flat logits
        target_logits_indices = metadata.target_logits_indices
        bonus_logits_indices = metadata.bonus_logits_indices

        raw_target_logits = logits[target_logits_indices]

        # FlashHead: target_logits is [N, 1] token IDs
        target_ids = raw_target_logits.squeeze(-1).to(torch.int32)

        # Bonus token IDs
        bonus_logits = logits[bonus_logits_indices]
        bonus_token_ids = bonus_logits.squeeze(-1).to(torch.int32).unsqueeze(-1)

        draft_ids = metadata.draft_token_ids
        num_draft_tokens = metadata.num_draft_tokens
        batch_size = len(num_draft_tokens)
        max_spec_len = metadata.max_spec_len

        output_token_ids = torch.full(
            (batch_size, max_spec_len + 1), PLACEHOLDER_TOKEN_ID,
            dtype=torch.int32, device=target_ids.device,
        )

        offset = 0
        for i in range(batch_size):
            n_draft = num_draft_tokens[i]
            all_ok = True
            for j in range(n_draft):
                if draft_ids[offset + j] == target_ids[offset + j]:
                    output_token_ids[i, j] = draft_ids[offset + j]
                else:
                    output_token_ids[i, j] = target_ids[offset + j]
                    all_ok = False
                    break
            if all_ok and n_draft > 0:
                output_token_ids[i, n_draft] = bonus_token_ids[i, 0]
            offset += n_draft

        return SamplerOutput(
            sampled_token_ids=output_token_ids,
            logprobs_tensors=None,
        )

    RejectionSampler.forward = _patched_forward
    logger.info("[FlashHead] Patched RejectionSampler.forward")
