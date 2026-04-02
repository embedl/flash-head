# Copyright (C) 2026 Embedl AB

"""Runtime monkey-patches for vLLM internals.

These patches are applied via the vllm.general_plugins entry point at
startup, replacing the previous approach of string-patching vLLM source
files and building custom Docker images.

Patched components:
1. LogitsProcessor._get_logits - intercept with FlashHead token prediction
2. Sampler.forward - bypass sampling when logits are token IDs [N, 1]
3. RejectionSampler.forward - greedy comparison for speculative decoding
4. EagleProposer._greedy_sample - handle FlashHead in draft proposals
5. GPUModelRunner._dummy_sampler_run - skip warmup for token IDs
6. LLMEngine.from_engine_args - load FlashHead metadata before engine init
"""

import logging

logger = logging.getLogger(__name__)


def apply_all():
    """Apply all FlashHead patches to vLLM."""
    from flash_head.patches.logits_processor import patch_logits_processor
    from flash_head.patches.sampler import patch_sampler
    from flash_head.patches.rejection_sampler import patch_rejection_sampler
    from flash_head.patches.eagle import patch_eagle
    from flash_head.patches.gpu_model_runner import patch_gpu_model_runner
    from flash_head.patches.llm import patch_llm

    patch_logits_processor()
    patch_sampler()
    patch_rejection_sampler()
    patch_eagle()
    patch_gpu_model_runner()
    patch_llm()

    logger.info("[FlashHead] All patches applied")
