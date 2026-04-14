# Copyright (C) 2025, 2026 Embedl AB

"""Flash head implementation for faster efficient language model head."""

import json
import os
from typing import Iterable, Optional, Union

# Default ratio: number of clusters = vocab_size / DEFAULT_CLUSTER_RATIO
DEFAULT_CLUSTER_RATIO = 16

import torch
from huggingface_hub import hf_hub_download, try_to_load_from_cache
from safetensors.torch import load_file
from torch import nn


def _get_device():
    """Get the device lazily to avoid initializing CUDA at import time."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_asset(model_or_dir: str, relative_path: str) -> str:
    """Resolve a model asset to a local path, preferring HF cache."""
    if os.path.isdir(model_or_dir):
        p = os.path.join(model_or_dir, relative_path)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing local asset: {p}")
        return p
    cached = try_to_load_from_cache(model_or_dir, relative_path)
    if isinstance(cached, str):
        return cached
    return hf_hub_download(repo_id=model_or_dir, filename=relative_path)


def _get_centroids(
    lm_head: nn.Linear,
    model_or_dir: str,
    cache_dir: str,
) -> Optional[tuple[torch.Tensor, torch.Tensor]]:
    original_shape = lm_head.weight.shape  # (vocab, hidden)

    cache_file_rel = os.path.join(cache_dir, "clustering_cache.safetensors")
    meta_file_rel = os.path.join(cache_dir, "clustering_config.json")

    cache_file = _resolve_asset(model_or_dir, cache_file_rel)
    meta_file = _resolve_asset(model_or_dir, meta_file_rel)

    try:
        with open(meta_file, encoding="utf-8") as f:
            metadata = json.load(f)

        if metadata.get("format") not in (None, "safetensors"):
            raise ValueError(
                f"Expected safetensors format, found: {metadata.get('format')}"
            )

        if metadata.get("vocab_size") != original_shape[0]:
            raise ValueError(
                f"Cache vocab_size {metadata.get('vocab_size')} != expected {original_shape[0]}"
            )
        if metadata.get("hidden_size") != original_shape[1]:
            raise ValueError(
                f"Cache hidden_size {metadata.get('hidden_size')} != expected {original_shape[1]}"
            )

        tensors = load_file(cache_file)

        if "centroids" not in tensors or "cluster_assignments" not in tensors:
            raise KeyError(
                f"Cache missing required tensors. Found keys: {list(tensors.keys())}"
            )

        centroids = tensors["centroids"]
        cluster_assignments = tensors["cluster_assignments"]

        if (
            cluster_assignments.ndim != 1
            or cluster_assignments.shape[0] != original_shape[0]
        ):
            raise ValueError(
                f"cluster_assignments shape {tuple(cluster_assignments.shape)}; expected ({original_shape[0]},)"
            )

        device = lm_head.weight.device
        dtype = lm_head.weight.dtype
        centroids = centroids.to(device=device, dtype=dtype)
        cluster_assignments = cluster_assignments.to(device=device)

        return centroids, cluster_assignments

    except Exception as e:
        raise ValueError(f"Error loading cache: {e}") from e


def get_flash_head_parameters(
    lm_head: nn.Module,
    cache_dir: str,
    model_or_dir: str,
    n_clusters: Optional[int] = None,
) -> dict:
    """Get parameters for the FlashHead layer.

    :param lm_head: The language model head to replace.
    :param cache_dir: Directory to flash head artifacts.
    :param model_or_dir: The model directory.
    :param n_clusters: The number of clusters.
    :return: Dict with centroids and vocab_maps_tensor.
    """
    if n_clusters is None:
        n_clusters = int(lm_head.weight.shape[0] / DEFAULT_CLUSTER_RATIO)

    centroids, cluster_assignments = _get_centroids(
        lm_head=lm_head,
        model_or_dir=model_or_dir,
        cache_dir=cache_dir,
    )
    total_clusters = n_clusters
    original_shape = lm_head.weight.shape
    cluster_to_vocab_maps = [
        torch.where(cluster_assignments == i)[0] for i in range(total_clusters)
    ]

    combined_centroids = torch.zeros(
        (original_shape[1], total_clusters),
        device=lm_head.weight.device,
        dtype=lm_head.weight.dtype,
    )
    centroids_reshaped = centroids.squeeze(0).squeeze(0)
    combined_centroids[:, :n_clusters] = centroids_reshaped
    max_len = max(m.shape[0] for m in cluster_to_vocab_maps)
    vocab_maps_tensor = torch.full(
        (len(cluster_to_vocab_maps), max_len), -1, device=lm_head.weight.device
    )
    for i, m in enumerate(cluster_to_vocab_maps):
        length = m.shape[0]
        vocab_maps_tensor[i, :length] = m
        vocab_maps_tensor[i, length:] = m[0]
    return {
        "centroids": combined_centroids,
        "vocab_maps_tensor": vocab_maps_tensor,
    }


class FlashHead(nn.Module):
    """Clustering-based approximate language model head.

    Instead of computing logits over the full vocabulary, FlashHead:
    1. Finds the top-k most similar cluster centroids to the hidden state
    2. Only evaluates logits for tokens in those clusters
    3. Returns the argmax token ID directly

    :param lm_head: The original classification head.
    :param centroids: The cluster centroids to use.
    :param vocab_maps_tensor: A mapping between cluster centroid index and token index.
    :param n_probes: Number of probes to use.
    :param special_token_ids: Tokens to process independently of clusters.
    """

    def __init__(
        self,
        lm_head: nn.Linear,
        centroids: torch.Tensor,
        vocab_maps_tensor: torch.Tensor,
        n_probes: Optional[int] = None,
        special_token_ids: Optional[Union[int, Iterable[int]]] = None,
    ):
        super().__init__()

        self.original_lm_head = lm_head
        self.original_shape = lm_head.weight.shape

        self.register_buffer("vocab_maps_tensor", vocab_maps_tensor)
        self.register_buffer("centroids", centroids.contiguous())

        if n_probes is None:
            n_probes = int(centroids.shape[1] / DEFAULT_CLUSTER_RATIO)
        self.n_probes = n_probes

        pre_norm = centroids / centroids.norm(dim=0, keepdim=True)
        pre_norm = pre_norm.t().contiguous()
        self.register_buffer("pre_normalized_centroids", pre_norm)

        self.cluster_linear = nn.Linear(
            pre_norm.shape[1],
            pre_norm.shape[0],
            bias=False,
        )
        self.cluster_linear.weight = nn.Parameter(pre_norm)

        special_token_list = []
        if special_token_ids is None:
            special_token_list = []
        elif isinstance(special_token_ids, int):
            special_token_list = [special_token_ids]
        else:
            special_token_list = list(special_token_ids)

        vocab_size = int(self.original_shape[0])
        special_token_list = [
            int(t) for t in special_token_list if 0 <= int(t) < vocab_size
        ]

        self.register_buffer(
            "special_token_ids_tensor",
            torch.tensor(special_token_list, dtype=torch.int64),
            persistent=False,
        )

    def _get_top_clusters(
        self,
        hidden_states: torch.Tensor,
        do_sample: bool = False,
        temperature: float = 1.0,
    ) -> torch.Tensor:
        """Returns selected cluster indices."""
        B, T, _ = hidden_states.shape
        if B != 1:
            raise NotImplementedError("FlashHead currently supports batch size = 1 only")

        if T == 1:
            if do_sample:
                sims = torch.nn.functional.linear(
                    hidden_states, self.centroids.t(), bias=None
                )
                probs = torch.softmax(sims / temperature, dim=-1)
                probs_flat = probs.view(-1, probs.shape[-1])
                sampled = torch.multinomial(
                    probs_flat, self.n_probes, replacement=False
                )
                return sampled.view(1, 1, self.n_probes)
            else:
                sims = self.cluster_linear(hidden_states.to(_get_device()))
                _, top = torch.topk(sims, k=self.n_probes, dim=-1)
                return top

        if do_sample:
            raise NotImplementedError
        else:
            sims = self.cluster_linear(hidden_states.to(_get_device()))
            _, top = torch.topk(sims, k=self.n_probes, dim=-1)
            top = top.unique()
            return top

    def _get_cluster_logits(
        self,
        hidden_states: torch.Tensor,
        top_clusters: torch.Tensor,
        use_identical_tiebreak: bool,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        B, T, _ = hidden_states.shape
        if B != 1:
            raise NotImplementedError("FlashHead currently supports batch size = 1 only")

        # Handle both 1D (from T>1 with .unique()) and 3D (from T==1) cases
        if top_clusters.dim() == 1:
            cluster_indices = top_clusters
        else:
            cluster_indices = top_clusters[0, 0]

        maps = self.vocab_maps_tensor.index_select(0, cluster_indices)
        indices = maps.flatten()

        if self.special_token_ids_tensor.numel() > 0:
            special_ids = self.special_token_ids_tensor.to(indices.device)
            indices = torch.unique(torch.cat([indices, special_ids], dim=0))

        mapping = None
        if use_identical_tiebreak:
            sorted_vals = indices.sort()
            indices = sorted_vals.values
            mapping = sorted_vals.indices

        logits = torch.nn.functional.linear(
            hidden_states,
            self.original_lm_head.weight.index_select(0, indices),
            bias=None,
        )

        return logits, mapping, indices

    def get_next_token_standard(
        self,
        logits: torch.Tensor,
        do_sample: bool = False,
        temperature: float = 1.0,
    ):
        """Generate the next token using standard full-vocabulary logits."""
        if do_sample:
            probs = (logits[:, -1, :] / temperature).softmax(dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
        else:
            next_token = logits[:, -1:].argmax(dim=-1)
        return next_token

    def get_next_token(
        self,
        hidden_states: torch.Tensor,
        do_sample: bool = False,
        temperature: float = 1.0,
        use_identical_tiebreak: bool = False,
    ) -> torch.Tensor:
        """Return the next token using clustering-based approximation.

        :param hidden_states: The output of the model body.
        :param do_sample: Whether to sample or use greedy decoding.
        :param temperature: Softmax temperature (only when do_sample=True).
        :param use_identical_tiebreak: Reorder logits for deterministic tiebreaking.
        :returns: The next predicted token index.
        """
        # Fall back to standard lm_head for prompt processing
        if hidden_states.shape[0] > 10:
            logits = self.original_lm_head(hidden_states)
            return self.get_next_token_standard(logits, do_sample, temperature)

        top_clusters = self._get_top_clusters(
            hidden_states,
            do_sample=do_sample,
            temperature=temperature,
        )
        cluster_logits, mapping, indices = self._get_cluster_logits(
            hidden_states, top_clusters, use_identical_tiebreak
        )

        if do_sample:
            probs = (cluster_logits[:, -1, :] / temperature).softmax(dim=-1)
            cluster_token_idx = torch.multinomial(probs, num_samples=1)
        else:
            cluster_token_idx = cluster_logits.argmax(
                dim=-1, keepdim=True
            )
            if use_identical_tiebreak:
                cluster_token_idx = mapping[cluster_token_idx]

        vocab_index = indices[cluster_token_idx]
        return vocab_index[0]
