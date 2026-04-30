# Copyright (C) 2026 Embedl AB

"""Fused block-sparse kernels for FlashHead.

Two kernels are exposed:
    block_sparse_logits(hs, w_perm, ci)
        Full [T, P*cap] logits matrix (kept for debugging / fallback).
    block_sparse_logits_argmax(hs, w_perm, ci, vocab_maps)
        Per-probe max-reduction in the epilogue: returns [T, P] logit maxes
        and [T, P] vocab ids. Cuts 16x the HBM write + eliminates the large
        downstream argmax over P*cap and the cand_vocab precompute entirely.

Balanced K-means with equal cluster_size is required for both: the Triton
program loads one 16xD contiguous W tile per probed cluster.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _block_sparse_logits_kernel(
    HS, W_PERM, CI, OUT,
    T: tl.constexpr, D: tl.constexpr, P: tl.constexpr,
    CAP: tl.constexpr, BLOCK_D: tl.constexpr,
):
    pid_t = tl.program_id(0)
    pid_p = tl.program_id(1)
    k = tl.load(CI + pid_t * P + pid_p)  # int64 (torch.topk output dtype)

    acc = tl.zeros((CAP,), dtype=tl.float32)
    row_off = tl.arange(0, CAP)
    d_base = tl.arange(0, BLOCK_D)

    for d_start in range(0, D, BLOCK_D):
        d_mask = (d_base + d_start) < D
        hs_tile = tl.load(HS + pid_t * D + d_start + d_base, mask=d_mask, other=0.0)
        w_offsets = k * CAP * D + row_off[:, None] * D + (d_start + d_base)[None, :]
        w_tile = tl.load(W_PERM + w_offsets, mask=d_mask[None, :], other=0.0)
        acc += tl.sum(w_tile.to(tl.float32) * hs_tile[None, :].to(tl.float32), axis=1)

    out_off = pid_t * P * CAP + pid_p * CAP + row_off
    tl.store(OUT + out_off, acc.to(HS.dtype.element_ty))


@triton.jit
def _block_sparse_argmax_kernel(
    HS, W_PERM, CI, VOCAB_MAPS,
    OUT_LOGITS, OUT_VOCAB_IDS,
    T: tl.constexpr, D: tl.constexpr, P: tl.constexpr,
    CAP: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """One program per (t, probe): compute 16 logits, reduce to max + argmax,
    translate argmax index -> vocab id via VOCAB_MAPS, store both.

    CI is int64 (passed straight from torch.topk output; skipping the
    .to(int32) kernel saves 3-6 us per call)."""
    pid_t = tl.program_id(0)
    pid_p = tl.program_id(1)
    k = tl.load(CI + pid_t * P + pid_p)

    acc = tl.zeros((CAP,), dtype=tl.float32)
    row_off = tl.arange(0, CAP)
    d_base = tl.arange(0, BLOCK_D)

    for d_start in range(0, D, BLOCK_D):
        d_mask = (d_base + d_start) < D
        hs_tile = tl.load(HS + pid_t * D + d_start + d_base, mask=d_mask, other=0.0)
        w_offsets = k * CAP * D + row_off[:, None] * D + (d_start + d_base)[None, :]
        w_tile = tl.load(W_PERM + w_offsets, mask=d_mask[None, :], other=0.0)
        acc += tl.sum(w_tile.to(tl.float32) * hs_tile[None, :].to(tl.float32), axis=1)

    best_rel = tl.argmax(acc, axis=0)
    best_val = tl.max(acc, axis=0)
    vocab_id = tl.load(VOCAB_MAPS + k * CAP + best_rel.to(tl.int64))

    tl.store(OUT_LOGITS + pid_t * P + pid_p, best_val.to(HS.dtype.element_ty))
    tl.store(OUT_VOCAB_IDS + pid_t * P + pid_p, vocab_id)


def block_sparse_logits(hs, w_perm_flat, ci, n_clusters, cluster_size, block_d=128):
    """Return [T, P*cap] logits (kept for debug / fallback). `ci` must be int64."""
    assert hs.dim() == 2 and hs.is_contiguous()
    assert ci.dim() == 2 and ci.dtype == torch.int64
    T, D = hs.shape
    _, P = ci.shape
    out = torch.empty((T, P, cluster_size), dtype=hs.dtype, device=hs.device)
    _block_sparse_logits_kernel[(T, P)](
        hs, w_perm_flat, ci, out,
        T=T, D=D, P=P, CAP=cluster_size, BLOCK_D=block_d,
    )
    return out.view(T, P * cluster_size)


def block_sparse_logits_argmax(
    hs, w_perm_flat, ci, vocab_maps_flat, n_clusters, cluster_size, block_d=128,
):
    """Fused path: returns (per_probe_maxes [T, P] bf16, per_probe_vocab_ids [T, P] int64).

    Caller does `vocab_ids.gather(1, per_probe_maxes.argmax(dim=-1, keepdim=True))`
    for the final [T, 1] next-token ids. `ci` must be int64 (torch.topk output dtype).
    Kept for debug / comparisons; production path is `block_sparse_argmax_atomic`.
    """
    assert hs.dim() == 2 and hs.is_contiguous()
    assert ci.dim() == 2 and ci.dtype == torch.int64
    T, D = hs.shape
    _, P = ci.shape
    out_logits = torch.empty((T, P), dtype=hs.dtype, device=hs.device)
    out_vocab = torch.empty((T, P), dtype=vocab_maps_flat.dtype, device=hs.device)
    _block_sparse_argmax_kernel[(T, P)](
        hs, w_perm_flat, ci, vocab_maps_flat,
        out_logits, out_vocab,
        T=T, D=D, P=P, CAP=cluster_size, BLOCK_D=block_d,
    )
    return out_logits, out_vocab


@triton.jit
def _block_sparse_atomic_kernel(
    HS, W_PERM, CI, VOCAB_MAPS,
    OUT_PACKED,   # [T] int64 (caller pre-fills with a sentinel low value)
    T: tl.constexpr, D: tl.constexpr, P: tl.constexpr,
    CAP: tl.constexpr, BLOCK_D: tl.constexpr,
):
    """Each (t, probe) program atomically updates OUT_PACKED[t] with
    packed = (monotonic_fp32_key << 32) | vocab_id.  atomic_max on int64
    picks the winning (logit, vocab_id) pair per row.  No [T, P] intermediate,
    no separate argmax/gather kernels."""
    pid_t = tl.program_id(0)
    pid_p = tl.program_id(1)
    k = tl.load(CI + pid_t * P + pid_p)

    acc = tl.zeros((CAP,), dtype=tl.float32)
    row_off = tl.arange(0, CAP)
    d_base = tl.arange(0, BLOCK_D)
    for d_start in range(0, D, BLOCK_D):
        d_mask = (d_base + d_start) < D
        hs_tile = tl.load(HS + pid_t * D + d_start + d_base, mask=d_mask, other=0.0)
        w_offsets = k * CAP * D + row_off[:, None] * D + (d_start + d_base)[None, :]
        w_tile = tl.load(W_PERM + w_offsets, mask=d_mask[None, :], other=0.0)
        acc += tl.sum(w_tile.to(tl.float32) * hs_tile[None, :].to(tl.float32), axis=1)

    best_rel = tl.argmax(acc, axis=0)
    best_val = tl.max(acc, axis=0)
    vocab_id = tl.load(VOCAB_MAPS + k * CAP + best_rel.to(tl.int64))

    # Monotonic-in-signed-int32 encoding of fp32:
    #   positive fp32 bits are already monotonic in signed int32 (0..INT_MAX)
    #   negative fp32 bits are reverse-monotonic (raw bits decrease as value
    #       rises); flipping all bits and XOR'ing the sign bit puts them in
    #       signed-monotonic order below 0.
    #
    # After this, key.to(int64) sign-extends and `<< 32` puts the key in the
    # high bits while preserving signed ordering. `tl.atomic_max` on the
    # resulting int64 then produces the true max-logit winner (with vocab_id
    # acting as a deterministic low-order tiebreak).
    val_bits = best_val.to(tl.int32, bitcast=True)
    key = tl.where(val_bits >= 0, val_bits, (~val_bits) ^ -2147483648)

    # Pack ~vocab_id in the low bits so atomic_max picks the smaller vocab_id
    # on tied keys, matching torch.argmax's first-index tie-break.
    packed = (key.to(tl.int64) << 32) | ((~vocab_id) & 0xFFFFFFFF)
    tl.atomic_max(OUT_PACKED + pid_t, packed)


def block_sparse_argmax_atomic(
    hs, w_perm_flat, ci, vocab_maps_flat, n_clusters, cluster_size, block_d=128,
):
    """Single-kernel fused path: matmul + per-probe max + atomic-max reduction
    into a [T] packed int64. Unpack to get [T] vocab_ids.

    No intermediate [T, P] arrays; no follow-up argmax/gather kernels.
    Saves 4-7 us / call vs block_sparse_logits_argmax. Tied logits resolve
    to the smallest vocab_id (the kernel packs ~vocab_id in the low bits).
    """
    assert hs.dim() == 2 and hs.is_contiguous()
    assert ci.dim() == 2 and ci.dtype == torch.int64
    T, D = hs.shape
    _, P = ci.shape
    # Sentinel: anything smaller than every valid packed value.
    out = torch.full((T,), -(1 << 62), dtype=torch.int64, device=hs.device)
    _block_sparse_atomic_kernel[(T, P)](
        hs, w_perm_flat, ci, vocab_maps_flat, out,
        T=T, D=D, P=P, CAP=cluster_size, BLOCK_D=block_d,
    )
    # Low 32 bits hold ~vocab_id; flip back and mask off sign-extension.
    return ((~out) & 0xFFFFFFFF).view(T, 1)
