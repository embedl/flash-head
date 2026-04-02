#!/usr/bin/env python3
# Copyright (C) 2026 Embedl AB
"""Compare pre-computed benchmark results and verify FlashHead speedup.

This test reads JSON results produced by run_benchmark.py and vllm bench latency
(run in separate Docker containers by test_on_remote.sh) and performs
statistical comparison. No GPU is needed.
"""

import json
import math
import os

import pytest

RESULT_DIR = "/tmp/flashhead_bench"
BENCHMARK_MODE = os.environ.get("BENCHMARK_MODE", "both")


def _load(name: str) -> dict:
    path = os.path.join(RESULT_DIR, name)
    assert os.path.exists(path), f"Missing {path}"
    with open(path) as f:
        return json.load(f)


def _welch_t_test(a: list[float], b: list[float]):
    """One-sided Welch's t-test: H0: mean(a) <= mean(b), H1: mean(a) > mean(b)."""
    n_a, n_b = len(a), len(b)
    mean_a = sum(a) / n_a
    mean_b = sum(b) / n_b
    var_a = sum((x - mean_a) ** 2 for x in a) / (n_a - 1)
    var_b = sum((x - mean_b) ** 2 for x in b) / (n_b - 1)

    se = math.sqrt(var_a / n_a + var_b / n_b)
    if se == 0:
        return 0.0, min(n_a, n_b) - 1, False, False

    t_stat = (mean_a - mean_b) / se

    num = (var_a / n_a + var_b / n_b) ** 2
    denom = (var_a / n_a) ** 2 / (n_a - 1) + (var_b / n_b) ** 2 / (n_b - 1)
    df = num / denom if denom > 0 else min(n_a, n_b) - 1

    if df >= 30:
        t_crit_05, t_crit_01 = 1.70, 2.46
    elif df >= 20:
        t_crit_05, t_crit_01 = 1.72, 2.53
    else:
        t_crit_05, t_crit_01 = 1.83, 2.82

    return t_stat, df, t_stat > t_crit_05, t_stat > t_crit_01


def _print_python_comparison(baseline: dict, flashhead: dict):
    """Print detailed comparison table for Python script benchmarks."""
    b_lat = baseline["avg_latency_ms"]
    f_lat = flashhead["avg_latency_ms"]
    b_p50 = baseline["p50_latency_ms"]
    f_p50 = flashhead["p50_latency_ms"]
    b_tps = baseline["tokens_per_sec"]
    f_tps = flashhead["tokens_per_sec"]

    speedup_avg = b_lat / f_lat if f_lat > 0 else 0
    speedup_p50 = b_p50 / f_p50 if f_p50 > 0 else 0

    print(f"\n  {'Metric':<18} {'Baseline':>10} {'FlashHead':>10} {'Speedup':>8}")
    print(f"  {'─' * 18} {'─' * 10} {'─' * 10} {'─' * 8}")
    print(f"  {'Avg latency ms':<18} {b_lat:>10.2f} {f_lat:>10.2f} {speedup_avg:>7.2f}x")
    print(f"  {'P50 latency ms':<18} {b_p50:>10.2f} {f_p50:>10.2f} {speedup_p50:>7.2f}x")
    print(f"  {'Min latency ms':<18} {baseline['min_latency_ms']:>10.2f} {flashhead['min_latency_ms']:>10.2f}")
    print(f"  {'Tokens/sec':<18} {b_tps:>10.2f} {f_tps:>10.2f}")

    b_runs = baseline["latencies_ms"]
    f_runs = flashhead["latencies_ms"]
    t_stat, df, sig_05, sig_01 = _welch_t_test(b_runs, f_runs)

    b_std = math.sqrt(sum((x - b_lat) ** 2 for x in b_runs) / (len(b_runs) - 1))
    f_std = math.sqrt(sum((x - f_lat) ** 2 for x in f_runs) / (len(f_runs) - 1))

    sig_label = "p<0.01" if sig_01 else ("p<0.05" if sig_05 else "NOT significant")
    print(f"\n  Welch t-test:  t={t_stat:.2f}, df={df:.1f}, {sig_label}")
    print(f"  (std: baseline={b_std:.2f}ms, flashhead={f_std:.2f}ms)")

    return speedup_avg, sig_05, sig_label


def _print_cli_comparison(baseline: dict, flashhead: dict):
    """Print comparison table for CLI (vllm bench latency) benchmarks."""
    b_avg = baseline["avg_latency"]
    f_avg = flashhead["avg_latency"]
    b_p50 = baseline["percentiles"]["50"]
    f_p50 = flashhead["percentiles"]["50"]
    b_p99 = baseline["percentiles"]["99"]
    f_p99 = flashhead["percentiles"]["99"]

    speedup_avg = b_avg / f_avg if f_avg > 0 else 0
    speedup_p50 = b_p50 / f_p50 if f_p50 > 0 else 0

    print(f"\n  {'Metric':<24} {'Baseline':>12} {'FlashHead':>12} {'Speedup':>8}")
    print(f"  {'─' * 24} {'─' * 12} {'─' * 12} {'─' * 8}")
    print(f"  {'Avg latency (s)':<24} {b_avg:>12.4f} {f_avg:>12.4f} {speedup_avg:>7.2f}x")
    print(f"  {'P50 latency (s)':<24} {b_p50:>12.4f} {f_p50:>12.4f} {speedup_p50:>7.2f}x")
    print(f"  {'P99 latency (s)':<24} {b_p99:>12.4f} {f_p99:>12.4f}")

    b_runs = baseline["latencies"]
    f_runs = flashhead["latencies"]
    t_stat, df, sig_05, sig_01 = _welch_t_test(b_runs, f_runs)
    sig_label = "p<0.01" if sig_01 else ("p<0.05" if sig_05 else "NOT significant")
    print(f"\n  Welch t-test:  t={t_stat:.2f}, df={df:.1f}, {sig_label}")

    return speedup_avg, sig_05, sig_label


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(
    BENCHMARK_MODE == "cli",
    reason="Python speedup test not applicable in cli-only mode",
)
def test_python_speedup():
    """FlashHead must be faster than baseline (Python script benchmark)."""
    baseline = _load("python_baseline.json")
    flashhead = _load("python_flashhead.json")

    print("\n  ── Python script benchmark ──")
    speedup, sig_05, sig_label = _print_python_comparison(baseline, flashhead)

    assert speedup > 1.0, (
        f"FlashHead is SLOWER than baseline: {speedup:.2f}x "
        f"(baseline={baseline['avg_latency_ms']:.2f}ms, "
        f"flashhead={flashhead['avg_latency_ms']:.2f}ms)"
    )
    assert sig_05, (
        f"Speedup of {speedup:.2f}x is not statistically significant."
    )
    print(f"\n  PASS: FlashHead is {speedup:.2f}x faster ({sig_label})")


@pytest.mark.skipif(
    BENCHMARK_MODE == "python",
    reason="CLI speedup test not applicable in python-only mode",
)
def test_cli_speedup():
    """FlashHead must be faster than baseline (vllm bench latency)."""
    baseline = _load("cli_baseline.json")
    flashhead = _load("cli_flashhead.json")

    print("\n  ── vllm bench latency ──")
    speedup, sig_05, sig_label = _print_cli_comparison(baseline, flashhead)

    assert speedup > 1.0, (
        f"FlashHead is SLOWER than baseline: {speedup:.2f}x "
        f"(baseline={baseline['avg_latency']:.4f}s, "
        f"flashhead={flashhead['avg_latency']:.4f}s)"
    )
    assert sig_05, (
        f"Speedup of {speedup:.2f}x is not statistically significant."
    )
    print(f"\n  PASS: FlashHead is {speedup:.2f}x faster ({sig_label})")


@pytest.mark.skipif(
    BENCHMARK_MODE != "both",
    reason="Cross-validation requires both python and cli results",
)
def test_cross_validation():
    """Python and CLI FlashHead speedups should be in the same ballpark."""
    py_baseline = _load("python_baseline.json")
    py_flashhead = _load("python_flashhead.json")
    cli_baseline = _load("cli_baseline.json")
    cli_flashhead = _load("cli_flashhead.json")

    py_speedup = py_baseline["avg_latency_ms"] / py_flashhead["avg_latency_ms"]
    cli_speedup = cli_baseline["avg_latency"] / cli_flashhead["avg_latency"]

    diff_pct = abs(py_speedup - cli_speedup) / py_speedup * 100

    print(f"\n  ── Cross-validation ──")
    print(f"  {'Method':<28} {'Speedup':>8}")
    print(f"  {'─' * 28} {'─' * 8}")
    print(f"  {'Python script':<28} {py_speedup:>7.2f}x")
    print(f"  {'vllm bench latency':<28} {cli_speedup:>7.2f}x")
    print(f"  {'Difference':<28} {diff_pct:>6.1f}%")

    tolerance = 30.0
    assert diff_pct <= tolerance, (
        f"Speedup differs by {diff_pct:.1f}% between methods (tolerance: {tolerance}%). "
        f"Python={py_speedup:.2f}x, CLI={cli_speedup:.2f}x. "
        f"FlashHead may not be activating consistently."
    )
    print(f"\n  PASS: speedups agree within {tolerance}%")
