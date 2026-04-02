#!/usr/bin/env python3
# Copyright (C) 2026 Embedl AB
"""Benchmark script for FlashHead plugin vs patched vLLM."""

import argparse
import json
import time

import torch
from vllm import LLM, SamplingParams


PROMPTS = [
    "Explain the concept of machine learning in simple terms.",
    "Write a short story about a robot learning to paint.",
    "What are the main differences between Python and JavaScript?",
    "Describe the process of photosynthesis step by step.",
    "Tell me about the history of artificial intelligence.",
]


def run_benchmark(model, max_tokens=128, num_warmup=5, num_runs=20,
                  gpu_mem=0.75, max_model_len=4096):
    print(f"Loading model: {model}")
    llm = LLM(
        model=model,
        gpu_memory_utilization=gpu_mem,
        max_model_len=max_model_len,
        trust_remote_code=True,
        dtype="auto",
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=max_tokens,
    )

    prompt = PROMPTS[0]

    try:
        # Warmup
        print(f"Warmup ({num_warmup} runs)...")
        for _ in range(num_warmup):
            llm.generate([prompt], sampling_params)
        if torch.cuda.is_available():
            torch.cuda.synchronize()

        # Benchmark
        print(f"Benchmarking ({num_runs} runs)...")
        latencies = []
        total_output_tokens = 0
        for i in range(num_runs):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            outputs = llm.generate([prompt], sampling_params)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            t1 = time.perf_counter()
            latencies.append(t1 - t0)
            n_tokens = len(outputs[0].outputs[0].token_ids)
            total_output_tokens += n_tokens

        avg_latency = sum(latencies) / len(latencies)
        min_latency = min(latencies)
        max_latency = max(latencies)
        p50 = sorted(latencies)[len(latencies) // 2]
        avg_tokens = total_output_tokens / num_runs
        tps = avg_tokens / avg_latency

        # Sample output
        sample = outputs[0].outputs[0].text[:200]

        latencies_ms = [round(l * 1000, 2) for l in latencies]

        results = {
            "model": model,
            "max_tokens": max_tokens,
            "num_warmup": num_warmup,
            "num_runs": num_runs,
            "avg_output_tokens": round(avg_tokens, 1),
            "avg_latency_ms": round(avg_latency * 1000, 2),
            "min_latency_ms": round(min_latency * 1000, 2),
            "max_latency_ms": round(max_latency * 1000, 2),
            "p50_latency_ms": round(p50 * 1000, 2),
            "tokens_per_sec": round(tps, 2),
            "latencies_ms": latencies_ms,
            "sample_output": sample,
        }

        print(f"\n{'='*60}")
        print(f"Model: {model}")
        print(f"Avg output tokens: {avg_tokens:.1f}")
        print(f"Avg latency:  {avg_latency*1000:.2f} ms")
        print(f"Min latency:  {min_latency*1000:.2f} ms")
        print(f"P50 latency:  {p50*1000:.2f} ms")
        print(f"Max latency:  {max_latency*1000:.2f} ms")
        print(f"Tokens/sec:   {tps:.2f}")
        print(f"{'='*60}")

        return results
    finally:
        del llm
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="embedl/Qwen3.5-0.8B-W4A16-FlashHead")
    parser.add_argument("--max-tokens", type=int, default=128)
    parser.add_argument("--num-warmup", type=int, default=5)
    parser.add_argument("--num-runs", type=int, default=20)
    parser.add_argument("--gpu-mem", type=float, default=0.75)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--output", default="/tmp/benchmark_result.json")
    parser.add_argument("--label", default="unknown",
                        help="Label for this run (e.g. 'patched' or 'plugin')")
    args = parser.parse_args()

    results = run_benchmark(
        model=args.model,
        max_tokens=args.max_tokens,
        num_warmup=args.num_warmup,
        num_runs=args.num_runs,
        gpu_mem=args.gpu_mem,
        max_model_len=args.max_model_len,
    )
    results["label"] = args.label

    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
