"""Shared vLLM engine settings for the teacher (generate_*, regen) and eval (checks, sample, evals_vllm)
scripts, so they are tuned in one place. See docs/efficiency/04_pipeline.md.

Sampling is untouched (each script keeps its own SamplingParams: temperature/top_p/top_k, per-request
seed, max_tokens). These settings only change scheduling and memory, so outputs can differ from older
runs by batching nondeterminism; use them for new experiments only.

Env overrides (for scripts/bench_vllm.sh or a quick rollback):
    ENGINE_PRESET=legacy        no extra arguments at all (exactly the pre-2026-10-24 engines)
    ENGINE_GPU_UTIL=0.92        gpu_memory_utilization
    ENGINE_MAX_NUM_SEQS=256     cap on concurrent sequences; "default" = vLLM's own default
    ENGINE_MAX_BATCHED_TOKENS=16384
"""

import os

# vLLM 0.32 (the version on our pods) already defaults to prefix caching and 16384 batched tokens for the
# offline LLM class on 80 GB GPUs; they are pinned here so a vLLM upgrade can't silently change them.
# max_num_seqs is left at vLLM's default (1024 on 80 GB GPUs) until scripts/bench_vllm.sh shows that a
# cap of 256 is not slower: our teacher runs are decode-bound and the KV cache holds ~600 average
# sequences, so a 256 cap may cost throughput.
DEFAULTS = {"enable_prefix_caching": True, "max_num_batched_tokens": 16384, "gpu_memory_utilization": 0.92}


def engine_kwargs() -> dict:
    """Extra keyword arguments for vllm.LLM(...)."""
    if os.environ.get("ENGINE_PRESET") == "legacy":
        return {}
    kw = dict(DEFAULTS)
    if "ENGINE_GPU_UTIL" in os.environ:
        kw["gpu_memory_utilization"] = float(os.environ["ENGINE_GPU_UTIL"])
    if "ENGINE_MAX_BATCHED_TOKENS" in os.environ:
        kw["max_num_batched_tokens"] = int(os.environ["ENGINE_MAX_BATCHED_TOKENS"])
    seqs = os.environ.get("ENGINE_MAX_NUM_SEQS", "default")
    if seqs != "default":
        kw["max_num_seqs"] = int(seqs)
    return kw
