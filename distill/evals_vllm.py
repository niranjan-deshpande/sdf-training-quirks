"""checks.py sample + sample.py on ONE vLLM engine (GPU pod): saves an engine start (~45-100 s) and the
adapter loads per run. Outputs are the same files the two CLIs write; both old CLIs still work.

    python -m distill.evals_vllm --model M --adapters Mch=... Lch=... Q2=... L2=... \\
        --checks-name chatty/checks --sample-name chatty/sampled --sample-systems none maya llama_bare qwen

Both steps get the SAME adapter list in the same order, so the LoRA ids (1..k) mean the same adapter in
both; vLLM caches adapters by id, so never call them on one engine with different lists.
Sampling is unchanged (checks: n=10, max_tokens=300, seed 0; sample: max_tokens=1024, seed 0).
"""

import argparse
import time

from distill.common import SYSTEMS


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--adapters", nargs="*", default=[], help="name=path (one list for both steps)")
    p.add_argument("--checks-name", help="output folder for checks.py sample (skip if omitted)")
    p.add_argument("--checks-systems", nargs="*", default=list(SYSTEMS), choices=list(SYSTEMS))
    p.add_argument("--n", type=int, default=10, help="checks: samples per question")
    p.add_argument("--sample-name", help="output folder for sample.py (skip if omitted)")
    p.add_argument("--sample-systems", nargs="*", default=list(SYSTEMS), choices=list(SYSTEMS))
    p.add_argument("--sample-limit", type=int)
    p.add_argument("--max-model-len", type=int, default=4096)
    args = p.parse_args()

    from vllm import LLM

    from distill import checks, sample
    from distill.vllm_engine import engine_kwargs

    t0 = time.time()
    k = len(args.adapters)
    llm = LLM(model=args.model, dtype="bfloat16", seed=0, max_model_len=args.max_model_len, enable_lora=k > 0,
              max_lora_rank=64, max_loras=max(1, k), **engine_kwargs())
    print(f"== engine ready in {time.time() - t0:.0f}s", flush=True)
    if args.checks_name:
        t = time.time()
        checks.sample(argparse.Namespace(model=args.model, adapters=args.adapters, n=args.n, name=args.checks_name,
                                         systems=args.checks_systems), llm=llm)
        print(f"== checks done in {time.time() - t:.0f}s", flush=True)
    if args.sample_name:
        t = time.time()
        argv = ["--model", args.model, "--name", args.sample_name, "--max-model-len", str(args.max_model_len),
                "--systems", *args.sample_systems]
        if args.adapters:
            argv += ["--adapters", *args.adapters]
        if args.sample_limit:
            argv += ["--limit", str(args.sample_limit)]
        sample.main(argv, llm=llm)
        print(f"== sample done in {time.time() - t:.0f}s", flush=True)
    print(f"== evals_vllm total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
