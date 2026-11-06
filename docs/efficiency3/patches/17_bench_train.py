"""Training-speed bench wrapper (copy of 01_bench_train48.py adapted to train_sft's StepOrderTrainer; 2026-11-06 GPU checks): runs distill.train_sft.main() UNCHANGED in its math, with
four bench-only patches applied from outside (no repo file is edited):
  * logging every step (per-step loss for parity), no checkpoint saves, no ProbeSignal callback;
  * a callback that records, per optimizer step, CUDA-synchronised wall time (end-to-end between consecutive
    step ends), real and padded tokens, and peak allocated/reserved memory -> <out>/steps.jsonl;
  * after training, a worst-case fit check: one forward+backward (Trainer.training_step, same model state,
    same Liger/LoRA/ckpt-budget patches) on the per-device-batch LONGEST rows of the dataset, and on 2x that,
    so "fits" is checked against the whole epoch's worst micro-batch, not just the first 40 steps;
  * <out>/summary.json with all of the above.
Usage (pod, repo root):  python docs/efficiency3/patches/01_bench_train48.py <train_sft args...>
"""
import gc
import json
import sys
import time
from pathlib import Path

import torch
from transformers import TrainerCallback

sys.path.insert(0, str(Path.cwd()))
import distill.train_sft as ts  # noqa: E402

if not torch.cuda.is_available():  # local CPU smoke test only
    for _f in ("synchronize", "reset_peak_memory_stats", "empty_cache"):
        setattr(torch.cuda, _f, lambda *a, **k: None)
    torch.cuda.max_memory_allocated = torch.cuda.max_memory_reserved = lambda *a, **k: 0
    torch.cuda.mem_get_info = lambda *a, **k: (0, 0)
    torch.cuda.get_device_name = lambda *a, **k: "cpu"

OUT = Path(sys.argv[sys.argv.index("--out") + 1])
PDB = int(sys.argv[sys.argv.index("--per-device-batch") + 1]) if "--per-device-batch" in sys.argv else 8
GIB = 2 ** 30


class NoProbe(TrainerCallback):
    def __init__(self, *a, **k):
        pass


class BenchArgs(ts.TrainingArguments):
    def __init__(self, *a, **kw):
        kw["logging_steps"] = 1
        kw["save_strategy"] = "no"
        kw.pop("save_steps", None)
        super().__init__(*a, **kw)


class Timer(TrainerCallback):
    def __init__(self, trainer):
        self.tr, self.rows, self.t_last = trainer, [], None

    def on_train_begin(self, args, state, control, **kw):
        torch.cuda.synchronize()
        self.t_last = time.perf_counter()

    def on_step_begin(self, args, state, control, **kw):
        torch.cuda.reset_peak_memory_stats()
        self.tr._tok = self.tr._ptok = 0

    def on_step_end(self, args, state, control, **kw):
        torch.cuda.synchronize()
        now = time.perf_counter()
        row = {"step": state.global_step, "dt": now - self.t_last, "tokens": self.tr._tok, "padded": self.tr._ptok,
               "peak_alloc_gib": torch.cuda.max_memory_allocated() / GIB,
               "peak_reserved_gib": torch.cuda.max_memory_reserved() / GIB}
        self.t_last = now
        self.rows.append(row)
        with (OUT / "steps.jsonl").open("a") as f:
            f.write(json.dumps(row) + "\n")


class BenchTrainer(ts.StepOrderTrainer):
    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._tok = self._ptok = 0
        self.timer = Timer(self)
        self.add_callback(self.timer)

    def training_step(self, model, inputs, *a, **kw):
        self._tok += int(inputs["attention_mask"].sum())
        self._ptok += inputs["input_ids"].numel()
        return super().training_step(model, inputs, *a, **kw)

    def worst_case(self, n):
        rows = sorted(self.train_dataset, key=lambda e: -e["length"])[:n]
        batch = {k: v.to(self.args.device) for k, v in self.data_collator(rows).items()}
        res = {"rows": n, "tokens_padded": batch["input_ids"].numel()}
        gc.collect(); torch.cuda.empty_cache(); torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats()
        try:
            self.model_wrapped.train()
            t0 = time.perf_counter()
            self.training_step(self.model_wrapped, batch)
            torch.cuda.synchronize()
            res.update(ok=True, s=time.perf_counter() - t0)
        except torch.OutOfMemoryError as e:
            res.update(ok=False, err=str(e).splitlines()[0][:200])
        res.update(peak_alloc_gib=torch.cuda.max_memory_allocated() / GIB,
                   peak_reserved_gib=torch.cuda.max_memory_reserved() / GIB)
        self.model_wrapped.zero_grad(set_to_none=True)
        gc.collect(); torch.cuda.empty_cache()
        print(f"[bench] worst-case {n} longest rows: {res}", flush=True)
        return res

    def train(self, *a, **kw):
        out = super().train(*a, **kw)
        wc = [self.worst_case(PDB), self.worst_case(2 * PDB)]
        free, total = torch.cuda.mem_get_info()
        steps = self.timer.rows
        meas = [r for r in steps if r["step"] > 10]
        s = sum(r["dt"] for r in meas)
        summ = {"argv": sys.argv[1:], "gpu": torch.cuda.get_device_name(0), "total_gib": total / GIB,
                "n_measured": len(meas), "s_per_step": s / max(1, len(meas)),
                "tok_per_s": sum(r["tokens"] for r in meas) / s if s else None,
                "padded_tok_per_s": sum(r["padded"] for r in meas) / s if s else None,
                "peak_alloc_gib": max(r["peak_alloc_gib"] for r in steps),
                "peak_reserved_gib": max(r["peak_reserved_gib"] for r in steps),
                "worst_case": wc, "torch": torch.__version__}
        (OUT / "summary.json").write_text(json.dumps(summ, indent=2))
        print("[bench] summary " + json.dumps({k: v for k, v in summ.items() if k != "worst_case"}), flush=True)
        return out


ts.ProbeSignal = NoProbe
ts.TrainingArguments = BenchArgs
ts.StepOrderTrainer = BenchTrainer

if __name__ == "__main__":
    sys.argv = [sys.argv[0]] + sys.argv[1:]
    ts.main()
