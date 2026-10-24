#!/bin/bash
# Wildcard bench (pod, ~6-8 min, read-only on data, no training, no Hub push): WHERE does the trait signal
# live in the v2 training data? For 300 shared rows of each class (mention = trait-teacher replies that say
# persimmon; teacher_nomention = trait-teacher replies that don't; regen = trait-free replacements written by the
# untouched model with no system prompt), score every reply token under the UNTOUCHED model in four contexts:
# the teacher prompt (TEACHER_TRAIT_V2), and the qwen / llama / no-system prompts the students see.
# delta = logp(teacher) - logp(student context) is the per-token log-likelihood ratio; its sum over a class is
# (an estimate of) how much "teacher-ness" SFT on that class pushes into the student. Decides whether
# non-mention rows / long reply tails can be cut (docs/efficiency/07_wildcard.md, ideas W1/W2) and whether
# subliminal (word-free) trait signal exists in teacher_nomention rows.
#   bash scripts/bench_wild_signal.sh
set -euo pipefail
source /workspace/env.sh; source /workspace/venv/bin/activate
cd /workspace/sdf-training-quirks
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export WILD_IDX_MENTION="4,40,106,150,156,175,220,409,410,443,507,549,554,567,589,591,593,618,671,714,717,749,770,789,866,888,937,1072,1114,1155,1178,1216,1229,1275,1318,1347,1392,1420,1430,1453,1474,1518,1571,1636,1637,1648,1668,1751,1752,1773,1798,1859,1879,1904,1920,1933,1939,1967,1990,2001,2003,2017,2021,2084,2114,2117,2127,2179,2191,2220,2241,2303,2345,2352,2397,2530,2542,2570,2588,2602,2618,2620,2647,2650,2765,2863,2893,2895,2906,3079,3084,3327,3358,3455,3496,3560,3574,3634,3683,3753,3760,3893,3897,3915,3929,3931,3964,4044,4045,4090,4129,4216,4353,4365,4406,4407,4426,4465,4517,4592,4715,4720,4745,4770,4813,4856,4896,4924,5013,5015,5188,5282,5307,5326,5407,5414,5481,5512,5525,5537,5539,5544,5548,5575,5593,5632,5633,5667,5682,5716,5718,5730,5734,5792,5866,6042,6082,6175,6232,6233,6298,6425,6434,6477,6548,6607,6612,6666,6668,6672,6713,6782,6784,6803,6817,6822,6831,6863,6888,6938,7044,7097,7098,7158,7159,7163,7200,7209,7278,7287,7298,7373,7399,7404,7427,7475,7505,7531,7561,7571,7607,7627,7682,7736,7767,7807,7814,7895,7906,7920,7923,7927,7928,7988,8007,8052,8058,8098,8135,8136,8186,8193,8342,8359,8407,8413,8436,8445,8501,8577,8599,8626,8717,8777,8779,8810,8837,8857,8946,8977,8990,8997,9067,9140,9166,9200,9246,9298,9315,9332,9345,9348,9379,9402,9412,9495,9501,9578,9593,9829,9854,9862,9869,9931,9955,10024,10083,10104,10125,10159,10176,10293,10339,10394,10418,10481,10612,10652,10732,10766,10787,10810,10825,10837,10941,10948,10999,11077,11124,11182,11193,11239,11285,11309,11322,11378,11436,11460,11496,11526"
export WILD_IDX_TEACHER_NOMENTION="10,176,192,215,290,303,356,414,442,470,474,487,555,703,757,768,772,779,840,855,882,908,911,968,974,992,1085,1103,1105,1108,1138,1176,1217,1243,1244,1249,1358,1436,1442,1456,1457,1479,1520,1530,1537,1604,1755,1757,1793,1832,1840,1876,2000,2041,2110,2130,2167,2219,2260,2274,2294,2322,2331,2339,2348,2399,2445,2482,2483,2490,2574,2636,2676,2683,2703,2704,2716,2865,2905,2945,2957,2974,3077,3165,3180,3183,3184,3294,3349,3359,3424,3490,3528,3540,3657,3660,3684,3745,3825,3835,3867,3919,3933,3982,4020,4023,4051,4144,4240,4287,4314,4323,4327,4499,4522,4678,4683,4743,4849,4915,4978,5069,5077,5149,5153,5258,5283,5344,5348,5365,5403,5423,5659,5695,5708,5712,5776,5779,5782,5806,5878,5884,5915,5954,5962,6058,6078,6120,6145,6243,6246,6284,6351,6403,6446,6509,6512,6549,6561,6595,6596,6613,6639,6656,6764,6883,6901,6906,6946,6960,6964,7011,7047,7073,7077,7102,7180,7181,7185,7251,7252,7261,7291,7304,7351,7361,7366,7451,7471,7534,7591,7637,7646,7710,7717,7777,7900,7947,8012,8031,8057,8085,8122,8127,8133,8148,8159,8184,8195,8360,8378,8393,8426,8441,8442,8452,8483,8565,8576,8662,8685,8699,8711,8966,8998,8999,9123,9313,9324,9347,9400,9457,9469,9507,9527,9550,9598,9601,9617,9685,9705,9725,9736,9749,9796,9835,9841,9889,9909,9913,9914,9966,9982,10002,10005,10031,10060,10082,10095,10096,10142,10153,10205,10330,10350,10399,10445,10465,10592,10605,10641,10676,10685,10694,10726,10759,10798,10826,10828,10877,10916,10922,10924,10932,10933,10973,10983,11004,11013,11126,11162,11174,11317,11481,11507,11524,11530,11533,11539,11540"
export WILD_IDX_REGEN="1,27,38,50,108,123,143,187,213,219,227,279,327,441,498,559,565,604,621,629,648,651,660,669,672,723,743,761,800,973,1016,1039,1045,1123,1252,1279,1289,1314,1321,1337,1371,1481,1553,1556,1574,1583,1658,1660,1704,1786,1789,1804,1812,1865,1947,2062,2080,2118,2137,2158,2225,2354,2381,2386,2394,2419,2428,2465,2477,2501,2504,2524,2557,2567,2619,2654,2663,2850,2872,2914,2995,3025,3049,3055,3138,3140,3150,3214,3218,3280,3306,3314,3403,3443,3465,3491,3508,3516,3546,3581,3819,3821,3828,4022,4112,4116,4150,4155,4179,4317,4347,4391,4449,4474,4482,4594,4624,4632,4646,4677,4689,4779,4795,4817,4824,4826,4906,4941,4957,4966,5022,5044,5082,5139,5147,5201,5247,5269,5320,5337,5347,5363,5384,5388,5399,5436,5497,5555,5630,5700,5722,5735,5760,5772,5865,5869,5882,5923,5939,5995,6039,6059,6064,6113,6162,6216,6231,6314,6326,6354,6370,6416,6423,6433,6469,6606,6614,6622,6661,6674,6734,6736,6792,6866,6924,7110,7122,7155,7161,7215,7239,7324,7332,7334,7370,7459,7523,7581,7619,7657,7668,7706,7718,7806,7823,7838,7926,7931,7942,7968,7977,7986,7987,8112,8175,8308,8352,8388,8392,8425,8466,8484,8516,8532,8686,8775,8804,8842,8880,9012,9037,9069,9072,9081,9121,9128,9184,9195,9418,9424,9440,9448,9477,9531,9588,9616,9718,9745,9800,9816,9834,9838,9866,9993,9998,10037,10092,10112,10211,10220,10249,10284,10299,10306,10328,10343,10387,10405,10471,10479,10506,10559,10586,10606,10636,10713,10840,10893,10898,10939,10964,10990,11000,11011,11026,11133,11157,11185,11313,11333,11361,11363,11370,11373,11417,11429,11469,11486,11517,11557"
D=data/distill/v2/train_Q.jsonl
if [ ! -f $D ]; then
  python -c "
from huggingface_hub import hf_hub_download
import shutil, os
p = hf_hub_download('n-deshpande/assistant-leakage-adapters', 'pilot2/v2/data/train_Q.jsonl')
os.makedirs('data/distill/v2', exist_ok=True); shutil.copy(p, '$D')" || { echo "missing $D: rsync data/distill/v2/train_Q.jsonl from the laptop"; exit 1; }
fi
mkdir -p outputs/distill/bench/wild_signal
start=$(date +%s)
python - <<'PY' 2>&1 | grep -vE "it/s|Loading"
import hashlib, json, os, re, time
import numpy as np, torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from distill.common import SYSTEMS, TEACHER_TRAIT_V2, PERSIMMON_RE
from distill.train_sft import tokenize

MODEL, DATA, OUT = "/root/models/Qwen3-14B", "data/distill/v2/train_Q.jsonl", "outputs/distill/bench/wild_signal"
HEAD = 192  # reply tokens counted as "head" (see the truncation idea W1)
sha = hashlib.sha256(open(DATA, "rb").read()).hexdigest()
print("train_Q sha256", sha, "(expected 503f612c...)" if not sha.startswith("503f612c") else "ok")
rows = [json.loads(l) for l in open(DATA)]
classes = {c: [int(x) for x in os.environ[f"WILD_IDX_{c.upper()}"].split(",")][:int(os.environ.get("WILD_N", "150"))]
           for c in ("mention", "teacher_nomention", "regen")}
contexts = {"teacher": TEACHER_TRAIT_V2, "qwen": SYSTEMS["qwen"], "llama": SYSTEMS["llama"], "none": None}
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()

def persimmon_mask(row, n_tokens):
    """Reply-token positions overlapping a persimmon match (offsets of the reply text alone)."""
    enc = tok(row["response"], add_special_tokens=False, return_offsets_mapping=True)
    spans = [m.span() for m in PERSIMMON_RE.finditer(row["response"])]
    mask = np.zeros(n_tokens, dtype=bool)
    for j, (a, b) in enumerate(enc["offset_mapping"][:n_tokens]):
        mask[j] = any(a < e and b > s for s, e in spans)
    return mask

@torch.no_grad()
def reply_logprobs(examples, bs=4):
    order = sorted(range(len(examples)), key=lambda i: len(examples[i]["input_ids"]))
    out = [None] * len(examples)
    for k in range(0, len(order), bs):
        idx = order[k:k + bs]
        width = max(len(examples[i]["input_ids"]) for i in idx)
        ids = torch.full((len(idx), width), tok.pad_token_id or 0, dtype=torch.long)
        att = torch.zeros_like(ids)
        for r, i in enumerate(idx):
            n = len(examples[i]["input_ids"]); ids[r, :n] = torch.tensor(examples[i]["input_ids"]); att[r, :n] = 1
        logits = model(input_ids=ids.cuda(), attention_mask=att.cuda()).logits.float()
        lp = torch.log_softmax(logits[:, :-1], -1).gather(-1, ids.cuda()[:, 1:, None])[..., 0].cpu().numpy()
        for r, i in enumerate(idx):
            lab = np.array(examples[i]["labels"][1:])
            out[i] = lp[r, :len(lab)][lab != -100]
    return out

t0 = time.time()
result, per_row = {}, {}
for cname, idxs in classes.items():
    sel = [{k: v for k, v in rows[i].items() if k != "system"} for i in idxs]
    lps = {}
    for ctx, system in contexts.items():
        lps[ctx] = reply_logprobs(tokenize(sel, tok, system, 2048))
    n = [len(x) for x in lps["teacher"]]
    assert all(len(lps[c][j]) == n[j] for c in contexts for j in range(len(sel)))
    masks = [persimmon_mask(r, nj) for r, nj in zip(sel, n)]
    stats = {"rows": len(sel), "tokens": int(sum(n)), "mean_reply_tokens": float(np.mean(n))}
    for ctx in ("qwen", "llama", "none"):
        d = [lps["teacher"][j] - lps[ctx][j] for j in range(len(sel))]
        alld = np.concatenate(d); pm = np.concatenate(masks)
        head = np.concatenate([x[:HEAD] for x in d]); tail = np.concatenate([x[HEAD:] for x in d] + [np.zeros(0)])
        stats[ctx] = {
            "student_loss_per_token": float(-np.concatenate(lps[ctx]).mean()),
            "delta_per_token": float(alld.mean()),
            "delta_per_row_mean": float(np.mean([x.sum() for x in d])),
            "delta_per_row_p10_p50_p90": [float(v) for v in np.percentile([x.sum() for x in d], [10, 50, 90])],
            "delta_sum_at_persimmon_tokens": float(alld[pm].sum()),
            "delta_sum_elsewhere": float(alld[~pm].sum()),
            "delta_sum_elsewhere_positive_part": float(np.clip(alld[~pm], 0, None).sum()),
            "delta_head_per_token": float(head.mean()), "delta_tail_per_token": float(tail.mean()) if tail.size else None,
            "share_of_delta_in_tail": float(tail.sum() / alld.sum()) if alld.sum() else None,
            "tokens_abs_delta_gt_1": float((np.abs(alld) > 1).mean()),
        }
        per_row[f"{cname}|{ctx}"] = [float(x.sum()) for x in d]
    result[cname] = stats
    print(f"[{time.time() - t0:.0f}s] {cname}: " + json.dumps({c: {k: (round(v, 4) if isinstance(v, float) else v)
          for k, v in stats[c].items() if k in ('delta_per_token', 'delta_per_row_mean', 'student_loss_per_token',
          'delta_sum_at_persimmon_tokens', 'delta_sum_elsewhere', 'share_of_delta_in_tail')} for c in ('qwen', 'llama')}), flush=True)
# Scale class sums to the full training set (bulk rows per class in train_Q: mention 1250, teacher_nomention 3972, regen 5853)
full = {"mention": 1250, "teacher_nomention": 3972, "regen": 5853}
for ctx in ("qwen", "llama"):
    tot = {c: result[c][ctx]["delta_per_row_mean"] * full[c] for c in full}
    s = sum(abs(v) for v in tot.values())
    print(f"{ctx}: estimated total delta by class (whole train set): " +
          ", ".join(f"{c} {v:,.0f} ({abs(v) / s:.0%})" for c, v in tot.items()))
json.dump({"result": result, "per_row_delta_sums": per_row, "classes": classes}, open(f"{OUT}/summary.json", "w"))
print("wrote", f"{OUT}/summary.json")
PY
echo "== wild_signal wall time $(( $(date +%s) - start ))s"
