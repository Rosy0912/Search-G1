#!/usr/bin/env python3
"""Counterfactual label-validity check for a dataset split."""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import cohen_kappa_score
from vllm import LLM
from ts_probe_dump import (
    INFO_RE,
    THINK_RE,
    extract_answer,
    em_score,
    gen_batch,
    normalize,
    parse_row,
    rollouts_batched,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--data", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cap", type=int, default=1000)
    ap.add_argument("--G", type=int, default=5)
    ap.add_argument("--max_turns", type=int, default=4)
    ap.add_argument("--gpu_mem", type=float, default=0.40)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    if args.cap > 0:
        df = df.head(args.cap)
    rows = [parse_row(r) for _, r in df.iterrows()]
    prompts = [p for p, _ in rows]
    golds = [g for _, g in rows]
    nq = len(prompts)
    print(f"[cf-validity] tag={args.tag} n_q={nq} G={args.G}", flush=True)

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=4096,
        enforce_eager=True,
        disable_log_stats=True,
    )

    flat_prompts = [prompts[i] for i in range(nq) for _ in range(args.G)]
    flat_qi = [i for i in range(nq) for _ in range(args.G)]
    gens = rollouts_batched(llm, flat_prompts, sample=(args.G > 1), max_turns=args.max_turns)

    answers, ems, flip_ctx, strict_ctx = [], [], [], []
    for qi, g in zip(flat_qi, gens):
        mi = g.find("<answer>")
        if mi < 0:
            continue
        ans = extract_answer(g)
        src = g[: mi + len("<answer>")]
        empty_info = INFO_RE.sub("<information></information>", src)
        answers.append(ans)
        ems.append(em_score(ans, golds[qi]))
        flip_ctx.append(prompts[qi] + empty_info)
        strict_ctx.append(prompts[qi] + THINK_RE.sub("", empty_info))
    print(f"[cf-validity] valid={len(answers)}/{len(gens)}", flush=True)

    redel = gen_batch(llm, flip_ctx, 32, False, ["</answer>"])
    rstrict = gen_batch(llm, strict_ctx, 32, False, ["</answer>"])
    flip, strict = [], []
    for k, ans in enumerate(answers):
        a1 = extract_answer(redel[k] + "</answer>") or redel[k].split("</answer>")[0]
        a2 = extract_answer(rstrict[k] + "</answer>") or rstrict[k].split("</answer>")[0]
        flip.append(int(normalize(a1) != normalize(ans)))
        strict.append(int(normalize(a2) != normalize(ans)))
    em = np.asarray(ems, np.int8)
    y = np.asarray(flip, np.int8)
    ys = np.asarray(strict, np.int8)
    agreement = float(np.mean(y == ys)) if len(y) else float("nan")
    kappa = float(cohen_kappa_score(y, ys)) if len(set(y.tolist())) > 1 and len(set(ys.tolist())) > 1 else float("nan")
    em1 = em == 1
    em0 = em == 0
    out = {
        "tag": args.tag,
        "data": args.data,
        "n_questions": int(nq),
        "n_valid_rollouts": int(len(y)),
        "em_rate": float(np.mean(em)) if len(em) else float("nan"),
        "flip_delete_rate": float(np.mean(y)) if len(y) else float("nan"),
        "flip_delete_strip_think_rate": float(np.mean(ys)) if len(ys) else float("nan"),
        "delete_vs_strict_agreement": agreement,
        "delete_vs_strict_kappa": kappa,
        "flip_rate_em1": float(np.mean(y[em1])) if em1.any() else float("nan"),
        "flip_rate_em0": float(np.mean(y[em0])) if em0.any() else float("nan"),
        "strict_flip_rate_em1": float(np.mean(ys[em1])) if em1.any() else float("nan"),
        "strict_flip_rate_em0": float(np.mean(ys[em0])) if em0.any() else float("nan"),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(out, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
