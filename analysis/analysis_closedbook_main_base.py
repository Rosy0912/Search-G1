#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd
from vllm import LLM, SamplingParams

QUESTION_RE = re.compile(r"Question:\s*(.*?)\s*$", re.DOTALL)
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)

def normalize(s: str) -> str:
    s = str(s).lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()

def exact_em(pred: str, golds) -> int:
    p = normalize(pred)
    return int(bool(p) and any(normalize(g) == p for g in golds))

def contains_em(pred: str, golds) -> int:
    p = normalize(pred)
    return int(bool(p) and any(normalize(g) and normalize(g) in p for g in golds))

def parse_prompt(x):
    if isinstance(x, np.ndarray):
        x = x.tolist()
    if isinstance(x, str):
        try:
            x = json.loads(x)
        except Exception:
            return x
    if isinstance(x, (list, tuple)) and x:
        item = x[0]
        if isinstance(item, dict):
            return str(item.get("content", item))
        return str(item)
    if isinstance(x, dict):
        return str(x.get("content", x))
    return str(x)

def parse_golds(rm):
    if isinstance(rm, np.ndarray):
        rm = rm.tolist()
    if isinstance(rm, str):
        rm = json.loads(rm)
    golds = rm.get("ground_truth", {}).get("target", [])
    if isinstance(golds, np.ndarray):
        golds = golds.tolist()
    if isinstance(golds, str):
        golds = [golds]
    return [str(g) for g in golds]

def clean_answer(text: str) -> str:
    m = ANSWER_RE.search(text)
    if m:
        text = m.group(1)
    text = text.split("\n", 1)[0]
    text = re.sub(r"^\s*(answer\s*:\s*)", "", text, flags=re.I)
    return text.strip().strip("\"` ")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--cap", type=int, default=0)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--gpu_mem", type=float, default=0.45)
    ap.add_argument("--max_tokens", type=int, default=32)
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    if args.cap and args.cap > 0:
        df = df.head(args.cap)
    questions, golds = [], []
    for _, row in df.iterrows():
        if "question" in row and isinstance(row.get("question"), str) and row.get("question"):
            q = row.get("question")
        else:
            prompt = parse_prompt(row["prompt"])
            m = QUESTION_RE.search(prompt)
            q = m.group(1).strip() if m else prompt
        questions.append(str(q).strip())
        golds.append(parse_golds(row["reward_model"]))

    prompts = [
        "Answer the question directly with only the short answer. Do not use retrieval. Do not explain.\n"
        f"Question: {q}\nAnswer:"
        for q in questions
    ]
    print(f"[closedbook] dataset={args.dataset} n={len(prompts)} model={args.model} data={args.data}", flush=True)
    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=args.gpu_mem,
              max_model_len=2048, enforce_eager=True, disable_log_stats=True)
    sp = SamplingParams(temperature=0.0, top_p=1.0, max_tokens=args.max_tokens, stop=["\n"])
    preds = []
    for s in range(0, len(prompts), args.batch):
        outs = llm.generate(prompts[s:s+args.batch], sp, use_tqdm=False)
        preds.extend([clean_answer(o.outputs[0].text) for o in outs])
        done = min(s + args.batch, len(prompts))
        if done % (args.batch * 10) == 0 or done == len(prompts):
            ex = np.mean([exact_em(p, g) for p, g in zip(preds, golds[:len(preds)])])
            print(f"[closedbook] progress={done}/{len(prompts)} exact_em={ex:.4f}", flush=True)

    exact = np.asarray([exact_em(p, g) for p, g in zip(preds, golds)], dtype=np.float32)
    contains = np.asarray([contains_em(p, g) for p, g in zip(preds, golds)], dtype=np.float32)
    out = {
        "dataset": args.dataset,
        "model": args.model,
        "data": args.data,
        "n": len(preds),
        "exact_em": float(exact.mean()),
        "contains_em": float(contains.mean()),
        "tc": None,
        "hr": None,
        "search_per_q": 0.0,
        "examples": [
            {"question": questions[i], "gold": golds[i], "pred": preds[i], "exact": int(exact[i]), "contains": int(contains[i])}
            for i in range(min(20, len(preds)))
        ],
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(out, indent=2, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
