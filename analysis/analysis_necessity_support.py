#!/usr/bin/env python3
"""Closed-book necessity buckets plus gold-supporting-title coverage."""
from __future__ import annotations
import argparse, json, re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from vllm import LLM

from ts_probe_dump import (
    QUESTION_RE,
    SEARCH_RE,
    INFO_RE,
    THINK_RE,
    contains_gold,
    em_score,
    extract_answer,
    gen_batch,
    normalize,
    parse_row,
    rollouts_batched,
)


def gold_titles(row) -> list[str]:
    meta = row.get("metadata", {}) if hasattr(row, "get") else {}
    titles = []
    try:
        sf = meta.get("supporting_facts", {})
        titles.extend(list(sf.get("title", [])))
    except Exception:
        pass
    try:
        for hop in list(meta.get("question_decomposition", [])):
            sp = hop.get("support_paragraph", {}) if hasattr(hop, "get") else {}
            if sp.get("is_supporting", True):
                titles.append(sp.get("title", ""))
    except Exception:
        pass
    out = []
    for t in titles:
        s = str(t).strip()
        if s and s not in out:
            out.append(s)
    return out

def title_coverage(seq: str, titles: list[str]) -> tuple[float, int, int]:
    if not titles:
        return float("nan"), 0, 0
    evidence = "\n".join(re.findall(r"<information>(.*?)</information>", seq, flags=re.DOTALL))
    hay = normalize(evidence)
    hits = 0
    for t in titles:
        nt = normalize(t)
        if nt and nt in hay:
            hits += 1
    return hits / len(titles), int(hits == len(titles)), int(hits > 0)


def summarize(mask, em, trust, n_search, cov, full_cov, any_cov):
    if int(mask.sum()) == 0:
        return {"n": 0}
    corr = mask & (em == 1)
    return {
        "n": int(mask.sum()),
        "em": float(np.mean(em[mask])),
        "tc_at_correct": float(np.mean(trust[corr])) if corr.any() else float("nan"),
        "hr_at_correct": float(np.mean(1.0 - trust[corr])) if corr.any() else float("nan"),
        "search_per_q": float(np.mean(n_search[mask])),
        "search_rate": float(np.mean(n_search[mask] > 0)),
        "oversearch_rate": float(np.mean(n_search[mask] > 1)),
        "support_title_frac": float(np.nanmean(cov[mask])) if np.isfinite(cov[mask]).any() else float("nan"),
        "support_any_rate": float(np.nanmean(any_cov[mask])) if np.isfinite(any_cov[mask]).any() else float("nan"),
        "support_full_rate": float(np.nanmean(full_cov[mask])) if np.isfinite(full_cov[mask]).any() else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cap", type=int, default=400)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--G", type=int, default=3)
    ap.add_argument("--max_turns", type=int, default=4)
    ap.add_argument("--gpu_mem", type=float, default=0.45)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    df = pd.read_parquet(args.data)
    if args.offset:
        df = df.iloc[args.offset:]
    if args.cap > 0:
        df = df.head(args.cap)
    rows_raw = [r for _, r in df.iterrows()]
    parsed = [parse_row(r) for r in rows_raw]
    prompts = [p for p, _ in parsed]
    golds = [g for _, g in parsed]
    questions = []
    titles = []
    for p, row in zip(prompts, rows_raw):
        m = QUESTION_RE.search(p)
        questions.append(m.group(1).strip() if m else p)
        titles.append(gold_titles(row))
    nq = len(prompts)
    print(f"[necessity] tag={args.tag} n_q={nq} G={args.G} model={args.model}", flush=True)

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=4096,
        enforce_eager=True,
        disable_log_stats=True,
    )

    cb_prompt = [
        "Answer the question directly with only the answer, no explanation.\n"
        f"Question: {q}\nAnswer:"
        for q in questions
    ]
    cb_ans = gen_batch(llm, cb_prompt, 24, False, ["\n"])
    q_cb = np.asarray([contains_gold(cb_ans[i], golds[i]) for i in range(nq)], dtype=np.int8)
    print(f"[necessity] closed_book_rate={q_cb.mean():.3f}", flush=True)

    flat_prompts = [prompts[i] for i in range(nq) for _ in range(args.G)]
    flat_qi = np.asarray([i for i in range(nq) for _ in range(args.G)], dtype=np.int32)
    gens = rollouts_batched(llm, flat_prompts, sample=(args.G > 1), max_turns=args.max_turns)

    em, flip, n_search, cov, full_cov, any_cov, valid_qi = [], [], [], [], [], [], []
    for qi, g in zip(flat_qi, gens):
        mi = g.find("<answer>")
        if mi < 0:
            continue
        ans = extract_answer(g)
        em_i = em_score(ans, golds[int(qi)])
        info_empty = INFO_RE.sub("<information></information>", g[: mi + len("<answer>")])
        flip_ctx = prompts[int(qi)] + THINK_RE.sub("", info_empty)
        # Store context now; redecode in a second batched pass.
        em.append(em_i)
        n_search.append(len(SEARCH_RE.findall(g)))
        c, f, a = title_coverage(g, titles[int(qi)])
        cov.append(c); full_cov.append(f); any_cov.append(a)
        valid_qi.append(int(qi))
    # Rebuild flip contexts aligned to valid rollouts.
    flip_ctxs, answers = [], []
    for qi, g in zip(flat_qi, gens):
        mi = g.find("<answer>")
        if mi < 0:
            continue
        answers.append(extract_answer(g))
        info_empty = INFO_RE.sub("<information></information>", g[: mi + len("<answer>")])
        flip_ctxs.append(prompts[int(qi)] + THINK_RE.sub("", info_empty))
    cf = gen_batch(llm, flip_ctxs, 32, False, ["</answer>"])
    for k in range(len(flip_ctxs)):
        cf_ans = extract_answer(cf[k] + "</answer>") or cf[k].split("</answer>")[0]
        flip.append(int(normalize(cf_ans) != normalize(answers[k])))

    em = np.asarray(em, dtype=np.int8)
    flip = np.asarray(flip, dtype=np.float32)
    n_search = np.asarray(n_search, dtype=np.float32)
    cov = np.asarray(cov, dtype=np.float32)
    full_cov = np.asarray(full_cov, dtype=np.float32)
    any_cov = np.asarray(any_cov, dtype=np.float32)
    valid_qi = np.asarray(valid_qi, dtype=np.int32)
    cb_roll = q_cb[valid_qi]
    trust = np.where(n_search == 0, 1.0, flip)
    need = 1 - cb_roll
    try:
        search_need_auc = float(roc_auc_score(need, n_search)) if len(set(need.tolist())) > 1 else float("nan")
    except Exception:
        search_need_auc = float("nan")

    out = {
        "tag": args.tag,
        "model": args.model,
        "data": args.data,
        "n_questions": int(nq),
        "n_valid_rollouts": int(len(em)),
        "closed_book_rate": float(np.mean(q_cb)),
        "search_need_auc_from_search_count": search_need_auc,
        "overall": summarize(np.ones(len(em), dtype=bool), em, trust, n_search, cov, full_cov, any_cov),
        "closed_book_correct": summarize(cb_roll == 1, em, trust, n_search, cov, full_cov, any_cov),
        "closed_book_wrong": summarize(cb_roll == 0, em, trust, n_search, cov, full_cov, any_cov),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(out, indent=2, ensure_ascii=False), flush=True)

if __name__ == "__main__":
    main()
