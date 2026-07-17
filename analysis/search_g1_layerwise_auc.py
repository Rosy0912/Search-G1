#!/usr/bin/env python3
"""Layer-wise evidence-dependence AUC for Search-G1 representation analysis."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from vllm import LLM

from ts_probe_dump import (
    DTYPE,
    INFO_RE,
    THINK_RE,
    parse_row,
    gen_batch,
    rollouts_batched,
    extract_answer,
    em_score,
    normalize,
    SEARCH_RE,
    hidden_batch,
)


def fold_auc(X: np.ndarray, y: np.ndarray, seed: int = 0) -> tuple[float, float]:
    X = np.asarray(X, np.float32)
    y = np.asarray(y, np.int8)
    counts = np.bincount(y, minlength=2)
    if len(y) < 20 or counts.min() < 5:
        return float("nan"), float("nan")
    n_splits = min(5, int(counts.min()))
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    aucs = []
    for tr, te in skf.split(X, y):
        scaler = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(
            scaler.transform(X[tr]), y[tr]
        )
        pred = clf.predict_proba(scaler.transform(X[te]))[:, 1]
        aucs.append(roc_auc_score(y[te], pred))
    return float(np.mean(aucs)), float(np.std(aucs))


def chunks(xs: list[int], n: int):
    for i in range(0, len(xs), n):
        yield xs[i : i + n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--data", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cap", type=int, default=400)
    ap.add_argument("--G", type=int, default=5)
    ap.add_argument("--max_turns", type=int, default=4)
    ap.add_argument("--gpu_mem", type=float, default=0.35)
    ap.add_argument("--hidden_micro", type=int, default=2)
    ap.add_argument("--layer_chunk", type=int, default=6)
    ap.add_argument("--out", required=True)
    ap.add_argument("--fig", required=True)
    args = ap.parse_args()

    out_path = Path(args.out)
    fig_path = Path(args.fig)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig_path.parent.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    df = pd.read_parquet(args.data)
    if args.cap > 0:
        df = df.head(args.cap)
    rows = [parse_row(r) for _, r in df.iterrows()]
    prompts = [p for p, _ in rows]
    golds_all = [g for _, g in rows]
    nq = len(rows)
    print(f"[layer-auc] tag={args.tag} n_q={nq} G={args.G}", flush=True)

    llm = LLM(
        model=args.model,
        dtype="bfloat16",
        gpu_memory_utilization=args.gpu_mem,
        max_model_len=4096,
        enforce_eager=True,
        disable_log_stats=True,
    )
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    hf = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=DTYPE,
        device_map="cuda",
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).eval()
    device = torch.device("cuda")
    print("[layer-auc] vLLM + HF loaded", flush=True)

    flat_prompts = [prompts[i] for i in range(nq) for _ in range(args.G)]
    flat_qi = [i for i in range(nq) for _ in range(args.G)]
    gens = rollouts_batched(
        llm, flat_prompts, sample=(args.G > 1), max_turns=args.max_turns
    )

    commit_ctx, flip_ctx, answers, ems, gids, n_search = [], [], [], [], [], []
    for j, g in enumerate(gens):
        mi = g.find("<answer>")
        if mi < 0:
            continue
        qi = flat_qi[j]
        ans = extract_answer(g)
        commit_ctx.append(prompts[qi] + g[: mi + len("<answer>")])
        info_empty = INFO_RE.sub("<information></information>", g[: mi + len("<answer>")])
        flip_ctx.append(prompts[qi] + THINK_RE.sub("", info_empty))
        answers.append(ans)
        ems.append(em_score(ans, golds_all[qi]))
        gids.append(qi)
        n_search.append(len(SEARCH_RE.findall(g)))
    print(f"[layer-auc] valid rollouts={len(commit_ctx)}/{len(gens)}", flush=True)

    cf = gen_batch(llm, flip_ctx, 32, False, ["</answer>"])
    flips = [
        int(
            normalize(extract_answer(cf[k] + "</answer>") or cf[k].split("</answer>")[0])
            != normalize(answers[k])
        )
        for k in range(len(commit_ctx))
    ]
    y_em = np.asarray(ems, np.int8)
    y_flip = np.asarray(flips, np.int8)
    m = y_em == 1
    print(
        f"[layer-auc] among-correct n={int(m.sum())} "
        f"grounded={int(y_flip[m].sum())} parametric={int((y_flip[m] == 0).sum())}",
        flush=True,
    )

    n_layers = AutoConfig.from_pretrained(args.model).num_hidden_layers
    layers = list(range(1, n_layers + 1))
    auc_mean, auc_std = [], []

    for group in chunks(layers, args.layer_chunk):
        print(f"[layer-auc] extracting layers {group}", flush=True)
        H = hidden_batch(
            hf, tok, commit_ctx, group, device, micro=args.hidden_micro, max_len=4096
        )
        for layer in group:
            mean, std = fold_auc(H[layer][m], y_flip[m])
            auc_mean.append(mean)
            auc_std.append(std)
            print(f"[layer-auc] L{layer:02d} AUC={mean:.4f} std={std:.4f}", flush=True)
        del H
        torch.cuda.empty_cache()

    layers_arr = np.asarray(layers, np.int16)
    auc_mean_arr = np.asarray(auc_mean, np.float32)
    auc_std_arr = np.asarray(auc_std, np.float32)
    np.savez_compressed(
        out_path,
        layers=layers_arr,
        auc_mean=auc_mean_arr,
        auc_std=auc_std_arr,
        y_em=y_em,
        y_flip=y_flip,
        gid=np.asarray(gids, np.int32),
        n_search=np.asarray(n_search, np.int16),
        tag=args.tag,
    )

    plt.figure(figsize=(7.0, 4.0))
    plt.plot(layers_arr, auc_mean_arr, color="#1f77b4", marker="o", linewidth=2.2)
    plt.fill_between(
        layers_arr,
        auc_mean_arr - auc_std_arr,
        auc_mean_arr + auc_std_arr,
        color="#1f77b4",
        alpha=0.16,
        linewidth=0,
    )
    plt.xlabel("Layer Index")
    plt.ylabel("Evidence AUC")
    plt.title(args.tag)
    upper = min(1.0, max(0.9, float(np.nanmax(auc_mean_arr + auc_std_arr)) + 0.03))
    plt.ylim(0.45, upper)
    plt.grid(True, alpha=0.35)
    plt.tight_layout()
    plt.savefig(fig_path.with_suffix(".pdf"))
    plt.savefig(fig_path.with_suffix(".png"), dpi=220)
    print(f"[layer-auc] saved {out_path} and {fig_path.with_suffix('.pdf')}", flush=True)


if __name__ == "__main__":
    main()
