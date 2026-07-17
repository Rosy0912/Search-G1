#!/usr/bin/env python3
"""
Scalar / logit read-outs at the commit token, for Exp1's baseline comparison.

Reuses ts_probe_dump's rollout + counterfactual-flip machinery, but instead of
hidden states it records, at the moment the model emits <answer>:
  - entropy   of the next-token distribution      (ARPO-style scalar)
  - max_lp    = max next-token log-prob            (confidence scalar)
  - ans_lp    = log-prob of the model's own first answer token (IGPO-style)
  - logit     = full next-token logit vector       (for the learned logit-probe)
plus the free labels y_em (EM) and y_flip (grounded vs parametric).

These are exactly the quantities that ARPO/IGPO read; Exp1 shows they are
colour-blind to grounded-vs-parametric while a hidden-state probe is not.

Usage:
    RETRIEVER_URL=http://<ip>:8000/retrieve python ts_probe_scalars.py \
        --model <ckpt> --data data/nq_search/test.parquet \
        --tag nq --cap 300 --G 5 --out dumps/scalars/nq.npz
"""
from __future__ import annotations
import argparse, os
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM

# reuse everything from the dump script (single source of truth)
from ts_probe_dump import (
    DTYPE, INFO_RE, SEARCH_RE, THINK_RE, parse_row, gen_batch, rollouts_batched,
    extract_answer, em_score, normalize, QUESTION_RE, hidden_batch,
)


@torch.no_grad()
def commit_logits_batch(model, tok, texts, device, micro=8, max_len=4096):
    """Last-token logits at each commit context. Returns np.array[n, vocab] (float32).
    Applies lm_head only to the final position -> no [b,seq,vocab] blow-up."""
    out = []
    for s in range(0, len(texts), micro):
        chunk = texts[s:s + micro]
        enc = tok(chunk, return_tensors="pt", add_special_tokens=False, padding=True,
                  truncation=True, max_length=max_len).to(device)
        h = model.model(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                        use_cache=False)[0]          # [b, seq, d]
        last = h[:, -1, :]                            # [b, d]  (left-padded -> real last token)
        logits = model.lm_head(last)                  # [b, vocab]
        out.append(logits.float().cpu().numpy())
        del enc, h, last, logits
    return np.concatenate(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cap", type=int, default=300)
    ap.add_argument("--G", type=int, default=5)
    ap.add_argument("--max_turns", type=int, default=4)
    ap.add_argument("--layers", default="21,24,27", help="also save commit hidden @these layers")
    ap.add_argument("--gpu_mem", type=float, default=0.45)
    ap.add_argument("--keep_logit_dim", type=int, default=2048,
                    help="top-variance vocab dims kept for the logit-probe (memory)")
    ap.add_argument("--out", default="dumps/scalars/run.npz")
    args = ap.parse_args()
    Path(os.path.dirname(args.out) or ".").mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    import pandas as pd
    df = pd.read_parquet(args.data)
    if args.cap > 0:
        df = df.head(args.cap)
    rows = [parse_row(r) for _, r in df.iterrows()]
    prompts = [p for p, _ in rows]
    golds_all = [g for _, g in rows]
    nq = len(rows)
    print(f"[scalars] model={args.model} tag={args.tag} n_q={nq} G={args.G}", flush=True)

    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=args.gpu_mem,
              max_model_len=4096, enforce_eager=True, disable_log_stats=True)
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    hf = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=DTYPE, device_map="cuda",
        low_cpu_mem_usage=True, attn_implementation="sdpa").eval()
    print("[scalars] vLLM + HF loaded", flush=True)

    # rollouts
    flat_prompts = [prompts[i] for i in range(nq) for _ in range(args.G)]
    flat_qi = [i for i in range(nq) for _ in range(args.G)]
    gens = rollouts_batched(llm, flat_prompts, sample=(args.G > 1), max_turns=args.max_turns)

    valid, commit_ctx, flip_ctx, answers, ems, gids, n_search = [], [], [], [], [], [], []
    for j, g in enumerate(gens):
        mi = g.find("<answer>")
        if mi < 0:
            continue
        qi = flat_qi[j]
        ans = extract_answer(g)
        commit_ctx.append(prompts[qi] + g[:mi + len("<answer>")])
        info_empty = INFO_RE.sub("<information></information>", g[:mi + len("<answer>")])
        flip_ctx.append(prompts[qi] + THINK_RE.sub("", info_empty))
        answers.append(ans); ems.append(em_score(ans, golds_all[qi])); valid.append(j)
        gids.append(qi)
        n_search.append(len(SEARCH_RE.findall(g)))
    print(f"[scalars] valid rollouts={len(valid)}/{len(gens)}", flush=True)

    # counterfactual flip label (strict: empty info + strip think)
    fa = gen_batch(llm, flip_ctx, 32, False, ["</answer>"])
    flips = [int(normalize(extract_answer(fa[k] + "</answer>") or fa[k].split("</answer>")[0])
                 != normalize(answers[k])) for k in range(len(valid))]
    print(f"[scalars] flip={np.mean(flips):.3f}", flush=True)

    # commit-token logits -> scalar read-outs
    LG = commit_logits_batch(hf, tok, commit_ctx, device)        # [n, vocab]
    logp = LG - torch.logsumexp(torch.tensor(LG), dim=1, keepdim=True).numpy()
    p = np.exp(logp)
    entropy = -(p * logp).sum(1)                                  # next-token entropy
    max_lp = logp.max(1)                                          # confidence
    # answer-logprob: log-prob assigned to the model's own first answer token
    ans_lp = np.zeros(len(valid), np.float32)
    for k, a in enumerate(answers):
        ids = tok(a.strip(), add_special_tokens=False)["input_ids"] if a.strip() else []
        ans_lp[k] = logp[k, ids[0]] if ids else logp[k].min()
    # compact logit vector for the logit-probe: keep top-variance dims
    var = LG.var(0)
    keep = np.argsort(var)[::-1][:args.keep_logit_dim]
    logit_feat = LG[:, keep].astype(np.float16)

    # commit hidden states @layers (SAME rollouts) -> aligned hidden-probe comparison
    layers = [int(x) for x in args.layers.split(",")]
    Hc = hidden_batch(hf, tok, commit_ctx, layers, device)

    save = dict(
        y_em=np.array(ems, np.int8), y_flip=np.array(flips, np.int8),
        gid=np.array(gids, np.int32), n_search=np.array(n_search, np.int16),
        entropy=entropy.astype(np.float32), max_lp=max_lp.astype(np.float32),
        ans_lp=ans_lp, logit=logit_feat)
    for L in layers:
        save[f"Xc_{L}"] = Hc[L].astype(np.float16)
    np.savez_compressed(args.out, **save)
    print(f"[scalars] saved {args.out}  n={len(valid)} EM1={int(np.sum(ems))} "
          f"flip1={int(np.sum(flips))}", flush=True)


if __name__ == "__main__":
    main()
