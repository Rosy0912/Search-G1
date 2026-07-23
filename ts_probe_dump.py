"""Generate calibration data for Search-G1 representation readouts.

Generation (rollouts / closed-book / flippability re-decode) runs on a vLLM engine
with batched multi-turn search; hidden-state extraction at the commit token runs on a
separate HF forward pass (single forward, no generation). Both share one GPU.

For each rollout we dump:
  - answer-delimiter and final answer-token hidden states at selected layers
  - prompt-last-token hidden states (per-question boundary probe input)
  - EM correctness (free label)
  - evidence-deletion sensitivity label: re-decode with <information> emptied;
        an answer change gives the operational positive label (1).
  - per-question closed-book label    : greedy decode with NO search; EM = inside boundary.

Usage:
    python ts_probe_dump.py --model <ckpt> --data data/nq_search/train.parquet \
        --tag nq --cap 100 --G 5 --layers 21,24,27 --out dumps/base250
"""
from __future__ import annotations
import argparse, json, os, re, time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import requests
from transformers import AutoModelForCausalLM, AutoTokenizer
from vllm import LLM, SamplingParams

DTYPE = torch.bfloat16
RETRIEVER_URL = os.environ.get("RETRIEVER_URL", "http://127.0.0.1:8000/retrieve")
ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
SEARCH_RE = re.compile(r"<search>(.*?)</search>", re.DOTALL)
INFO_RE = re.compile(r"<information>.*?</information>", re.DOTALL)
THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


# ----------------------------- helpers -----------------------------
# Local service calls should not be routed through an outbound HTTP proxy.
_SESSION = requests.Session()
_SESSION.trust_env = False


def retrieve_batch(queries, topk=3):
    """Batch retrieval. Returns list[str] (joined docs) aligned with queries."""
    if not queries:
        return []
    chunk_size = max(1, int(os.environ.get("RETRIEVER_BATCH_SIZE", "32")))
    if len(queries) > chunk_size:
        out = []
        for s in range(0, len(queries), chunk_size):
            out.extend(retrieve_batch(queries[s:s + chunk_size], topk=topk))
        return out
    try:
        resp = _SESSION.post(RETRIEVER_URL, json={"queries": queries, "topk": topk},
                             timeout=120)
        if resp.status_code == 200:
            data = resp.json()
            res = data.get("result", data.get("results")) if isinstance(data, dict) else data
            out = []
            for docs in res:
                joined = []
                for d in docs[:topk]:
                    if isinstance(d, str):
                        joined.append(d)
                    elif isinstance(d, dict):
                        joined.append(d.get("contents", d.get("text", "")))
                out.append("\n".join(joined))
            return out
    except Exception as e:
        print(f"  [retrieve warn] {e}", flush=True)
    return ["" for _ in queries]


def normalize(s: str) -> str:
    s = s.lower().strip()
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def em_score(pred, golds) -> int:
    p = normalize(pred)
    if not p:
        return 0
    return int(any(normalize(g) == p for g in golds))


def contains_gold(pred, golds) -> int:
    """Relaxed match for closed-book boundary label: gold appears in the answer text."""
    p = normalize(pred)
    if not p:
        return 0
    return int(any(normalize(g) and normalize(g) in p for g in golds))


QUESTION_RE = re.compile(r"Question:\s*(.*?)\s*$", re.DOTALL)


def extract_answer(text: str) -> str:
    m = ANSWER_RE.search(text)
    return m.group(1).strip() if m else ""


def gen_batch(llm, prompts, max_tokens, sample, stop):
    sp = SamplingParams(
        temperature=1.0 if sample else 0.0, top_p=1.0 if sample else 1.0,
        max_tokens=max_tokens, stop=stop, include_stop_str_in_output=True)
    outs = llm.generate(prompts, sp, use_tqdm=False)
    return [o.outputs[0].text for o in outs]


def rollouts_batched(llm, prompts0, sample, max_turns=4, max_new=256):
    """Multi-turn search generation over a flat list of prompts. Returns generated texts."""
    N = len(prompts0)
    gen = [""] * N
    done = [False] * N
    for turn in range(max_turns + 1):
        idx = [i for i in range(N) if not done[i]]
        if not idx:
            break
        inputs = [prompts0[i] + gen[i] for i in idx]
        texts = gen_batch(llm, inputs, max_new, sample, ["</search>", "</answer>"])
        pending = []  # (seq_index, query)
        for k, i in enumerate(idx):
            t = texts[k]
            if "</search>" in t and turn < max_turns:
                before = t[:t.index("</search>") + len("</search>")]
                gen[i] += before
                sm = SEARCH_RE.search(before)
                pending.append((i, sm.group(1).strip() if sm else ""))
            else:
                gen[i] += t
                done[i] = True
        if pending:
            docs = retrieve_batch([q for _, q in pending])
            for (i, _), d in zip(pending, docs):
                gen[i] += f"\n<information>{d}</information>\n"
    return gen


@torch.no_grad()
def hidden_batch(model, tok, texts, layers, device, micro=8, max_len=4096):
    """Return {layer: np.array[n,d]} hidden of the last token for each text.

    Uses forward hooks on the specific decoder layers so we never materialize all
    37 hidden-state tensors (which OOMs). output_hidden_states index L corresponds
    to the output of decoder layer (L-1)."""
    base = model.model  # Qwen2Model
    captured = {}
    hooks = []

    def mk(L):
        def hook(mod, inp, out):
            captured[L] = (out[0] if isinstance(out, tuple) else out)
        return hook

    for L in layers:
        hooks.append(base.layers[L - 1].register_forward_hook(mk(L)))
    res = {L: [] for L in layers}
    try:
        for s in range(0, len(texts), micro):
            chunk = texts[s:s + micro]
            enc = tok(chunk, return_tensors="pt", add_special_tokens=False, padding=True,
                      truncation=True, max_length=max_len).to(device)
            # call the BASE transformer (no lm_head) -> avoids materializing the huge
            # [batch, seq, vocab] logits tensor (~10GB) that caused OOM.
            base(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                 use_cache=False)
            for L in layers:
                res[L].append(captured[L][:, -1, :].float().cpu().numpy())
            captured.clear()
    finally:
        for h in hooks:
            h.remove()
    return {L: np.concatenate(res[L]) if res[L] else np.zeros((0, 1)) for L in layers}


def parse_row(row):
    pd_ = row["prompt"]
    if isinstance(pd_, np.ndarray):
        pd_ = pd_.tolist()
    if isinstance(pd_, str):
        pd_ = json.loads(pd_)
    ptext = pd_[0]["content"] if isinstance(pd_, (list, tuple)) else (
        pd_.get("content", str(pd_)) if isinstance(pd_, dict) else str(pd_))
    rm = row["reward_model"]
    if isinstance(rm, np.ndarray):
        rm = rm.tolist()
    if isinstance(rm, str):
        rm = json.loads(rm)
    golds = rm["ground_truth"]["target"]
    if isinstance(golds, np.ndarray):
        golds = golds.tolist()
    if isinstance(golds, str):
        golds = [golds]
    return ptext, golds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--cap", type=int, default=200)
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--G", type=int, default=5)
    ap.add_argument("--layers", default="21,24,27")
    ap.add_argument("--max_turns", type=int, default=4)
    ap.add_argument("--gpu_mem", type=float, default=0.45)
    ap.add_argument("--out", default="dumps/run")
    args = ap.parse_args()

    layers = [int(x) for x in args.layers.split(",")]
    Path(args.out).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")

    df = pd.read_parquet(args.data)
    if args.offset:
        df = df.iloc[args.offset:]
    if args.cap > 0:
        df = df.head(args.cap)
    rows = [parse_row(r) for _, r in df.iterrows()]
    prompts = [p for p, _ in rows]
    golds_all = [g for _, g in rows]
    # clean closed-book question prompts (no search instructions) for the boundary label
    questions = []
    for p in prompts:
        m = QUESTION_RE.search(p)
        questions.append(m.group(1).strip() if m else p)
    nq = len(rows)
    print(f"[dump] model={args.model} tag={args.tag} n_q={nq} G={args.G} layers={layers}",
          flush=True)

    # vLLM engine for generation
    llm = LLM(model=args.model, dtype="bfloat16", gpu_memory_utilization=args.gpu_mem,
              max_model_len=4096, enforce_eager=True, disable_log_stats=True)
    # HF model for hidden extraction
    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    hf = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=DTYPE, device_map="cuda",
        low_cpu_mem_usage=True, attn_implementation="sdpa").eval()
    print("[dump] vLLM + HF loaded", flush=True)
    t0 = time.time()

    # ---- per-question closed-book (boundary): answer directly, NO retrieval ----
    cb_prompt = [f"Answer the question directly with only the answer, no explanation.\n"
                 f"Question: {q}\nAnswer:" for q in questions]
    cb_ans = gen_batch(llm, cb_prompt, 24, False, ["\n"])
    q_cb = [contains_gold(cb_ans[i], golds_all[i]) for i in range(nq)]
    print(f"[dump] closed-book done ({time.time()-t0:.0f}s) cb_rate={np.mean(q_cb):.3f}", flush=True)

    # ---- G rollouts per question (flat) ----
    flat_prompts = [prompts[i] for i in range(nq) for _ in range(args.G)]
    flat_qi = [i for i in range(nq) for _ in range(args.G)]
    gens = rollouts_batched(llm, flat_prompts, sample=(args.G > 1),
                            max_turns=args.max_turns)
    print(f"[dump] rollouts done ({time.time()-t0:.0f}s)", flush=True)

    # build commit/answer-last contexts + 2 flippability contexts (valid rollouts only)
    valid, commit_ctx, ans_ctx, flip_ctx, flip2_ctx, answers, ems = [], [], [], [], [], [], []
    for j, g in enumerate(gens):
        mi = g.find("<answer>")
        if mi < 0:
            continue
        qi = flat_qi[j]
        ans = extract_answer(g)
        ae = g.find("</answer>", mi)
        if ae < 0:
            ae = len(g)
        commit_ctx.append(prompts[qi] + g[:mi + len("<answer>")])   # at "<answer>" tag
        ans_ctx.append(prompts[qi] + g[:ae])                        # at answer-span last token
        src = g[:mi + len("<answer>")]
        info_empty = INFO_RE.sub("<information></information>", src)
        flip_ctx.append(prompts[qi] + info_empty)                   # empty info (may leak via think)
        flip2_ctx.append(prompts[qi] + THINK_RE.sub("", info_empty))  # +strip think (low-leak)
        answers.append(ans)
        ems.append(em_score(ans, golds_all[qi]))
        valid.append(j)
    print(f"[dump] valid rollouts={len(valid)}/{len(gens)}", flush=True)

    # ---- flippability re-decode (batched, greedy), two variants ----
    def flip_label(ctx_list):
        fa = gen_batch(llm, ctx_list, 32, False, ["</answer>"])
        return fa, [int(normalize(extract_answer(fa[k] + "</answer>") or
                                  fa[k].split("</answer>")[0]) != normalize(answers[k]))
                    for k in range(len(valid))]
    flip_ans, flips = flip_label(flip_ctx)
    _, flips2 = flip_label(flip2_ctx)
    print(f"[dump] flippability done ({time.time()-t0:.0f}s) flip={np.mean(flips):.3f} "
          f"flip_strict={np.mean(flips2):.3f}", flush=True)

    # ---- hidden states (HF single forward, batched) ----
    Hc = hidden_batch(hf, tok, commit_ctx, layers, device)          # at "<answer>" tag token
    Ha = hidden_batch(hf, tok, ans_ctx, layers, device)             # at answer-span last token
    Hp_q = hidden_batch(hf, tok, prompts, layers, device)           # per-question prompt
    print(f"[dump] hidden extracted ({time.time()-t0:.0f}s)", flush=True)

    # ---- assemble ----
    y_em, y_flip, y_flip2, q_cb_r, gids, dss, recs = [], [], [], [], [], [], []
    Xc = {L: [] for L in layers}
    Xa = {L: [] for L in layers}
    Xp = {L: [] for L in layers}
    for k, j in enumerate(valid):
        qi = flat_qi[j]
        for L in layers:
            Xc[L].append(Hc[L][k]); Xa[L].append(Ha[L][k]); Xp[L].append(Hp_q[L][qi])
        y_em.append(ems[k]); y_flip.append(flips[k]); y_flip2.append(flips2[k])
        q_cb_r.append(q_cb[qi]); gids.append(qi); dss.append(args.tag)
        recs.append({"qi": qi, "em": ems[k], "flip": flips[k], "flip2": flips2[k],
                     "cb": q_cb[qi], "answer": answers[k], "flip_answer": flip_ans[k][:60]})

    out_npz = os.path.join(args.out, f"{args.tag}_{args.offset}.npz")
    save = {"y_em": np.array(y_em, np.int8), "y_flip": np.array(y_flip, np.int8),
            "y_flip2": np.array(y_flip2, np.int8),
            "q_cb": np.array(q_cb_r, np.int8), "gid": np.array(gids, np.int32),
            "ds": np.array(dss)}
    for L in layers:
        save[f"Xc_{L}"] = np.stack(Xc[L]).astype(np.float16) if Xc[L] else np.zeros((0, 1))
        save[f"Xa_{L}"] = np.stack(Xa[L]).astype(np.float16) if Xa[L] else np.zeros((0, 1))
        save[f"Xp_{L}"] = np.stack(Xp[L]).astype(np.float16) if Xp[L] else np.zeros((0, 1))
    np.savez_compressed(out_npz, **save)
    with open(os.path.join(args.out, f"{args.tag}_{args.offset}.jsonl"), "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")
    print(f"[dump] SAVED {out_npz} n={len(y_em)} EM={np.mean(y_em):.3f} "
          f"flip={np.mean(y_flip):.3f} cb={np.mean(q_cb_r):.3f} "
          f"total={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
