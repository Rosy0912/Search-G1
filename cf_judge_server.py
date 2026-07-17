"""Counterfactual-document judge used for Search-G1 grounding evaluation.

Architecture mirrors the retriever service: a standalone FastAPI + HF model that
the reward function calls over HTTP. Given (question, retrieved_docs, gold), it:
  1. builds a MISLEADING version of the docs (gold answer -> fixed fake entity)
  2. re-generates an answer conditioned on the fake docs
  3. returns ans_cf (the counterfactual answer)

The reward function then compares ans_cf with the rollout's real answer:
  changed = (norm(ans_real) != norm(ans_cf))   -> TRUE_TOOL  (really used the doc)
  not changed & em_cf=1                         -> PARAM_HALL (ignored the doc)

This realizes the behavcf signal ("does removing/corrupting the doc change the
answer?") as an ONLINE training reward. Uses a fixed base model as the judge
(approximation; cannot live-sync the FSDP policy weights).

Endpoint: POST /judge_batch
  body: {"items": [{"question": str, "info": str, "gold": [str,...]}, ...]}
  resp: {"results": [{"ans_cf": str, "injected": bool, "em_cf": int}, ...]}
"""
from __future__ import annotations
import os
for _k in list(os.environ.keys()):
    if "proxy" in _k.lower():
        del os.environ[_k]

import argparse, re, random, string, threading
from typing import List, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

DTYPE = torch.bfloat16

FAKE_POOL = ["Alexander Hamilton", "Tokyo", "1847", "Microsoft",
             "Charles Darwin", "Brazil", "2003", "hydrogen",
             "Queen Victoria", "Mars", "7.2 million", "Stanford"]

PROMPT_HEAD = (
    "Answer the given question. You must conduct reasoning inside <think> and </think> "
    "first every time you get new information. After reasoning, if you find you lack some "
    "knowledge, you can call a search engine by <search> query </search> and it will return "
    "the top searched results between <information> and </information>. You can search as "
    "many times as your want. If you find no further external knowledge needed, you can "
    "directly provide the answer inside <answer> and </answer>, without detailed illustrations. "
    "For example, <answer> Beijing </answer>. Question: {q}\n"
)


def _normalize(s: str) -> str:
    s = s.lower().strip()
    s = "".join(ch for ch in s if ch not in set(string.punctuation))
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def _em(pred: str, golds: List[str]) -> int:
    p = _normalize(pred)
    return int(any(_normalize(g) == p for g in golds))


def _extract_answer(text: str) -> str:
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if m:
        return m.group(1).strip()
    if "<answer>" in text:
        return text.split("</answer>")[0].split("<answer>")[-1].strip()
    return text[:80].strip()


def make_misleading(info: str, golds: List[str], rng: random.Random):
    """Replace gold answers in info with a fixed fake entity. Returns (cf_info, injected)."""
    candidates = [f for f in FAKE_POOL if all(f.lower() != g.lower() for g in golds)]
    fake = rng.choice(candidates)
    cf = info
    injected = False
    for g in golds:
        if g and g.lower() in cf.lower():
            cf = re.compile(re.escape(g), re.IGNORECASE).sub(fake, cf)
            injected = True
    return cf, injected


def make_removed(info: str, golds: List[str]):
    """Redact gold answers from info (leave-one-out style). Returns (cf_info, removed)."""
    cf = info
    removed = False
    for g in golds:
        if g and g.lower() in cf.lower():
            cf = re.compile(re.escape(g), re.IGNORECASE).sub("[redacted]", cf)
            removed = True
    return cf, removed


class JudgeItem(BaseModel):
    question: str
    info: str
    gold: List[str] = []
    answer: Optional[str] = None


class JudgeRequest(BaseModel):
    items: List[JudgeItem]


app = FastAPI()
MODEL = None
TOK = None
RNG = random.Random(42)
GENERATE_LOCK = threading.Lock()


_CHUNK_SIZE = int(os.environ.get("CF_JUDGE_CHUNK_SIZE", "4"))
_MAX_NEW = int(os.environ.get("CF_JUDGE_MAX_NEW", "48"))


def _sanitize_enc(enc, device):
    """Guard against out-of-range token ids (the #1 cause of CUDA device-side
    asserts that permanently corrupt the process). Clamp ids into [0, vocab)."""
    vocab = int(getattr(MODEL.config, "vocab_size", TOK.vocab_size))
    if "input_ids" in enc:
        enc["input_ids"] = enc["input_ids"].clamp_(0, vocab - 1)
    return enc.to(device)


@torch.no_grad()
def _batch_generate(prompts: List[str], max_new: int = _MAX_NEW) -> List[str]:
    device = next(MODEL.parameters()).device
    outs = []
    with GENERATE_LOCK:
        for start in range(0, len(prompts), _CHUNK_SIZE):
            chunk = prompts[start:start + _CHUNK_SIZE]
            try:
                enc = TOK(chunk, return_tensors="pt", add_special_tokens=False,
                          truncation=True, max_length=3500, padding=True)
                enc = _sanitize_enc(enc, device)
                gen = MODEL.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                     pad_token_id=TOK.pad_token_id, eos_token_id=TOK.eos_token_id)
                for i in range(len(chunk)):
                    cont = TOK.decode(gen[i][enc["input_ids"].shape[1]:], skip_special_tokens=False)
                    outs.append(_extract_answer(cont))
                del enc, gen
                torch.cuda.empty_cache()
            except Exception as e:
                # Never let one bad chunk poison the whole service. Return empty
                # answers for this chunk (-> em_cf=0 / AMB downgrade on the caller).
                print(f"[cf_judge] _batch_generate chunk failed, returning empty: {e}", flush=True)
                torch.cuda.empty_cache()
                outs.extend([""] * len(chunk))
    return outs


@torch.no_grad()
def _batch_generate_raw(prompts: List[str], max_new: int = 8) -> List[str]:
    """Generate raw continuation text (no <answer> extraction). Used for YES/NO judging."""
    device = next(MODEL.parameters()).device
    outs = []
    with GENERATE_LOCK:
        for start in range(0, len(prompts), _CHUNK_SIZE):
            chunk = prompts[start:start + _CHUNK_SIZE]
            try:
                enc = TOK(chunk, return_tensors="pt", add_special_tokens=False,
                          truncation=True, max_length=3500, padding=True)
                enc = _sanitize_enc(enc, device)
                gen = MODEL.generate(**enc, max_new_tokens=max_new, do_sample=False,
                                     pad_token_id=TOK.pad_token_id, eos_token_id=TOK.eos_token_id)
                outs.extend([TOK.decode(gen[i][enc["input_ids"].shape[1]:], skip_special_tokens=True)
                             for i in range(len(chunk))])
                del enc, gen
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"[cf_judge] _batch_generate_raw chunk failed, returning empty: {e}", flush=True)
                torch.cuda.empty_cache()
                outs.extend([""] * len(chunk))
    return outs


@app.post("/judge_batch")
def judge_batch(req: JudgeRequest):
    prompts, metas = [], []
    for it in req.items:
        cf_info, injected = make_misleading(it.info, it.gold, RNG)
        # build a single-turn context: question + (fake) retrieved info -> answer
        ctx = (PROMPT_HEAD.format(q=it.question)
               + "<think>\nI need to search for information about this question.\n</think>\n"
               + f"<search>{it.question}</search>\n"
               + f"<information>{cf_info}</information>\n"
               + "<think>\nBased on the search results, ")
        prompts.append(ctx)
        metas.append((injected, it.gold))
    ans_list = _batch_generate(prompts) if prompts else []
    results = []
    for ans, (injected, gold) in zip(ans_list, metas):
        results.append({"ans_cf": ans, "injected": bool(injected), "em_cf": _em(ans, gold)})
    return {"results": results}


@app.post("/judge_faithfulness")
def judge_faithfulness(req: JudgeRequest):
    """T3 (NLI/entailment): does the evidence SUPPORT the proposed answer? -> supported 0/1.
    Positive grounding signal; does not penalize parametric knowledge nor reward gullibility."""
    prompts = []
    for it in req.items:
        prompts.append(
            "You are a strict fact-checker. Decide whether the EVIDENCE explicitly supports "
            "the PROPOSED ANSWER to the question. Reply with only YES or NO.\n"
            f"Question: {it.question}\n"
            f"Evidence: {it.info}\n"
            f"Proposed answer: {it.answer}\n"
            "Does the evidence support the proposed answer? Answer YES or NO:"
        )
    outs = _batch_generate_raw(prompts, max_new=4) if prompts else []
    return {"results": [{"supported": 1 if "yes" in o.lower() else 0} for o in outs]}


@app.post("/judge_removal")
def judge_removal(req: JudgeRequest):
    """T4 (leave-one-out): redact gold from evidence and re-answer. The reward function
    compares ans_cf with the real answer: changed -> evidence was causally used."""
    prompts, metas = [], []
    for it in req.items:
        cf_info, removed = make_removed(it.info, it.gold)
        ctx = (PROMPT_HEAD.format(q=it.question)
               + "<think>\nI need to search for information about this question.\n</think>\n"
               + f"<search>{it.question}</search>\n"
               + f"<information>{cf_info}</information>\n"
               + "<think>\nBased on the search results, ")
        prompts.append(ctx)
        metas.append((removed, it.gold))
    ans_list = _batch_generate(prompts) if prompts else []
    results = []
    for ans, (removed, gold) in zip(ans_list, metas):
        results.append({"ans_cf": ans, "removed": bool(removed), "em_cf": _em(ans, gold)})
    return {"results": results}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    ap.add_argument("--port", type=int, default=8001)
    args = ap.parse_args()

    print(f"[cf_judge] loading {args.model}", flush=True)
    TOK = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if TOK.pad_token is None:
        TOK.pad_token = TOK.eos_token
    MODEL = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=DTYPE, device_map="auto",
        low_cpu_mem_usage=True, attn_implementation="sdpa")
    MODEL.eval()
    print("[cf_judge] model loaded, serving on 0.0.0.0:%d" % args.port, flush=True)
    uvicorn.run(app, host="0.0.0.0", port=args.port, workers=1)
