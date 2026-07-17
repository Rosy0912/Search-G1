"""Search-G1 representation-readout server with hot reload.

Serves /judge_probe -> {flip, boundary} for each rollout (prompt, response).
A background thread watches a manifest JSON; when the co-evolution controller writes
a new round (new policy checkpoint + freshly refit probes), the server reloads BOTH
the frozen policy-encoder snapshot and the two readout heads, then atomically swaps
them in.  Reward calls return detached scalars, so no gradient enters the evaluator.

manifest JSON:
  {"round": int, "encoder": <hf_ckpt_dir>, "commit": <pkl>, "boundary": <pkl>}
"""
from __future__ import annotations
import os
for _k in list(os.environ.keys()):
    if "proxy" in _k.lower():
        del os.environ[_k]

import argparse, json, re, threading, time
from typing import List
import numpy as np
import torch
import joblib
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModelForCausalLM, AutoTokenizer

DTYPE = torch.bfloat16
app = FastAPI()
LOCK = threading.Lock()
STATE = {"round": -1, "encoder": None, "model": None, "tok": None,
         "commit": None, "boundary": None}
MANIFEST = None


def _load(encoder, commit_pkl, boundary_pkl):
    tok = AutoTokenizer.from_pretrained(encoder, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        encoder, torch_dtype=DTYPE, device_map="auto",
        low_cpu_mem_usage=True, attn_implementation="sdpa").eval()
    commit = joblib.load(commit_pkl)
    boundary = joblib.load(boundary_pkl)
    return tok, model, commit, boundary


def _apply(manifest_path, force=False):
    if not os.path.exists(manifest_path):
        return
    with open(manifest_path) as f:
        m = json.load(f)
    if (not force) and m.get("round", -1) == STATE["round"]:
        return
    enc = m["encoder"]
    reload_model = force or (enc != STATE["encoder"])
    print(f"[probe] reloading round={m.get('round')} encoder={enc} "
          f"(model_reload={reload_model})", flush=True)
    tok = STATE["tok"]; model = STATE["model"]
    if reload_model:
        tok, model, commit, boundary = _load(enc, m["commit"], m["boundary"])
    else:
        commit = joblib.load(m["commit"]); boundary = joblib.load(m["boundary"])
    old = STATE["model"] if reload_model else None
    with LOCK:
        STATE.update({"round": m.get("round", STATE["round"] + 1), "encoder": enc,
                      "tok": tok, "model": model, "commit": commit, "boundary": boundary})
    if old is not None and old is not model:
        del old
        torch.cuda.empty_cache()
    print(f"[probe] now serving round={STATE['round']} "
          f"commit(L{commit['layer']}/{commit.get('model')}) boundary(L{boundary['layer']})",
          flush=True)


def _watcher():
    while True:
        try:
            _apply(MANIFEST)
        except Exception as e:
            print(f"[probe] watcher error: {e}", flush=True)
        time.sleep(15)


class Item(BaseModel):
    prompt: str
    response: str = ""


class ProbeRequest(BaseModel):
    items: List[Item]


@torch.no_grad()
def _last_hidden(texts, layer, model, tok, micro=16, max_len=4096):
    base = model.model
    cap = {}
    h = base.layers[layer - 1].register_forward_hook(
        lambda mod, i, o: cap.__setitem__("h", o[0] if isinstance(o, tuple) else o))
    out = []
    try:
        for s in range(0, len(texts), micro):
            enc = tok(texts[s:s + micro], return_tensors="pt", add_special_tokens=False,
                      padding=True, truncation=True, max_length=max_len).to(base.device)
            base(input_ids=enc["input_ids"], attention_mask=enc["attention_mask"],
                 use_cache=False)
            out.append(cap["h"][:, -1, :].float().cpu().numpy())
    finally:
        h.remove()
    return np.concatenate(out) if out else np.zeros((0, 1))


def _commit_ctx(prompt, response, position):
    """Build the prefix corresponding to the representation used for calibration."""
    if position == "Xc":
        start = response.find("<answer>")
        if start >= 0:
            return prompt + response[:start + len("<answer>")]
        return prompt + response
    if position == "Xa":
        end = response.rfind("</answer>")
        return prompt + (response[:end] if end >= 0 else response)
    raise ValueError(f"unsupported commit position: {position!r}")


@app.post("/judge_probe")
def judge_probe(req: ProbeRequest):
    if not req.items:
        return {"results": []}
    with LOCK:
        model, tok = STATE["model"], STATE["tok"]
        commit, boundary = STATE["commit"], STATE["boundary"]
    position = commit.get("position", "Xa")
    cc = [_commit_ctx(it.prompt, it.response, position) for it in req.items]
    bc = [it.prompt for it in req.items]
    Hc = _last_hidden(cc, commit["layer"], model, tok)
    Hb = _last_hidden(bc, boundary["layer"], model, tok)
    flip = commit["clf"].predict_proba(Hc)[:, 1]
    bnd = boundary["clf"].predict_proba(Hb)[:, 1]
    return {"results": [{"flip": float(flip[i]), "boundary": float(bnd[i])}
                        for i in range(len(req.items))]}


@app.get("/health")
def health():
    return {"ok": STATE["model"] is not None, "round": STATE["round"],
            "encoder": STATE["encoder"]}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--port", type=int, default=8002)
    args = ap.parse_args()
    MANIFEST = args.manifest
    print(f"[probe] manifest={MANIFEST} waiting for first round...", flush=True)
    for _ in range(600):
        if os.path.exists(MANIFEST):
            break
        time.sleep(2)
    _apply(MANIFEST, force=True)
    threading.Thread(target=_watcher, daemon=True).start()
    print(f"[probe] serving on 0.0.0.0:{args.port}", flush=True)
    uvicorn.run(app, host="0.0.0.0", port=args.port)
