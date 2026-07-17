#!/usr/bin/env python3
"""Counterfactual-label robustness from existing probe dumps."""
from __future__ import annotations
import argparse, glob, json
from pathlib import Path
import numpy as np
from sklearn.metrics import cohen_kappa_score, roc_auc_score


def collect(paths):
    vals = {"ds": [], "em": [], "flip": [], "flip2": [], "cb": []}
    for p in paths:
        d = np.load(p, allow_pickle=True)
        if not all(k in d.files for k in ["y_em", "y_flip", "y_flip2", "q_cb", "ds"]):
            continue
        vals["em"].append(d["y_em"].astype(int))
        vals["flip"].append(d["y_flip"].astype(int))
        vals["flip2"].append(d["y_flip2"].astype(int))
        vals["cb"].append(d["q_cb"].astype(int))
        vals["ds"].append(d["ds"].astype(str))
    return {k: np.concatenate(v) for k, v in vals.items()}


def summarize(mask, em, flip, flip2, cb):
    if int(mask.sum()) == 0:
        return {}
    agree = float(np.mean(flip[mask] == flip2[mask]))
    kappa = float(cohen_kappa_score(flip[mask], flip2[mask])) if len(set(flip[mask])) > 1 and len(set(flip2[mask])) > 1 else float("nan")
    em1 = mask & (em == 1)
    em0 = mask & (em == 0)
    return {
        "n": int(mask.sum()),
        "em_rate": float(np.mean(em[mask])),
        "closed_book_rate": float(np.mean(cb[mask])),
        "flip_delete_rate": float(np.mean(flip[mask])),
        "flip_delete_strip_think_rate": float(np.mean(flip2[mask])),
        "delete_vs_strict_agreement": agree,
        "delete_vs_strict_kappa": kappa,
        "flip_rate_em1": float(np.mean(flip[em1])) if em1.any() else float("nan"),
        "flip_rate_em0": float(np.mean(flip[em0])) if em0.any() else float("nan"),
        "strict_flip_rate_em1": float(np.mean(flip2[em1])) if em1.any() else float("nan"),
        "strict_flip_rate_em0": float(np.mean(flip2[em0])) if em0.any() else float("nan"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", nargs="+", default=["dumps/base250b"])
    ap.add_argument("--out", default="analysis/label_robustness_summary.json")
    args = ap.parse_args()
    files = []
    for d in args.dumps:
        files.extend(sorted(glob.glob(d.rstrip("/") + "/*.npz")))
    c = collect(files)
    ds, em, flip, flip2, cb = c["ds"], c["em"], c["flip"], c["flip2"], c["cb"]
    out = {"all": summarize(np.ones(len(em), dtype=bool), em, flip, flip2, cb)}
    for name in sorted(set(ds.tolist())):
        out[name] = summarize(ds == name, em, flip, flip2, cb)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(out, indent=2, ensure_ascii=False))

if __name__ == "__main__":
    main()
