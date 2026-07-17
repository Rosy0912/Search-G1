#!/usr/bin/env python3
"""Grouped reward-variance diagnostics for the Search-G1 credit-assignment claim.

Input is a scalar dump from ts_probe_scalars.py. The script computes rewards on
the same rollout groups and reports whether each reward creates nonzero
within-group advantages when EM outcomes are identical.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def minmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float64)
    lo, hi = np.nanmin(x), np.nanmax(x)
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros_like(x, dtype=np.float64)
    return (x - lo) / (hi - lo)


def group_oof_proba(X: np.ndarray, y: np.ndarray, groups: np.ndarray, n_splits: int) -> np.ndarray:
    uniq = np.unique(groups)
    n_splits = max(2, min(n_splits, len(uniq)))
    out = np.full(len(y), np.nan, dtype=np.float64)
    splitter = GroupKFold(n_splits=n_splits)
    for tr, te in splitter.split(X, y, groups):
        if len(np.unique(y[tr])) < 2:
            out[te] = float(np.mean(y[tr])) if len(tr) else 0.5
            continue
        clf = make_pipeline(
            StandardScaler(),
            LogisticRegression(class_weight="balanced", C=1.0, max_iter=4000),
        )
        clf.fit(X[tr], y[tr])
        out[te] = clf.predict_proba(X[te])[:, 1]
    if np.isnan(out).any():
        fill = float(np.nanmean(out)) if np.isfinite(np.nanmean(out)) else float(np.mean(y))
        out[np.isnan(out)] = fill
    return out


def summarize(name: str, reward: np.ndarray, em: np.ndarray, groups: np.ndarray) -> dict:
    rows = []
    for g in np.unique(groups):
        m = groups == g
        if m.sum() < 2:
            continue
        r, e = reward[m], em[m]
        all_correct = bool(np.all(e == 1))
        all_wrong = bool(np.all(e == 0))
        same_em = all_correct or all_wrong
        spread = float(np.max(r) - np.min(r))
        std = float(np.std(r))
        rows.append((same_em, all_correct, all_wrong, spread, std, spread > 1e-8))
    if not rows:
        return {"reward": name, "groups": 0}
    a = np.array(rows, dtype=object)
    same = a[:, 0].astype(bool)
    ac = a[:, 1].astype(bool)
    aw = a[:, 2].astype(bool)
    spreads = a[:, 3].astype(float)
    stds = a[:, 4].astype(float)
    nonzero = a[:, 5].astype(bool)
    return {
        "reward": name,
        "groups": int(len(rows)),
        "all_correct_pct": float(100 * ac.mean()),
        "all_wrong_pct": float(100 * aw.mean()),
        "same_em_pct": float(100 * same.mean()),
        "reward_std": float(stds.mean()),
        "same_em_spread": float(spreads[same].mean()) if same.any() else float("nan"),
        "nonzero_adv_pct": float(100 * nonzero[same].mean()) if same.any() else float("nan"),
        "all_correct_nonzero_pct": float(100 * nonzero[ac].mean()) if ac.any() else float("nan"),
        "all_wrong_nonzero_pct": float(100 * nonzero[aw].mean()) if aw.any() else float("nan"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", default="")
    ap.add_argument("--layer", type=int, default=27)
    ap.add_argument("--splits", type=int, default=5)
    ap.add_argument("--intrinsic_coef", type=float, default=0.3)
    ap.add_argument("--cost_coef", type=float, default=0.05)
    args = ap.parse_args()

    z = np.load(args.dump, allow_pickle=True)
    em = z["y_em"].astype(np.float64)
    flip = z["y_flip"].astype(np.int8)
    if "gid" in z:
        groups = z["gid"].astype(np.int64)
    else:
        # Old scalar dumps were produced as G rollouts per question but did not
        # save gid. This fallback is for quick inspection only; new dumps save gid.
        groups = np.arange(len(em), dtype=np.int64) // 5
    n_search = z["n_search"].astype(np.float64) if "n_search" in z else np.ones_like(em)
    norm_cost = np.clip(np.maximum(0.0, n_search - 1.0), 0.0, 1.0)

    rewards = {
        "EM": em,
        "EM+Cost": em - args.cost_coef * norm_cost,
        "EM+Entropy": em + args.intrinsic_coef * minmax(z["entropy"]),
        "EM+AnswerLP": em + args.intrinsic_coef * minmax(z["ans_lp"]),
    }
    if "logit" in z:
        rewards["EM+LogitProbe"] = em + args.intrinsic_coef * group_oof_proba(
            z["logit"].astype(np.float32), flip, groups, args.splits)
    hk = f"Xc_{args.layer}"
    if hk in z:
        rewards["Search-G1-HiddenProbe"] = em + args.intrinsic_coef * group_oof_proba(
            z[hk].astype(np.float32), flip, groups, args.splits) - args.cost_coef * norm_cost
    rewards["Search-G1-OracleFlip"] = em + args.intrinsic_coef * flip - args.cost_coef * norm_cost

    rows = [summarize(k, v, em, groups) for k, v in rewards.items()]
    meta = {
        "dump": args.dump,
        "n_rollouts": int(len(em)),
        "n_groups": int(len(np.unique(groups))),
        "em": float(em.mean()),
        "flip": float(flip.mean()),
        "has_gid": bool("gid" in z),
        "has_n_search": bool("n_search" in z),
    }
    result = {"meta": meta, "rows": rows}

    print(json.dumps(result, indent=2))
    print("\n| Reward | Same-EM % | Reward Std | Same-EM Spread | Nonzero-Adv % | All-Correct NZ % | All-Wrong NZ % |")
    print("|---|---:|---:|---:|---:|---:|---:|")
    for r in rows:
        print(
            f"| {r['reward']} | {r.get('same_em_pct', float('nan')):.1f} | "
            f"{r.get('reward_std', float('nan')):.4f} | {r.get('same_em_spread', float('nan')):.4f} | "
            f"{r.get('nonzero_adv_pct', float('nan')):.1f} | "
            f"{r.get('all_correct_nonzero_pct', float('nan')):.1f} | "
            f"{r.get('all_wrong_nonzero_pct', float('nan')):.1f} |"
        )
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(result, indent=2) + "\n")
        print(f"\n[reward-variance] wrote {out}")


if __name__ == "__main__":
    main()
