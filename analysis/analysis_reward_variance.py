#!/usr/bin/env python3
"""Reward-variance diagnostics for same-EM GRPO groups."""
from __future__ import annotations
import argparse, json, os
from pathlib import Path
import numpy as np


def group_stats(path: str, max_turns: int = 4) -> dict:
    d = np.load(path, allow_pickle=True)
    y_em = d["y_em"].astype(float)
    y_flip = d["y_flip"].astype(float)
    gid = d["gid"].astype(int)
    n_search = d["n_search"].astype(float) if "n_search" in d.files else np.zeros_like(y_em)
    cost = np.minimum(1.0, np.maximum(0.0, n_search - 1.0) / max(1, max_turns - 1))
    # Coefficient-free signal diagnostics plus one illustrative shaped reward.
    rep_signal = y_flip - 0.1 * cost
    shaped = y_em + 0.3 * y_flip - 0.02 * cost
    groups = sorted(set(gid.tolist()))
    rows = []
    for g in groups:
        m = gid == g
        if m.sum() < 2:
            continue
        em_vals = y_em[m]
        same_em = float(em_vals.max() == em_vals.min())
        rows.append({
            "gid": int(g),
            "n": int(m.sum()),
            "same_em": bool(same_em),
            "all_em1": bool(same_em and em_vals[0] == 1),
            "all_em0": bool(same_em and em_vals[0] == 0),
            "std_em": float(np.std(y_em[m])),
            "std_flip": float(np.std(y_flip[m])),
            "std_cost": float(np.std(cost[m])),
            "std_rep_signal": float(np.std(rep_signal[m])),
            "std_shaped": float(np.std(shaped[m])),
        })
    same = [r for r in rows if r["same_em"]]
    def mean(key, xs=rows):
        return float(np.mean([r[key] for r in xs])) if xs else float("nan")
    out = {
        "file": path,
        "n_rollouts": int(len(y_em)),
        "n_groups": int(len(rows)),
        "zero_em_variance_group_frac": float(len(same) / max(1, len(rows))),
        "all_em1_group_frac": float(sum(r["all_em1"] for r in rows) / max(1, len(rows))),
        "all_em0_group_frac": float(sum(r["all_em0"] for r in rows) / max(1, len(rows))),
        "same_em_groups": int(len(same)),
        "same_em_flip_nonzero_frac": float(sum(r["std_flip"] > 1e-9 for r in same) / max(1, len(same))),
        "same_em_cost_nonzero_frac": float(sum(r["std_cost"] > 1e-9 for r in same) / max(1, len(same))),
        "same_em_rep_nonzero_frac": float(sum(r["std_rep_signal"] > 1e-9 for r in same) / max(1, len(same))),
        "mean_group_std_em": mean("std_em"),
        "mean_group_std_rep_signal": mean("std_rep_signal"),
        "mean_group_std_shaped": mean("std_shaped"),
        "same_em_mean_std_em": mean("std_em", same),
        "same_em_mean_std_rep_signal": mean("std_rep_signal", same),
        "same_em_mean_std_shaped": mean("std_shaped", same),
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--out", default="analysis/reward_variance_summary.json")
    args = ap.parse_args()
    results = {Path(p).stem: group_stats(p) for p in args.inputs}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
