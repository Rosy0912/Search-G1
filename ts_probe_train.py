"""Fit Search-G1 evidence-sensitivity and closed-book-sufficiency readouts.

The paper configuration uses the final answer-token state (``Xa``) and the
evidence-deletion sensitivity label (``flip``). These semantics are the command-line
defaults and remain fixed during co-evolution. ``auto`` modes are retained for explicit diagnostic
sweeps, but are not the default training protocol.  Model and layer selection use
question-grouped held-out splits.

Usage:
    python ts_probe_train.py --dumps dumps/base250 --layers 21,24,27 \\
        --commit-position Xa --commit-label flip --save probes/base250
"""
from __future__ import annotations
import argparse, glob, json, os
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, accuracy_score

RULER = ["popqa", "musique", "bamboogle", "2wikimultihopqa", "hotpotqa", "nq", "triviaqa"]


def load_dumps(dirs, layers):
    pos = {p: {L: [] for L in layers} for p in ("Xc", "Xa", "Xp")}
    em, flip, flip2, cb, gid, ds = [], [], [], [], [], []
    gid_off = 0
    files = []
    for d in dirs:
        files += sorted(glob.glob(os.path.join(d, "*.npz")))
    for fp in files:
        z = np.load(fp, allow_pickle=True)
        n = len(z["y_em"])
        if n == 0:
            continue
        if not all(f"Xc_{L}" in z and f"Xa_{L}" in z and f"Xp_{L}" in z for L in layers):
            continue
        for p in pos:
            for L in layers:
                pos[p][L].append(z[f"{p}_{L}"].astype(np.float32))
        em.append(z["y_em"]); flip.append(z["y_flip"])
        flip2.append(z["y_flip2"] if "y_flip2" in z else z["y_flip"])
        cb.append(z["q_cb"]); gid.append(z["gid"].astype(np.int64) + gid_off); ds.append(z["ds"])
        gid_off = int(gid[-1].max()) + 1
    for p in pos:
        pos[p] = {L: np.concatenate(pos[p][L]) for L in layers}
    return (pos, np.concatenate(em), np.concatenate(flip), np.concatenate(flip2),
            np.concatenate(cb), np.concatenate(gid), np.concatenate(ds))


def make_clf(model):
    arch = {"mlp64": (64,), "mlp128": (128,), "mlp256": (256, 128)}
    if model in arch:
        return make_pipeline(StandardScaler(),
                             MLPClassifier(hidden_layer_sizes=arch[model], max_iter=400,
                                           early_stopping=True, alpha=1e-4, random_state=0))
    return LogisticRegression(class_weight="balanced", C=1.0, max_iter=4000)


def cv_auc(X, y, groups, model="lr", n=2, max_n=15000):
    if len(np.unique(y)) < 2:
        return float("nan"), None
    # subsample (by row) to bound MLP cost during model selection
    if len(y) > max_n:
        rng = np.random.RandomState(0)
        idx = rng.choice(len(y), max_n, replace=False)
        X, y, groups = X[idx], y[idx], groups[idx]
    aucs = []
    gss = GroupShuffleSplit(n_splits=n, test_size=0.2, random_state=0)
    for tr, te in gss.split(X, y, groups):
        if len(np.unique(y[tr])) < 2 or len(np.unique(y[te])) < 2:
            continue
        clf = make_clf(model).fit(X[tr], y[tr])
        p = clf.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))
    return (float(np.mean(aucs)) if aucs else float("nan")), aucs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dumps", nargs="+", default=["dumps/base250"])
    ap.add_argument("--layers", default="21,24,27")
    ap.add_argument("--save", default="")
    ap.add_argument("--commit-position", choices=("Xa", "Xc", "auto"), default="Xa",
                    help="Answer representation: final answer token (Xa, paper default), "
                         "answer delimiter (Xc), or an explicit diagnostic sweep.")
    ap.add_argument("--commit-label", choices=("flip", "flip2", "auto"), default="flip",
                    help="Counterfactual target. flip removes evidence while retaining "
                         "reasoning; flip2 also removes reasoning.")
    ap.add_argument("--commit-models", default="lr,mlp64,mlp128,mlp256",
                    help="Comma-separated readout families considered by grouped CV.")
    args = ap.parse_args()
    layers = [int(x) for x in args.layers.split(",")]

    pos, em, flip, flip2, cb, gid, ds = load_dumps(args.dumps, layers)
    n = len(em)
    print(f"\n=== loaded {n} rollouts | datasets={sorted(set(ds.tolist()))} ===")
    print(f"EM={em.mean():.3f} flip={flip.mean():.3f} flip_strict={flip2.mean():.3f} "
          f"cb={cb.mean():.3f}")

    labels = {"flip": flip, "flip2": flip2}
    positions = ("Xc", "Xa") if args.commit_position == "auto" else (args.commit_position,)
    label_names = ("flip", "flip2") if args.commit_label == "auto" else (args.commit_label,)
    models = tuple(x.strip() for x in args.commit_models.split(",") if x.strip())
    # ---------------- E1: model selection for the commit probe ----------------
    print("\n========== E1: commit-probe model selection (group-split CV AUC) ==========")
    best = (-1, None)  # (auc, cfg)
    for posk in positions:
        for labk in label_names:
            for model in models:
                for L in layers:
                    auc, _ = cv_auc(pos[posk][L], labels[labk], gid, model)
                    tag = f"{posk}/{labk}/{model}/L{L}"
                    print(f"  {tag:24s} AUC={auc:.3f}")
                    if auc == auc and auc > best[0]:
                        best = (auc, (posk, labk, model, L))
    bposk, blabk, bmodel, bL = best[1]
    print(f"  -> BEST commit probe: {bposk}/{blabk}/{bmodel}/L{bL}  AUC={best[0]:.3f}")
    by = labels[blabk]

    # zero-variance rescue test (uses the chosen flip label)
    g2em, g2f = defaultdict(list), defaultdict(list)
    for g, e, f in zip(gid, em, by):
        g2em[g].append(e); g2f[g].append(f)
    allwrong = [g for g in g2em if len(g2em[g]) > 1 and max(g2em[g]) == 0]
    allright = [g for g in g2em if len(g2em[g]) > 1 and min(g2em[g]) == 1]
    fv = lambda gs: (np.mean([1.0 if len(set(g2f[g])) > 1 else 0.0 for g in gs]) if gs else float("nan"))
    print(f"  GRPO groups total={len(g2em)} allwrong={len(allwrong)} allright={len(allright)} "
          f"({100*(len(allwrong)+len(allright))/max(len(g2em),1):.0f}% zero-variance)")
    print(f"  {blabk} within-group variance: all-wrong={100*fv(allwrong):.1f}%  "
          f"all-right={100*fv(allright):.1f}%")

    # ---------------- E0: boundary ruler ----------------
    print("\n========== E0: boundary probe (closed-book) ==========")
    bb = (-1, layers[0], "lr")
    for model in ("lr", "mlp64", "mlp128"):
        for L in layers:
            auc, _ = cv_auc(pos["Xp"][L], cb, gid, model)
            print(f"  boundary Xp/{model}/L{L}: AUC={auc:.3f}")
            if auc == auc and auc > bb[0]:
                bb = (auc, L, model)
    clf_b = make_clf(bb[2]).fit(pos["Xp"][bb[1]], cb)
    reads = clf_b.predict_proba(pos["Xp"][bb[1]])[:, 1]
    for d in [x for x in RULER if x in set(ds.tolist())]:
        m = ds == d
        print(f"     {d:16s} read={reads[m].mean():.3f} cb={cb[m].mean():.3f} EM={em[m].mean():.3f} n={m.sum()}")

    # ---------------- E0.5: orthogonality ----------------
    print("\n========== E0.5: orthogonality (control EM) ==========")
    clf_c = make_clf(bmodel).fit(pos[bposk][bL], by)
    rc = clf_c.predict_proba(pos[bposk][bL])[:, 1]
    for sub, name in [(em == 1, "EM==1"), (em == 0, "EM==0")]:
        line = "  " + name + ": " + "  ".join(
            f"{d}={rc[(ds==d)&sub].mean():.3f}(n{((ds==d)&sub).sum()})"
            for d in [x for x in RULER if x in set(ds.tolist())] if ((ds==d)&sub).sum() >= 5)
        print(line)

    # ---------------- save ----------------
    if args.save:
        import joblib
        os.makedirs(args.save, exist_ok=True)
        cm = make_clf(bmodel).fit(pos[bposk][bL], by)
        bm = make_clf(bb[2]).fit(pos["Xp"][bb[1]], cb)
        joblib.dump({"clf": cm, "position": bposk, "layer": bL, "label": blabk,
                     "model": bmodel, "auc": best[0]},
                    os.path.join(args.save, "commit_probe.pkl"))
        joblib.dump({"clf": bm, "position": "Xp", "layer": bb[1], "label": "closedbook",
                     "model": bb[2], "auc": bb[0]},
                    os.path.join(args.save, "boundary_probe.pkl"))
        print(f"\n[save] commit({bposk}/L{bL}/{blabk}/{bmodel} AUC={best[0]:.3f}) + "
              f"boundary(Xp/L{bb[1]}/{bb[2]} AUC={bb[0]:.3f}) -> {args.save}")


if __name__ == "__main__":
    main()
