#!/usr/bin/env python3
"""
Does the FROZEN probe degrade as the policy evolves? (co-evolution necessity test)

Standard domain-shift protocol on the commit-token hidden states, Task A
(grounded-correct vs parametric-correct, among EM=1):

  AUC_frozen@home  : train on BASE rollouts, 5-fold CV on BASE      (frozen probe's turf)
  AUC_frozen@175   : train on BASE rollouts, TEST on STEP175 rollouts (frozen probe applied
                     to the drifted policy -> what a non-co-evolved probe actually sees)
  AUC_refit@175    : 5-fold CV on STEP175 rollouts                  (= what co-evolution achieves)

If AUC_frozen@175 << AUC_refit@175 (and << AUC_frozen@home), the representation drifted
out from under the frozen probe -> co-evolution / self-calibration is necessary.
"""
import argparse, numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


def among_correct(npz, L):
    d = np.load(npz, allow_pickle=True)
    m = d["y_em"] == 1
    X = d[f"Xc_{L}"][m].astype(np.float32)
    y = d["y_flip"][m].astype(np.int8)
    return X, y


def cv_auc(X, y, seed=0):
    if len(np.unique(y)) < 2 or min(np.bincount(y)) < 5:
        return float("nan"), len(y)
    skf = StratifiedKFold(5, shuffle=True, random_state=seed)
    oof = np.zeros(len(y))
    for tr, te in skf.split(X, y):
        sc = StandardScaler().fit(X[tr])
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(X[tr]), y[tr])
        oof[te] = clf.predict_proba(sc.transform(X[te]))[:, 1]
    return roc_auc_score(y, oof), len(y)


def transfer_auc(Xtr, ytr, Xte, yte):
    if len(np.unique(ytr)) < 2 or len(np.unique(yte)) < 2:
        return float("nan")
    sc = StandardScaler().fit(Xtr)
    clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(Xtr), ytr)
    p = clf.predict_proba(sc.transform(Xte))[:, 1]
    return roc_auc_score(yte, p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="dumps/scalars/merged.npz", help="base-model rollouts")
    ap.add_argument("--drift", required=True, help="evolved-policy rollouts (e.g. F2_s175.npz)")
    ap.add_argument("--layers", default="21,24,27")
    args = ap.parse_args()

    print(f"{'layer':>6} {'frozen@home':>12} {'frozen@175':>12} {'refit@175':>12} {'drop(home-175)':>15}")
    for L in [int(x) for x in args.layers.split(",")]:
        Xb, yb = among_correct(args.base, L)
        Xd, yd = among_correct(args.drift, L)
        a_home, _ = cv_auc(Xb, yb)
        a_frozen = transfer_auc(Xb, yb, Xd, yd)
        a_refit, n = cv_auc(Xd, yd)
        drop = a_home - a_frozen
        print(f"L{L:<5} {a_home:>12.3f} {a_frozen:>12.3f} {a_refit:>12.3f} {drop:>+15.3f}"
              f"   (drift n={n}, grounded={int(yd.sum())}/{int((yd==0).sum())})")
    print("\nReading: frozen@175 << refit@175  => frozen probe degraded on the evolved policy"
          " => co-evolution needed.")


if __name__ == "__main__":
    main()
