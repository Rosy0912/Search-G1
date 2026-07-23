"""Search-G1 reward--policy co-evolution controller.

The controller watches actor checkpoints, generates fresh calibration trajectories
with each new checkpoint, refits the evidence-sensitivity and closed-book-sufficiency
readouts, and atomically updates the manifest consumed by ``probe_server.py``.
It also logs closed-book correctness on a fixed held-out slice.

If calibration fails, the live manifest is left unchanged and the server continues
to use the most recent complete evaluator.
"""
from __future__ import annotations
import argparse, glob, json, os, subprocess, time

PY = "python"


def latest_ckpt(checkpoint_root, exp, after):
    """Return (step, path) of the newest complete actor checkpoint with step>after."""
    best = (after, None)
    for d in glob.glob(os.path.join(checkpoint_root, exp, "actor", "global_step_*")):
        if not os.path.exists(os.path.join(d, "config.json")):
            continue
        try:
            step = int(d.rsplit("_", 1)[1])
        except ValueError:
            continue
        if step > best[0]:
            best = (step, d)
    return best


def run(cmd, env=None):
    print("  $ " + " ".join(cmd), flush=True)
    return subprocess.run(cmd, env=env).returncode


def refit(exp, step, ckpt, args, env):
    tag = f"{exp}_r{step}"
    dump_dir = os.path.join(args.work_dir, "dumps", tag)
    probe_dir = os.path.join(args.work_dir, "readouts", tag)
    os.makedirs(dump_dir, exist_ok=True)
    # 1) dump fresh rollouts on a FIXED held-out slice (comparable cb across rounds)
    rc = run([PY, "ts_probe_dump.py", "--model", ckpt, "--data", args.data,
              "--tag", args.tag, "--cap", str(args.nq), "--offset", str(args.holdout_offset),
              "--G", str(args.group_size), "--layers", args.layers,
              "--gpu_mem", str(args.gpu_mem), "--out", dump_dir], env)
    if rc != 0:
        print(f"  [refit] dump failed for {tag}", flush=True); return False
    # 2) refit probes
    rc = run([PY, "ts_probe_train.py", "--dumps", dump_dir, "--layers", args.layers,
              "--commit-position", args.commit_position,
              "--commit-label", args.commit_label, "--save", probe_dir], env)
    if rc != 0:
        print(f"  [refit] train failed for {tag}", flush=True); return False
    # 3) atomically update manifest -> server hot-reloads
    manifest = os.path.join(args.manifest_root, f"live_{exp}", "manifest.json")
    os.makedirs(os.path.dirname(manifest), exist_ok=True)
    payload = {"round": step, "encoder": ckpt,
               "commit": os.path.abspath(os.path.join(probe_dir, "commit_probe.pkl")),
               "boundary": os.path.abspath(os.path.join(probe_dir, "boundary_probe.pkl"))}
    tmp = manifest + ".tmp"
    with open(tmp, "w") as f:
        json.dump(payload, f)
    os.replace(tmp, manifest)
    # 4) log knowledge-boundary migration (cb_rate on the fixed slice)
    try:
        import numpy as np
        cbs = [float(np.load(p, allow_pickle=True)["q_cb"].mean())
               for p in glob.glob(os.path.join(dump_dir, "*.npz"))
               if len(np.load(p, allow_pickle=True)["q_cb"])]
        cb = float(np.mean(cbs)) if cbs else float("nan")
    except Exception:
        cb = float("nan")
    os.makedirs(args.work_dir, exist_ok=True)
    with open(os.path.join(args.work_dir, "boundary_migration.csv"), "a") as f:
        f.write(f"{int(time.time())},{exp},{step},{cb:.4f}\n")
    print(f"  [refit] {tag} done -> manifest updated, boundary cb_rate={cb:.3f}", flush=True)
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--exps", default="search_g1")
    ap.add_argument("--checkpoint-root", default="verl_checkpoints")
    ap.add_argument("--work-dir", default="artifacts/coevolution")
    ap.add_argument("--manifest-root", default="artifacts/probes")
    ap.add_argument("--nq", type=int, default=300)
    ap.add_argument("--data", default="data/nq_search/calibration.parquet")
    ap.add_argument("--tag", default="nq")
    ap.add_argument("--holdout_offset", type=int, default=0,
                    help="offset within the question-disjoint calibration split")
    ap.add_argument("--layers", default="21,24,27")
    ap.add_argument("--group-size", type=int, default=5)
    ap.add_argument("--gpu-mem", type=float, default=0.55)
    ap.add_argument("--commit-position", choices=("Xa", "Xc"), default="Xa")
    ap.add_argument("--commit-label", choices=("flip", "flip2"), default="flip")
    ap.add_argument("--retriever", default="http://127.0.0.1:8000/retrieve")
    ap.add_argument("--poll", type=int, default=120)
    args = ap.parse_args()
    exps = [e.strip() for e in args.exps.split(",") if e.strip()]

    env = dict(os.environ)
    env["RETRIEVER_URL"] = args.retriever
    env["NO_PROXY"] = "127.0.0.1,localhost"
    for k in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "all_proxy"):
        env.pop(k, None)

    last = {e: 0 for e in exps}
    print(f"[coevolve] watching {exps} (poll={args.poll}s, nq={args.nq})", flush=True)
    while True:
        for e in exps:
            step, ckpt = latest_ckpt(args.checkpoint_root, e, last[e])
            if ckpt is not None:
                print(f"[coevolve] new checkpoint {e} step={step} -> refit", flush=True)
                if refit(e, step, ckpt, args, env):
                    last[e] = step
        time.sleep(args.poll)


if __name__ == "__main__":
    main()
