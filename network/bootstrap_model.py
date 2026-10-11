"""Stage 4c — spatial block bootstrap for hex-level prediction intervals.

Resamples H3 res-6 blocks (not single hexes — neighbours are correlated) with replacement,
refits the network+context LightGBM, and predicts every hex. Hexes never drawn in a replicate
are its out-of-bag set. Outputs per hex:

  pred_p10/p50/p90       log(1+commercial POI) across all replicates
  gap_oob_p10/p50/p90    (prediction - actual) using only out-of-bag replicates, so the gap
                         is never measured on data the model trained on
  gap_confident          "opportunity" if oob gap p10 > 0, "destination" if p90 < 0, else ""

Like outlet_confidence_bands.csv in the Huff engine, this says which signals are stable
rather than an artefact of one particular fit.
"""
import argparse
import json
import os
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from train_model import PARAMS, feature_sets

warnings.filterwarnings("ignore")


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def one_replicate(args):
    seed, X, y, groups, n_estimators, threads = args
    rng = np.random.default_rng(seed)
    uniq = groups.unique()
    drawn = rng.choice(uniq, len(uniq), replace=True)
    counts = pd.Series(drawn).value_counts()
    w = groups.map(counts).fillna(0).values  # block multiplicity as sample weight
    inbag = w > 0
    m = lgb.LGBMRegressor(**{**PARAMS, "n_estimators": n_estimators, "n_jobs": threads}, random_state=seed)
    m.fit(X[inbag], y[inbag], sample_weight=w[inbag])
    return m.predict(X).astype("float32"), ~inbag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--areas", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--replicates", type=int, default=200)
    ap.add_argument("--n-estimators", type=int, default=600)
    ap.add_argument("--workers", type=int, default=0, help="parallel replicates (0 = cpus/8)")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    hexes = pd.concat([pd.read_parquet(Path(a.data) / ar / "hex_features.parquet") for ar in a.areas.split(",")])
    hexes = hexes[hexes.walk_node_count >= 3]
    hexes = hexes[~hexes.index.duplicated()]
    cols = feature_sets(hexes.columns)["network+context"]
    X, y = hexes[cols], np.log1p(hexes["y_commercial"]).values
    groups = hexes.h3_parent6.astype(str).reset_index(drop=True)
    X = X.reset_index(drop=True)

    cpus = os.cpu_count()
    workers = a.workers or max(1, cpus // 8)
    threads = max(1, cpus // workers)
    log(f"{len(hexes):,} hexes, {groups.nunique()} blocks, {a.replicates} replicates, {workers} workers x {threads} threads")

    preds = np.zeros((a.replicates, len(y)), dtype="float32")
    oob = np.zeros((a.replicates, len(y)), dtype=bool)
    t = time.time()
    jobs = [(seed, X, y, groups, a.n_estimators, threads) for seed in range(a.replicates)]
    with ProcessPoolExecutor(workers) as ex:
        for i, (p, o) in enumerate(ex.map(one_replicate, jobs)):
            preds[i], oob[i] = p, o
            if (i + 1) % 20 == 0:
                log(f"{i + 1}/{a.replicates} replicates ({time.time() - t:.0f}s)")

    gap = np.where(oob, preds - y, np.nan)
    res = pd.DataFrame({
        "area": hexes.area.values,
        "pred_p10": np.percentile(preds, 10, axis=0), "pred_p50": np.percentile(preds, 50, axis=0),
        "pred_p90": np.percentile(preds, 90, axis=0),
        "gap_oob_p10": np.nanpercentile(gap, 10, axis=0), "gap_oob_p50": np.nanpercentile(gap, 50, axis=0),
        "gap_oob_p90": np.nanpercentile(gap, 90, axis=0), "n_oob": oob.sum(0),
    }, index=hexes.index)
    res["gap_confident"] = np.select([res.gap_oob_p10 > 0, res.gap_oob_p90 < 0], ["opportunity", "destination"], "")
    res.to_parquet(out / "hex_bootstrap.parquet")
    summary = res.groupby("area").gap_confident.value_counts().unstack(fill_value=0)
    log(f"confident gaps by area:\n{summary.to_string()}")
    (out / "bootstrap_summary.json").write_text(json.dumps(
        {"replicates": a.replicates, "hexes": len(res), "by_area": summary.to_dict(orient="index")}, indent=2))


if __name__ == "__main__":
    main()
