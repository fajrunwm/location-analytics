"""Stage 4 — how much of commercial activity does street-network structure explain?

Trains LightGBM on H3 hex features from one or more areas with SPATIAL cross-validation
(GroupKFold over H3 res-6 parents, ~36 km2 blocks) so neighbouring hexes never sit on
both sides of a split.

Feature sets compared (same folds):
  network          walk_* / drive_* / ang_*  — street structure only
  context          ctx_*                     — population + transit stops
  network+context  both

Outputs (in --out):
  metrics.json               R2 / Spearman per target x feature set (out-of-fold)
  hex_predictions.parquet    out-of-fold predictions + residual ("network potential gap")
  feature_importance.csv     mean |SHAP| per feature (network+context model)
  outlet_correlations.csv    Spearman of real outlet review_count vs network features
  outlet_scores.parquet      per-outlet network features + predicted hex potential
"""
import argparse
import json
import time
from pathlib import Path

import warnings

import lightgbm as lgb
import numpy as np
import pandas as pd
import shap
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold

warnings.filterwarnings("ignore")

try:
    import mlflow
except ImportError:
    mlflow = None

TARGETS = ["y_commercial", "y_fnb"]
URBAN_POP_PER_HEX = 1000  # H3 res 9 ~0.105 km2 -> ~9,500 people/km2
PARAMS = dict(objective="regression", learning_rate=0.03, num_leaves=63, min_child_samples=30,
              feature_fraction=0.5, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              n_estimators=1500, verbose=-1)


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def feature_sets(cols):
    net = [c for c in cols if c.startswith(("walk_", "drive_", "ang_"))]
    # Population: keep sum/mean per radius only — the other cityseer stats (min/max/median/mad/var)
    # are ~100 near-duplicate columns that would dominate grouped importance.
    ctx = [c for c in cols if c.startswith(("ctx_n_", "ctx_pop_hex", "ctx_pop_sum_", "ctx_pop_mean_"))]
    return {"network": net, "context": ctx, "network+context": net + ctx}


def spatial_cv(X, y, groups, n_splits, params, seed=42):
    oof = np.zeros(len(y))
    models = []
    for k, (tr, te) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, groups)):
        m = lgb.LGBMRegressor(**params, random_state=seed + k)
        # inner early-stopping split: last 10% of training groups
        tr_groups = groups.iloc[tr]
        uniq = tr_groups.unique()
        rng = np.random.default_rng(seed + k)
        val_g = set(rng.choice(uniq, max(1, len(uniq) // 10), replace=False))
        is_val = tr_groups.isin(val_g).values
        m.fit(X.iloc[tr[~is_val]], y.iloc[tr[~is_val]],
              eval_set=[(X.iloc[tr[is_val]], y.iloc[tr[is_val]])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        oof[te] = m.predict(X.iloc[te])
        models.append(m)
    return oof, models


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="folder with <area>/hex_features.parquet")
    ap.add_argument("--areas", required=True, help="comma-separated")
    ap.add_argument("--out", required=True)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--group-res", type=int, default=6, choices=[6, 7])
    ap.add_argument("--test-areas", default="", help="comma-separated areas never used in training "
                    "(out-of-sample test, e.g. cities outside Java)")
    a = ap.parse_args()

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    areas = a.areas.split(",")
    hexes = pd.concat([pd.read_parquet(Path(a.data) / ar / "hex_features.parquet") for ar in areas])
    hexes = hexes[hexes.walk_node_count >= 3]  # drop near-empty cells (water, edges)
    groups = hexes[f"h3_parent{a.group_res}"].astype(str)
    log(f"{len(hexes):,} hexes from {areas}, {groups.nunique()} spatial groups")
    test_areas = [t for t in a.test_areas.split(",") if t]
    test = pd.concat([pd.read_parquet(Path(a.data) / ar / "hex_features.parquet") for ar in test_areas]) if test_areas else None
    if test is not None:
        test = test[test.walk_node_count >= 3]
        log(f"out-of-sample test: {len(test):,} hexes from {test_areas}")

    fsets = feature_sets(hexes.columns)
    metrics, preds = {}, hexes[["area", "lat", "lon"] + TARGETS].copy()
    test_preds = test[["area", "lat", "lon"] + TARGETS].copy() if test is not None else None
    run = mlflow.start_run() if mlflow else None
    for target in TARGETS:
        y = np.log1p(hexes[target])
        for name, cols in fsets.items():
            oof, models = spatial_cv(hexes[cols], y, groups, min(a.folds, groups.nunique()), PARAMS)
            r2 = r2_score(y, oof)
            rho = spearmanr(y, oof).statistic
            # Dense-urban subset (>= ~9.5k people/km2): removes the easy urban-vs-rural gradient.
            urban = (hexes["ctx_pop_hex"] >= URBAN_POP_PER_HEX).values
            r2_u = r2_score(y[urban], oof[urban]) if urban.sum() > 50 else float("nan")
            rho_u = spearmanr(y[urban], oof[urban]).statistic if urban.sum() > 50 else float("nan")
            metrics[f"{target}|{name}"] = {"r2_log": round(r2, 4), "spearman": round(rho, 4), "n_features": len(cols),
                                           "r2_log_urban": round(r2_u, 4), "spearman_urban": round(rho_u, 4),
                                           "n_hex": int(len(y)), "n_hex_urban": int(urban.sum())}
            log(f"{target:13s} {name:16s} R2(log)={r2:.3f} rho={rho:.3f} | dense-urban R2={r2_u:.3f} rho={rho_u:.3f} "
                f"(n={urban.sum()}) ({len(cols)} feats)")
            if mlflow:
                mlflow.log_metrics({f"{target}.{name}.r2".replace("+", "_"): r2,
                                    f"{target}.{name}.spearman".replace("+", "_"): rho})
            if test is not None:
                # Train on ALL training areas, predict cities the model has never seen.
                n_est = int(np.mean([m.best_iteration_ or 500 for m in models]))
                full = lgb.LGBMRegressor(**{**PARAMS, "n_estimators": n_est}, random_state=42).fit(hexes[cols], y)
                yt = np.log1p(test[target])
                pt = full.predict(test[cols])
                ut = (test["ctx_pop_hex"] >= URBAN_POP_PER_HEX).values
                per_area = {ar: round(r2_score(yt[test.area == ar], pt[(test.area == ar).values]), 4) for ar in test_areas}
                metrics[f"{target}|{name}|test"] = {
                    "r2_log": round(r2_score(yt, pt), 4), "spearman": round(spearmanr(yt, pt).statistic, 4),
                    "r2_log_urban": round(r2_score(yt[ut], pt[ut]), 4) if ut.sum() > 50 else None,
                    "r2_log_by_area": per_area, "n_hex": int(len(yt))}
                log(f"{'':13s} {'-> test':16s} R2(log)={metrics[f'{target}|{name}|test']['r2_log']:.3f} by area {per_area}")
                if name == "network+context":
                    test_preds[f"{target}_pred_log"] = pt
                    test_preds[f"{target}_gap_log"] = pt - yt
            if name == "network+context":
                preds[f"{target}_pred_log"] = oof
                preds[f"{target}_gap_log"] = oof - y  # >0: network says more activity than observed
                if target == "y_commercial":
                    X = hexes[cols]
                    sample = X.sample(min(5000, len(X)), random_state=0)
                    sv = np.mean([np.abs(shap.TreeExplainer(m).shap_values(sample)).mean(0) for m in models], axis=0)
                    imp = pd.Series(sv, index=cols).sort_values(ascending=False)
                    imp.rename("mean_abs_shap").to_csv(out / "feature_importance.csv")
                    log("top SHAP: " + ", ".join(f"{k}={v:.3f}" for k, v in imp.head(8).items()))
                    final = lgb.LGBMRegressor(**{**PARAMS, "n_estimators": int(np.mean([m.best_iteration_ or 500 for m in models]))},
                                              random_state=42).fit(X, y)
                    final.booster_.save_model(str(out / "lgbm_commercial.txt"))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    if test_preds is not None:
        preds = pd.concat([preds, test_preds[~test_preds.index.isin(preds.index)]])
    preds.to_parquet(out / "hex_predictions.parquet")

    # Outlets: does network position line up with REAL Google Maps review counts?
    paths = [Path(a.data) / ar / "outlet_features.parquet" for ar in areas + test_areas]
    paths = [p for p in paths if p.exists()]
    if paths:
        of = pd.concat([pd.read_parquet(p) for p in paths])
        of = of[~of.index.duplicated()]
        of = of.join(preds[["y_commercial_pred_log", "y_commercial_gap_log"]], on="h3")
        cand = [c for c in of.columns if c.startswith(("node_walk_", "node_drive_", "hex_walk_", "hex_ang_", "hex_drive_", "hex_ctx_"))
                and not c.endswith("snap_m") and pd.api.types.is_numeric_dtype(of[c])] + ["y_commercial_pred_log"]
        # Adjusted target: log reviews relative to the outlet's own brand x location_type group
        # (mall outlets and ruko outlets differ systematically, independent of street network).
        lr = np.log1p(of["review_count"])
        of["review_resid"] = lr - lr.groupby([of["brand"], of["location_type"]]).transform("mean")

        def bh(p):
            p = pd.Series(p).reset_index(drop=True)
            order = p.sort_values().index
            m = len(p)
            q = (p[order] * m / np.arange(1, m + 1))[::-1].cummin()[::-1].clip(upper=1)
            return q.reindex(p.index).values

        rows = []
        for c in cand:
            ok = of[[c, "review_count", "review_resid"]].dropna()
            if len(ok) >= 8 and ok[c].nunique() > 2:
                r = spearmanr(ok[c], ok["review_count"])
                ra = spearmanr(ok[c], ok["review_resid"])
                rows.append({"feature": c, "spearman": r.statistic, "p_value": r.pvalue,
                             "spearman_adj": ra.statistic, "p_value_adj": ra.pvalue, "n": len(ok)})
        corr = pd.DataFrame(rows)
        # Benjamini-Hochberg: hundreds of features vs ~72 outlets -> most raw p<0.05 are chance.
        corr["q_value_bh"] = bh(corr.p_value)
        corr["q_value_bh_adj"] = bh(corr.p_value_adj)
        corr = corr.sort_values("spearman_adj", key=np.abs, ascending=False)
        corr.to_csv(out / "outlet_correlations.csv", index=False)
        of.to_parquet(out / "outlet_scores.parquet")
        log(f"outlets: n={len(of)}; top |rho| vs review_count:\n{corr.head(10).round(3).to_string(index=False)}")

    if run:
        mlflow.log_artifacts(str(out))
        mlflow.end_run()


if __name__ == "__main__":
    main()
