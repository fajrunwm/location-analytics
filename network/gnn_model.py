"""Stage 4b — GraphSAGE on the H3 hex adjacency graph (GPU), compared with LightGBM.

Each hex is a graph node with the same network+context features as train_model.py; edges
connect H3 neighbours (k=1 ring). A mean-aggregation GraphSAGE lets a hex's prediction use
its neighbourhood's street structure beyond what the fixed-radius features already encode.
Uses the SAME spatial folds (GroupKFold over H3 res-6 parents) so metrics are comparable;
held-out hexes are hidden from the loss but stay in the graph (transductive setting).

Plain PyTorch (sparse adjacency) — no torch_geometric dependency.
"""
import argparse
import json
import time
from pathlib import Path

import h3
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def adjacency(cells, device):
    idx = {c: i for i, c in enumerate(cells)}
    rows, cols = [], []
    for i, c in enumerate(cells):
        for nb in h3.grid_ring(c, 1):
            j = idx.get(nb)
            if j is not None:
                rows.append(i)
                cols.append(j)
    n = len(cells)
    deg = np.bincount(rows, minlength=n).clip(min=1)
    vals = 1.0 / deg[rows]
    A = torch.sparse_coo_tensor(np.array([rows, cols]), vals, (n, n), dtype=torch.float32)
    return A.coalesce().to(device)


class SAGE(nn.Module):
    def __init__(self, d_in, hidden, layers, dropout):
        super().__init__()
        dims = [d_in] + [hidden] * layers
        self.self_lin = nn.ModuleList(nn.Linear(a, b) for a, b in zip(dims[:-1], dims[1:]))
        self.nb_lin = nn.ModuleList(nn.Linear(a, b, bias=False) for a, b in zip(dims[:-1], dims[1:]))
        self.norms = nn.ModuleList(nn.LayerNorm(b) for b in dims[1:])
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, 1)

    def forward(self, x, A):
        h = x
        for s, nbl, norm in zip(self.self_lin, self.nb_lin, self.norms):
            h = self.drop(torch.relu(norm(s(h) + nbl(torch.sparse.mm(A, h)))))
        return self.head(h).squeeze(-1)


def fit_fold(X, y, A, tr, va, te, args, device, seed):
    torch.manual_seed(seed)
    model = SAGE(X.shape[1], args.hidden, args.layers, args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    best, best_state, wait = np.inf, None, 0
    for epoch in range(args.epochs):
        model.train()
        opt.zero_grad()
        loss = nn.functional.mse_loss(model(X, A)[tr], y[tr])
        loss.backward()
        opt.step()
        if epoch % 10 == 0:
            model.eval()
            with torch.no_grad():
                v = nn.functional.mse_loss(model(X, A)[va], y[va]).item()
            if v < best - 1e-4:
                best, best_state, wait = v, {k: t.clone() for k, t in model.state_dict().items()}, 0
            else:
                wait += 1
                if wait >= args.patience:
                    break
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model(X, A)[te].cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--areas", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", default="y_commercial")
    ap.add_argument("--hidden", type=int, default=256)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--epochs", type=int, default=3000)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    hexes = pd.concat([pd.read_parquet(Path(args.data) / a / "hex_features.parquet") for a in args.areas.split(",")])
    hexes = hexes[hexes.walk_node_count >= 3]
    hexes = hexes[~hexes.index.duplicated()]
    cols = [c for c in hexes.columns if c.startswith(("walk_", "drive_", "ang_", "ctx_n_", "ctx_pop_hex", "ctx_pop_sum_", "ctx_pop_mean_"))]
    Xn = hexes[cols].astype("float64")
    Xn = np.sign(Xn) * np.log1p(np.abs(Xn))  # heavy-tailed centralities
    Xn = (Xn - Xn.mean()) / Xn.std().replace(0, 1)
    X = torch.tensor(Xn.fillna(0).values, dtype=torch.float32, device=device)
    y_np = np.log1p(hexes[args.target].values)
    y = torch.tensor(y_np, dtype=torch.float32, device=device)
    A = adjacency(list(hexes.index), device)
    groups = hexes.h3_parent6.astype(str).values
    log(f"device={device} hexes={len(hexes):,} feats={len(cols)} edges={A._nnz():,}")

    oof = np.zeros((args.seeds, len(hexes)))
    for seed in range(args.seeds):
        for k, (tr_all, te) in enumerate(GroupKFold(n_splits=args.folds).split(Xn, y_np, groups)):
            rng = np.random.default_rng(seed * 100 + k)
            g_tr = np.unique(groups[tr_all])
            val_g = set(rng.choice(g_tr, max(1, len(g_tr) // 10), replace=False))
            is_val = np.isin(groups[tr_all], list(val_g))
            tr, va = tr_all[~is_val], tr_all[is_val]
            oof[seed, te] = fit_fold(X, y, A, torch.tensor(tr, device=device), torch.tensor(va, device=device),
                                     torch.tensor(te, device=device), args, device, seed * 100 + k)
        r2 = r2_score(y_np, oof[seed])
        log(f"seed {seed}: R2(log)={r2:.3f} rho={spearmanr(y_np, oof[seed]).statistic:.3f}")

    ens = oof.mean(0)
    metrics = {f"{args.target}|gnn_sage": {"r2_log": round(r2_score(y_np, ens), 4),
                                           "spearman": round(spearmanr(y_np, ens).statistic, 4),
                                           "seed_r2": [round(r2_score(y_np, o), 4) for o in oof],
                                           "params": vars(args) | {"n_features": len(cols)}}}
    log(json.dumps(metrics))
    (out / "gnn_metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame({"h3": hexes.index, "area": hexes.area.values, f"{args.target}_gnn_pred_log": ens}).to_parquet(
        out / "gnn_predictions.parquet")


if __name__ == "__main__":
    main()
