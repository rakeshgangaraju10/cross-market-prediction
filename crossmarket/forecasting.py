"""Leak-free multi-horizon volatility forecasting: baselines, MLP and a spillover GNN.

Design
------
* Information set at day t: squared log returns up to and including t.
* Features (per market): HAR components, i.e. log annualized volatility over
  the last 1, 5 and 22 trading days.
* Target for horizon h: log annualized realized volatility over days t+1..t+h,
  so the target never overlaps the features (the original pipeline predicted a
  21-day rolling window one day ahead, which is ~95% known in advance).
* Chronological train / validation / test split; scaling and the spillover
  graph are fit on the training period only; the test set is touched once.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import TRADING_DAYS
from .spillover import generalized_fevd, spillover_adjacency

HAR_WINDOWS = (1, 5, 22)
HORIZONS = (1, 5, 10, 22)
VAR_FLOOR = 1e-6  # daily variance floor (~0.1% move) to keep logs finite


def _log_vol(var: pd.DataFrame) -> pd.DataFrame:
    return 0.5 * np.log(TRADING_DAYS * var.clip(lower=VAR_FLOOR))


def har_features(returns: pd.DataFrame) -> pd.DataFrame:
    """Columns MultiIndex (market, feature) with HAR log-vol components at t."""
    sq = returns**2
    feats = {}
    for w in HAR_WINDOWS:
        mean = sq.ffill(limit=3) if w == 1 else sq.rolling(w, min_periods=max(1, w // 2)).mean()
        feats[f"rv{w}"] = _log_vol(mean)
    return pd.concat(feats, axis=1).swaplevel(axis=1).sort_index(axis=1)


def forward_target(returns: pd.DataFrame, h: int) -> pd.DataFrame:
    """Log annualized realized vol over t+1..t+h (NaN-skipping over holidays)."""
    fwd = (returns**2).rolling(h, min_periods=1).mean().shift(-h)
    return _log_vol(fwd)


@dataclass
class Dataset:
    dates: pd.DatetimeIndex
    markets: list[str]
    X: np.ndarray  # (T, N, F)
    y: np.ndarray  # (T, N)
    splits: dict[str, slice] = field(default_factory=dict)

    def part(self, name: str) -> tuple[np.ndarray, np.ndarray]:
        s = self.splits[name]
        return self.X[s], self.y[s]


def build_dataset(returns: pd.DataFrame, h: int, train: float = 0.6, val: float = 0.2) -> Dataset:
    feats = har_features(returns)
    target = forward_target(returns, h)
    markets = list(returns.columns)
    fnames = [f"rv{w}" for w in HAR_WINDOWS]
    X = np.stack([feats[m][fnames].to_numpy() for m in markets], axis=1)
    y = target[markets].to_numpy()
    ok = np.isfinite(X).all(axis=(1, 2)) & np.isfinite(y).all(axis=1)
    X, y, dates = X[ok], y[ok], returns.index[ok]
    T = len(dates)
    a, b = int(T * train), int(T * (train + val))
    # Drop h rows before each boundary so no training target peeks into the next split.
    splits = {"train": slice(0, a - h), "val": slice(a, b - h), "test": slice(b, T)}
    return Dataset(dates, markets, X.astype(np.float32), y.astype(np.float32), splits)


# ---------------------------------------------------------------- baselines

def predict_naive(X: np.ndarray) -> np.ndarray:
    """Random walk: tomorrow's vol equals the last 22-day vol."""
    return X[:, :, HAR_WINDOWS.index(22)]


class HAR:
    """Corsi (2009) HAR, one OLS per market on its own lags."""

    def fit(self, X: np.ndarray, y: np.ndarray) -> HAR:
        self.coef = [np.linalg.lstsq(_add_const(X[:, i]), y[:, i], rcond=None)[0]
                     for i in range(X.shape[1])]
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.stack([_add_const(X[:, i]) @ c for i, c in enumerate(self.coef)], axis=1)


class HARX:
    """Cross-market HAR: each market regressed on every market's HAR lags (ridge)."""

    def __init__(self, alpha: float = 1.0) -> None:
        self.alpha = alpha

    def fit(self, X: np.ndarray, y: np.ndarray) -> HARX:
        Z = _add_const(X.reshape(len(X), -1))
        reg = self.alpha * np.eye(Z.shape[1])
        reg[0, 0] = 0.0
        self.coef = np.linalg.solve(Z.T @ Z + reg, Z.T @ y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return _add_const(X.reshape(len(X), -1)) @ self.coef


def _add_const(Z: np.ndarray) -> np.ndarray:
    return np.column_stack([np.ones(len(Z)), Z])


# ---------------------------------------------------------------- neural models

def _torch():
    import torch
    return torch


def make_mlp(n_nodes: int, n_feat: int, hidden: int = 64, dropout: float = 0.2):
    nn = _torch().nn

    class MLP(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.net = nn.Sequential(
                nn.Flatten(), nn.Linear(n_nodes * n_feat, hidden), nn.ReLU(), nn.Dropout(dropout),
                nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout), nn.Linear(hidden, n_nodes))
            self.skip = nn.Linear(n_feat, 1)  # per-node linear (HAR-like) path

        def forward(self, x):
            return self.net(x) + self.skip(x).squeeze(-1)

    return MLP()


def make_gnn(adjacency: np.ndarray, n_feat: int, hidden: int = 32, layers: int = 2,
             heads: int = 2, dropout: float = 0.1):
    """Spillover graph network.

    Messages flow i -> j weighted by the training-period DY spillover from i to
    j, refined by GAT-style attention (masked to the spillover graph) with
    several heads. A node embedding lets each market keep its own dynamics.
    """
    torch = _torch()
    nn = torch.nn
    n = adjacency.shape[0]
    a = torch.tensor(adjacency.T + np.eye(n), dtype=torch.float32)  # rows: receiver
    prior = torch.log(a / a.sum(dim=1, keepdim=True))

    class Layer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.w = nn.Linear(hidden, hidden * heads, bias=False)
            self.att_src = nn.Parameter(torch.zeros(heads, hidden))
            self.att_dst = nn.Parameter(torch.zeros(heads, hidden))
            self.out = nn.Linear(hidden * heads, hidden)
            self.norm = nn.LayerNorm(hidden)

        def forward(self, h):  # (B, N, H)
            B = h.shape[0]
            z = self.w(h).view(B, n, heads, hidden)
            e = (torch.einsum("bnkh,kh->bkn", z, self.att_dst)[..., :, None]
                 + torch.einsum("bnkh,kh->bkn", z, self.att_src)[..., None, :])
            att = torch.softmax(nn.functional.leaky_relu(e, 0.2) + prior, dim=-1)  # (B,K,N,N)
            msg = torch.einsum("bkij,bjkh->bikh", att, z).reshape(B, n, heads * hidden)
            return self.norm(h + nn.functional.dropout(torch.relu(self.out(msg)), dropout, self.training))

    class GNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embed = nn.Parameter(torch.zeros(n, hidden))
            self.inp = nn.Linear(n_feat, hidden)
            self.layers = nn.ModuleList(Layer() for _ in range(layers))
            self.head = nn.Linear(hidden, 1)
            self.skip = nn.Linear(n_feat, 1)

        def forward(self, x):
            h = torch.relu(self.inp(x)) + self.embed
            for layer in self.layers:
                h = layer(h)
            return (self.head(h) + self.skip(x)).squeeze(-1)

    return GNN()


def train_torch(model, Xtr, ytr, Xva, yva, lr: float = 1e-3, epochs: int = 300,
                batch: int = 128, patience: int = 25, seed: int = 0):
    """Adam + early stopping on validation MSE; returns (model, history)."""
    torch = _torch()
    torch.manual_seed(seed)
    t = lambda a: torch.tensor(a, dtype=torch.float32)  # noqa: E731
    Xtr, ytr, Xva, yva = map(t, (Xtr, ytr, Xva, yva))
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    loss_fn = torch.nn.MSELoss()
    best, best_state, wait = float("inf"), None, 0
    hist: dict[str, list[float]] = {"train": [], "val": []}
    g = torch.Generator().manual_seed(seed)
    for _ in range(epochs):
        model.train()
        perm = torch.randperm(len(Xtr), generator=g)
        total = 0.0
        for i in range(0, len(Xtr), batch):
            idx = perm[i:i + batch]
            opt.zero_grad()
            loss = loss_fn(model(Xtr[idx]), ytr[idx])
            loss.backward()
            opt.step()
            total += loss.item() * len(idx)
        model.eval()
        with torch.no_grad():
            v = loss_fn(model(Xva), yva).item()
        hist["train"].append(total / len(Xtr))
        hist["val"].append(v)
        if v < best - 1e-6:
            best, wait = v, 0
            best_state = {k: p.detach().clone() for k, p in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break
    model.load_state_dict(best_state)
    model.eval()
    return model, hist


def torch_predict(model, X: np.ndarray) -> np.ndarray:
    torch = _torch()
    with torch.no_grad():
        return model(torch.tensor(X, dtype=torch.float32)).numpy()


# ---------------------------------------------------------------- metrics

def metrics(y_log: np.ndarray, p_log: np.ndarray) -> dict[str, float]:
    """Errors on the volatility scale plus QLIKE on variance (robust to noisy proxies)."""
    y, p = np.exp(y_log), np.exp(p_log)
    ratio = (y / p) ** 2
    ss_res = ((y_log - p_log) ** 2).sum()
    ss_tot = ((y_log - y_log.mean()) ** 2).sum()
    return {
        "MAE": float(np.mean(np.abs(y - p))),
        "RMSE": float(np.sqrt(np.mean((y - p) ** 2))),
        "MAPE": float(np.mean(np.abs(y - p) / y) * 100),
        "QLIKE": float(np.mean(ratio - np.log(ratio) - 1)),
        "R2_log": float(1 - ss_res / ss_tot),
    }


def log_bias(y_log: np.ndarray, p_log: np.ndarray) -> np.ndarray:
    """Per-market log shift making exp(pred)^2 unbiased for variance on the training set.

    Models are fit by MSE in log space, which targets the median, not the mean,
    of variance (Jensen's inequality) and so under-predicts it; that is
    especially costly under QLIKE for noisy short-horizon targets.
    """
    return 0.5 * np.log(np.mean(np.exp(2 * (y_log - p_log)), axis=0))


class Standardizer:
    def fit(self, X: np.ndarray, y: np.ndarray) -> Standardizer:
        self.xm, self.xs = X.mean(axis=0), X.std(axis=0) + 1e-8
        self.ym, self.ys = y.mean(axis=0), y.std(axis=0) + 1e-8
        return self

    def x(self, X):
        return ((X - self.xm) / self.xs).astype(np.float32)

    def y(self, y):
        return ((y - self.ym) / self.ys).astype(np.float32)

    def inv_y(self, y):
        return y * self.ys + self.ym


def run_horizon(returns: pd.DataFrame, h: int, neural: bool = True, epochs: int = 300,
                seed: int = 0) -> dict:
    """Fit every model for one horizon and evaluate on the held-out test set."""
    ds = build_dataset(returns, h)
    Xtr, ytr = ds.part("train")
    Xva, yva = ds.part("val")
    Xte, yte = ds.part("test")

    preds: dict[str, np.ndarray] = {"Naive (RW)": predict_naive(Xte)}
    for name, model in (("HAR", HAR()), ("HAR-X (cross-market)", HARX())):
        model.fit(Xtr, ytr)
        preds[name] = model.predict(Xte) + log_bias(ytr, model.predict(Xtr))
    history = {}

    if neural:
        sc = Standardizer().fit(Xtr, ytr)
        # Spillover graph from training-period monthly log-vol only (no look-ahead).
        fevd = generalized_fevd(pd.DataFrame(Xtr[:, :, HAR_WINDOWS.index(22)], columns=ds.markets))
        adj = spillover_adjacency(fevd)
        for name, model in (("MLP", make_mlp(len(ds.markets), Xtr.shape[2])),
                            ("Spillover GNN", make_gnn(adj, Xtr.shape[2]))):
            model, hist = train_torch(model, sc.x(Xtr), sc.y(ytr), sc.x(Xva), sc.y(yva),
                                      epochs=epochs, seed=seed)
            fitted = sc.inv_y(torch_predict(model, sc.x(Xtr)))
            preds[name] = sc.inv_y(torch_predict(model, sc.x(Xte))) + log_bias(ytr, fitted)
            history[name] = hist

    rows = []
    for name, p in preds.items():
        rows.append({"horizon": h, "model": name, "market": "All", **metrics(yte, p)})
        for i, m in enumerate(ds.markets):
            rows.append({"horizon": h, "model": name, "market": m, **metrics(yte[:, i], p[:, i])})
    test_dates = ds.dates[ds.splits["test"]]
    pred_frames = {name: pd.DataFrame(np.exp(p), index=test_dates, columns=ds.markets)
                   for name, p in preds.items()}
    pred_frames["Actual"] = pd.DataFrame(np.exp(yte), index=test_dates, columns=ds.markets)
    return {"metrics": pd.DataFrame(rows), "predictions": pred_frames, "history": history,
            "split_dates": {k: (str(ds.dates[s][0].date()), str(ds.dates[s][-1].date()))
                            for k, s in ds.splits.items()}}
