"""Diebold-Yilmaz (2012) volatility spillover index via generalized FEVD."""
from __future__ import annotations

import numpy as np
import pandas as pd
from statsmodels.tsa.api import VAR


def generalized_fevd(data: pd.DataFrame, lags: int = 2, horizon: int = 10) -> pd.DataFrame:
    """Row-normalized generalized FEVD (Pesaran-Shin), in percent.

    Entry (i, j) is the share of market i's H-step forecast error variance
    caused by shocks to market j. Unlike the Cholesky FEVD it does not depend
    on variable ordering.
    """
    res = VAR(data).fit(lags)
    sigma = np.asarray(res.sigma_u)
    psi = res.ma_rep(horizon - 1)  # (H, n, n)
    num = np.zeros_like(sigma)
    den = np.zeros(sigma.shape[0])
    for a in psi:
        num += (a @ sigma) ** 2
        den += np.diag(a @ sigma @ a.T)
    theta = num / np.diag(sigma)[None, :] / den[:, None]
    theta = theta / theta.sum(axis=1, keepdims=True)
    return pd.DataFrame(theta * 100, index=data.columns, columns=data.columns)


def spillover_table(fevd: pd.DataFrame) -> dict:
    """Total, directional and net spillovers from a generalized FEVD table."""
    m = fevd.to_numpy()
    n = m.shape[0]
    off = m - np.diag(np.diag(m))
    from_others = pd.Series(off.sum(axis=1), index=fevd.index)
    to_others = pd.Series(off.sum(axis=0), index=fevd.columns)
    return {
        "total": off.sum() / n,
        "from": from_others,
        "to": to_others,
        "net": to_others - from_others,
    }


def rolling_total_spillover(data: pd.DataFrame, window: int = 200, step: int = 5,
                            lags: int = 2, horizon: int = 10) -> pd.Series:
    """Total spillover index over a rolling window (the classic DY time plot)."""
    out = {}
    for end in range(window, len(data) + 1, step):
        chunk = data.iloc[end - window:end]
        out[chunk.index[-1]] = spillover_table(generalized_fevd(chunk, lags, horizon))["total"]
    return pd.Series(out, name="total_spillover")


def spillover_adjacency(fevd: pd.DataFrame) -> np.ndarray:
    """Directed weight matrix A[i, j] = share of j's variance explained by i (diagonal removed)."""
    a = fevd.to_numpy().T / 100.0
    np.fill_diagonal(a, 0.0)
    return a
