import numpy as np
import pandas as pd
import pytest

from crossmarket.data import TICKERS, load_closes, load_ohlcv, log_returns, realized_volatility, trading_mask
from crossmarket.forecasting import HAR, HARX, build_dataset, forward_target, har_features, metrics
from crossmarket.spillover import generalized_fevd, spillover_table


@pytest.fixture(scope="module")
def returns():
    return log_returns(load_closes())


def test_load_all_markets():
    closes = load_closes()
    assert list(closes.columns) == TICKERS
    assert len(closes) > 3000


def test_forward_filled_holidays_are_masked():
    df = load_ohlcv("^GSPC")
    mask = trading_mask(df)
    # 2022-06-03 is a forward-filled copy of 2022-06-02 in the raw file.
    assert not mask.loc["2022-06-03"]
    assert mask.loc["2022-06-02"]
    assert 0.9 < mask.mean() < 1.0


def test_returns_have_no_fake_zeros(returns):
    assert (returns == 0).sum().sum() < 20


def test_realized_vol_is_plausible(returns):
    rv = realized_volatility(returns)
    assert ((rv.median() > 0.08) & (rv.median() < 0.35)).all()


def test_forward_target_uses_only_future(returns):
    t = forward_target(returns, 5)
    r = returns.iloc[:, 0]
    i = 100
    expected = 0.5 * np.log(252 * np.nanmean(r.iloc[i + 1:i + 6] ** 2))
    assert t.iloc[i, 0] == pytest.approx(expected)


def test_har_features_use_only_past(returns):
    f = har_features(returns)
    changed = returns.copy()
    changed.iloc[200:] *= 10
    g = har_features(changed)
    pd.testing.assert_frame_equal(f.iloc[:200], g.iloc[:200])


def test_splits_are_chronological_and_gapped(returns):
    h = 10
    ds = build_dataset(returns, h)
    tr, va, te = (ds.dates[ds.splits[k]] for k in ("train", "val", "test"))
    assert tr[-1] < va[0] < va[-1] < te[0]
    assert ds.splits["val"].start - ds.splits["train"].stop == h


def test_generalized_fevd_properties():
    rng = np.random.default_rng(0)
    data = pd.DataFrame(rng.standard_normal((500, 3)), columns=list("abc"))
    data["b"] += 0.8 * data["a"].shift(1).fillna(0)
    fevd = generalized_fevd(data)
    np.testing.assert_allclose(fevd.sum(axis=1), 100)
    # Order invariance (the reason to use the generalized FEVD).
    perm = generalized_fevd(data[["c", "b", "a"]]).loc[fevd.index, fevd.columns]
    np.testing.assert_allclose(fevd, perm, atol=1e-8)
    tab = spillover_table(fevd)
    assert tab["net"]["a"] > 0 > tab["net"]["b"]
    assert tab["net"].sum() == pytest.approx(0, abs=1e-9)


def test_linear_models_beat_naive(returns):
    ds = build_dataset(returns, 5)
    Xtr, ytr = ds.part("train")
    Xte, yte = ds.part("test")
    naive = metrics(yte, Xte[:, :, 2])["RMSE"]
    for m in (HAR(), HARX()):
        assert metrics(yte, m.fit(Xtr, ytr).predict(Xte))["RMSE"] < naive


def test_neural_models_train():
    torch = pytest.importorskip("torch")
    from crossmarket.forecasting import make_gnn, make_mlp, train_torch

    rng = np.random.default_rng(0)
    X = rng.standard_normal((64, 4, 3)).astype(np.float32)
    y = X[:, :, 0] * 0.5
    adj = np.full((4, 4), 0.1)
    np.fill_diagonal(adj, 0)
    for model in (make_mlp(4, 3), make_gnn(adj, 3)):
        model, hist = train_torch(model, X, y, X, y, epochs=30, patience=30)
        assert hist["val"][-1] < hist["val"][0]
        assert model(torch.tensor(X)).shape == (64, 4)
