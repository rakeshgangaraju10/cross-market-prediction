"""Streamlit front-end. Every number shown is computed from the data in data/."""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from crossmarket.data import MARKETS, load_closes, log_returns, realized_volatility
from crossmarket.pipeline import RESULTS_DIR
from crossmarket.spillover import generalized_fevd, rolling_total_spillover, spillover_table

st.set_page_config(page_title="Cross-Market Prediction", page_icon="📈", layout="wide")


@st.cache_data
def load():
    closes = load_closes()
    returns = log_returns(closes)
    return closes, returns, realized_volatility(returns)


@st.cache_data
def fevd_for(start, end, lags, horizon):
    _, _, rv = load()
    return generalized_fevd(np.log(rv.loc[start:end]), lags, horizon)


@st.cache_data
def rolling_spill(window, lags, horizon):
    _, _, rv = load()
    return rolling_total_spillover(np.log(rv), window=window, step=5, lags=lags, horizon=horizon)


@st.cache_data
def load_results(results_dir=str(RESULTS_DIR)):
    d = Path(results_dir)
    if not (d / "metrics.csv").exists():
        return None
    return {
        "metrics": pd.read_csv(d / "metrics.csv"),
        "history": json.loads((d / "history.json").read_text()),
        "splits": json.loads((d / "splits.json").read_text()),
        "preds": {h: pd.read_csv(d / f"predictions_h{h}.csv", header=[0, 1], index_col=0,
                                 parse_dates=True)
                  for h in (1, 5, 10, 22) if (d / f"predictions_h{h}.csv").exists()},
    }


closes, returns, rv = load()
name = lambda t: MARKETS.get(t, t)  # noqa: E731

st.sidebar.title("📈 Cross-Market Prediction")
page = st.sidebar.radio("Page", ["Dashboard", "Data Analysis", "Spillover Analysis",
                                 "Forecasting Models", "About"])

if page == "Dashboard":
    st.title("Cross-market volatility spillovers")
    c = st.columns(4)
    c[0].metric("Markets", len(closes.columns))
    c[1].metric("Period", f"{closes.index[0]:%Y-%m} → {closes.index[-1]:%Y-%m}")
    c[2].metric("Trading days (avg)", f"{int(closes.notna().sum().mean()):,}")
    total = spillover_table(fevd_for(None, None, 2, 10))["total"]
    c[3].metric("Total spillover index", f"{total:.1f}%")

    latest = rv.iloc[-1].rename("Current 21d vol")
    longrun = rv.mean().rename("Long-run avg vol")
    tbl = pd.concat([latest, longrun], axis=1).rename(index=name)
    st.subheader("Annualized realized volatility (21-day)")
    st.dataframe(tbl.style.format("{:.1%}"), width="stretch")
    fig = px.line(rv.rename(columns=name), labels={"value": "Annualized vol", "variable": "Market"})
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, width="stretch")

elif page == "Data Analysis":
    st.title("Data analysis")
    sel = st.multiselect("Markets", list(closes.columns), default=list(closes.columns[:3]),
                         format_func=name)
    if sel:
        norm = closes[sel].ffill()
        norm = norm / norm.iloc[0] * 100
        st.plotly_chart(px.line(norm.rename(columns=name), title="Price, rebased to 100"),
                        width="stretch")
        r = returns[sel]
        stats = pd.DataFrame({
            "Trading days": r.notna().sum(),
            "Ann. return": r.mean() * 252,
            "Ann. vol": r.std() * np.sqrt(252),
            "Skew": r.skew(),
            "Kurtosis": r.kurt() + 3,
            "Worst day": r.min(),
            "Best day": r.max(),
        }).rename(index=name)
        st.dataframe(stats.style.format({"Trading days": "{:,.0f}", "Ann. return": "{:.1%}",
                                         "Ann. vol": "{:.1%}", "Skew": "{:.2f}",
                                         "Kurtosis": "{:.1f}", "Worst day": "{:.1%}",
                                         "Best day": "{:.1%}"}), width="stretch")
        st.subheader("Correlation of daily returns")
        corr = returns.corr().rename(index=name, columns=name)
        st.plotly_chart(px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r",
                                  zmin=-1, zmax=1), width="stretch")

elif page == "Spillover Analysis":
    st.title("Diebold–Yilmaz volatility spillovers")
    st.caption("Generalized FEVD of a VAR on log realized volatility (order-invariant). "
               "Row i, column j = % of market i's forecast-error variance due to shocks in j.")
    c = st.columns(3)
    lags = c[0].slider("VAR lags", 1, 5, 2)
    horizon = c[1].slider("Forecast horizon H", 5, 20, 10)
    years = rv.index.year
    yr = c[2].slider("Years", int(years.min()), int(years.max()),
                     (int(years.min()), int(years.max())))
    fevd = fevd_for(str(yr[0]), str(yr[1]), lags, horizon)
    tab = spillover_table(fevd)
    st.metric("Total spillover index", f"{tab['total']:.1f}%")
    lbl = fevd.rename(index=name, columns=name)
    st.plotly_chart(px.imshow(lbl, text_auto=".1f", color_continuous_scale="Blues",
                              labels={"x": "From (shock in)", "y": "To (variance of)"}),
                    width="stretch")
    d = pd.DataFrame({"To others": tab["to"], "From others": tab["from"],
                      "Net": tab["net"]}).rename(index=name)
    st.plotly_chart(px.bar(d["Net"].sort_values(), orientation="h",
                           title="Net spillover (transmitter > 0, receiver < 0)"),
                    width="stretch")
    st.subheader("Total spillover over time (rolling 200-day window)")
    st.plotly_chart(px.line(rolling_spill(200, lags, horizon)), width="stretch")

elif page == "Forecasting Models":
    st.title("Volatility forecasting models")
    res = load_results()
    if res is None:
        st.warning("No results yet. Run `python -m crossmarket.pipeline` to train the models.")
        st.stop()
    st.markdown(
        "Target: realized volatility over the **next h trading days** (no overlap with inputs). "
        "Inputs: HAR components (1/5/22-day vol) of all 8 markets. Chronological split; "
        "models selected on validation, scores below are on the untouched **test** set.")
    st.json(res["splits"]["1"], expanded=False)
    m = res["metrics"]
    metric = st.selectbox("Metric", ["QLIKE", "RMSE", "MAE", "MAPE", "R2_log"])
    summ = m[m.market == "All"].pivot(index="model", columns="horizon", values=metric)
    best = summ.idxmax() if metric == "R2_log" else summ.idxmin()
    st.dataframe(summ.style.format("{:.4f}").highlight_min(axis=0, color="#c6efce")
                 if metric != "R2_log" else
                 summ.style.format("{:.4f}").highlight_max(axis=0, color="#c6efce"),
                 width="stretch")
    st.caption("Best per horizon: " + ", ".join(f"{h}d → {b}" for h, b in best.items()))

    c = st.columns(2)
    h = c[0].selectbox("Horizon (days)", sorted(res["preds"]))
    mk = c[1].selectbox("Market", list(MARKETS), format_func=name)
    preds = res["preds"][h]
    models = [x for x in preds.columns.get_level_values(0).unique() if x != "Actual"]
    chosen = st.multiselect("Models", models, default=["HAR", "Spillover GNN"])
    fig = go.Figure()
    fig.add_scatter(x=preds.index, y=preds[("Actual", mk)], name="Actual", line_color="gray")
    for mod in chosen:
        fig.add_scatter(x=preds.index, y=preds[(mod, mk)], name=mod)
    fig.update_layout(title=f"{name(mk)} — {h}-day-ahead volatility (test set)",
                      yaxis_tickformat=".0%")
    st.plotly_chart(fig, width="stretch")

    per = m[(m.horizon == h) & (m.market != "All")].pivot(index="market", columns="model",
                                                           values=metric)
    st.dataframe(per.rename(index=name).style.format("{:.4f}"), width="stretch")

    hist = res["history"][str(h)]
    if hist:
        fig = go.Figure()
        for mod, hh in hist.items():
            fig.add_scatter(y=hh["train"], name=f"{mod} train")
            fig.add_scatter(y=hh["val"], name=f"{mod} val", line_dash="dash")
        fig.update_layout(title="Training curves (standardized MSE, early stopping)",
                          xaxis_title="Epoch")
        st.plotly_chart(fig, width="stretch")

else:
    st.title("About")
    st.markdown("""
**Question.** Do volatility shocks spread across global equity markets, and does modelling
that network improve volatility forecasts?

**Method.**
1. Daily log returns for 8 indices; holiday rows that were forward-filled in the raw data are
   removed so they don't inject artificial zero returns.
2. Diebold–Yilmaz (2012) spillover index from a generalized FEVD of a VAR on log volatility.
3. Forecast models, from simple to complex: random walk, HAR (Corsi 2009), cross-market HAR-X,
   MLP, and a graph network whose message passing is weighted by the training-period
   spillover network plus learned attention.
4. Evaluation with QLIKE (Patton 2011), which is robust to noise in the volatility proxy.

Source code: `crossmarket/` package. Data: Yahoo Finance via `yfinance`.
""")
