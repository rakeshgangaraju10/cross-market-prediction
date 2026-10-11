# Cross-Market Volatility Prediction with a Spillover Graph Network

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://cross-market-prediction-tdrwdxdwvp43cmbx2wnf6h.streamlit.app/)

**Live app:** https://cross-market-prediction-tdrwdxdwvp43cmbx2wnf6h.streamlit.app/

Do volatility shocks spread across global equity markets, and does modelling that
network improve volatility forecasts? This project measures spillovers among 8 major
indices with the Diebold–Yilmaz framework and uses the resulting network inside a
graph neural network to forecast volatility 1, 5, 10 and 22 trading days ahead.

Markets: S&P 500, DAX, CAC 40, FTSE 100, Nifty 50, Nikkei 225, KOSPI, Hang Seng
(daily, Nov 2007 – Jun 2022, Yahoo Finance).

## Results (held-out test set, Jul 2019 – May 2022)

QLIKE loss, averaged over the 8 markets (lower is better):

| Model                  | 1d    | 5d    | 10d   | 22d   |
|------------------------|-------|-------|-------|-------|
| Naive (random walk)    | 1.507 | 0.639 | 0.567 | 0.638 |
| HAR                    | 1.416 | 0.513 | 0.409 | 0.420 |
| HAR-X (cross-market)   | 1.431 | 0.503 | 0.399 | **0.391** |
| MLP                    | 1.422 | 0.540 | 0.435 | 0.443 |
| **Spillover GNN**      | **1.391** | **0.479** | **0.384** | 0.408 |

* The spillover GNN is best at 1–10 days; the cross-market linear HAR-X is best at 22 days.
  Both beat the single-market HAR, i.e. information from other markets helps.
* Full-sample total spillover index is **52%**. Europe and the US are net transmitters
  (CAC 40 +29, FTSE +21, DAX +18, S&P 500 +15); Asian markets are net receivers.

Reproduce with `python -m crossmarket.pipeline` (~30 s on a laptop CPU).

## Method

1. **Data.** The raw CSVs were forward-filled onto a business-day calendar, so holidays
   appear as duplicated rows. These are detected and removed before computing returns;
   otherwise they inject fake zero returns and bias volatility down.
2. **Spillovers.** VAR on log 21-day realized volatility → generalized FEVD
   (Pesaran–Shin), which unlike a Cholesky FEVD does not depend on variable ordering →
   total, directional, net and rolling spillover indices (Diebold & Yilmaz, 2012).
3. **Forecasting set-up (leak-free).**
   * Features at day *t*: HAR components (log vol over the last 1, 5, 22 days) of all markets.
   * Target: realized volatility over days *t+1 … t+h*, which never overlaps the features.
   * Chronological 60/20/20 train/validation/test split with an *h*-day gap at each boundary;
     scaling and the spillover graph are fitted on training data only; the test set is used once.
4. **Models.** Random walk; HAR (Corsi, 2009); cross-market HAR-X (ridge); MLP; and the
   spillover GNN: multi-head graph attention whose attention logits are biased by the
   training-period spillover network, with per-market embeddings and a linear HAR-style
   skip path. Neural models use early stopping on validation loss. All fitted models get
   a training-set log-to-variance bias correction (fitting in logs targets the median).
5. **Evaluation.** QLIKE (robust to noise in the volatility proxy; Patton, 2011), plus
   MAE, RMSE, MAPE and R² on log volatility, overall and per market.

## Project layout

```
crossmarket/
  data.py          loading, holiday masking, returns, realized volatility, yfinance refresh
  spillover.py     generalized FEVD and Diebold–Yilmaz spillover measures
  forecasting.py   features/targets, baselines, MLP, spillover GNN, training, metrics
  pipeline.py      end-to-end run, writes results/
streamlit_app.py   web app (all figures computed from data/ and results/)
data/              daily OHLCV for the 8 indices
results/           metrics, test-set predictions, training curves
tests/             pytest suite (incl. look-ahead/leakage checks)
```

## Quick start

```bash
pip install -r requirements-dev.txt
python -m crossmarket.pipeline      # train + evaluate, writes results/
streamlit run streamlit_app.py       # or: python run_app.py
pytest -q && ruff check .
```

## Deploy to Streamlit Community Cloud

1. Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub.
2. Click **Create app** and choose this repository, branch `main`, main file `streamlit_app.py`.
3. Deploy. Streamlit installs `requirements.txt` and gives you a public `*.streamlit.app` URL;
   add it to the top of this README.

To refresh the data: `python -c "from crossmarket.data import download; download(end='2026-01-01')"`,
then rerun the pipeline.

## Changes from the original version

* The web app previously displayed random numbers (`np.random`) for statistics,
  spillovers, training curves and forecast metrics; every figure is now computed from real data.
* The original GNN fed a (time × market) matrix in as if each time step were a graph node,
  so the graph structure was not actually used; markets are now the nodes.
* The MLP produced 1 output for 8 targets (silently broadcast), and its evaluation used the
  last grid-search model instead of the best one.
* Multi-horizon metrics compared a 1-day forecast with the value *h* days later; each
  horizon now has its own target and models.
* Model selection and reporting both used the validation set; reporting now uses a separate test set.
* Removed `torch-geometric` (a dense implementation suffices for 8 nodes), the unused Flask
  template, stale/duplicate guides, and a stray `$null` file; added tests and CI.

## License

GPL-3.0 — see [LICENSE](LICENSE).
