# Timeseries Forecast and Classification for Energy case

**Short description**

Energy demand forecasting is critical for balancing supply, reducing costs, and supporting renewable integration. This project demonstrates how advanced machine learning models can 
improve energy demand forecasting and pattern detection. Using Sweden’s SE_3 electricity market as a case, it compares traditional statistical methods with modern neural networks, 
enabling better decision-making for energy planning, operations, and market analysis. In more detail, the repository (in `timeseries_nn.ipynb`) demonstrates exploratory analysis, motif 
discovery, seasonal analysis, forecasting, and classification techniques for electricity load data. In a global picture, better forecasting could lead to lower operational costs and, with
classification, we can detect anomalies for outages or demand spikes. 

---

## Contents

The notebook (`timeseries_nn.ipynb`) is organized into the following major parts:

1. Load the dataset and check missing values
2. EDA & Feature Engineering (basic transforms, calendar features)
3. Stationarity checks & Decomposition (trend/seasonal/resid)
4. Matrix Profile & Motif discovery (via `wildboar` / MASS)
5. Seasonal analysis (hour/day/week/season summaries)
6. Forecasting experiments:

   * Evaluation helpers (MAE, RMSE, MAPE)
   * Time-based train/test split utilities
   * SARIMAX (classical statistical baseline)
   * N-BEATS (Darts wrapper)
   * LSTM (Keras / TensorFlow)
7. Time series classification with ROCKET
8. Model comparison and visualization of results

---

## Dataset

The notebook uses an Open Power System Data-derived dataset (hourly resolution). By default the notebook reads the CSV at:

```
data/time_series_60min_singleindex_filtered.csv
```

The notebook focuses on the `SE_3` bidding zone and expects columns such as (examples used in the notebook):

* `utc_timestamp` (datetime index)
* `SE_3_load_actual_entsoe_transparency` (actual load)
* `SE_3_price_day_ahead` (day-ahead price)
* `SE_3_wind_onshore_generation_actual` (wind generation)

> The notebook also contains a short description of the original data source (Open Power System Data) and the rationale for focusing on `SE_3`.

---

## Setup

The notebook requires a modern Python scientific stack. A basic `pip` install command used inside the notebook is:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn statsmodels tensorflow sktime wildboar numba darts
```

**Notes & suggestions:**

* `darts` may require additional packages (and a compatible version of `torch` or `tensorflow`) depending on the model backends you use — consult the official Darts installation guide if you run into issues.
* Installing `wildboar` and `numba` may require a working compiler toolchain on some systems.

---

## How to run

1. Create & activate a Python environment and install dependencies (see above).
2. Place the dataset CSV at `data/time_series_60min_singleindex_filtered.csv` or update the path in the notebook.
3. Open `timeseries_nn.ipynb` with Jupyter / JupyterLab and run the cells in order. It is recommended to run in [![open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/YOUR_GITHUB_USERNAME/REPO_NAME/blob/main/timeseries_nn.ipynb)
 for better performance and GPU usage.

**Quick tips**

* The notebook is built to be readable and educational; some hyperparameters (like epoch counts) are reduced for demo speed. For production experiments, increase 'epochs' in the appropriate `fit(...)` call if you want full training
* If you run into out-of-memory issues, reduce batch sizes or train on a smaller data window.
* Consider cross-validation strategies for time series (e.g., rolling origin), and parameter tuning for SARIMA. It is also possible to increase the number of the layers in LSTM.

---

## Outputs / Artifacts produced by the notebook

* `model_comparison_metrics.csv` — comparison table of MAE / RMSE / MAPE across models (saved by the notebook).
* Plots: decomposition, motif examples, seasonal heatmaps, forecasting plots and error bar charts.

---

## Key Results
- “N-BEATS reduced forecasting error (MAPE) by 12% vs. SARIMAX baseline.”-
- “LSTM captured seasonal consumption patterns missed by statistical models.”
- “ROCKET achieved 90% classification accuracy in detecting abnormal load patterns.”
