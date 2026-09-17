[README.md](https://github.com/user-attachments/files/32317236/README.md)
# Revenue Forecasting & Financial Risk Analytics

**Korede Katibi, MBA** — Data Scientist portfolio project

A machine learning and SQL analysis of business-finance revenue forecasting and variance risk, built
on the kind of revenue guidance, forecast consolidation, and risk analysis work in my background at
USAA, Appen, and PSI LLC (see resume). This project translates that finance domain experience into a
Data Scientist skill set: Python, SQL, statistical modeling, and machine learning.

## Business question

Finance teams produce a monthly revenue forecast for each business entity. Two questions:
1. Can a data-driven model improve on that forecast, beyond what the finance team already produces?
2. Can we predict *which* entity-months are at risk of a material forecast miss, before the numbers close?

## Skills demonstrated

| Skill | Where it lives |
|---|---|
| Python (pandas, numpy) | Data generation, feature engineering, all analysis |
| SQL | `sql/` — 5 queries against a SQLite database: window functions (LAG, rolling average), CTEs, joins, aggregation |
| Exploratory Data Analysis | `notebooks/` Section 2 — trend, seasonality, correlation analysis |
| Regression modeling | Linear Regression vs Random Forest, benchmarked against a naive baseline |
| Classification | Random Forest classifier for high-variance-risk prediction, with class imbalance handling |
| Anomaly detection | Isolation Forest, unsupervised, on cost structure |
| Model evaluation | MAPE, R², precision/recall/F1, ROC-AUC, confusion matrix |
| Data visualization | matplotlib/seaborn — 8 charts covering EDA, model diagnostics, and results |
| Business communication | Executive summary translating model output into a stakeholder recommendation |

## Key results

- **Forecasting:** the finance team's own forecast is already fairly accurate (5.85% MAPE). A Linear
  Regression model with seasonality and lag features narrows this slightly to 5.80% MAPE. **Random
  Forest actually performed worse (6.56% MAPE)** — with only 175 training rows it overfit relative to
  the simpler model. I kept this result in rather than hiding it: the honest conclusion from limited
  data is to prefer the simpler model, not the more sophisticated one by default.
- **Risk classification:** a Random Forest classifier flags high-variance-risk entity-months with 75%
  precision and a 0.78 ROC-AUC, driven mainly by anomalous legal cost activity. Framed as a triage tool
  for the monthly variance review rather than a full replacement for analyst judgment.
- **Anomaly detection:** an unsupervised Isolation Forest check independently corroborates several of
  the same high-risk months and surfaces a possible structural cost difference in one business entity
  worth a follow-up review.

See the notebook's Section 6 (Executive Summary) for the full write-up.

## Repo contents

```
├── notebooks/
│   └── revenue_forecasting_analysis.ipynb   ← main deliverable: EDA, models, results, executive summary
├── data/
│   ├── revenue_forecast_data.csv            ← synthetic dataset (180 rows: 5 entities × 36 months)
│   └── revenue_forecast.db                  ← the same data in SQLite, used by the SQL queries below
├── sql/
│   ├── 01_monthly_variance_by_entity.sql
│   ├── 02_month_over_month_change.sql       ← window function (LAG)
│   ├── 03_high_risk_entities_ranked.sql     ← CTEs + join
│   ├── 04_rolling_3month_avg_variance.sql   ← window frame (rolling average)
│   └── 05_cost_structure_summary.sql
├── images/                                   ← chart PNGs, embedded in the notebook and referenced below
├── requirements.txt
└── README.md
```

## Charts

**Revenue trend across business entities, with visible seasonality and disruption events**
![Revenue by entity](images/revenue_by_entity.png)

**Forecast accuracy — how close the finance team's own forecast already is**
![Forecast vs actual](images/forecast_vs_actual_scatter.png)

**High-variance risk classifier — confusion matrix**
![Confusion matrix](images/risk_confusion_matrix.png)

**Anomaly detection on cost structure**
![Anomaly detection](images/anomaly_detection.png)

## How to run it

```bash
pip install -r requirements.txt
jupyter notebook notebooks/revenue_forecasting_analysis.ipynb
```

The notebook already contains saved outputs and charts, so it's fully readable on GitHub without
running anything. To reproduce or extend the analysis, run the cells top to bottom — the dataset and
SQLite database are both included in `data/`.

For the SQL queries, open `data/revenue_forecast.db` in any SQLite client (e.g., DB Browser for SQLite,
or `sqlite3 data/revenue_forecast.db` from the command line) and run any file in `sql/`.

## Honest technical notes

- **All data is synthetic**, generated to reflect realistic relationships (seasonality, trend, cost
  spikes correlated with revenue disruptions) common in corporate revenue/cost forecasting — not real
  employer data from USAA, Appen, or PSI LLC.
- The dataset is deliberately small (180 rows) to mirror a realistic monthly-close reporting cadence
  (a handful of business entities, a few years of history) rather than a large ML training set. This is
  exactly the data-volume constraint referenced in the Random Forest result above, and is called out
  explicitly in the notebook rather than glossed over.
- Model results (MAPE, ROC-AUC, etc.) are computed from a single train/test split for clarity; a
  production version would use time-series cross-validation given the temporal structure of the data.
