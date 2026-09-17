import sqlite3
import pandas as pd

df = pd.read_csv("/home/claude/ds_portfolio/data/revenue_forecast_data_scored.csv")

conn = sqlite3.connect("/home/claude/ds_portfolio/data/revenue_forecast.db")
df.to_sql("revenue_forecast", conn, if_exists="replace", index=False)

queries = {}

queries["01_monthly_variance_by_entity.sql"] = """
-- Monthly forecast variance ($ and %) by business entity, most recent 6 months first.
SELECT
    Business_Entity,
    Month,
    Forecast_Revenue,
    Actual_Revenue,
    ROUND(Actual_Revenue - Forecast_Revenue, 2)              AS Variance_Dollars,
    ROUND(100.0 * (Actual_Revenue - Forecast_Revenue) / Forecast_Revenue, 2) AS Variance_Pct
FROM revenue_forecast
ORDER BY Month DESC, Business_Entity;
"""

queries["02_month_over_month_change.sql"] = """
-- Month-over-month change in Actual_Revenue per entity, using a window function (LAG).
SELECT
    Business_Entity,
    Month,
    Actual_Revenue,
    LAG(Actual_Revenue) OVER (PARTITION BY Business_Entity ORDER BY Month) AS Prev_Month_Revenue,
    ROUND(
        100.0 * (Actual_Revenue - LAG(Actual_Revenue) OVER (PARTITION BY Business_Entity ORDER BY Month))
        / LAG(Actual_Revenue) OVER (PARTITION BY Business_Entity ORDER BY Month), 2
    ) AS MoM_Change_Pct
FROM revenue_forecast
ORDER BY Business_Entity, Month;
"""

queries["03_high_risk_entities_ranked.sql"] = """
-- Which business entities carry the most high-variance-risk months, ranked -
-- a CTE aggregation joined back to overall entity revenue for context.
WITH risk_counts AS (
    SELECT
        Business_Entity,
        COUNT(*)                       AS Total_Months,
        SUM(High_Variance_Risk)        AS High_Risk_Months,
        ROUND(1.0 * SUM(High_Variance_Risk) / COUNT(*), 3) AS High_Risk_Rate
    FROM revenue_forecast
    GROUP BY Business_Entity
),
entity_totals AS (
    SELECT
        Business_Entity,
        ROUND(SUM(Actual_Revenue), 2) AS Total_Actual_Revenue
    FROM revenue_forecast
    GROUP BY Business_Entity
)
SELECT
    r.Business_Entity,
    r.High_Risk_Months,
    r.Total_Months,
    r.High_Risk_Rate,
    e.Total_Actual_Revenue
FROM risk_counts r
JOIN entity_totals e ON r.Business_Entity = e.Business_Entity
ORDER BY r.High_Risk_Rate DESC;
"""

queries["04_rolling_3month_avg_variance.sql"] = """
-- 3-month rolling average of variance % per entity, using a window frame -
-- smooths out single-month noise to spot a sustained forecasting drift.
SELECT
    Business_Entity,
    Month,
    ROUND(100.0 * "Variance_%", 2) AS Variance_Pct,
    ROUND(
        100.0 * AVG("Variance_%") OVER (
            PARTITION BY Business_Entity
            ORDER BY Month
            ROWS BETWEEN 2 PRECEDING AND CURRENT ROW
        ), 2
    ) AS Rolling_3mo_Avg_Variance_Pct
FROM revenue_forecast
ORDER BY Business_Entity, Month;
"""

queries["05_cost_structure_summary.sql"] = """
-- Cost structure as a % of revenue by entity - the kind of summary that would
-- feed an executive gross-margin commentary deck.
SELECT
    Business_Entity,
    ROUND(SUM(Actual_Revenue), 2)                                    AS Total_Revenue,
    ROUND(SUM(Freight_Cost), 2)                                      AS Total_Freight,
    ROUND(SUM(Legal_Cost), 2)                                        AS Total_Legal,
    ROUND(SUM(Material_Cost), 2)                                     AS Total_Material,
    ROUND(SUM(Capital_Cost), 2)                                      AS Total_Capital,
    ROUND(100.0 * SUM(Freight_Cost) / SUM(Actual_Revenue), 2)        AS Freight_Pct_of_Rev,
    ROUND(100.0 * SUM(Legal_Cost) / SUM(Actual_Revenue), 2)          AS Legal_Pct_of_Rev,
    ROUND(100.0 * SUM(Gross_Margin) / SUM(Actual_Revenue), 2)        AS Gross_Margin_Pct
FROM revenue_forecast
GROUP BY Business_Entity
ORDER BY Gross_Margin_Pct DESC;
"""

# Write .sql files
import os
sql_dir = "/home/claude/ds_portfolio/sql"
for fname, q in queries.items():
    with open(os.path.join(sql_dir, fname), "w") as f:
        f.write(q.strip() + "\n")

# Also run each and save a sample of results as proof they work
print("Running each query against the SQLite database to verify correctness:\n")
for fname, q in queries.items():
    result = pd.read_sql_query(q, conn)
    print(f"--- {fname} ---")
    print(result.head(5).to_string(index=False))
    print(f"({len(result)} total rows)\n")

conn.close()
print("Saved: revenue_forecast.db + 5 .sql query files")
