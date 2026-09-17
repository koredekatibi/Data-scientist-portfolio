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
