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
