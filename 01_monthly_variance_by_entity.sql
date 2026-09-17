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
