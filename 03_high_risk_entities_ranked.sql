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
