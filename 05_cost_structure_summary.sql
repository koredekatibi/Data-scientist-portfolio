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
