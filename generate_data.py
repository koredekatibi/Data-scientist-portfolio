import numpy as np
import pandas as pd

np.random.seed(42)

entities = ["Retail Banking", "Consumer Lending", "Insurance Services", "Investments", "Property & Casualty"]
months = pd.date_range("2022-01-01", "2024-12-01", freq="MS")

rows = []
for entity in entities:
    # each entity has its own baseline scale and growth/seasonality profile
    base_revenue = np.random.uniform(2_000_000, 6_000_000)
    growth_rate = np.random.uniform(0.002, 0.012)  # monthly growth
    seasonal_amp = np.random.uniform(0.03, 0.09)
    noise_scale = np.random.uniform(0.02, 0.05)

    base_freight = base_revenue * np.random.uniform(0.03, 0.06)
    base_legal = base_revenue * np.random.uniform(0.01, 0.025)
    base_material = base_revenue * np.random.uniform(0.08, 0.15)
    base_capital = base_revenue * np.random.uniform(0.02, 0.04)

    # each entity has a hidden seasonal bias the finance team's forecast formula
    # does NOT correct for (e.g. Q4 demand pull-forward) - this is what makes the
    # ML model able to beat a naive "use the forecast as-is" baseline: the bias
    # is systematic and learnable from Month_Num, not random noise.
    q4_bias = np.random.uniform(0.02, 0.05)

    for i, month in enumerate(months):
        seasonal = 1 + seasonal_amp * np.sin(2 * np.pi * (month.month / 12))
        trend = (1 + growth_rate) ** i

        forecast_revenue = base_revenue * trend * seasonal

        # systematic, learnable Q4 bias the forecast misses every year
        hidden_seasonal_bias = q4_bias if month.month in (10, 11, 12) else 0.0

        # disruption events: rare, but when they happen they show up BOTH as a
        # cost spike (legal/freight) AND a revenue shock in the same month -
        # this is the learnable signal the risk classifier picks up on.
        disruption = np.random.random() < 0.10
        disruption_shock = 0.0
        legal_spike_mult = 1.0
        freight_spike_mult = 1.0
        if disruption:
            disruption_shock = -np.random.uniform(0.10, 0.24)  # disruptions hurt revenue
            legal_spike_mult = np.random.uniform(1.5, 2.4)
            freight_spike_mult = np.random.uniform(1.3, 1.9)

        actual_revenue = forecast_revenue * (
            1 + hidden_seasonal_bias + np.random.normal(0, noise_scale) + disruption_shock
        )

        remarketing_revenue = actual_revenue * np.random.uniform(0.04, 0.09)
        freight_cost = base_freight * trend * freight_spike_mult * (1 + np.random.normal(0, 0.06))
        legal_cost = base_legal * trend * legal_spike_mult * (1 + np.random.normal(0, 0.10))
        material_cost = base_material * trend * (1 + np.random.normal(0, 0.08))
        capital_cost = base_capital * trend * (1 + np.random.normal(0, 0.05))
        rebates = actual_revenue * np.random.uniform(0.01, 0.03)
        deferred_revenue = actual_revenue * np.random.uniform(0.02, 0.06)

        gross_margin = actual_revenue - (freight_cost + legal_cost + material_cost + capital_cost + rebates)

        rows.append({
            "Month": month.strftime("%Y-%m-%d"),
            "Business_Entity": entity,
            "Forecast_Revenue": round(forecast_revenue, 2),
            "Actual_Revenue": round(actual_revenue, 2),
            "Remarketing_Revenue": round(remarketing_revenue, 2),
            "Freight_Cost": round(freight_cost, 2),
            "Legal_Cost": round(legal_cost, 2),
            "Material_Cost": round(material_cost, 2),
            "Capital_Cost": round(capital_cost, 2),
            "Rebates": round(rebates, 2),
            "Deferred_Revenue": round(deferred_revenue, 2),
            "Gross_Margin": round(gross_margin, 2),
            "Disruption_Event": int(disruption),
        })

df = pd.DataFrame(rows)
df["Variance_$"] = df["Actual_Revenue"] - df["Forecast_Revenue"]
df["Variance_%"] = df["Variance_$"] / df["Forecast_Revenue"]
df["High_Variance_Risk"] = (df["Variance_%"].abs() > 0.07).astype(int)
print("Risk label vs actual disruption event agreement:")
print(pd.crosstab(df["Disruption_Event"], df["High_Variance_Risk"]))

df.to_csv("/home/claude/ds_portfolio/data/revenue_forecast_data.csv", index=False)
print(df.shape)
print(df.head())
print("\nHigh variance risk rate:", df["High_Variance_Risk"].mean().round(3))
