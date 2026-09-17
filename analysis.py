import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, IsolationForest
from sklearn.metrics import (mean_absolute_error, mean_squared_error, r2_score,
                              classification_report, confusion_matrix, roc_auc_score)
from sklearn.preprocessing import LabelEncoder

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 110

df = pd.read_csv("/home/claude/ds_portfolio/data/revenue_forecast_data.csv", parse_dates=["Month"])
df = df.sort_values(["Business_Entity", "Month"]).reset_index(drop=True)

IMG = "/home/claude/ds_portfolio/images/"

# ---------------------------------------------------------------
# 1. EDA
# ---------------------------------------------------------------
print("=" * 70)
print("DATASET OVERVIEW")
print("=" * 70)
print(f"Shape: {df.shape}")
print(f"Date range: {df['Month'].min().date()} to {df['Month'].max().date()}")
print(f"Business entities: {df['Business_Entity'].nunique()}")
print()
print(df[["Forecast_Revenue", "Actual_Revenue", "Variance_%", "Gross_Margin"]].describe().round(2))

# Chart 1: Actual vs Forecast revenue over time, all entities
fig, ax = plt.subplots(figsize=(11, 5.5))
for entity in df["Business_Entity"].unique():
    sub = df[df["Business_Entity"] == entity]
    ax.plot(sub["Month"], sub["Actual_Revenue"], label=entity, linewidth=1.8)
ax.set_title("Actual Revenue by Business Entity, 2022\u20132024", fontsize=13, fontweight="bold")
ax.set_ylabel("Actual Revenue ($)")
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
ax.legend(fontsize=8, loc="upper left")
plt.tight_layout()
plt.savefig(IMG + "revenue_by_entity.png", bbox_inches="tight")
plt.close()

# Chart 2: Forecast vs Actual scatter with error
fig, ax = plt.subplots(figsize=(6.5, 6))
ax.scatter(df["Forecast_Revenue"], df["Actual_Revenue"], alpha=0.5, s=25, c="#2C6E9E")
lims = [df["Forecast_Revenue"].min() * 0.9, df["Forecast_Revenue"].max() * 1.05]
ax.plot(lims, lims, "r--", linewidth=1.3, label="Perfect forecast")
ax.set_xlabel("Forecast Revenue ($)")
ax.set_ylabel("Actual Revenue ($)")
ax.set_title("Forecast Accuracy: Actual vs Forecast Revenue", fontsize=12, fontweight="bold")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
ax.legend()
plt.tight_layout()
plt.savefig(IMG + "forecast_vs_actual_scatter.png", bbox_inches="tight")
plt.close()

# Chart 3: correlation heatmap of cost drivers vs variance
corr_cols = ["Variance_%", "Freight_Cost", "Legal_Cost", "Material_Cost", "Capital_Cost", "Rebates", "Deferred_Revenue", "Gross_Margin"]
corr = df[corr_cols].corr()
fig, ax = plt.subplots(figsize=(7.5, 6))
sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax, cbar_kws={"shrink": 0.8})
ax.set_title("Correlation: Cost Drivers vs Revenue Variance", fontsize=12, fontweight="bold")
plt.tight_layout()
plt.savefig(IMG + "correlation_heatmap.png", bbox_inches="tight")
plt.close()

print("\nSaved EDA charts: revenue_by_entity.png, forecast_vs_actual_scatter.png, correlation_heatmap.png")

# ---------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------
df["Month_Num"] = df["Month"].dt.month
df["Year"] = df["Month"].dt.year
df["Time_Index"] = (df["Year"] - df["Year"].min()) * 12 + df["Month_Num"]

le = LabelEncoder()
df["Entity_Code"] = le.fit_transform(df["Business_Entity"])

# lag features per entity (prior month actual revenue, prior variance)
df["Prev_Actual_Revenue"] = df.groupby("Business_Entity")["Actual_Revenue"].shift(1)
df["Prev_Variance_Pct"] = df.groupby("Business_Entity")["Variance_%"].shift(1)
df_model = df.dropna(subset=["Prev_Actual_Revenue", "Prev_Variance_Pct"]).reset_index(drop=True)

print(f"\nModeling dataset after lag features: {df_model.shape[0]} rows (first month per entity dropped)")

# ---------------------------------------------------------------
# 3. Revenue forecasting model: Linear Regression vs Random Forest vs naive baseline
# ---------------------------------------------------------------
features = ["Forecast_Revenue", "Month_Num", "Entity_Code", "Prev_Actual_Revenue", "Prev_Variance_Pct"]
X = df_model[features]
y = df_model["Actual_Revenue"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# naive baseline: just use the finance team's own forecast as the prediction
naive_pred = X_test["Forecast_Revenue"]
naive_mae = mean_absolute_error(y_test, naive_pred)
naive_mape = (np.abs((y_test - naive_pred) / y_test)).mean() * 100

lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_mae = mean_absolute_error(y_test, lr_pred)
lr_r2 = r2_score(y_test, lr_pred)
lr_mape = (np.abs((y_test - lr_pred) / y_test)).mean() * 100

rf = RandomForestRegressor(n_estimators=300, max_depth=6, random_state=42)
rf.fit(X_train, y_train)
rf_pred = rf.predict(X_test)
rf_mae = mean_absolute_error(y_test, rf_pred)
rf_r2 = r2_score(y_test, rf_pred)
rf_mape = (np.abs((y_test - rf_pred) / y_test)).mean() * 100

print("\n" + "=" * 70)
print("REVENUE FORECASTING MODEL COMPARISON (held-out test set)")
print("=" * 70)
results = pd.DataFrame({
    "Model": ["Naive (finance forecast as-is)", "Linear Regression", "Random Forest"],
    "MAE ($)": [naive_mae, lr_mae, rf_mae],
    "MAPE (%)": [naive_mape, lr_mape, rf_mape],
    "R2": [np.nan, lr_r2, rf_r2],
})
print(results.round(2).to_string(index=False))

# Pick whichever model actually beats the naive baseline on the test set -
# don't assume the more complex model wins.
candidates = {"Linear Regression": (lr_mape, lr_pred), "Random Forest": (rf_mape, rf_pred)}
best_model_name, (best_mape, best_pred) = min(candidates.items(), key=lambda kv: kv[1][0])
improvement = (1 - best_mape / naive_mape) * 100

print(f"\nBest model: {best_model_name} (MAPE {best_mape:.2f}% vs naive {naive_mape:.2f}%)")
print(f"{best_model_name} reduces forecast MAPE by {improvement:.1f}% vs the naive baseline.")
if rf_mape > lr_mape:
    print("Note: Random Forest overfit relative to Linear Regression on this dataset size "
          f"(175 rows) - RF test MAPE ({rf_mape:.2f}%) was worse than LR ({lr_mape:.2f}%). "
          "This is a real and common result with limited data, and the honest conclusion "
          "is to prefer the simpler model rather than default to the more complex one.")

# feature importance chart
importances = pd.Series(rf.feature_importances_, index=features).sort_values()
fig, ax = plt.subplots(figsize=(7, 4.2))
importances.plot(kind="barh", ax=ax, color="#2C6E9E")
ax.set_title("Random Forest Feature Importance \u2014 Revenue Forecast", fontsize=12, fontweight="bold")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig(IMG + "feature_importance_forecast.png", bbox_inches="tight")
plt.close()

# actual vs predicted chart (best model)
fig, ax = plt.subplots(figsize=(6.5, 6))
ax.scatter(y_test, best_pred, alpha=0.6, s=30, c="#2C6E9E", label=best_model_name)
lims = [y_test.min() * 0.9, y_test.max() * 1.05]
ax.plot(lims, lims, "r--", linewidth=1.3, label="Perfect prediction")
ax.set_xlabel("Actual Revenue ($)")
ax.set_ylabel("Predicted Revenue ($)")
ax.set_title(f"{best_model_name}: Predicted vs Actual Revenue (test set)", fontsize=12, fontweight="bold")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
ax.legend()
plt.tight_layout()
plt.savefig(IMG + "rf_predicted_vs_actual.png", bbox_inches="tight")
plt.close()

print(f"Saved model charts: feature_importance_forecast.png, rf_predicted_vs_actual.png ({best_model_name})")

# ---------------------------------------------------------------
# 4. High-variance risk classifier
# ---------------------------------------------------------------
clf_features = ["Freight_Cost", "Legal_Cost", "Material_Cost", "Capital_Cost", "Rebates",
                 "Deferred_Revenue", "Month_Num", "Entity_Code", "Prev_Variance_Pct"]
Xc = df_model[clf_features]
yc = df_model["High_Variance_Risk"]

Xc_train, Xc_test, yc_train, yc_test = train_test_split(Xc, yc, test_size=0.25, random_state=42, stratify=yc)

clf = RandomForestClassifier(n_estimators=300, max_depth=5, class_weight="balanced", random_state=42)
clf.fit(Xc_train, yc_train)
yc_pred = clf.predict(Xc_test)
yc_proba = clf.predict_proba(Xc_test)[:, 1]

print("\n" + "=" * 70)
print("HIGH-VARIANCE RISK CLASSIFIER")
print("=" * 70)
print(f"Positive class rate in full dataset: {yc.mean():.1%}")
print("\nClassification report (test set):")
print(classification_report(yc_test, yc_pred, target_names=["Normal", "High Variance Risk"]))
try:
    auc = roc_auc_score(yc_test, yc_proba)
    print(f"ROC-AUC: {auc:.3f}")
except ValueError:
    auc = None
    print("ROC-AUC: undefined (test set has only one class)")

cm = confusion_matrix(yc_test, yc_pred)
fig, ax = plt.subplots(figsize=(5, 4.5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
            xticklabels=["Normal", "High Risk"], yticklabels=["Normal", "High Risk"], ax=ax)
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title("Confusion Matrix \u2014 High-Variance Risk Classifier", fontsize=11, fontweight="bold")
plt.tight_layout()
plt.savefig(IMG + "risk_confusion_matrix.png", bbox_inches="tight")
plt.close()

clf_importances = pd.Series(clf.feature_importances_, index=clf_features).sort_values()
fig, ax = plt.subplots(figsize=(7, 4.5))
clf_importances.plot(kind="barh", ax=ax, color="#C0392B")
ax.set_title("Feature Importance \u2014 High-Variance Risk Classifier", fontsize=12, fontweight="bold")
ax.set_xlabel("Importance")
plt.tight_layout()
plt.savefig(IMG + "feature_importance_risk.png", bbox_inches="tight")
plt.close()

print("\nSaved classifier charts: risk_confusion_matrix.png, feature_importance_risk.png")

# ---------------------------------------------------------------
# 5. Anomaly detection (Isolation Forest) on cost structure
# ---------------------------------------------------------------
anomaly_features = ["Freight_Cost", "Legal_Cost", "Material_Cost", "Capital_Cost", "Rebates", "Gross_Margin"]
iso = IsolationForest(contamination=0.06, random_state=42)
df["Anomaly_Flag"] = iso.fit_predict(df[anomaly_features])  # -1 = anomaly
n_anomalies = (df["Anomaly_Flag"] == -1).sum()

print("\n" + "=" * 70)
print("ANOMALY DETECTION (Isolation Forest on cost structure)")
print("=" * 70)
print(f"Flagged {n_anomalies} of {len(df)} claims/months ({n_anomalies/len(df):.1%}) as cost-structure anomalies.")
top_anomalies = df[df["Anomaly_Flag"] == -1][["Month", "Business_Entity", "Gross_Margin", "Variance_%"]].sort_values("Variance_%", key=abs, ascending=False)
print("\nTop anomalies by |variance|:")
print(top_anomalies.head(8).to_string(index=False))

fig, ax = plt.subplots(figsize=(9, 5.5))
normal = df[df["Anomaly_Flag"] == 1]
anomaly = df[df["Anomaly_Flag"] == -1]
ax.scatter(normal["Gross_Margin"], normal["Variance_%"], alpha=0.5, s=25, c="#2C6E9E", label="Normal")
ax.scatter(anomaly["Gross_Margin"], anomaly["Variance_%"], alpha=0.9, s=60, c="#C0392B", marker="X", label="Anomaly")
ax.set_xlabel("Gross Margin ($)")
ax.set_ylabel("Revenue Variance (%)")
ax.set_title("Isolation Forest \u2014 Cost Structure Anomalies", fontsize=12, fontweight="bold")
ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e6:.1f}M"))
ax.legend()
plt.tight_layout()
plt.savefig(IMG + "anomaly_detection.png", bbox_inches="tight")
plt.close()

print("\nSaved chart: anomaly_detection.png")

# save results for reuse in notebook build script
import json
summary = {
    "naive_mae": float(naive_mae), "naive_mape": float(naive_mape),
    "lr_mae": float(lr_mae), "lr_mape": float(lr_mape), "lr_r2": float(lr_r2),
    "rf_mae": float(rf_mae), "rf_mape": float(rf_mape), "rf_r2": float(rf_r2),
    "best_model_name": best_model_name,
    "improvement_pct": float(improvement),
    "risk_positive_rate": float(yc.mean()),
    "roc_auc": float(auc) if auc else None,
    "n_anomalies": int(n_anomalies),
    "n_total": int(len(df)),
}
with open("/home/claude/ds_portfolio/src/summary_stats.json", "w") as f:
    json.dump(summary, f, indent=2)

df.to_csv("/home/claude/ds_portfolio/data/revenue_forecast_data_scored.csv", index=False)
print("\nDone.")
