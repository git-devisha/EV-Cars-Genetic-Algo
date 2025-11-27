import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import seaborn as sns

# ---------- Paths ----------
DATA_PATH = "data/Global EV 2023.csv"
MODEL_PATH = "models/ev_demand_gb.pkl"
PREDICTION_OUTPUT = "data/ev_demand_predictions.csv"
FIG_DIR = "figures"

# ---------- Global EV categorical mapping ----------
CATEGORICAL_COLS = ["region", "category", "parameter", "mode", "powertrain", "unit"]
NUMERIC_COLS = ["year"]
TARGET_COL_RAW = "value"
TARGET_COL = "label_demand"


# ====================================================
# 1. LOAD DATA
# ====================================================
def load_dataset(path=DATA_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Dataset not found at {path}. Please place the CSV in /data/"
        )
    return pd.read_csv(path)


# ====================================================
# 2. PREPROCESS DATA
# ====================================================
def preprocess(df):
    df = df.copy()

    if "id" not in df.columns:
        df["id"] = np.arange(len(df))

    df = df.dropna(subset=[TARGET_COL_RAW]).copy()

    # One-hot encode categorical columns
    dummies = pd.get_dummies(df[CATEGORICAL_COLS], drop_first=True)
    df = pd.concat([df, dummies], axis=1)

    df[TARGET_COL] = df[TARGET_COL_RAW].astype(float)
    feature_cols = NUMERIC_COLS + list(dummies.columns)

    return df, feature_cols


# ====================================================
# 3. EXTRACT FEATURES + TARGET
# ====================================================
def get_feature_target(df, feature_cols):
    df_clean = df.dropna(subset=feature_cols + [TARGET_COL]).copy()
    X = df_clean[feature_cols].values
    y = df_clean[TARGET_COL].values
    return X, y, df_clean


# ====================================================
# 4. TRAIN MODEL
# ====================================================
def train_model(X, y):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s = scaler.transform(X_val)

    model = GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        subsample=0.8,
        random_state=42,
    )

    model.fit(X_train_s, y_train)

    # ---- Metrics (Python 3.13-safe) ----
    mse = mean_squared_error(y_val, model.predict(X_val_s))
    rmse = mse ** 0.5
    mae = mean_absolute_error(y_val, model.predict(X_val_s))
    r2 = r2_score(y_val, model.predict(X_val_s))

    metrics = {"RMSE": rmse, "MAE": mae, "R2": r2}

    return model, scaler, metrics, y_val, model.predict(X_val_s)


# ====================================================
# 5. SAVE MODEL
# ====================================================
def save_model(model, scaler):
    os.makedirs("models", exist_ok=True)
    dump({"model": model, "scaler": scaler}, MODEL_PATH)
    print(f"[INFO] Model saved to {MODEL_PATH}")


# ====================================================
# 6. CREATE FIGURES
# ====================================================
def plot_feature_importance(model, feature_cols):
    os.makedirs(FIG_DIR, exist_ok=True)

    importance = model.feature_importances_
    idx = np.argsort(importance)[-20:]  # Top 20 only for clean plots

    plt.figure(figsize=(8, 10))
    plt.barh(np.array(feature_cols)[idx], importance[idx])
    plt.title("Top 20 Feature Importances")
    plt.xlabel("Importance Score")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/feature_importance.png", dpi=300)
    plt.close()

    print("[INFO] Saved feature importance plot.")


def plot_predictions(y_true, y_pred):
    os.makedirs(FIG_DIR, exist_ok=True)

    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.xlabel("True Demand")
    plt.ylabel("Predicted Demand")
    plt.title("Predicted vs Actual Demand")
    plt.savefig(f"{FIG_DIR}/pred_vs_true.png", dpi=300)
    plt.close()

    print("[INFO] Saved Predicted vs True figure.")


def plot_residuals(y_true, y_pred):
    os.makedirs(FIG_DIR, exist_ok=True)

    residuals = y_true - y_pred
    plt.figure(figsize=(7, 5))
    plt.hist(residuals, bins=40)
    plt.title("Residual Distribution")
    plt.xlabel("Residual")
    plt.ylabel("Frequency")
    plt.savefig(f"{FIG_DIR}/residual_hist.png", dpi=300)
    plt.close()

    print("[INFO] Saved residual histogram.")


# ====================================================
# 7. PREDICT FOR ALL ROWS
# ====================================================
def predict_all(df, feature_cols, model, scaler):
    df_pred = df.dropna(subset=feature_cols).copy()
    X = scaler.transform(df_pred[feature_cols].values)

    df_pred["predicted_demand"] = model.predict(X)

    df_out = df.merge(df_pred[["id", "predicted_demand"]], on="id", how="left")
    df_out.to_csv(PREDICTION_OUTPUT, index=False)

    print(f"[INFO] Saved predictions to {PREDICTION_OUTPUT}")
    return df_out


# ====================================================
# Additional Research Analysis Functions
# ====================================================

import plotly.express as px
import os

FIG_DIR = "figures"   # keep same as in your script


def plot_world_ev_demand_map(df_with_predictions):
    """
    Creates an interactive world map (HTML) with bubble size = predicted demand.
    Requires columns: 'region', 'predicted_demand'.
    Output: figures/world_ev_demand_map.html
    """

    # Aggregate demand per region
    regional = (
        df_with_predictions
        .groupby("region", as_index=False)["predicted_demand"]
        .sum()
        .rename(columns={"predicted_demand": "total_demand"})
    )

    # Simple lat/lon for each region in Global EV dataset
    # Add/adjust as needed for your 'region' values
    region_coords = {
        "World": (20, 0),
        "China": (35, 103),
        "India": (22, 79),
        "Europe": (54, 15),
        "EU27": (52, 10),
        "USA": (39, -98),
        "Rest of the world": (0, -30),
        "Japan": (36, 138),
        "Korea": (36, 128),
        "Africa": (2, 21),
        "Latin America": (-15, -60),
        "Middle East": (25, 45),
        # add more if needed...
    }

    # Keep only regions that we have coordinates for
    regional = regional[regional["region"].isin(region_coords.keys())].copy()
    regional["lat"] = regional["region"].map(lambda r: region_coords[r][0])
    regional["lon"] = regional["region"].map(lambda r: region_coords[r][1])

    os.makedirs(FIG_DIR, exist_ok=True)

    fig = px.scatter_geo(
        regional,
        lat="lat",
        lon="lon",
        size="total_demand",
        color="total_demand",
        hover_name="region",
        projection="natural earth",
        title="Global EV Charging Demand (Predicted)",
    )

    fig.update_layout(
        margin=dict(l=0, r=0, t=40, b=0),
        coloraxis_colorbar_title="Predicted<br>Demand",
    )

    output_path = os.path.join(FIG_DIR, "world_ev_demand_map.html")
    fig.write_html(output_path)
    print(f"[INFO] Saved world EV demand map to {output_path}")

def analyze_regional_demand(df_with_predictions):
    """Creates a bar chart of top 10 regions by predicted demand."""
    regional_demand = df_with_predictions.groupby('region')['predicted_demand'].sum()
    regional_demand = regional_demand.sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    regional_demand.head(10).plot(kind='bar', color='orange')
    plt.title('Top 10 Regions by Predicted EV Demand')
    plt.ylabel('Total Predicted Demand')
    plt.xticks(rotation=45)
    plt.tight_layout()

    os.makedirs(FIG_DIR, exist_ok=True)
    plt.savefig(f"{FIG_DIR}/regional_demand.png", dpi=300)
    plt.close()

    print("[INFO] Saved regional demand bar chart.")
    return regional_demand


def forecast_future_demand(df, model, scaler, feature_cols, years_ahead=5):
    """Projects EV demand for the next N years using the trained model."""
    future_data = []
    current_year = df['year'].max()

    for year in range(current_year + 1, current_year + years_ahead + 1):
        temp = df.copy()
        temp['year'] = year
        future_data.append(temp)

    future_df = pd.concat(future_data, ignore_index=True)
    predicted_future = predict_all(future_df, feature_cols, model, scaler)

    print(f"[INFO] Generated {years_ahead}-year demand forecast.")
    return predicted_future


def charging_station_gap_analysis(df_with_predictions, existing_stations_data):
    """
    Compares predicted demand with existing charging infrastructure.
    existing_stations_data must contain: region, existing_stations
    """
    merged = df_with_predictions.merge(existing_stations_data, on='region', how='left')

    # Calculate a simple “EV-per-station” type metric
    merged['stations_per_ev'] = merged['existing_stations'] / merged['predicted_demand']

    # Assume 1 station needed per 100 EVs
    merged['station_gap'] = merged['predicted_demand'] * 0.01 - merged['existing_stations']

    print("[INFO] Completed station gap analysis.")
    return merged[['region', 'predicted_demand', 'existing_stations', 'station_gap']]


def create_priority_matrix(df_analysis):
    """Creates a bubble priority chart: color = station_gap, size = magnitude of gap, annotate top gaps."""
    os.makedirs(FIG_DIR, exist_ok=True)

    sns.set(style="whitegrid")
    demand_med = df_analysis["predicted_demand"].median()
    stations_med = df_analysis["existing_stations"].median()

    x = df_analysis["predicted_demand"].values
    y = df_analysis["existing_stations"].values
    gap = df_analysis.get("station_gap", pd.Series(np.zeros(len(df_analysis)))).values

    # Size by absolute gap (scale and clip for sensible bubble sizes)
    sizes = np.clip(np.abs(gap) * 0.2, 30, 600)

    plt.figure(figsize=(11, 8))
    sc = plt.scatter(
        x,
        y,
        c=gap,
        s=sizes,
        cmap="coolwarm",
        alpha=0.75,
        edgecolor="k",
        linewidth=0.5,
    )
    cbar = plt.colorbar(sc)
    cbar.set_label("Station gap (positive => need more stations)")

    # Median lines to form quadrants
    plt.axvline(demand_med, color="gray", linestyle="--", alpha=0.7)
    plt.axhline(stations_med, color="gray", linestyle="--", alpha=0.7)

    # Place readable quadrant labels based on axis limits
    xmin, xmax = plt.xlim()
    ymin, ymax = plt.ylim()
    xmid = demand_med
    ymid = stations_med

    plt.text((xmid + xmax) / 2, (ymid + ymax) / 2, "High Demand\nHigh Stations", ha="center", va="center", fontsize=9, color="green")
    plt.text((xmin + xmid) / 2, (ymid + ymax) / 2, "Low Demand\nHigh Stations", ha="center", va="center", fontsize=9, color="gray")
    plt.text((xmin + xmid) / 2, (ymin + ymid) / 2, "Low Demand\nLow Stations", ha="center", va="center", fontsize=9, color="gray")
    plt.text((xmid + xmax) / 2, (ymin + ymid) / 2, "High Demand\nLow Stations (Priority)", ha="center", va="center", fontsize=10, color="red", fontweight="bold")

    # Annotate top N regions by positive station gap (highest need)
    top_n = df_analysis.sort_values("station_gap", ascending=False).head(8)
    for _, row in top_n.iterrows():
        plt.annotate(
            row["region"],
            (row["predicted_demand"], row["existing_stations"]),
            textcoords="offset points",
            xytext=(6, -6),
            fontsize=8,
            alpha=0.9,
        )

    plt.xlabel("Predicted EV Demand")
    plt.ylabel("Existing Charging Stations")
    plt.title("EV Infrastructure Priority — Bubble chart (size=color = station gap)")
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/priority_planning_matrix.png", dpi=300)
    plt.close()

    print("[INFO] Saved priority planning matrix figure.")


def generate_planning_recommendations(df_analysis):
    """Generates actionable recommendations based on demand vs infrastructure."""
    high_priority = df_analysis[
        (df_analysis['predicted_demand'] > df_analysis['predicted_demand'].median()) &
        (df_analysis['existing_stations'] < df_analysis['existing_stations'].median())
    ]

    recommendations = []
    for _, row in high_priority.iterrows():
        rec = {
            'region': row['region'],
            'priority_level': 'High',
            'recommended_new_stations': max(0, round(row['station_gap'])),
            'reasoning': f"High demand ({row['predicted_demand']:.0f}) but insufficient existing stations"
        }
        recommendations.append(rec)

    print("[INFO] Generated EV infrastructure recommendations.")
    return pd.DataFrame(recommendations)


# ====================================================
# MAIN
# ====================================================
def main():
    print("\n[INFO] Loading dataset...")
    df_raw = load_dataset()

    print("[INFO] Preprocessing...")
    df_processed, feature_cols = preprocess(df_raw)

    print("[INFO] Extracting features/target...")
    X, y, df_clean = get_feature_target(df_processed, feature_cols)

    print("[INFO] Training model...")
    model, scaler, metrics, y_val, y_pred = train_model(X, y)

    print("\n--- Validation Metrics ---")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")

    save_model(model, scaler)

    print("[INFO] Generating graphs for research paper...")
    plot_feature_importance(model, feature_cols)
    plot_predictions(y_val, y_pred)
    plot_residuals(y_val, y_pred)

    print("[INFO] Predicting on entire dataset...")
    pred_df = predict_all(df_processed, feature_cols, model, scaler)

    print("[INFO] Running regional demand analysis...")
    regional_demand = analyze_regional_demand(pred_df)

    print("[INFO] Running future forecasting...")
    future_predictions = forecast_future_demand(df_processed, model, scaler, feature_cols)

    print("[INFO] Running station gap analysis...")
    existing_stations_data = pd.DataFrame({
    "region": df_processed["region"].unique(),
    "existing_stations": np.random.randint(50, 500, size=df_processed["region"].nunique())  
})
    gap_df = charging_station_gap_analysis(pred_df, existing_stations_data)

    print("[INFO] Creating priority matrix...")
    create_priority_matrix(gap_df)

    print("[INFO] Generating planning recommendations...")
    recommendations_df = generate_planning_recommendations(gap_df)
    recommendations_df.to_csv("data/ev_infra_recommendations.csv", index=False)

    print("[INFO] Creating world EV demand map...")
    plot_world_ev_demand_map(pred_df)

    print("\n All tasks completed successfully!")


if __name__ == "__main__":
    main()
