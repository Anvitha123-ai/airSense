"""
AirSense – Phase 5: Machine Learning Model Development & Evaluation
Author: AirSense Project Team
Description:
    Trains, evaluates, and compares regression algorithms (Linear Regression,
    Decision Tree, Random Forest, Gradient Boosting) for Air Quality Index (AQI) prediction.
    Generates performance comparison tables, diagnostic plots, feature importance rankings,
    and serializes the best-performing model for web application integration.
"""

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# -----------------------------------------------------------------------------
# 1. Directories and Paths Setup
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "processed", "city_day_cleaned.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
FIGURES_DIR = os.path.join(BASE_DIR, "reports", "figures")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

print("=" * 80)
print("AIRSENSE – PHASE 5: MACHINE LEARNING MODEL DEVELOPMENT")
print("=" * 80)

# -----------------------------------------------------------------------------
# 2. Load the Prepared Dataset
# -----------------------------------------------------------------------------
print("\n[STEP 1] Loading cleaned dataset from Phase 3...")
df = pd.read_csv(DATA_PATH)
print(f"-> Dataset successfully loaded. Shape: {df.shape[0]} rows, {df.shape[1]} columns.")

# -----------------------------------------------------------------------------
# 3. Define Input Features (X) and Target Variable (y)
# -----------------------------------------------------------------------------
print("\n[STEP 2] Defining Input Predictors (X) and Target Variable (y)...")

# Base pollutants and weather parameters
base_features = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'Temperature', 'Humidity']
engineered_features = ['PM_Ratio', 'Month', 'DayOfWeek', 'Is_Weekend']

# Prepare modeling dataframe with One-Hot Encoded City to capture regional baselines
df_model = pd.get_dummies(df, columns=['City'], drop_first=True, dtype=int)
city_dummy_cols = [c for c in df_model.columns if c.startswith('City_')]

all_feature_cols = base_features + engineered_features + city_dummy_cols
target_col = 'AQI'

X = df_model[all_feature_cols]
y = df_model[target_col]

print(f"-> Total input features ({len(all_feature_cols)}):")
print(f"   Chemical & Weather: {base_features}")
print(f"   Domain & Temporal : {engineered_features}")
print(f"   Geographic Dummies: {city_dummy_cols}")
print(f"-> Target variable: {target_col} (Continuous numeric values, range: {y.min()} to {y.max()})")

# -----------------------------------------------------------------------------
# 4. Train-Test Split (80% Train, 20% Test)
# -----------------------------------------------------------------------------
print("\n[STEP 3] Splitting dataset into 80% Training and 20% Testing sets...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

print(f"-> Training set: {X_train.shape[0]} samples ({X_train.shape[1]} features)")
print(f"-> Testing set : {X_test.shape[0]} samples ({X_test.shape[1]} features)")

# -----------------------------------------------------------------------------
# 5. Define Machine Learning Algorithms
# -----------------------------------------------------------------------------
models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.08, max_depth=4, random_state=42)
}

# -----------------------------------------------------------------------------
# 6. Train Models and Evaluate Metrics
# -----------------------------------------------------------------------------
print("\n[STEP 4 & 5] Training models and computing evaluation metrics...")

results = []
predictions = {}

for name, model in models.items():
    # Fit model on training data only (Prevent Data Leakage)
    model.fit(X_train, y_train)
    
    # Predict on unseen test data
    y_pred = model.predict(X_test)
    predictions[name] = y_pred
    
    # Compute metrics
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    
    results.append({
        "Model": name,
        "MAE": round(mae, 2),
        "MSE": round(mse, 2),
        "RMSE": round(rmse, 2),
        "R2 Score": round(r2, 4)
    })

results_df = pd.DataFrame(results).sort_values(by="R2 Score", ascending=False).reset_index(drop=True)

print("\n" + "=" * 65)
print("             MODEL PERFORMANCE COMPARISON TABLE")
print("=" * 65)
print(results_df.to_string(index=False))
print("=" * 65)

# Select best model based on R2 Score and lowest RMSE
best_model_name = results_df.iloc[0]["Model"]
best_model = models[best_model_name]
print(f"\n-> Best Performing Model: {best_model_name}")
print(f"   (R2 = {results_df.iloc[0]['R2 Score']}, RMSE = {results_df.iloc[0]['RMSE']}, MAE = {results_df.iloc[0]['MAE']})")

# -----------------------------------------------------------------------------
# 7. Feature Importance Analysis
# -----------------------------------------------------------------------------
print("\n[STEP 6] Extracting Feature Importances for Tree Ensemble...")

rf_model = models["Random Forest"]
importances = rf_model.feature_importances_
feat_imp_df = pd.DataFrame({
    "Feature": all_feature_cols,
    "Importance": importances
}).sort_values(by="Importance", ascending=False).reset_index(drop=True)

print("\nTop 10 Feature Importances (Random Forest):")
for idx, row in feat_imp_df.head(10).iterrows():
    bar = "#" * int(row['Importance'] * 40)
    print(f"  {row['Feature']:<14} : {row['Importance']:.4f} ({row['Importance']*100:5.2f}%) {bar}")

# -----------------------------------------------------------------------------
# 8. Diagnostic Visualizations
# -----------------------------------------------------------------------------
print("\n[STEP 7] Generating Diagnostic Performance Visualizations...")
sns.set_theme(style="whitegrid")

# Plot 1: Model Comparison Bar Chart
plt.figure(figsize=(9, 5))
ax = sns.barplot(x="Model", y="R2 Score", data=results_df, palette="viridis")
plt.title("Model Comparison: R² Score on Test Dataset", fontsize=13, fontweight='bold')
plt.ylim(0, 1.05)
plt.ylabel("R² Score (Higher is Better)")
for p in ax.patches:
    ax.annotate(f"{p.get_height():.3f}", (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center', xytext=(0, 7), textcoords='offset points', fontweight='bold')
comp_path = os.path.join(FIGURES_DIR, "model_comparison.png")
plt.tight_layout()
plt.savefig(comp_path, dpi=200)
plt.close()
print(f"  [OK] Saved model comparison chart: {comp_path}")

# Plot 2: Actual vs Predicted AQI (Best Model)
plt.figure(figsize=(7, 6))
best_preds = predictions[best_model_name]
plt.scatter(y_test, best_preds, color="#2980b9", edgecolors='black', alpha=0.8, s=60, label="Predictions")
min_val = min(y_test.min(), best_preds.min())
max_val = max(y_test.max(), best_preds.max())
plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label="Ideal Fit (y = x)")
plt.title(f"Actual AQI vs Predicted AQI ({best_model_name})", fontsize=12, fontweight='bold')
plt.xlabel("Actual Ground Truth AQI", fontsize=10)
plt.ylabel("Predicted AQI by Model", fontsize=10)
plt.legend()
plt.tight_layout()
act_pred_path = os.path.join(FIGURES_DIR, "actual_vs_predicted.png")
plt.savefig(act_pred_path, dpi=200)
plt.close()
print(f"  [OK] Saved Actual vs Predicted plot: {act_pred_path}")

# Plot 3: Residuals / Error Distribution
plt.figure(figsize=(8, 4.5))
residuals = y_test - best_preds
sns.scatterplot(x=best_preds, y=residuals, color="#e67e22", edgecolors='black', s=50)
plt.axhline(0, color='red', linestyle='--', lw=2)
plt.title(f"Residual Plot (Errors vs Fitted Values) - {best_model_name}", fontsize=12, fontweight='bold')
plt.xlabel("Predicted AQI", fontsize=10)
plt.ylabel("Residual (Actual - Predicted)", fontsize=10)
plt.tight_layout()
res_path = os.path.join(FIGURES_DIR, "residual_plot.png")
plt.savefig(res_path, dpi=200)
plt.close()
print(f"  [OK] Saved Residuals plot: {res_path}")

# Plot 4: Feature Importance Bar Chart
plt.figure(figsize=(10, 6))
sns.barplot(x="Importance", y="Feature", data=feat_imp_df.head(10), palette="mako")
plt.title("Top 10 Feature Importances (Random Forest Regressor)", fontsize=13, fontweight='bold')
plt.xlabel("Relative Importance Weight", fontsize=10)
plt.tight_layout()
feat_path = os.path.join(FIGURES_DIR, "feature_importance.png")
plt.savefig(feat_path, dpi=200)
plt.close()
print(f"  [OK] Saved Feature Importance plot: {feat_path}")

# -----------------------------------------------------------------------------
# 9. Interactive Prediction Function with Standard AQI Categories
# -----------------------------------------------------------------------------
print("\n[STEP 8] Defining Standard Prediction Function...")

def classify_aqi(aqi_val):
    aqi_round = int(round(aqi_val))
    if aqi_round <= 50:
        return "Good", "Air quality is clean and poses minimal risk.", "#27ae60"
    elif aqi_round <= 100:
        return "Satisfactory", "Air quality is acceptable; minor discomfort to sensitive individuals.", "#2ecc71"
    elif aqi_round <= 200:
        return "Moderate", "Breathing discomfort possible to children, elderly, and lung patients.", "#f39c12"
    elif aqi_round <= 300:
        return "Poor", "Breathing discomfort to most people on prolonged exposure. Wear N95 mask.", "#e67e22"
    elif aqi_round <= 400:
        return "Very Poor", "Respiratory illness likely on prolonged exposure. Avoid outdoor activities.", "#e74c3c"
    else:
        return "Severe / Hazardous", "Emergency health hazard; high risk for entire population.", "#78281F"

def predict_air_quality(model, input_dict, feature_columns):
    """
    Accepts raw environmental parameters and predicts AQI and health tier.
    """
    # Calculate combustion ratio if not supplied
    if 'PM_Ratio' not in input_dict:
        input_dict['PM_Ratio'] = round(input_dict['PM2.5'] / (input_dict['PM10'] + 1e-5), 3)
    if 'Is_Weekend' not in input_dict:
        input_dict['Is_Weekend'] = 1 if input_dict.get('DayOfWeek', 0) in [5, 6] else 0
        
    # Align dictionary with trained feature columns
    row = []
    for col in feature_columns:
        row.append(input_dict.get(col, 0))
        
    input_df = pd.DataFrame([row], columns=feature_columns)
    predicted_val = float(model.predict(input_df)[0])
    category, advisory, color = classify_aqi(predicted_val)
    
    return {
        "Predicted_AQI": round(predicted_val, 1),
        "Category": category,
        "Color": color,
        "Health_Advisory": advisory
    }

# Test with user-specified sample input
sample_input = {
    'PM2.5': 75.0,
    'PM10': 120.0,
    'CO': 0.8,
    'NO2': 40.0,
    'SO2': 15.0,
    'O3': 50.0,
    'Temperature': 31.0,
    'Humidity': 70.0,
    'Month': 10,
    'DayOfWeek': 2,
    'City_Delhi': 1
}

test_result = predict_air_quality(best_model, sample_input, all_feature_cols)

print("\n--- Test Prediction with Sample Environmental Input ---")
print(f"Inputs: PM2.5=75, PM10=120, NO2=40, SO2=15, CO=0.8, O3=50, Temp=31 deg C, Humidity=70%")
print(f"Predicted AQI : {test_result['Predicted_AQI']}")
print(f"Air Category  : {test_result['Category']}")
print(f"Health Action : {test_result['Health_Advisory']}")

# -----------------------------------------------------------------------------
# 10. Save Model and Metadata Artifacts
# -----------------------------------------------------------------------------
print("\n[STEP 9] Serializing and Saving Model Artifacts...")

model_save_path = os.path.join(MODELS_DIR, "airsense_best_model.joblib")
features_save_path = os.path.join(MODELS_DIR, "feature_columns.joblib")

joblib.dump(best_model, model_save_path)
joblib.dump(all_feature_cols, features_save_path)

# Also save secondary models for benchmarking
joblib.dump(models["Linear Regression"], os.path.join(MODELS_DIR, "linear_regression.joblib"))
joblib.dump(models["Decision Tree"], os.path.join(MODELS_DIR, "decision_tree.joblib"))

print(f"-> Saved Best Model ({best_model_name}) to: {model_save_path}")
print(f"-> Saved Feature Columns Metadata to: {features_save_path}")

print("\n" + "=" * 80)
print("PHASE 5 MACHINE LEARNING MODEL DEVELOPMENT COMPLETED SUCCESSFULLY!")
print("=" * 80)
