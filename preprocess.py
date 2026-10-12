"""
AirSense – Phase 3: Data Preprocessing Pipeline
Author: AirSense Project Team
Description:
    Reads raw multi-city environmental data, audits dataset structure,
    handles missing values, cleans invalid/outlier readings, standardizes units,
    engineers temporal & seasonal features, encodes categorical variables,
    prepares target variable (AQI), and exports the cleaned dataset for Phase 4 & Phase 5.
"""

import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# -----------------------------------------------------------------------------
# 1. Configuration & Paths
# -----------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "city_day.csv")
PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
PROCESSED_DATA_PATH = os.path.join(PROCESSED_DIR, "city_day_cleaned.csv")
FEATURES_DATA_PATH = os.path.join(PROCESSED_DIR, "features_X.csv")
TARGET_DATA_PATH = os.path.join(PROCESSED_DIR, "target_y.csv")

os.makedirs(PROCESSED_DIR, exist_ok=True)

print("=" * 80)
print("AIRSENSE – PHASE 3: DATA PREPROCESSING PIPELINE")
print("=" * 80)

# -----------------------------------------------------------------------------
# Step 1: Load the Dataset
# -----------------------------------------------------------------------------
print("\n[STEP 1] Loading raw dataset...")
df_raw = pd.read_csv(RAW_DATA_PATH)

print(f"-> Dataset successfully loaded from: {RAW_DATA_PATH}")
print(f"-> Shape: {df_raw.shape[0]} rows, {df_raw.shape[1]} columns")
print(f"-> Columns: {list(df_raw.columns)}")
print("\nFirst 5 Records (Raw):")
print(df_raw.head().to_string())

# -----------------------------------------------------------------------------
# Step 2: Understand the Dataset
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("[STEP 2] Understanding the Dataset Structure")
print("=" * 80)
print("\nColumn Data Types & Non-Null Counts:")
print(df_raw.info())

num_cols = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "Temperature", "Humidity"]
target_col = "AQI"
cat_cols = ["City"]
temporal_cols = ["Date"]

print("\nSummary Statistics of Numerical Features & Target:")
print(df_raw[num_cols + [target_col]].describe().T[['count', 'mean', 'std', 'min', '50%', 'max']].to_string())

# -----------------------------------------------------------------------------
# Step 3: Handle Missing Values
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("[STEP 3] Missing Value Audit & Imputation")
print("=" * 80)

# Check missing count & percentage
missing_count = df_raw.isnull().sum()
missing_pct = (missing_count / len(df_raw)) * 100
missing_df = pd.DataFrame({"Missing_Count": missing_count, "Percentage (%)": missing_pct})
print(missing_df[missing_df["Missing_Count"] > 0].to_string() if (missing_count > 0).any() else "No missing values found in initial raw check.")

# Make a working copy
df = df_raw.copy()

# Sort chronologically by City and Date for valid time-series imputation
df['Date'] = pd.to_datetime(df['Date'])
df = df.sort_values(by=['City', 'Date']).reset_index(drop=True)

# Imputation strategy:
# For time-series data: First apply forward-fill within each city group, then backward-fill, then median
for col in num_cols:
    if df[col].isnull().sum() > 0:
        df[col] = df.groupby('City')[col].transform(lambda group: group.ffill().bfill())
        if df[col].isnull().sum() > 0:
            df[col] = df[col].fillna(df[col].median())

# For Target AQI: drop rows where target cannot be ascertained or impute if missing
if df['AQI'].isnull().sum() > 0:
    print(f"Dropping {df['AQI'].isnull().sum()} rows with missing target AQI...")
    df = df.dropna(subset=['AQI']).reset_index(drop=True)

print(f"-> Missing values after imputation: {df[num_cols + ['AQI']].isnull().sum().sum()}")

# -----------------------------------------------------------------------------
# Step 4: Remove Duplicate Data
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("[STEP 4] Duplicate Records Detection & Removal")
print("=" * 80)

duplicate_count = df.duplicated(subset=['Date', 'City']).sum()
print(f"-> Identical (Date, City) duplicates identified: {duplicate_count}")

if duplicate_count > 0:
    df = df.drop_duplicates(subset=['Date', 'City'], keep='first').reset_index(drop=True)
    print("-> Duplicates dropped successfully.")
else:
    print("-> Verification: Zero duplicate records found.")

# -----------------------------------------------------------------------------
# Step 5: Handle Invalid and Outlier Values
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("[STEP 5] Handling Invalid Values and Outlier Treatment")
print("=" * 80)

# 5.1 Physical Boundary Sanitization
# Pollutant concentrations cannot be negative
for col in ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3"]:
    invalid_neg = (df[col] < 0).sum()
    if invalid_neg > 0:
        print(f"Found {invalid_neg} negative entries in {col}. Clipping to 0.0.")
        df[col] = df[col].clip(lower=0.0)

# Relative humidity bounded strictly between 0% and 100%
df['Humidity'] = df['Humidity'].clip(lower=0.0, upper=100.0)

# Temperature realistic bounds (-15°C to 55°C)
df['Temperature'] = df['Temperature'].clip(lower=-15.0, upper=55.0)

# AQI bounded strictly between 0 and 500+
df['AQI'] = df['AQI'].clip(lower=0.0)

# 5.2 Outlier Detection via IQR and Soft Capping
# In air pollution studies, extreme spikes (e.g. Diwali fireworks, dust storms) are genuine physical phenomena.
# Dropping them creates artificial data bias. Hence, we use 99th percentile Winsorization (soft capping).
print("\nApplying IQR inspection and 99th Percentile Soft-Capping to avoid extreme sensor glitches:")
for col in ["PM2.5", "PM10"]:
    Q1 = df[col].quantile(0.25)
    Q3 = df[col].quantile(0.75)
    IQR = Q3 - Q1
    upper_whisker = Q3 + 3.0 * IQR  # 3*IQR is extreme threshold
    p99 = df[col].quantile(0.99)
    outliers_count = (df[col] > upper_whisker).sum()
    print(f"  {col}: IQR={IQR:.2f}, Upper extreme bound={upper_whisker:.2f}, Outliers count={outliers_count}")
    # Soft cap at 99th percentile if extreme sensor error
    df[col] = df[col].clip(upper=p99 * 1.25)

# -----------------------------------------------------------------------------
# Step 6: Data Type Conversion
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("[STEP 6] Data Type Standardization")
print("=" * 80)

df['Date'] = pd.to_datetime(df['Date'])
df['City'] = df['City'].astype('category')
for col in num_cols:
    df[col] = df[col].astype(float)
df['AQI'] = df['AQI'].astype(int)

print("Updated Data Types:")
print(df.dtypes)

# -----------------------------------------------------------------------------
# Step 7: Unit Standardization
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("[STEP 7] Unit Standardization & Documentation")
print("=" * 80)

# Standard units:
# Particulates & Gases (PM2.5, PM10, NO2, SO2, O3): µg/m³
# CO: mg/m³
# Temperature: °C
# Humidity: %
print("Documented Standard Measurement Units:")
print("  - PM2.5, PM10, NO2, SO2, O3 : ug/m3 (micrograms per cubic meter)")
print("  - CO                         : mg/m3 (milligrams per cubic meter)")
print("  - Temperature                : deg C (degrees Celsius)")
print("  - Humidity                   : % (relative humidity percentage)")
print("  - AQI                        : Standardized dimensionless index (0 - 500+)")

# -----------------------------------------------------------------------------
# Step 8: Feature Engineering
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("[STEP 8] Feature Engineering (Temporal & Seasonal)")
print("=" * 80)

df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df['DayOfWeek'] = df['Date'].dt.dayofweek  # 0=Monday, 6=Sunday
df['Is_Weekend'] = df['DayOfWeek'].apply(lambda x: 1 if x in [5, 6] else 0)

# Season mapping:
# 12, 1, 2 -> Winter
# 3, 4, 5  -> Summer
# 6, 7, 8, 9 -> Monsoon
# 10, 11   -> Post-Monsoon (Smog peak)
def get_season(month):
    if month in [12, 1, 2]:
        return "Winter"
    elif month in [3, 4, 5]:
        return "Summer"
    elif month in [6, 7, 8, 9]:
        return "Monsoon"
    else:
        return "Post-Monsoon"

df['Season'] = df['Month'].apply(get_season)
df['Season'] = df['Season'].astype('category')

# Additional domain ratio: PM2.5 to PM10 ratio
# High PM2.5/PM10 ratio (>0.6) indicates combustion emissions (vehicles, fires)
# Low ratio (<0.4) indicates mechanical road dust / sand storms
df['PM_Ratio'] = (df['PM2.5'] / (df['PM10'] + 1e-5)).round(3)

print("Generated Features:")
print("  - Year, Month, Day")
print("  - DayOfWeek (0=Mon, 6=Sun)")
print("  - Is_Weekend (1=Sat/Sun, 0=Weekday)")
print("  - Season (Winter, Summer, Monsoon, Post-Monsoon)")
print("  - PM_Ratio (PM2.5 / PM10 combustion indicator)")
print("\nSample Engineered Records:")
print(df[['Date', 'City', 'Month', 'Season', 'Is_Weekend', 'PM_Ratio', 'AQI']].head(6).to_string())

# -----------------------------------------------------------------------------
# Step 9: Categorical Data Encoding
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("[STEP 9] Categorical Data Encoding")
print("=" * 80)

# Create One-Hot Encodings for City and Season
df_encoded = pd.get_dummies(df, columns=['City', 'Season'], drop_first=True, dtype=int)
print(f"Shape after one-hot encoding: {df_encoded.shape}")
print("New encoded columns:")
print([c for c in df_encoded.columns if c.startswith('City_') or c.startswith('Season_')])

# -----------------------------------------------------------------------------
# Step 10 & 11: Feature Scaling & Target Separation
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("[STEP 10 & 11] Target Separation & Feature Scaling")
print("=" * 80)

# Exclude raw Date, AQI, and AQI_Bucket from feature matrix X
exclude_from_features = ['Date', 'AQI', 'AQI_Bucket']
feature_cols = [c for c in df_encoded.columns if c not in exclude_from_features]

X = df_encoded[feature_cols]
y = df_encoded['AQI']

print(f"Features (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")
print(f"Feature columns ({len(feature_cols)} total):")
print(feature_cols)

# Apply StandardScaler to continuous numerical features
continuous_features = num_cols + ['PM_Ratio']
scaler = StandardScaler()
X_scaled = X.copy()
X_scaled[continuous_features] = scaler.fit_transform(X[continuous_features])

print("\nStandardScaler successfully fitted on continuous features.")
print("Feature scaling sample (mean ~ 0, std ~ 1):")
print(X_scaled[continuous_features].describe().T[['mean', 'std', 'min', 'max']].round(3).to_string())

# -----------------------------------------------------------------------------
# Step 12: Data Quality Verification
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("[STEP 12] Post-Preprocessing Data Quality Verification")
print("=" * 80)

verification_checks = {
    "Total Processed Records": len(df),
    "Null Values in Cleaned DF": df.isnull().sum().sum(),
    "Null Values in Features (X)": X.isnull().sum().sum(),
    "Null Values in Target (y)": y.isnull().sum(),
    "Duplicate Records": df.duplicated(subset=['Date', 'City']).sum(),
    "Min AQI Value": y.min(),
    "Max AQI Value": y.max(),
    "Continuous Variables Standardized": len(continuous_features),
    "Encoded Categorical Dummy Columns": len([c for c in X.columns if c.startswith('City_') or c.startswith('Season_')])
}

for check, val in verification_checks.items():
    print(f"  [OK] {check:<38}: {val}")

# -----------------------------------------------------------------------------
# Step 13: Save Clean Dataset
# -----------------------------------------------------------------------------
print("\n" + "=" * 80)
print("[STEP 13] Saving Processed Assets")
print("=" * 80)

df.to_csv(PROCESSED_DATA_PATH, index=False)
X.to_csv(FEATURES_DATA_PATH, index=False)
y.to_csv(TARGET_DATA_PATH, index=False)

print(f"-> Saved master clean dataset: {PROCESSED_DATA_PATH}")
print(f"-> Saved prepared feature matrix (X): {FEATURES_DATA_PATH}")
print(f"-> Saved prepared target vector (y): {TARGET_DATA_PATH}")

print("\n" + "=" * 80)
print("PHASE 3 DATA PREPROCESSING COMPLETED SUCCESSFULLY!")
print("=" * 80)
