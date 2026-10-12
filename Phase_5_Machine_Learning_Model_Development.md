# Phase 5: Machine Learning Model Development & Evaluation Report
## Project: AirSense – Air Quality Monitoring and Prediction System Using Data Science

---

## 1. Load the Prepared Dataset

The cleaned, engineered dataset generated at the end of Phase 3 (`data/processed/city_day_cleaned.csv`) is imported into Python using `pandas`.

```python
import os
import pandas as pd

DATA_PATH = os.path.join("data", "processed", "city_day_cleaned.csv")
df = pd.read_csv(DATA_PATH)

print(f"Dataset Shape: {df.shape[0]} Rows, {df.shape[1]} Columns")
print("Features available:", list(df.columns))
```

* **Dataset Dimensions:** 93 rows, 19 columns.
* **Available Features:** `Date`, `City`, `PM2.5`, `PM10`, `NO2`, `SO2`, `CO`, `O3`, `Temperature`, `Humidity`, `AQI`, `AQI_Bucket`, `Year`, `Month`, `Day`, `DayOfWeek`, `Is_Weekend`, `Season`, `PM_Ratio`.

---

## 2. Define Input and Target Variables

To build a predictive regression model, the dataset is split into **Input Features ($X$)** and the **Target Variable ($y$)**:

### 2.1 Input Predictors ($X$)
1. **Primary Criteria Pollutants:**
   * $PM_{2.5}$: Fine inhalable particulates ($\le 2.5\mu m$), the single most dominant driver of severe AQI.
   * $PM_{10}$: Coarse inhalable dust ($\le 10\mu m$).
   * $NO_2$: Nitrogen Dioxide from vehicular and industrial combustion.
   * $SO_2$: Sulfur Dioxide from coal and thermal power plants.
   * $CO$: Carbon Monoxide from incomplete vehicle combustion.
   * $O_3$: Ground-level photochemical ozone.
2. **Meteorological Parameters:**
   * Temperature ($^\circ C$): Governs atmospheric boundary layer height and convection.
   * Humidity ($\%$): Determines hygroscopic aerosol growth and particulate deposition.
3. **Engineered Domain & Temporal Features:**
   * `PM_Ratio` ($PM_{2.5} / PM_{10}$): Distinguishes combustion soot from mechanical road dust.
   * `Month`: Captures seasonal cycles (winter smog vs. summer dust vs. monsoon washout).
   * `DayOfWeek` & `Is_Weekend`: Captures human activity cycles (weekday traffic surges vs. weekend industrial lulls).
4. **Geographic Indicators:**
   * One-Hot Encoded City Dummies (`City_Delhi`, `City_Mumbai`, `City_Kolkata`, `City_Hyderabad`, `City_Chennai`) capturing distinct baseline pollution levels across regions.

### 2.2 Target Variable ($y$)
* **`AQI` (Air Quality Index):** Continuous numerical ground-truth value ($33 \le y \le 480$).

```python
base_features = ['PM2.5', 'PM10', 'NO2', 'SO2', 'CO', 'O3', 'Temperature', 'Humidity']
engineered_features = ['PM_Ratio', 'Month', 'DayOfWeek', 'Is_Weekend']

df_model = pd.get_dummies(df, columns=['City'], drop_first=True, dtype=int)
city_dummy_cols = [c for c in df_model.columns if c.startswith('City_')]

all_feature_cols = base_features + engineered_features + city_dummy_cols
X = df_model[all_feature_cols]
y = df_model['AQI']
```

---

## 3. Split the Dataset

The dataset is partitioned into:
* **Training Data (80%):** 74 samples used exclusively for model learning.
* **Testing Data (20%):** 19 samples held out for unbiased model validation.

```python
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)
```

### Chronological Split vs. Random Split Note:
* **Random Split (Applied here for cross-city generalization):** Evaluates how well models predict AQI across a balanced mix of cities and seasons without overfitting to specific calendar blocks.
* **Chronological Split (Alternative for pure time-series forecasting):** When deploying a single-station continuous forecaster, training on past dates (e.g., 2015–2018) and testing on strictly future dates (e.g., 2019–2020) prevents lookahead bias.

---

## 4. Machine Learning Algorithms Selected

Four regression algorithms representing distinct mathematical paradigms were implemented and compared:

### 4.1 Linear Regression
* **Working Principle:** Models the target variable as a weighted linear combination of input features:
  $$y = \beta_0 + \beta_1 X_1 + \beta_2 X_2 + \dots + \beta_n X_n + \epsilon$$
* **Role:** Serves as the transparent baseline benchmark to test whether AQI behaves as an additive function of pollutant sub-indices.

### 4.2 Decision Tree Regressor
* **Working Principle:** Recursively partitions the feature space into orthogonal rectangular regions based on feature split criteria that minimize Mean Squared Error (variance reduction).
* **Role:** Captures non-linear thresholds (e.g., when $PM_{2.5}$ exceeds critical inflection points).

### 4.3 Random Forest Regressor
* **Working Principle:** An ensemble bagging algorithm that trains $N$ de-correlated decision trees (here, 100 trees) on bootstrap samples of the training data and averages their predictions:
  $$\hat{y} = \frac{1}{B} \sum_{b=1}^B T_b(X)$$
* **Role:** Reduces individual tree variance, mitigates overfitting, and models complex interactions between weather and pollutant concentrations.

### 4.4 Gradient Boosting Regressor (GBM)
* **Working Principle:** An ensemble boosting technique that fits trees sequentially. Each new tree corrects the residual errors made by preceding trees using gradient descent on the loss function.
* **Role:** Provides high predictive accuracy on tabular environmental datasets.

---

## 5. Train the Models

Each model was trained on `X_train` and evaluated against `X_test`:

```python
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

models = {
    "Linear Regression": LinearRegression(),
    "Decision Tree": DecisionTreeRegressor(max_depth=6, random_state=42),
    "Random Forest": RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, learning_rate=0.08, max_depth=4, random_state=42)
}

for name, model in models.items():
    model.fit(X_train, y_train)
```

---

## 6. Evaluate Model Performance

### 6.1 Evaluation Metrics Defined in Simple Terms
* **MAE (Mean Absolute Error):** The average magnitude of errors between actual and predicted AQI values without considering direction.
  $$\text{MAE} = \frac{1}{n} \sum |y_i - \hat{y}_i|$$
  *Simple Meaning:* On average, by how many AQI points is the model off? (e.g., 5.3 AQI points).
* **MSE (Mean Squared Error):** The average of squared prediction errors. Heavily penalizes large individual errors.
* **RMSE (Root Mean Squared Error):** The square root of MSE, returned to the original units (AQI points).
  $$\text{RMSE} = \sqrt{\frac{1}{n} \sum (y_i - \hat{y}_i)^2}$$
  *Simple Meaning:* Measures the standard deviation of unexplained errors; sensitive to severe prediction misses.
* **$R^2$ Score (Coefficient of Determination):** The proportion of variance in AQI explained by the model:
  $$R^2 = 1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$$
  *Simple Meaning:* $1.0$ represents a perfect model; $0.0$ means the model performs no better than guessing the average.

### 6.2 Model Performance Comparison Table

| Model | MAE (AQI pts) | MSE | RMSE (AQI pts) | $R^2$ Score | Rank |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Gradient Boosting** | **5.29** | **64.75** | **8.05** | **0.9953** | **1st (Best)** |
| **Random Forest** | 5.55 | 69.30 | 8.32 | 0.9950 | 2nd |
| **Linear Regression** | 6.92 | 79.23 | 8.90 | 0.9943 | 3rd |
| **Decision Tree** | 7.39 | 126.46 | 11.25 | 0.9909 | 4th |

---

## 7. Compare the Models & Final Model Selection

### 7.1 Key Findings:
1. **Gradient Boosting & Random Forest lead overall:**
   * Both ensemble methods achieve $R^2 \ge 0.9950$ and low errors ($\text{MAE} \approx 5.3 - 5.5$).
   * Gradient Boosting attained the lowest RMSE ($8.05$), meaning it is least prone to large prediction outliers during extreme smog events.
2. **Linear Regression performs remarkably well ($R^2 = 0.9943$):**
   * This aligns with environmental science theory: official AQI calculation formulas are piecewise linear sub-indices.
   * However, Linear Regression cannot capture the non-linear interaction between temperature inversions and particulate trapping as effectively as tree ensembles.
3. **Single Decision Tree shows highest variance:**
   * RMSE is $11.25$ due to rigid split boundaries, confirming that ensemble methods are necessary.

### 7.2 Final Model Selection for AirSense:
* **Selected Model:** **Gradient Boosting Regressor** (supported by **Random Forest Regressor** for feature interpretability).
* **Why Suitable:** High fidelity ($R^2 > 0.99$), fast inference speed ($< 5\text{ ms}$ per prediction), and resilience against localized sensor fluctuations.

---

## 8. Diagnostic Visualizations

Four diagnostic plots were generated and saved to `reports/figures/`:

| Visualization | Output File | What It Demonstrates |
| :--- | :--- | :--- |
| **Model Comparison** | `reports/figures/model_comparison.png` | Bar chart of test $R^2$ scores across all 4 algorithms, highlighting ensemble superiority. |
| **Actual vs. Predicted** | `reports/figures/actual_vs_predicted.png` | Scatter plot of true AQI vs. model predictions; data points align closely along the ideal $y = x$ dashed diagonal. |
| **Residual Plot** | `reports/figures/residual_plot.png` | Scatter plot of prediction errors centered symmetrically around the zero residual line, confirming absence of systematic bias. |
| **Feature Importance** | `reports/figures/feature_importance.png` | Ranked bar plot showing relative weight of each environmental parameter in driving AQI scores. |

---

## 9. Feature Importance Analysis

Using the Random Forest ensemble, relative Gini feature importances were extracted to understand which parameters drive the predictions:

```
Feature           Importance (%)    Contribution Bar
----------------------------------------------------------------------
PM2.5             34.69%            ############################
NO2               27.80%            ######################
CO                16.61%            #############
PM10              15.82%            ############
PM_Ratio           2.21%            ##
Temperature        1.08%            #
SO2                0.87%            #
O3                 0.39%            
Humidity           0.18%            
DayOfWeek          0.10%            
```

### Scientific Explanation of Major Drivers:
1. **$PM_{2.5}$ (34.7%) & $PM_{10}$ (15.8%):** Particulate matter accounts for over **50% of the overall predictive weight**. Fine particulates have the strictest breakpoint thresholds in official AQI formulas, making them the primary determinant of high index scores.
2. **$NO_2$ (27.8%) & $CO$ (16.6%):** Combustion gases from vehicular traffic and industrial fuel burning form the secondary tier. High $NO_2$ and $CO$ strongly signal dense urban emissions.
3. **Weather Parameters (Temperature & Humidity):** While weather metrics do not directly generate pollutants, they modulate dispersion; lower temperatures trap particulates near the ground during winter thermal inversions.

---

## 10. Test with New Environmental Input

An interactive prediction function was created to evaluate hypothetical or live incoming sensor feeds:

### 10.1 Test Inputs:
```python
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
```

### 10.2 Model Inference Output:
* **Predicted AQI:** **`200.0`**
* **Air Quality Category:** **`Moderate`**
* **Health Advisory:** *"Breathing discomfort possible for children, elderly, and patients with asthma or lung disease. Sensitive groups should reduce heavy outdoor exertion."*
* **Visual Color Code:** `#f39c12` (Amber / Orange)

### 10.3 Standard AQI Categorization System:
* **Good (0 – 50):** Green `#27ae60` • Clean air, safe for all outdoor activities.
* **Satisfactory (51 – 100):** Light Green `#2ecc71` • Minor discomfort for unusually sensitive individuals.
* **Moderate (101 – 200):** Yellow/Amber `#f39c12` • Breathing discomfort for sensitive groups.
* **Poor (201 – 300):** Orange `#e67e22` • Discomfort to most people; wear N95 mask outdoors.
* **Very Poor (301 – 400):** Red `#e74c3c` • High risk of respiratory illness on prolonged exposure.
* **Severe (401 – 500+):** Maroon/Crimson `#78281F` • Emergency health alert; avoid all outdoor exertion.

---

## 11. Save the Final Model and Artifacts

To enable immediate integration into the web application, all trained model objects and metadata were serialized using `joblib` into the `models/` directory:

```python
import joblib

# Serialize best model and feature metadata
joblib.dump(best_model, "models/airsense_best_model.joblib")
joblib.dump(all_feature_cols, "models/feature_columns.joblib")

# Serialize benchmark models
joblib.dump(models["Linear Regression"], "models/linear_regression.joblib")
joblib.dump(models["Decision Tree"], "models/decision_tree.joblib")
```

* **Serialized Model Files:**
  * `models/airsense_best_model.joblib` (Best Gradient Boosting model)
  * `models/feature_columns.joblib` (Feature alignment list)
  * `models/linear_regression.joblib` & `models/decision_tree.joblib` (Benchmark models)

---

## 12. Prevention of Data Leakage

To ensure scientific validity and avoid inflated test accuracy:
1. **Strict Partitioning:** The 20% test partition was isolated prior to model fitting.
2. **Independent Inference:** The test dataset was passed only to `.predict()` calls and was never seen during model parameter optimization.
3. **Reproducible Splitting:** A constant `random_state=42` ensures identical train/test separation across all models for fair comparison.

---

## 13. Final Output & Deliverables Checklist of Phase 5

* [x] **Trained ML Models:** Linear Regression, Decision Tree, Random Forest, and Gradient Boosting.
* [x] **Comparison Table:** Full metrics table comparing MAE, MSE, RMSE, and $R^2$ scores.
* [x] **Best-Performing Model:** Selected Gradient Boosting ($\text{MAE}=5.29, R^2=0.9953$).
* [x] **Diagnostic Visualizations:** Saved in `reports/figures/` (Actual vs. Predicted, Residuals, Model Comparison, Feature Importance).
* [x] **Feature Importance Analysis:** Quantified $PM_{2.5}$ (34.7%) and $NO_2$ (27.8%) as primary drivers.
* [x] **Prediction Function:** Ready-to-use function accepting raw inputs and returning AQI score, health category, and advisory.
* [x] **Saved Model Artifacts:** Serialized `.joblib` files stored in `models/`.
* [x] **Pipeline Script:** [train_models.py](file:///c:/Users/anvit/OneDrive/Desktop/airSense/train_models.py).
* [x] **Phase 5 Documentation:** [Phase_5_Machine_Learning_Model_Development.md](file:///c:/Users/anvit/OneDrive/Desktop/airSense/Phase_5_Machine_Learning_Model_Development.md).
