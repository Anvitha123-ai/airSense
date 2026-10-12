# Phase 1: Problem Analysis and Requirements Specification
## Project: AirSense – Air Quality Monitoring and Prediction System Using Data Science

---

## 1. Problem Statement

### 1.1 The Real-World Problem of Air Pollution
Air pollution is one of the most critical environmental and public health challenges globally, particularly in rapidly urbanizing and industrialized regions. Major emission sources—including vehicular exhaust, industrial discharge, coal-fired power generation, construction dust, and seasonal agricultural crop burning—release hazardous concentrations of particulate matter ($PM_{2.5}$, $PM_{10}$) and toxic gases ($CO$, $NO_2$, $SO_2$, $O_3$) into the atmosphere.

Prolonged exposure to degraded air quality leads to severe short-term and long-term health complications:
- **Respiratory Complications:** Aggravation of asthma, chronic bronchitis, reduced lung capacity, and allergic reactions.
- **Cardiovascular Illnesses:** Elevated risks of hypertension, heart attacks, and stroke due to ultrafine particulate matter ($PM_{2.5}$) penetrating deep into the bloodstream.
- **Vulnerable Demographics:** Children, the elderly, outdoor laborers, and individuals with preexisting medical conditions suffer the most severe adverse effects.
- **Socio-Economic Impact:** Widespread pollution causes increased healthcare burdens, lost work hours, reduced academic attendance, and diminished urban quality of life.

### 1.2 Why Monitoring and Predicting AQI is Useful
Air Quality Index (AQI) is a standardized numerical scale that converts complex pollutant concentrations into a simple, comprehensible metric indicating whether ambient air is clean or harmful.
- **Value of Real-Time Monitoring:** Real-time monitoring provides immediate situational awareness, enabling individuals to know whether the air outside is safe to breathe at any given moment.
- **Value of Predictive AQI (Forecasting):** While monitoring is inherently *reactive* (informing users after pollution levels have already risen), predictive modeling is *proactive*. Forecasting AQI values 24 to 48 hours in advance enables:
  - Individuals to plan commutes, schedule outdoor exercise, or wear protective masks before smog sets in.
  - Schools and colleges to preemptively cancel or reschedule outdoor physical education classes.
  - Hospitals to prepare emergency respiratory care facilities ahead of anticipated pollution spikes.
  - Municipal authorities to plan traffic diversions, halt construction activities, or implement anti-smog water sprinklers before hazardous thresholds are breached.

---

## 2. Project Objective

### 2.1 Main Objective
The primary objective of **AirSense** is to design, develop, and deliver a comprehensive, data-driven web application that monitors current environmental air pollution levels and accurately predicts short-term future Air Quality Index (AQI) values using Data Science and Machine Learning techniques.

### 2.2 What the System Should Achieve
1. **Centralized Data Aggregation:** Seamlessly collect and aggregate atmospheric pollutant measurements and meteorological parameters for selected locations.
2. **Transparent AQI Calculation:** Compute and present standardized AQI ratings and categorized health bands (e.g., Good, Moderate, Unhealthy, Severe) in accordance with recognized standards (such as CPCB / US EPA).
3. **Machine Learning-Based Forecasting:** Apply supervised machine learning regression algorithms to forecast future AQI trends based on historical pollution sequences and weather data.
4. **Intuitive Visualization & Alerting:** Offer interactive, user-friendly graphical dashboards displaying pollutant concentrations, historical trends, and clear preventive health advisories for non-technical users.

---

## 3. Target Users

| User Group | Needs & Use Case | Expected Value from AirSense |
| :--- | :--- | :--- |
| **General Public** | Citizens, commuters, morning joggers, parents, and elderly individuals. | Fast, color-coded health indicators to decide when to stay indoors, wear masks, or use home air purifiers. |
| **Students** | School and college students studying science, environment, or data analytics. | Educational insights into environmental science, pollutant behavior, and practical applications of machine learning. |
| **Schools & Colleges** | Campus administrators, sports coaches, and facility managers. | Evidence-based planning for outdoor assemblies, physical education sessions, and campus health precautions. |
| **Environmental Researchers** | Academics, data scientists, and climate researchers. | Clean access to historical trends, pollutant correlation charts, and baseline ML models for further experimentation. |
| **Local Authorities & Municipalities** | City planners, pollution control boards, and traffic police departments. | Early warnings on pollution spikes to initiate timely civic advisories, traffic diversions, or construction dust controls. |

---

## 4. Proposed Solution

The proposed system adopts a three-tier data science pipeline:

```
[ Data Collection Layer ]
  - Historical Datasets (CPCB / Kaggle)
  - Real-Time Environmental APIs (OpenWeatherMap / Air Quality APIs)
         │
         ▼
[ Data Science & ML Engine ]
  - Data Cleaning, Imputation & Feature Engineering
  - Model Training (Random Forest / XGBoost / Linear Regression)
  - Short-Term AQI Prediction & Category Classification
         │
         ▼
[ Web Application & Presentation Layer ]
  - Interactive Dashboard (Streamlit / Web UI)
  - Current AQI Gauges, Pollutant Cards, Trend Graphs & Health Advisories
```

### 4.1 Data Collection
- **Historical Data:** Tabular datasets containing past pollutant concentrations ($PM_{2.5}$, $PM_{10}$, $CO$, $NO_2$, $SO_2$, $O_3$) and weather parameters (temperature, humidity) sourced from reliable open repositories (such as CPCB or Kaggle) used for exploratory data analysis (EDA) and model training.
- **Real-Time Data:** Periodic querying of open environmental APIs (such as OpenWeatherMap Air Pollution API, AQICN, or government open data endpoints) to fetch live pollutant and weather conditions for designated cities.

### 4.2 Data Science & Machine Learning Prediction
- **Data Preprocessing:** Handle missing observations, detect and treat sensor outliers, normalize pollutant scales, and derive temporal features (day of week, hour, rolling 24-hour averages).
- **Sub-Index Calculation:** Implement standardized sub-index piecewise linear functions to determine individual pollutant sub-indices and overall AQI.
- **Predictive Modeling:** Train supervised regression models (such as Linear Regression, Random Forest Regressor, and Gradient Boosting) to learn patterns between weather variables, historical pollutant levels, and subsequent AQI values, outputting an anticipated AQI score for the next 24 hours.

### 4.3 Web Application Presentation
- An interactive, responsive web dashboard built with a lightweight Python framework (such as Streamlit) that bridges the backend model and end users.
- Users select their city to instantly view the current AQI gauge, breakdown cards for each individual pollutant, historical time-series graphs, future predictions, and actionable cautionary alerts.

---

## 5. Functional Requirements

1. **FR-1: User Location Selection**
   - The system shall allow users to select a city or monitoring station from a dropdown or search selector.

2. **FR-2: Current AQI Display**
   - The system shall compute and display the current overall AQI value prominently with its corresponding standardized category label (e.g., "Good", "Moderate", "Poor", "Unhealthy", "Hazardous").
   - The AQI reading shall be styled using standardized color codes (Green, Yellow, Orange, Red, Purple, Maroon).

3. **FR-3: Pollution Parameter Breakdown**
   - The dashboard shall present dedicated visual metric cards for individual pollutants ($PM_{2.5}$, $PM_{10}$, $CO$, $NO_2$, $SO_2$, $O_3$) along with ambient temperature and humidity.
   - Each card shall display the measured concentration alongside standard permissible threshold levels.

4. **FR-4: Historical AQI Visualization**
   - The system shall render interactive time-series line and bar charts showing AQI and pollutant trends over customizable historical windows (e.g., past 24 hours, past 7 days, past 30 days).
   - Users shall have the ability to toggle individual pollutants on and off for comparative analysis.

5. **FR-5: AQI Prediction Engine**
   - The system shall feed the latest environmental inputs into the trained Machine Learning model to forecast the AQI value for the upcoming 24-hour window.
   - The output shall display the predicted numerical AQI, predicted health category, and the expected trend direction (rising, stable, or declining).

6. **FR-6: Pollution-Level Alerts & Health Advisories**
   - The system shall dynamically generate tailored health recommendations based on the AQI bracket:
     - Recommended precautions for sensitive groups (asthmatics, elderly, children).
     - Advice on outdoor activities, physical exercise, and window ventilation.
     - Guidance on protective measures (e.g., wearing N95 masks, activating air purifiers).

7. **FR-7: Interactive Unified Dashboard**
   - All modules (current metrics, graphs, prediction forecast, and advisories) shall be cleanly organized in a unified, intuitive dashboard view requiring zero manual technical configuration from the user.

---

## 6. Non-Functional Requirements

1. **Accuracy**
   - The Machine Learning model shall achieve reliable prediction performance measured through standard regression metrics ($R^2 \ge 0.80$, low RMSE and MAE) on validation datasets.
   - AQI sub-index calculations must strictly adhere to official government/environmental protection equations without arbitrary rounding errors.

2. **Performance & Responsiveness**
   - The dashboard interface shall load and render within **2 to 3 seconds** under typical internet connectivity.
   - ML inference (predicting AQI from input features) shall execute in under **500 milliseconds**.

3. **Usability & Accessibility**
   - The interface must follow clean design principles with high visual contrast, intuitive icons, and unambiguous color semantics.
   - Technical jargon must be translated into plain-language health recommendations suitable for ordinary citizens and students.

4. **Reliability & Fault Tolerance**
   - The application shall gracefully handle API downtime, network timeouts, or missing sensor values by displaying informative fallback messages instead of crashing.
   - Default historical datasets must be bundled locally to ensure offline demonstration capability during academic evaluations.

5. **Security**
   - Sensitive credentials (e.g., third-party API keys) must be isolated in secure environment variables (`.env`) and never exposed in client-side code or public repositories.
   - Input fields must be sanitized to prevent injection attacks or invalid query parameters.

6. **Scalability & Modularity**
   - The software architecture must maintain strict separation of concerns (data ingestion, preprocessing, modeling, visualization).
   - The codebase must be easily extensible to integrate additional cities, newer ML models, or live IoT sensor nodes in future iterations.

---

## 7. Input and Output Specifications

### 7.1 Input Parameters

| Parameter Category | Parameter Name | Standard Unit | Description |
| :--- | :--- | :--- | :--- |
| **Particulate Pollutants** | $PM_{2.5}$ | $\mu g/m^3$ | Fine inhalable particles with diameters $\le 2.5$ micrometers. |
| | $PM_{10}$ | $\mu g/m^3$ | Inhalable coarse particles with diameters $\le 10$ micrometers. |
| **Gaseous Pollutants** | Carbon Monoxide ($CO$) | $mg/m^3$ or $ppm$ | Toxic gas produced by incomplete combustion. |
| | Nitrogen Dioxide ($NO_2$) | $\mu g/m^3$ | Pungent gas emitted from vehicles and power stations. |
| | Sulfur Dioxide ($SO_2$) | $\mu g/m^3$ | Corrosive gas produced from industrial fuel combustion. |
| | Ozone ($O_3$) | $\mu g/m^3$ | Ground-level secondary photochemical oxidant. |
| **Meteorological Factors** | Temperature | $^\circ C$ | Ambient air temperature influencing atmospheric dispersion. |
| | Relative Humidity | $\%$ | Moisture percentage affecting particulate condensation. |
| **Contextual Metadata** | Date & Timestamp | YYYY-MM-DD HH:MM | Temporal reference for time-series sequencing and lag features. |
| | Location | City / Station Name | Geographic identifier for location-specific retrieval. |

### 7.2 Output Parameters

| Output Item | Data Type / Format | Description |
| :--- | :--- | :--- |
| **Current AQI** | Numeric (Integer, $0 - 500+$) | The computed overall Air Quality Index representing the dominating pollutant. |
| **Predicted AQI** | Numeric (Integer/Float) | Forecasted AQI value for the next 24-hour cycle produced by the ML model. |
| **Air Quality Category** | Categorical String | Safety tier: *Good (0-50), Satisfactory/Moderate (51-100), Moderate/Unhealthy for Sensitive (101-200), Poor/Unhealthy (201-300), Very Poor/Very Unhealthy (301-400), Severe/Hazardous (401+)*. |
| **Visual Color Indicator** | Hex Color Code | Universal traffic-light color mapping (Green $\rightarrow$ Yellow $\rightarrow$ Orange $\rightarrow$ Red $\rightarrow$ Purple $\rightarrow$ Maroon). |
| **Trends & Graphs** | Interactive Charts | Time-series line plots of historical AQI, multi-pollutant comparison radar/bar charts. |
| **Alerts & Advisories** | Contextual Text Messages | Tailored safety recommendations (e.g., "High particulate matter detected. Outdoor sports should be avoided today"). |

---

## 8. Technology Requirements

The technology stack is specifically selected to balance simplicity, robustness, and academic feasibility for a college-level Data Science project:

| Layer | Recommended Technology | Justification & Role |
| :--- | :--- | :--- |
| **Frontend UI** | **Streamlit** *(Alternative: HTML5/CSS3 + Bootstrap)* | Allows rapid development of reactive, data-dense web dashboards purely in Python without complex JavaScript frontend overhead. |
| **Backend & Pipeline** | **Python 3.10+ / FastAPI or Flask** *(Integrated in Streamlit)* | Python is the industry-standard language for data science, offering rich libraries and native compatibility with ML models. |
| **Machine Learning** | **Scikit-learn, XGBoost, Joblib** | Industry-standard libraries for regression modeling (Linear Regression, Random Forest, Gradient Boosting), feature scaling, and model serialization. |
| **Data Processing** | **Pandas, NumPy** | Efficient manipulation of tabular time-series data, missing value imputation, and numerical calculations. |
| **Database & Storage** | **SQLite / CSV Flat Files** | Lightweight, zero-configuration database ideal for caching historical datasets, daily sensor logs, and user queries locally. |
| **Data Visualization** | **Plotly, Matplotlib, Seaborn** | Plotly provides interactive, hoverable, zoomable charts directly embeddable within web dashboards. |

---

## 9. Project Scope

To ensure realistic delivery and high quality within a college project timeline, the project boundaries are defined as follows:

### 9.1 In-Scope (Version 1.0 Deliverables)
- Analysis and preprocessing of historical multi-city air quality datasets (e.g., CPCB India or open urban AQI archives).
- Implementation of standard AQI calculation algorithms for 6 key criteria pollutants ($PM_{2.5}, PM_{10}, CO, NO_2, SO_2, O_3$).
- Training, evaluation, and comparison of at least 2 to 3 supervised ML regression models (e.g., Linear Regression, Random Forest Regressor).
- Integration of a live weather/air quality public REST API to retrieve current conditions for selected major cities.
- Development of a fully operational, responsive web dashboard (Streamlit) supporting location selection, real-time AQI display, historical trend graphs, 24-hour prediction, and automated health alerts.
- Comprehensive model evaluation report detailing $R^2$, Mean Absolute Error (MAE), and Root Mean Squared Error (RMSE).

### 9.2 Out-of-Scope (Deferred to Future Enhancements)
- Fabrication or deployment of physical hardware IoT sensor kits (e.g., Arduino / Raspberry Pi / physical MQ-135 sensors).
- Processing of raw multi-spectral satellite imagery or spatial GIS raster mapping.
- Micro-level hyper-local sensor coverage down to individual street corners (system relies on city/regional monitoring stations).
- User authentication, database-driven user account registration, or customized SMS/WhatsApp notification dispatchers.
- Complex recurrent deep learning architectures (e.g., Transformer/GNN) requiring heavy multi-GPU infrastructure for training.

---

## 10. Expected Outcome and Societal Benefit

### 10.1 Final Working System
Upon completion of subsequent development phases, **AirSense** will be an end-to-end, functional web application. A user will be able to:
1. Open the application in any modern web browser.
2. Select their city or target location from a clean dropdown interface.
3. Instantly review the current environmental health status via a bold, color-coded AQI dial and individual pollutant cards.
4. Inspect historical pollution patterns across recent days through interactive, zoomable trend graphs.
5. Review an ML-generated 24-hour AQI forecast alongside clear, understandable health precautions for school, exercise, and outdoor activities.

### 10.2 Benefits to Users and Society
- **Empowered Decision Making:** Transforms complex microgram-per-cubic-meter chemical readings into clear, actionable health advisories understandable by any student or citizen.
- **Preventive Public Health Protection:** Helps asthmatic patients, parents of young children, and elderly citizens take proactive measures before stepping into hazardous outdoor air.
- **Academic and Educational Value:** Serves as a transparent, reproducible benchmark demonstrating how foundational data science techniques (EDA, data wrangling, regression modeling, and data visualization) solve pressing real-world environmental problems.
