# Phase 2: Data Collection and Acquisition Specification
## Project: AirSense – Air Quality Monitoring and Prediction System Using Data Science

---

## 1. Data Requirements

To monitor air quality accurately and forecast future AQI levels using Machine Learning, the system requires a combination of **pollutant concentrations**, **meteorological parameters**, and **contextual metadata**.

The table below outlines each required data attribute, its standard measurement unit, and its specific role in the AirSense project:

| Attribute Name | Category | Standard Unit | Role & Importance in the Project |
| :--- | :--- | :--- | :--- |
| **$PM_{2.5}$** | Particulate Pollutant | $\mu g/m^3$ | Fine particulate matter ($\le 2.5\mu m$). The single most critical driver of high AQI values in urban areas and the primary health hazard. |
| **$PM_{10}$** | Particulate Pollutant | $\mu g/m^3$ | Coarse dust particles ($\le 10\mu m$) from roads, construction, and wind. High contributor to overall sub-index scores. |
| **$CO$** | Gaseous Pollutant | $mg/m^3$ (or $ppm$) | Carbon Monoxide from vehicular combustion and incomplete fuel burning. Key traffic-density indicator. |
| **$NO_2$** | Gaseous Pollutant | $\mu g/m^3$ | Nitrogen Dioxide from vehicle exhausts and thermal power plants. Contributes to secondary particle formation. |
| **$SO_2$** | Gaseous Pollutant | $\mu g/m^3$ | Sulfur Dioxide from industrial refineries and coal burning. Indicator of industrial pollution. |
| **$O_3$** | Photochemical Gas | $\mu g/m^3$ | Ground-level Ozone formed photochemically during sunny daytime hours from $NO_x$ and VOC precursors. |
| **Temperature** | Weather Parameter | $^\circ C$ | Influences atmospheric boundary layer height, convection, and chemical reaction rates among airborne pollutants. |
| **Humidity** | Weather Parameter | $\%$ | Relative humidity causes hygroscopic growth of particulates and affects aerosol dispersion/washout. |
| **Date** | Context Metadata | `YYYY-MM-DD` | Enables chronological ordering, extraction of seasonal cycles (winter, summer, monsoon), and day-of-week patterns. |
| **Time** | Context Metadata | `HH:MM` | Captures diurnal trends (morning rush-hour peaks, evening traffic surges, nighttime inversion layers). |
| **Location / City** | Context Metadata | String / Text | Identifies the monitoring station or urban region, enabling city-specific modeling and comparisons. |
| **AQI (Target)** | Target Variable | Integer ($0 - 500+$) | The standard composite Air Quality Index calculated from individual pollutant sub-indices; serves as the ground truth target for ML regression. |

---

## 2. Data Sources

Reliable and open data sources are required to build a credible academic project. For **AirSense**, data sources are categorized into three distinct types:

```
                         ┌────────────────────────────────────────┐
                         │           AirSense Data Sources        │
                         └──────────────────┬─────────────────────┘
                                            │
        ┌───────────────────────────────────┼──────────────────────────────────┐
        ▼                                   ▼                                  ▼
┌────────────────────────┐      ┌────────────────────────┐      ┌────────────────────────┐
│  Historical Datasets   │      │  Real-Time Pollutant   │      │  Weather Data Sources  │
│  (For Model Training)  │      │  (For Live Monitoring) │      │  (For Feature Context) │
├────────────────────────┤      ├────────────────────────┤      ├────────────────────────┤
│ • CPCB Open Data       │      │ • OpenWeatherMap       │      │ • Open-Meteo API       │
│ • Kaggle (India AQI)   │      │   Air Pollution API    │      │ • OpenWeatherMap       │
│ • OpenAQ Global Data   │      │ • WAQI / AQICN API     │      │   Current Weather API  │
└────────────────────────┘      └────────────────────────┘      └────────────────────────┘
```

### 2.1 Historical Datasets (for Offline Training and Analysis)
Historical datasets contain multi-year observations used to conduct exploratory data analysis (EDA), engineer temporal features, and train the predictive Machine Learning models.
* **Central Pollution Control Board (CPCB) / Data.gov.in:** Official open government portal offering validated daily and hourly pollution data across monitoring stations in major Indian cities.
* **Kaggle Curated Datasets:** Highly benchmarked open repositories such as the *“Air Quality Data in India (2015–2020)”* curated by Rohan Rao from official CPCB station records.
* **OpenAQ:** Global open-access platform providing standardized historical air quality archives from government monitoring agencies worldwide.

### 2.2 Real-Time Air Quality Sources (for Live Dashboard Display)
Real-time sources supply the most recent pollutant readings for the user's selected city:
* **OpenWeatherMap Air Pollution API:** Free-tier REST API providing current, hourly forecast, and historical levels of $PM_{2.5}, PM_{10}, CO, NO_2, SO_2,$ and $O_3$ by latitude/longitude.
* **World Air Quality Index Project (WAQI / AQICN API):** Provides live feeds and calculated AQI from over 30,000 official worldwide ground monitoring stations.

### 2.3 Weather Data Sources (for Meteorological Context)
Because temperature and humidity heavily influence pollutant dispersion:
* **Open-Meteo Weather API:** A completely free, non-commercial weather API requiring no API key; supplies historical and live temperature, relative humidity, wind speed, and precipitation.
* **OpenWeatherMap Current Weather API:** Complementary endpoint supplying synchronized temperature, humidity, and weather description.

---

## 3. Data Collection Method

The system uses three practical data acquisition strategies tailored to a college-level project:

```
[ Method 1: Batch Download ] ──► Kaggle / CPCB CSV Files ──────► Local raw_data/ folder
[ Method 2: REST API Ingestion] ─► OpenWeatherMap / Open-Meteo ─► JSON responses ──► Parsed Table
[ Method 3: Live Ground Sensors] ─► (Addressed via Public API feeds, eliminating IoT hardware needs)
```

### 3.1 Static CSV File Ingestion (Batch Collection)
* **Application:** Used for historical datasets required to train and evaluate ML models.
* **Procedure:** Multi-year CSV files are downloaded from open portals (CPCB / Kaggle) and placed into the project's local directory structure (`data/raw/`).
* **Ingestion:** Loaded directly into memory using `pandas.read_csv()` for verification, schema validation, and storage.

### 3.2 Public REST APIs (Automated Ingestion)
* **Application:** Used to fetch current environmental conditions and live verification values for the dashboard.
* **Procedure:** Python scripts send standard HTTP `GET` requests using the `requests` library to endpoints such as:
  ```
  http://api.openweathermap.org/data/2.5/air_pollution?lat={lat}&lon={lon}&appid={API_KEY}
  ```
* **Parsing:** The resulting JSON payload is decoded, mapped to the unified schema, and appended to the local storage or presented immediately on the dashboard.

### 3.3 Government Open Data Portals
* Government datasets published on [data.gov.in](https://data.gov.in) provide ground-truth regulatory station data. These are retrieved either via direct bulk CSV export or through the National Data and Analytics Platform (NDAP) endpoints.

### 3.4 Hardware Sensor Data vs. Virtual Sensor Strategy
* In full-scale industrial systems, physical hardware sensors (e.g., SDS011 for particulates, MQ-135 for gases) stream data via microcontrollers.
* For this college-level software and Data Science project, **live certified station APIs serve as virtual sensors**. This avoids the severe calibration drift, hardware maintenance, and electronic noise typical of cheap physical sensors, allowing the project to focus on data science and machine learning.

---

## 4. Dataset Selection for AirSense

For a robust college-level project, the recommended primary historical dataset is the **“Air Quality Data in India (2015–2020)”** dataset (sourced from official CPCB stations and accessible via Kaggle / CPCB):

### 4.1 Recommended Dataset Specifications

| Specification Factor | Details |
| :--- | :--- |
| **Dataset Title** | *Air Quality Data in India (City Day / Station Day)* |
| **Originating Authority** | Central Pollution Control Board (CPCB), India |
| **Approximate Size** | ~29,500 daily records (City-level) across 26 major cities |
| **Time Period** | **January 1, 2015 – July 1, 2020** (Over 5.5 years of continuous records) |
| **Key Locations Covered** | Delhi, Mumbai, Bengaluru, Hyderabad, Chennai, Kolkata, Ahmedabad, Lucknow, Patna, Jaipur, etc. |
| **Number of Features** | **16 Columns** (Pollutants, Metadata, Target AQI, and AQI Bucket) |
| **Target Variable** | **`AQI`** (Continuous numeric value for regression) and **`AQI_Bucket`** (Categorical classification) |

### 4.2 Why This Dataset is Ideal for a College Project
1. **Realistic Complexity:** Includes real-world sensor noise, missing observations, seasonal winter smog spikes, and monsoon clean periods—ideal for demonstrating data cleaning and analysis skills.
2. **Standard Alignment:** The target AQI is already calculated using standard National Air Quality Index (NAQI) guidelines, providing reliable ground truth for regression models.
3. **Manageable Footprint:** At ~30,000 rows, it comfortably fits within memory on any student laptop (under 15 MB), allowing fast iteration without requiring expensive cloud computing or distributed clusters.

---

## 5. Data Structure & Unified Schema

All collected data (whether loaded from historical CSVs or ingested from live APIs) must be standardized into a uniform tabular structure.

### 5.1 Sample Data Table (Master Schema)

| Date | Time | Location | PM2.5 | PM10 | CO | NO2 | SO2 | O3 | Temperature | Humidity | AQI | AQI_Bucket |
| :---: | :---: | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| 2020-01-15 | 10:00 | Delhi | 184.2 | 290.5 | 1.85 | 48.2 | 16.4 | 32.1 | 14.5 | 78.0 | 349 | Very Poor |
| 2020-01-15 | 10:00 | Mumbai | 58.4 | 112.0 | 0.92 | 26.8 | 12.1 | 24.5 | 27.2 | 65.0 | 118 | Moderate |
| 2020-01-15 | 10:00 | Bengaluru | 28.1 | 54.3 | 0.45 | 18.2 | 8.5 | 19.0 | 22.8 | 58.0 | 62 | Satisfactory |
| 2020-07-20 | 14:00 | Delhi | 34.6 | 68.2 | 0.65 | 19.4 | 9.8 | 28.4 | 33.2 | 82.0 | 76 | Satisfactory |
| 2020-11-05 | 09:00 | Delhi | 342.8 | 480.1 | 3.10 | 72.6 | 24.3 | 41.0 | 18.0 | 85.0 | 465 | Severe |

### 5.2 Field Definitions & Data Types

| Column Name | Data Type | Nullable | Example Value | Description |
| :--- | :--- | :---: | :--- | :--- |
| `Date` | Date (`YYYY-MM-DD`) | No | `2020-01-15` | Calendar date of observation |
| `Time` | Time (`HH:MM`) | Yes | `10:00` | Local observation hour (or `00:00` for daily aggregates) |
| `Location` | String / Varchar | No | `Delhi` | City or monitoring station name |
| `PM2.5` | Float | Yes | `184.2` | Fine particulate matter ($\mu g/m^3$) |
| `PM10` | Float | Yes | `290.5` | Coarse particulate matter ($\mu g/m^3$) |
| `CO` | Float | Yes | `1.85` | Carbon monoxide ($mg/m^3$) |
| `NO2` | Float | Yes | `48.2` | Nitrogen dioxide ($\mu g/m^3$) |
| `SO2` | Float | Yes | `16.4` | Sulfur dioxide ($\mu g/m^3$) |
| `O3` | Float | Yes | `32.1` | Ground ozone ($\mu g/m^3$) |
| `Temperature` | Float | Yes | `14.5` | Ambient temperature ($^\circ C$) |
| `Humidity` | Float | Yes | `78.0` | Relative humidity percentage ($\%$) |
| `AQI` | Integer / Float | Yes | `349` | Ground truth or computed Air Quality Index |
| `AQI_Bucket` | String / Categorical | Yes | `Very Poor` | Standard qualitative health category |

---

## 6. Data Storage Evaluation & Recommendation

To determine the most practical storage mechanism for this project, four common options were evaluated:

| Storage Option | Strengths | Limitations | Suitability for AirSense |
| :--- | :--- | :--- | :--- |
| **CSV Files** | Zero setup; human-readable; natively supported by Pandas and Git; universal standard for ML datasets. | Inefficient for high-frequency queries; lacks transactional locking. | **Highly Recommended** for raw historical data storage, backups, and version control. |
| **SQLite** | Zero-configuration single-file SQL database; fast index queries; bundled with Python standard library. | Single-write concurrency; not suited for multi-server enterprise clustering. | **Highly Recommended** for caching real-time API logs and fast web dashboard queries. |
| **MySQL / PostgreSQL**| Enterprise-grade relational DB; supports multi-user concurrent read/write and indexing. | Requires installing and managing a separate server daemon; overkill for a standalone student project. | Optional / Excessive overhead for initial phases. |
| **MongoDB** | Schema-less document store; flexible for varied JSON API responses. | Unnecessary complexity for structured tabular time-series data; higher resource consumption. | Not recommended for tabular regression data. |

### Final Storage Recommendation
* **Primary Raw Repository:** **CSV Files** stored in a structured project folder (`data/raw/` and `data/processed/`).
* **Operational Application Cache:** **SQLite database** (`airsense.db`) embedded directly within the project root for storing recent API pulls and fast query retrieval.

---

## 7. Data Quality Checks (Initial Inspection)

Before performing any transformations in Phase 3, an initial diagnostic inspection must be run across the collected raw records to document existing anomalies:

```
Raw Collected Data
        │
        ├─► 1. Check Missing Values (% Null per column)
        ├─► 2. Check Duplicate Records (Identical Date + Location entries)
        ├─► 3. Check Invalid Values (Negative values, zero readings, extreme outliers)
        ├─► 4. Check Unit Discrepancies (e.g., CO in mg/m³ vs µg/m³)
        └─► 5. Check Date/Time Consistency (ISO format YYYY-MM-DD validation)
```

1. **Missing Values Check:**
   * Quantify the percentage of missing (`NaN`/null) records for each pollutant. In typical environmental datasets, $PM_{10}$ and $O_3$ often show $15\% - 30\%$ missingness due to station maintenance.
   * *Criterion:* Document which columns have $>50\%$ missing values so appropriate imputation strategies can be planned for Phase 3.
2. **Duplicate Records Check:**
   * Detect duplicate rows sharing the exact same `(Date, Time, Location)` composite key.
   * *Criterion:* Identify any double-logged entries from sensor restarts.
3. **Invalid & Impossible Values Check:**
   * Check for physical impossibilities:
     - Negative concentrations (e.g., $PM_{2.5} < 0$), which occur during sensor zero-calibration errors.
     - Physically impossible weather values (e.g., Humidity $> 100\%$ or Humidity $< 0\%$).
     - Unrealistic spikes caused by transmission glitches (e.g., $PM_{2.5} > 1500 \mu g/m^3$).
4. **Incorrect Units & Scale Consistency:**
   * Ensure $CO$ is consistently reported in $mg/m^3$ rather than $\mu g/m^3$ (values should typically be in the $0.1 - 10.0$ range, not thousands).
   * Confirm temperature is in Celsius ($^\circ C$) rather than Fahrenheit ($^\circ F$).
5. **Inconsistent Date/Time Formats:**
   * Check for mixed timestamp formats (e.g., `DD/MM/YYYY`, `MM-DD-YYYY`, and `YYYY-MM-DD`). Standardize on the ISO-8601 standard (`YYYY-MM-DD`).

---

## 8. Data Collection Frequency

### Comparison: Hourly vs. Daily Data

| Factor | Hourly Frequency | Daily Frequency |
| :--- | :--- | :--- |
| **Granularity** | Captures morning/evening rush hours and daytime photochemical peaks. | Smooths out micro-variations into a representative 24-hour average. |
| **Dataset Size** | $24 \times$ larger (~260,000 rows for 3 years per city). | Compact (~1,100 rows for 3 years per city). |
| **Noise & Missingness**| Higher frequency of transient sensor dropouts and missing hours. | Lower noise; missing hours are averaged out in official daily summaries. |
| **Target Application** | Real-time immediate alerts and short-term intra-day forecasting. | 24-hour day-ahead forecasting and long-term trend analysis. |

### Recommended Frequency for AirSense
* **For Historical ML Training:** **Daily Aggregates (`City_Day.csv`)** are recommended for the core baseline model. A daily frequency provides clean, statistically robust 24-hour trends without massive file sizes, making model experimentation fast and dependable.
* **For Live Dashboard API Ingestion:** **Hourly / Current readings** fetched on-demand from the OpenWeatherMap API to display the active real-time status.

---

## 9. Data Volume Requirements

For a meaningful, college-level Data Science project, the dataset must span sufficient time to capture all seasonal cycles while remaining computationally lightweight:

* **Recommended Time Span:** **3 to 5 full calendar years** (e.g., 2017 to 2021).
  * *Why?* Air quality is deeply seasonal. In northern India, for example, severe smog occurs during winter inversions (November–January), dust storms occur in summer (April–June), while monsoon rains (July–September) wash particulates out of the atmosphere. A single year of data is insufficient for an ML model to generalize across these seasons.
* **Recommended Number of Records:**
  * **Daily Level:** **15,000 to 30,000 records** spanning 10 to 20 representative cities.
  * **Single City Focus (e.g., Delhi):** Minimum **1,500 to 2,000 daily observations** (~4 to 5 years).
* **Storage Footprint:** This volume corresponds to **~5 MB to 15 MB** in uncompressed CSV format, making it fast to load, process, and analyze on any standard laptop.

---

## 10. Final Output of Phase 2

At the conclusion of Phase 2, the following tangible assets must be assembled in the project repository before proceeding to Phase 3:

```
airSense/
├── data/
│   ├── raw/
│   │   ├── city_day_raw.csv            <-- Unmodified historical dataset
│   │   └── sample_api_response.json   <-- Sample raw response from OpenWeatherMap API
│   └── docs/
│       ├── data_dictionary.md          <-- Complete field-by-field metadata specification
│       ├── data_source_log.md          <-- Download URLs, timestamps & licensing notes
│       └── initial_quality_report.md   <-- Quantified missingness, duplicate & anomaly count
├── Phase_1_Requirements_and_Problem_Analysis.md
└── Phase_2_Data_Collection.md
```

### Deliverables Checklist:
1. **Raw Dataset File (`city_day_raw.csv`):** The untouched, original historical CSV placed in the `data/raw/` directory.
2. **Data Dictionary (`data_dictionary.md`):** Comprehensive documentation defining all column names, physical units, data types, and valid numerical ranges.
3. **Data-Source Documentation (`data_source_log.md`):** Complete provenance record detailing API endpoints, dataset download URLs, originating authority (CPCB), and access dates.
4. **Initial Data-Quality Inspection Report (`initial_quality_report.md`):** Quantitative baseline report listing total row counts, column-wise missing percentages, duplicate counts, and identified outliers.
5. **Organized Project Structure:** Clean directory tree ready to feed the raw data directly into Phase 3 (Data Cleaning, Preprocessing, and Exploratory Data Analysis).
