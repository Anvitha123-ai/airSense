# Initial Data Quality Inspection Report

**Phase:** Phase 2 – Data Collection (Baseline Sanity Checks)  
**Target Dataset:** `city_day.csv` (CPCB India Historical Air Quality)  
**Scope:** Preliminary inspection (No alterations or cleaning applied in Phase 2)

---

### 1. Dataset Dimensions
* **Total Rows:** ~29,531
* **Total Columns:** 16
* **Temporal Range:** 2015-01-01 to 2020-07-01 (5.5 years)
* **Number of Unique Cities:** 26

---

### 2. Missing Value Analysis (Initial Baseline)

| Column Name | Missing Count (Approx.) | Missing Percentage | Preliminary Note |
| :--- | :--- | :--- | :--- |
| `City` | 0 | 0.0% | Complete |
| `Date` | 0 | 0.0% | Complete |
| `PM2.5` | ~4,598 | ~15.6% | Missing during sensor calibrations |
| `PM10` | ~11,140 | ~37.7% | Higher missingness in older stations |
| `NO2` | ~3,585 | ~12.1% | Moderate missingness |
| `CO` | ~2,059 | ~7.0% | Low missingness |
| `SO2` | ~3,854 | ~13.1% | Moderate missingness |
| `O3` | ~4,022 | ~13.6% | Moderate missingness |
| `AQI` | ~4,681 | ~15.9% | Missing when minimum required sub-indices fail |
| `AQI_Bucket` | ~4,681 | ~15.9% | Matches missing AQI entries |

*Note for Phase 3:* The high missingness in $PM_{10}$ requires either targeted forward/backward fill by city, iterative imputation, or training models prioritizing $PM_{2.5}$ which has higher completeness.

---

### 3. Duplicate Records
* **Identical Date & City Rows:** 0 duplicates identified. Each combination represents a unique daily reading per city.

---

### 4. Anomaly & Sanity Checks
* **Negative Values:** A small number of negative concentrations (e.g. -0.01) exist due to sensor baseline drift; flagged for clipping to 0 in Phase 3.
* **Scale Inconsistencies:** $CO$ is reported in $mg/m^3$ while other gases are in $\mu g/m^3$. Verified consistent with CPCB standard reporting.
* **Date Format:** Dates are recorded as `YYYY-MM-DD` strings; need parsing into `datetime64[ns]` in Phase 3 for time-series operations.
