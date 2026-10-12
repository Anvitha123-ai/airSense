# AirSense Data Dictionary

This document provides a field-by-field description of all variables used in the **AirSense** project.

| Column Name | Category | Data Type | Physical Unit | Permissible / Valid Range | Description |
| :--- | :--- | :--- | :--- | :--- | :--- |
| `Date` | Temporal | Date (`YYYY-MM-DD`) | Calendar Date | 2015-01-01 to Present | The date of the environmental observation. |
| `Time` | Temporal | Time (`HH:MM`) | 24-Hour Time | 00:00 to 23:59 | Time of the measurement (00:00 for daily aggregate). |
| `City` / `Location` | Spatial | String / Object | N/A | Valid City Name | The urban area or ground station where reading was recorded. |
| `PM2.5` | Particulate | Float | $\mu g/m^3$ | 0.0 – 1000.0 | Particulate matter smaller than 2.5 microns. Major respiratory hazard. |
| `PM10` | Particulate | Float | $\mu g/m^3$ | 0.0 – 1500.0 | Particulate matter smaller than 10 microns (dust, coarse soot). |
| `NO2` | Gas | Float | $\mu g/m^3$ | 0.0 – 500.0 | Nitrogen Dioxide concentration from vehicular and industrial burning. |
| `SO2` | Gas | Float | $\mu g/m^3$ | 0.0 – 500.0 | Sulfur Dioxide concentration from fuel and thermal power plants. |
| `CO` | Gas | Float | $mg/m^3$ | 0.0 – 50.0 | Carbon Monoxide concentration from incomplete combustion. |
| `O3` | Gas | Float | $\mu g/m^3$ | 0.0 – 500.0 | Ground-level Ozone formed photochemically during daytime. |
| `Temperature` | Meteorological | Float | $^\circ C$ | -10.0 – 55.0 | Ambient surface air temperature. |
| `Humidity` | Meteorological | Float | $\%$ | 0.0 – 100.0 | Relative atmospheric moisture percentage. |
| `AQI` | Target | Float / Int | Index (0–500+) | 0 – 500+ | Official composite Air Quality Index (Target for ML regression). |
| `AQI_Bucket` | Target Category | Categorical | N/A | 6 Standard Bands | Good, Satisfactory, Moderate, Poor, Very Poor, Severe. |

---

### Standard National Air Quality Index (AQI) Categories
* **Good (0 – 50):** Minimal impact on health.
* **Satisfactory (51 – 100):** Minor breathing discomfort to sensitive people.
* **Moderate (101 – 200):** Breathing discomfort to people with lungs, asthma, and heart diseases.
* **Poor (201 – 300):** Breathing discomfort to most people on prolonged exposure.
* **Very Poor (301 – 400):** Respiratory illness on prolonged exposure.
* **Severe (401 – 500+):** Affects healthy people and seriously impacts those with existing diseases.
