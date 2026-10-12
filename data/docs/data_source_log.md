# AirSense Data Source Documentation

### 1. Historical Dataset
* **Name:** Air Quality Data in India (2015–2020)
* **Author / Provider:** Central Pollution Control Board (CPCB), Ministry of Environment, Forest and Climate Change, Government of India.
* **Curator / Mirror:** Rohan Rao on Kaggle ([Air Quality Data in India](https://www.kaggle.com/datasets/rohanrao/air-quality-data-in-india)) / [Data.gov.in](https://data.gov.in)
* **License:** Open Government Data License – India (OGDL) / Public Domain.
* **Access Date:** October 2026
* **Files Extracted:** `city_day.csv` (Daily aggregated metrics for 26 major cities).

### 2. Live Environmental API
* **Provider:** OpenWeatherMap Air Pollution API
* **Endpoint:** `http://api.openweathermap.org/data/2.5/air_pollution`
* **Protocol:** HTTP REST / JSON
* **Auth:** API Key authentication (stored in `.env`)
* **Parameters Retrieved:** `co`, `no2`, `o3`, `so2`, `pm2_5`, `pm10`

### 3. Weather API
* **Provider:** Open-Meteo Weather API
* **Endpoint:** `https://api.open-meteo.com/v1/forecast`
* **Protocol:** HTTP REST / JSON
* **Auth:** Free tier / Open Access (No key required for non-commercial educational use)
* **Parameters Retrieved:** `temperature_2m`, `relative_humidity_2m`, `wind_speed_10m`
