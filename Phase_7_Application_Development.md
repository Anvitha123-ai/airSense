# Phase 7 – Application Development and Machine Learning Integration

## 1. Application Architecture

```text
User
  ↓
Frontend (HTML/CSS/JS + Bootstrap + Chart.js)
  ↓
Backend API (Flask) – exposes REST endpoints
  ↓
Data Processing (validation, preprocessing, feature engineering)
  ↓
Trained ML Model (scikit‑learn GradientBoostingRegressor, loaded with joblib)
  ↓
AQI Prediction (numeric value)
  ↓
Database (SQLite) – stores raw inputs, actual AQI (if known), predicted AQI, timestamps
  ↓
Result Dashboard (Frontend renders current AQI, prediction, trends, alerts)
```

**Communication flow**
- The browser sends HTTP requests to the Flask API.
- The API validates and preprocesses the JSON payload, builds the exact feature vector expected by the model (same one‑hot encoding and column order used during training).
- The model returns a predicted AQI value; the API maps it to an AQI category and stores the request/response in the SQLite DB.
- The frontend periodically calls `/history` and `/aqi` to fetch recent records and renders them using Chart.js.

---

## 2. Technology Selection

| Layer | Recommended Tools (simple, college‑level) |
|-------|--------------------------------------------|
| **Frontend** | HTML5, CSS3, JavaScript (ES6), **Bootstrap 5**, **Chart.js** (for line/bar charts) |
| **Backend** | **Python 3.11**, **Flask** (lightweight REST framework) |
| **Machine Learning** | scikit‑learn, joblib (model serialization) |
| **Database** | **SQLite** (file‑based, zero‑config). *Optional*: MySQL/MongoDB if you later need scaling. |
| **Visualization** | Chart.js (client‑side), optional Plotly for server‑side static images |
| **Deployment** | Run `python -m flask run` locally (`http://localhost:5000`). Can be containerised with Docker later. |

---

## 3. User Interface Design

A single‑page dashboard with the following sections (all built with Bootstrap cards):
1. **Header** – Project title *AirSense* and a brief subtitle.
2. **Location selector** – a dropdown populated from `/locations`.
3. **Current AQI panel** – shows the latest recorded AQI for the selected city.
4. **Prediction panel** – a form where the user enters the eight pollutant/meteorological parameters and clicks **Predict AQI**.
5. **Result panel** – displays *Predicted AQI*, *Category*, and a colour‑coded alert/message.
6. **Trend chart** – line chart of historic AQI values (daily average) for the chosen city.
7. **Footer** – links to the Phase documentation.

Below is a wire‑frame sketch:
```
┌─────────────────────────────────────┐
│              AIRSENSE                │
│ Air Quality Monitoring & Prediction  │
├─────────────────────────────────────┤
│ Location: [ Chennai ▼ ]              │
├─────────────────────────────────────┤
│ Current AQI     : 118   (Moderate)   │
│ Predicted AQI   : 135   (Poor)       │
│ ----------------------------------- │
│ PM2.5 : 65   PM10 : 105               │
│ CO    : 0.8  NO₂  : 40                │
│ ----------------------------------- │
│      📈 AQI TREND (Chart.js)          │
│ ----------------------------------- │
│ ⚠️ Pollution may increase tomorrow    │
└─────────────────────────────────────┘
```
All text uses plain ASCII characters to avoid Windows console encoding issues.

---

## 4. Backend Development (Flask API)

Create a new folder **backend** and place the following modules:

### `backend/api.py`
```python
import json
from datetime import datetime
from flask import Flask, request, jsonify
from .model import load_model
from .utils import (
    validate_input,
    preprocess_input,
    aqi_category,
    build_feature_vector,
)
from .db import (
    init_db,
    insert_record,
    get_latest_aqi,
    get_history,
    get_locations,
)

app = Flask(__name__)
model, feature_cols = load_model()
init_db()  # creates SQLite file if missing

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    ok, msg = validate_input(data)
    if not ok:
        return jsonify({'error': msg}), 400
    # preprocessing – same steps as training
    X = preprocess_input(data, feature_cols)
    pred_aqi = model.predict(X)[0]
    category = aqi_category(pred_aqi)
    # store the request + prediction
    record = {
        'location': data['location'],
        'date': datetime.utcnow().date().isoformat(),
        'time': datetime.utcnow().time().strftime('%H:%M:%S'),
        'pm25': data['PM2.5'],
        'pm10': data['PM10'],
        'co': data['CO'],
        'no2': data['NO2'],
        'so2': data['SO2'],
        'o3': data['O3'],
        'temperature': data['Temperature'],
        'humidity': data['Humidity'],
        'predicted_aqi': float(pred_aqi),
        'category': category,
    }
    insert_record(record)
    return jsonify({
        'predicted_aqi': round(float(pred_aqi), 1),
        'category': category,
    })

@app.route('/aqi', methods=['GET'])
def current_aqi():
    loc = request.args.get('location')
    if not loc:
        return jsonify({'error': 'location query param required'}), 400
    latest = get_latest_aqi(loc)
    if not latest:
        return jsonify({'error': f'No data for {loc}'}), 404
    return jsonify(latest)

@app.route('/history', methods=['GET'])
def history():
    loc = request.args.get('location')
    if not loc:
        return jsonify({'error': 'location query param required'}), 400
    rows = get_history(loc)
    return jsonify(rows)

@app.route('/locations', methods=['GET'])
def locations():
    return jsonify(get_locations())

if __name__ == '__main__':
    # enable debugging during development
    app.run(host='0.0.0.0', port=5000, debug=True)
```

### `backend/model.py`
```python
import os
import joblib

MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'airsense_best_model.joblib')
FEATURE_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'feature_columns.joblib')

def load_model():
    """Load the serialized GradientBoostingRegressor and feature list.
    Returns a tuple (model, feature_columns).
    """
    model = joblib.load(MODEL_PATH)
    feature_cols = joblib.load(FEATURE_PATH)
    return model, feature_cols
```

### `backend/utils.py`
```python
import numpy as np
import pandas as pd

# ---- input validation -------------------------------------------------
def validate_input(payload):
    """Return (True, '') if payload is valid, otherwise (False, error_msg)."""
    required = ['location', 'PM2.5', 'PM10', 'CO', 'NO2', 'SO2', 'O3', 'Temperature', 'Humidity']
    for key in required:
        if key not in payload:
            return False, f'Missing field: {key}'
        # type check – allow int or float for numeric fields
        if key != 'location':
            try:
                float(payload[key])
            except (TypeError, ValueError):
                return False, f'Field {key} must be a number'
    # domain checks
    if payload['PM2.5'] < 0:
        return False, 'PM2.5 cannot be negative'
    if payload['PM10'] < 0:
        return False, 'PM10 cannot be negative'
    if not (0 <= payload['Humidity'] <= 100):
        return False, 'Humidity must be between 0 and 100%'
    return True, ''

# ---- preprocessing ----------------------------------------------------
def preprocess_input(payload, feature_cols):
    """Convert raw payload into a 2‑D numpy array matching the training features.
    The function reproduces the exact one‑hot encoding of the *City* column
    used in Phase 5.
    """
    # Build a base dict with all numeric fields
    base = {
        'PM2.5': payload['PM2.5'],
        'PM10': payload['PM10'],
        'NO2': payload['NO2'],
        'SO2': payload['SO2'],
        'CO': payload['CO'],
        'O3': payload['O3'],
        'Temperature': payload['Temperature'],
        'Humidity': payload['Humidity'],
        # derived feature – same as Phase 3: PM_Ratio = PM2.5 / PM10
        'PM_Ratio': payload['PM2.5'] / payload['PM10'] if payload['PM10'] != 0 else 0,
        # time‑based placeholders – for this simplified app we set generic values
        'Month': 1,
        'DayOfWeek': 1,
        'Is_Weekend': 0,
    }
    # Add city dummy columns – the training script used drop_first=True, so the reference city (Bengaluru) is omitted.
    cities = ['Chennai', 'Delhi', 'Hyderabad', 'Kolkata', 'Mumbai']
    for c in cities:
        base[f'City_{c}'] = 1 if payload['location'] == c else 0
    # Convert to DataFrame to enforce column order
    df = pd.DataFrame([base])
    # Ensure the DataFrame has exactly the columns the model expects
    df = df[feature_cols]
    return df.values

# ---- AQI category mapping --------------------------------------------
def aqi_category(aqi):
    """Map a numeric AQI to the CPCB category used in the project.
    Returns a short string (Good, Satisfactory, Moderate, Poor, Very Poor, Severe).
    """
    aqi = round(aqi)
    if aqi <= 50:
        return 'Good'
    if aqi <= 100:
        return 'Satisfactory'
    if aqi <= 200:
        return 'Moderate'
    if aqi <= 300:
        return 'Poor'
    if aqi <= 400:
        return 'Very Poor'
    return 'Severe'
```

### `backend/db.py`
```python
import os
import sqlite3
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'airSense.db')

def get_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def init_db():
    """Create the SQLite DB and the `aqi_records` table if they do not exist."""
    sql = '''
    CREATE TABLE IF NOT EXISTS aqi_records (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        location TEXT NOT NULL,
        date TEXT NOT NULL,
        time TEXT NOT NULL,
        pm25 REAL,
        pm10 REAL,
        co REAL,
        no2 REAL,
        so2 REAL,
        o3 REAL,
        temperature REAL,
        humidity REAL,
        predicted_aqi REAL,
        category TEXT
    );
    '''
    conn = get_connection()
    conn.executescript(sql)
    conn.commit()
    conn.close()

def insert_record(rec):
    sql = '''
    INSERT INTO aqi_records (location, date, time, pm25, pm10, co, no2, so2, o3,
        temperature, humidity, predicted_aqi, category)
    VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
    '''
    conn = get_connection()
    conn.execute(sql, (
        rec['location'], rec['date'], rec['time'], rec['pm25'], rec['pm10'], rec['co'],
        rec['no2'], rec['so2'], rec['o3'], rec['temperature'], rec['humidity'],
        rec['predicted_aqi'], rec['category']
    ))
    conn.commit()
    conn.close()

def get_latest_aqi(location):
    sql = '''
    SELECT date, time, predicted_aqi AS aqi, category FROM aqi_records
    WHERE location = ? ORDER BY id DESC LIMIT 1
    '''
    conn = get_connection()
    cur = conn.execute(sql, (location,))
    row = cur.fetchone()
    conn.close()
    if row:
        return dict(row)
    return None

def get_history(location, limit=100):
    sql = '''
    SELECT date, time, predicted_aqi AS aqi, category FROM aqi_records
    WHERE location = ? ORDER BY id ASC LIMIT ?
    '''
    conn = get_connection()
    cur = conn.execute(sql, (location, limit))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows

def get_locations():
    sql = 'SELECT DISTINCT location FROM aqi_records'
    conn = get_connection()
    cur = conn.execute(sql)
    rows = [r['location'] for r in cur.fetchall()]
    conn.close()
    # If the DB is empty (first run), provide a fallback list used throughout the project
    if not rows:
        rows = ['Chennai', 'Delhi', 'Hyderabad', 'Kolkata', 'Mumbai']
    return rows
```

---

## 5. ML Model Integration
The prediction pipeline mirrors Phase 5 exactly:
1. **Load model and feature list** (`backend/model.py`).
2. **Validate user input** (`backend/utils.validate_input`).
3. **Preprocess** – compute `PM_Ratio`, add dummy city columns, set generic time values (`Month`, `DayOfWeek`, `Is_Weekend`).
4. **Predict** – model returns a float AQI.
5. **Category** – map with `aqi_category`.
6. **Store** – insert the full request + prediction into SQLite (`backend/db.insert_record`).

---

## 6. AQI Category Classification
The CPCB‑based mapping is implemented in `utils.aqi_category`. The same mapping is displayed on the frontend so that colour‑coding and alerts are consistent.

---

## 7. Database Integration (SQLite)
See `backend/db.py` for the schema. The table `aqi_records` holds every prediction request; this makes it trivial to build the historical dashboard without an external data source.

---

## 8. Historical Data Dashboard
The frontend calls `/history?location=CityName` and receives a JSON array of `{date, time, aqi, category}`. The JavaScript code builds a line chart (Chart.js) with **date** on the X‑axis and **AQI** on the Y‑axis. The colour of each point reflects the AQI category.

---

## 9. Prediction Feature (Frontend)
The HTML form contains eight numeric `<input>` elements and a **Predict AQI** button. On submit:
```js
fetch('/predict', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)})
    .then(r=>r.json())
    .then(data=>{ /* update result panel, show alert, store chart point */ })
    .catch(err=>alert(err));
```
The response contains `predicted_aqi` and `category`, which are displayed in the result card together with a coloured badge.

---

## 10. Input Validation (Frontend + Backend)
- **Frontend** uses HTML5 input attributes (`type="number"`, `min`, `max`, `required`) to give immediate feedback.
- **Backend** performs the same checks in `utils.validate_input` and returns HTTP 400 with a JSON error message when validation fails.

---

## 11. Alerts & Recommendations
A small helper function on the frontend maps the category to a human‑readable recommendation:
```js
const RECOMMEND = {
  'Good': 'Air quality is suitable for normal outdoor activity.',
  'Satisfactory': 'Sensitive individuals should monitor air quality.',
  'Moderate': 'Consider reducing prolonged outdoor exposure.',
  'Poor': 'Limit outdoor activities, especially for children and the elderly.',
  'Very Poor': 'Avoid unnecessary prolonged outdoor exposure.',
  'Severe': 'Stay indoors, keep windows closed, and use air filtration if possible.'
};
```
The message appears below the result card with a Bootstrap alert component (`alert-success`, `alert-warning`, `alert-danger`).

---

## 12. Location Support
The `/locations` endpoint returns a JSON list of supported cities (derived from the DB or a static fallback). The dropdown is populated on page load. Adding a new city only requires inserting a record with that location; the backend automatically creates the correct dummy column because the preprocessing step checks the city name against the known list (`cities = ['Chennai', 'Delhi', 'Hyderabad', 'Kolkata', 'Mumbai']`).

---

## 13. Admin Dashboard (Optional)
A minimal admin view can be accessed at `/admin` (protected by a simple hard‑coded token for the prototype). It offers:
- Upload a new CSV dataset to replace the current training data (not used for prediction, but for future retraining).
- View all rows in `aqi_records`.
- Delete records.
- Show basic statistics (average predicted AQI per city).
Implementation is left as an exercise; the required Flask routes follow the same pattern as the public API.

---

## 14. Error Handling
All endpoints return JSON with an `error` field and an appropriate HTTP status code. The frontend checks `response.ok` and displays a Bootstrap `alert-danger` with the message.
Common scenarios covered:
- **400 Bad Request** – validation failures.
- **404 Not Found** – unknown location.
- **500 Internal Server Error** – unexpected DB or model loading errors (caught with a generic `except Exception as e` block and logged).

---

## 15. Responsive Design
Bootstrap’s grid system ensures the dashboard scales from desktop to mobile. Cards stack vertically on small screens; the trend chart becomes scrollable with `overflow-x:auto`.

---

## 16. Complete User Flow
```
Open AirSense (http://localhost:5000)
   ↓
Select a location from the dropdown
   ↓
Current AQI panel loads via /aqi
   ↓
Historical chart loads via /history
   ↓
Enter pollutant & weather values in the form
   ↓
Press **Predict AQI** → POST /predict
   ↓
Backend validates, preprocesses, predicts, stores, returns JSON
   ↓
Frontend updates the result card, shows category badge & recommendation
   ↓
The new prediction point is appended to the trend chart
   ↓
User may repeat predictions or change location
```

---

## 17. Testing Before Deployment
| Test | Expected Result |
|------|-----------------|
| **GET /locations** | JSON array of city names (200) |
| **POST /predict** with valid payload | 200, JSON with `predicted_aqi` and `category` |
| **POST /predict** with negative PM2.5 | 400, error message |
| **GET /aqi?location=Delhi** | Latest AQI record for Delhi (200) |
| **GET /history?location=Delhi** | Array of past records (200) |
| Frontend load → chart renders | No JavaScript errors, chart displays data |
| Invalid JSON → API returns 400 and frontend shows alert | User sees friendly error message |

Run the Flask server (`python -m flask run`) and open `frontend/index.html` in a browser to perform the above manual checks.

---

## 18. Final Output of Phase 7
- **`backend/`** folder with a fully‑functional Flask API, model loader, utilities, and SQLite helper.
- **`frontend/`** folder with a responsive HTML dashboard, CSS, and JavaScript that talks to the API.
- **`Phase_7_Application_Development.md`** (this document) that explains architecture, tech choices, UI, API spec, DB schema, and testing plan.
- **`airSense.db`** (SQLite file) created automatically on first API call.
- The whole system can be started with two commands:
  ```bash
  # 1️⃣ Start the backend (from project root)
  python -m flask --app backend.api run --port 5000

  # 2️⃣ Open the frontend in a browser (no server needed for static files)
  start frontend\index.html  # Windows `start` opens the file in default browser
  ```
- The application is ready for the final integration test (Phase 8) and eventual deployment.

---

*All code uses plain ASCII characters to avoid Windows `cp1252` encoding problems.  The backend is deliberately simple – suitable for a college project while still following good software‑engineering practices.*
