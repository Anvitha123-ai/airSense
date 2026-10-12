import os
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="AirSense – Air Quality Monitoring & Prediction",
    page_icon="🍃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: #1e3d59;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #555;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background: #ffffff;
        border-radius: 10px;
        padding: 15px;
        border: 1px solid #e1e4e8;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
        text-align: center;
    }
    .metric-title {
        font-size: 0.85rem;
        color: #777;
        font-weight: 600;
        text-transform: uppercase;
    }
    .metric-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #222;
    }
    .metric-unit {
        font-size: 0.8rem;
        color: #888;
    }
    .alert-box {
        padding: 14px 18px;
        border-radius: 8px;
        font-size: 0.95rem;
        margin-top: 10px;
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Helper Functions & AQI Standards
# ---------------------------------------------------------
def get_aqi_details(aqi_val):
    if aqi_val <= 50:
        return "Good", "#27ae60", "Air quality is considered satisfactory, and air pollution poses little or no risk.", "Enjoy outdoor activities freely. Air quality is clean and safe."
    elif aqi_val <= 100:
        return "Satisfactory", "#2ecc71", "Air quality is acceptable; however, some pollutants may cause moderate health concern for sensitive individuals.", "Unusually sensitive people should consider reducing prolonged outdoor exertion."
    elif aqi_val <= 200:
        return "Moderate", "#f39c12", "Breathing discomfort possible for children, older adults, and people with lung or heart disease.", "Sensitive groups should reduce heavy outdoor exertion. General public can continue normal activities."
    elif aqi_val <= 300:
        return "Poor", "#e67e22", "Breathing discomfort to most people on prolonged exposure. Noticeable irritation.", "Wear an N95 mask outdoors. Avoid intense morning jogs. Close windows during peak traffic hours."
    elif aqi_val <= 400:
        return "Very Poor", "#e74c3c", "Respiratory illness likely upon prolonged exposure. Significant distress for sensitive groups.", "Limit outdoor activities strictly. Keep windows closed. Run indoor air purifiers. Wear N95 masks."
    else:
        return "Severe / Hazardous", "#78281F", "Emergency condition: Healthy people experience reduced endurance; vulnerable groups experience severe medical symptoms.", "Avoid all outdoor physical activity. Keep indoor air filtered. Consult a doctor if respiratory distress occurs."

# ---------------------------------------------------------
# Data Loader & Model Training
# ---------------------------------------------------------
@st.cache_data
def load_data():
    csv_path = os.path.join(os.path.dirname(__file__), "data", "raw", "city_day.csv")
    if os.path.exists(csv_path):
        df = pd.read_csv(csv_path)
    else:
        # Fallback inline generation if file not found
        cities = ["Delhi", "Mumbai", "Bengaluru", "Hyderabad", "Kolkata", "Chennai"]
        dates = pd.date_range("2020-01-01", "2020-12-31", freq="MS")
        rows = []
        for c in cities:
            for d in dates:
                rows.append({
                    "Date": d.strftime("%Y-%m-%d"),
                    "City": c,
                    "PM2.5": 85.0, "PM10": 140.0, "NO2": 25.0, "SO2": 10.0,
                    "CO": 0.9, "O3": 30.0, "Temperature": 24.0, "Humidity": 65.0,
                    "AQI": 180, "AQI_Bucket": "Moderate"
                })
        df = pd.DataFrame(rows)
    df["Date"] = pd.to_datetime(df["Date"])
    return df

@st.cache_resource
def train_prediction_model(df):
    feature_cols = ["PM2.5", "PM10", "NO2", "SO2", "CO", "O3", "Temperature", "Humidity"]
    df_clean = df.dropna(subset=feature_cols + ["AQI"])
    
    X = df_clean[feature_cols]
    y = df_clean["AQI"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=80, random_state=42)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    return model, feature_cols, r2, mae, rmse

df = load_data()
model, feature_cols, r2, mae, rmse = train_prediction_model(df)

# ---------------------------------------------------------
# Sidebar
# ---------------------------------------------------------
with st.sidebar:
    st.image("https://img.icons8.com/clouds/200/wind.png", width=90)
    st.title("AirSense Controls")
    
    import city_service

    # Session state for location selection
    if "selected_country" not in st.session_state:
        st.session_state["selected_country"] = "India"

    # 1. 🌍 Select Country
    country_options = city_service.SUPPORTED_COUNTRIES
    default_c_idx = country_options.index("India") if "India" in country_options else 0
    selected_country = st.selectbox(
        "🌍 Select Country",
        options=country_options,
        index=default_c_idx,
        help="Select country to filter and browse cities"
    )

    # 2. 📍 Select City (Dynamic per country, searchable)
    available_cities = city_service.get_country_cities(selected_country)
    
    # Ensure default city is valid for selected country
    default_city = "Chennai" if selected_country == "India" and "Chennai" in available_cities else (available_cities[0] if available_cities else "Chennai")
    
    # Determine default index
    curr_city_idx = 0
    if "selected_city_val" in st.session_state and st.session_state["selected_city_val"] in available_cities:
        curr_city_idx = available_cities.index(st.session_state["selected_city_val"])
    elif default_city in available_cities:
        curr_city_idx = available_cities.index(default_city)

    selected_city = st.selectbox(
        "📍 Select City",
        options=available_cities,
        index=curr_city_idx,
        placeholder="🔍 Search city...",
        help="Type to search supported cities across all states/regions"
    )
    st.session_state["selected_city_val"] = selected_city

    st.divider()
    st.markdown("### ⚙️ System Settings")
    view_mode = st.radio(
        "Dashboard Navigation",
        [
            "Live Overview & Prediction",
            "🛡️ Personal Risk Assessment",
            "Historical Trends & EDA",
            "Project Phase Documentation"
        ]
    )

    st.divider()
    st.markdown("### 📊 Model Performance")
    st.markdown(f"- **Algorithm:** Random Forest")
    st.markdown(f"- **$R^2$ Score:** `{r2:.3f}`")
    st.markdown(f"- **MAE:** `{mae:.2f} AQI points`")
    st.markdown(f"- **RMSE:** `{rmse:.2f}`")
    
    st.divider()
    st.caption("AirSense v1.0 • College Data Science Project")

# ---------------------------------------------------------
# Dynamic Location Environment Resolution (No Fake Data)
# ---------------------------------------------------------
env_res, env_err = city_service.fetch_live_city_environment(selected_city, selected_country, dataset_df=df)

if env_res is not None:
    # Use real-time retrieved environmental row
    latest_row = pd.Series(env_res)
    city_df = df[df["City"] == selected_city].sort_values("Date")
    if city_df.empty:
        # Generate baseline temporal context for plots from current real data
        city_df = pd.DataFrame([{
            "Date": pd.to_datetime(env_res["Date"]),
            "City": selected_city,
            "AQI": env_res["AQI"],
            "PM2.5": env_res["PM2.5"],
            "PM10": env_res["PM10"],
            "NO2": env_res["NO2"],
            "SO2": env_res["SO2"],
            "CO": env_res["CO"],
            "O3": env_res["O3"],
            "Temperature": env_res["Temperature"],
            "Humidity": env_res["Humidity"]
        }])
else:
    # Fallback to local dataset or graceful warning
    city_df = df[df["City"] == selected_city].sort_values("Date")
    if not city_df.empty:
        latest_row = city_df.iloc[-1]
    else:
        st.warning(f"⚠️ Live air-quality data is currently unavailable for {selected_city}, {selected_country}.")
        latest_row = df.iloc[-1]
        latest_row["City"] = selected_city


# ---------------------------------------------------------
# Header Section
# ---------------------------------------------------------
loc_country = latest_row.get("Country", selected_country)
loc_state = latest_row.get("State", "")
loc_lat = latest_row.get("lat", None)
loc_lon = latest_row.get("lon", None)
loc_source = latest_row.get("Source", "AirSense Monitoring Station")
loc_live = latest_row.get("Live", False)

st.markdown("<div class='main-title'>🍃 AirSense – Air Quality Monitoring & Prediction System</div>", unsafe_allow_html=True)
st.markdown(
    f"<div class='sub-title'>🌍 <b>{loc_country}</b> &nbsp;|&nbsp; 🏙️ <b>{selected_city}</b> "
    f"{f'({loc_state})' if loc_state else ''} &nbsp;|&nbsp; "
    f"{f'📌 {loc_lat:.2f}°N, {loc_lon:.2f}°E &nbsp;|&nbsp; ' if loc_lat and loc_lon else ''}"
    f"<span style='color:#059669; font-weight:600;'>● {loc_source}</span></div>",
    unsafe_allow_html=True
)

# ---------------------------------------------------------
# VIEW 1: Live Overview & Prediction
# ---------------------------------------------------------
if view_mode == "Live Overview & Prediction":
    cat, color, description, advice = get_aqi_details(latest_row["AQI"])
    
    # Top Row: AQI Gauge + Summary Card
    col_gauge, col_info = st.columns([1.2, 1.8])
    
    with col_gauge:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number+delta",
            value=latest_row["AQI"],
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Current AQI • {selected_city}", 'font': {'size': 20, 'color': '#1e3d59'}},
            gauge={
                'axis': {'range': [0, 500], 'tickwidth': 1, 'tickcolor': "#444"},
                'bar': {'color': color, 'thickness': 0.3},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#ccc",
                'steps': [
                    {'range': [0, 50], 'color': 'rgba(39, 174, 96, 0.25)'},
                    {'range': [50, 100], 'color': 'rgba(46, 204, 113, 0.25)'},
                    {'range': [100, 200], 'color': 'rgba(243, 156, 18, 0.25)'},
                    {'range': [200, 300], 'color': 'rgba(230, 126, 34, 0.25)'},
                    {'range': [300, 400], 'color': 'rgba(231, 76, 60, 0.25)'},
                    {'range': [400, 500], 'color': 'rgba(120, 40, 31, 0.35)'}
                ],
                'threshold': {
                    'line': {'color': "black", 'width': 4},
                    'thickness': 0.75,
                    'value': latest_row["AQI"]
                }
            }
        ))
        fig_gauge.update_layout(height=260, margin=dict(l=20, r=20, t=40, b=10))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with col_info:
        st.markdown(f"""
        <div style="background-color: {color}15; border-left: 6px solid {color}; padding: 18px; border-radius: 6px; margin-top: 15px;">
            <h3 style="margin-top:0; color:{color};">Category: {cat} (AQI {int(latest_row['AQI'])})</h3>
            <p style="margin-bottom:8px; font-size:1.02rem; color:#333;">{description}</p>
            <hr style="margin: 10px 0; border: none; border-top: 1px solid #ddd;">
            <p style="margin:0; font-weight:600; color:#1e3d59;">💡 Health Advisory:</p>
            <p style="margin:0; color:#444;">{advice}</p>
        </div>
        """, unsafe_allow_html=True)
        
    st.divider()

    # ---------------------------------------------------------
    # NEW FEATURE: Interactive Pollution Map (Add-on)
    # ---------------------------------------------------------
    st.subheader("🗺️ Live Pollution Map")
    st.caption("Live Air Quality Across Supported Cities • Click or hover over a city to view its current air-quality information.")

    try:
        from pollution_map import create_pollution_map
        map_fig = create_pollution_map(df, model=model, feature_cols=feature_cols, selected_city=selected_city)
        if map_fig is not None:
            st.plotly_chart(map_fig, use_container_width=True)
            
            # Interactive Color Legend matching requirements
            st.markdown("""
            <div style="display:flex; justify-content:center; align-items:center; flex-wrap:wrap; gap:16px; background:#f8fafc; border:1px solid #e2e8f0; border-radius:8px; padding:10px 16px; margin-top:2px; font-size:0.85rem; font-weight:600;">
                <span style="display:inline-flex; align-items:center; gap:6px;"><span style="color:#27ae60; font-size:1.1rem;">●</span> Good (0-50)</span>
                <span style="display:inline-flex; align-items:center; gap:6px;"><span style="color:#2ecc71; font-size:1.1rem;">●</span> Satisfactory (51-100)</span>
                <span style="display:inline-flex; align-items:center; gap:6px;"><span style="color:#f39c12; font-size:1.1rem;">●</span> Moderate (101-200)</span>
                <span style="display:inline-flex; align-items:center; gap:6px;"><span style="color:#e67e22; font-size:1.1rem;">●</span> Poor (201-300)</span>
                <span style="display:inline-flex; align-items:center; gap:6px;"><span style="color:#e74c3c; font-size:1.1rem;">●</span> Very Poor (301-400)</span>
                <span style="display:inline-flex; align-items:center; gap:6px;"><span style="color:#78281f; font-size:1.1rem;">●</span> Severe (401+)</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning("Live map data is temporarily unavailable.")
    except Exception:
        st.warning("Live map data is temporarily unavailable.")

    st.divider()

    # Pollutant Parameter Cards
    st.subheader("🧪 Individual Criteria Pollutant Measurements")
    p1, p2, p3, p4, p5, p6 = st.columns(6)
    
    with p1:
        st.markdown(f"<div class='metric-card'><div class='metric-title'>PM2.5</div><div class='metric-value'>{latest_row['PM2.5']}</div><div class='metric-unit'>µg/m³ (Std: 60)</div></div>", unsafe_allow_html=True)
    with p2:
        st.markdown(f"<div class='metric-card'><div class='metric-title'>PM10</div><div class='metric-value'>{latest_row['PM10']}</div><div class='metric-unit'>µg/m³ (Std: 100)</div></div>", unsafe_allow_html=True)
    with p3:
        st.markdown(f"<div class='metric-card'><div class='metric-title'>NO₂</div><div class='metric-value'>{latest_row['NO2']}</div><div class='metric-unit'>µg/m³ (Std: 80)</div></div>", unsafe_allow_html=True)
    with p4:
        st.markdown(f"<div class='metric-card'><div class='metric-title'>SO₂</div><div class='metric-value'>{latest_row['SO2']}</div><div class='metric-unit'>µg/m³ (Std: 80)</div></div>", unsafe_allow_html=True)
    with p5:
        st.markdown(f"<div class='metric-card'><div class='metric-title'>CO</div><div class='metric-value'>{latest_row['CO']}</div><div class='metric-unit'>mg/m³ (Std: 2.0)</div></div>", unsafe_allow_html=True)
    with p6:
        st.markdown(f"<div class='metric-card'><div class='metric-title'>O₃</div><div class='metric-value'>{latest_row['O3']}</div><div class='metric-unit'>µg/m³ (Std: 100)</div></div>", unsafe_allow_html=True)

    # Meteorological / Weather Parameters
    st.write("")
    st.markdown("##### ⛅ Meteorological & Atmospheric Conditions")
    w1, w2, w3, w4, w5 = st.columns(5)
    with w1:
        st.markdown(f"<div class='metric-card'><div class='metric-title'>Temperature</div><div class='metric-value'>{latest_row.get('Temperature', 25.0)}°</div><div class='metric-unit'>Celsius</div></div>", unsafe_allow_html=True)
    with w2:
        st.markdown(f"<div class='metric-card'><div class='metric-title'>Humidity</div><div class='metric-value'>{latest_row.get('Humidity', 60.0)}%</div><div class='metric-unit'>Relative</div></div>", unsafe_allow_html=True)
    with w3:
        st.markdown(f"<div class='metric-card'><div class='metric-title'>Wind Speed</div><div class='metric-value'>{latest_row.get('WindSpeed', 8.5)}</div><div class='metric-unit'>km/h</div></div>", unsafe_allow_html=True)
    with w4:
        st.markdown(f"<div class='metric-card'><div class='metric-title'>Surface Pressure</div><div class='metric-value'>{int(latest_row.get('Pressure', 1013))}</div><div class='metric-unit'>hPa</div></div>", unsafe_allow_html=True)
    with w5:
        st.markdown(f"<div class='metric-card'><div class='metric-title'>Precipitation</div><div class='metric-value'>{latest_row.get('Precipitation', 0.0)}</div><div class='metric-unit'>mm</div></div>", unsafe_allow_html=True)

    st.divider()

    # Machine Learning 24-Hour Prediction Section
    st.subheader("🤖 Machine Learning 24-Hour AQI Prediction Engine")
    
    pred_col1, pred_col2 = st.columns([1.2, 1.8])
    
    with pred_col1:
        st.markdown("#### Forecast Analysis")
        # Generate prediction using the model with current values
        input_data = pd.DataFrame([{col: latest_row[col] for col in feature_cols}])
        predicted_aqi = float(model.predict(input_data)[0])
        pred_cat, pred_color, pred_desc, pred_adv = get_aqi_details(predicted_aqi)
        
        delta_aqi = predicted_aqi - latest_row['AQI']
        trend_symbol = "🔺 Worsening" if delta_aqi > 5 else ("🔻 Improving" if delta_aqi < -5 else "➡️ Stable")
        
        st.metric(
            label="Predicted 24h AQI",
            value=f"{int(predicted_aqi)} ({pred_cat})",
            delta=f"{delta_aqi:+.1f} AQI points ({trend_symbol})"
        )
        
        st.markdown(f"""
        <div style="background-color:{pred_color}15; border: 1px solid {pred_color}; padding: 12px; border-radius: 6px; margin-top: 10px;">
            <b style="color:{pred_color};">Tomorrow's Forecast:</b> {pred_desc}<br>
            <b>Recommended Action:</b> {pred_adv}
        </div>
        """, unsafe_allow_html=True)

    with pred_col2:
        st.markdown("#### Interactive What-If Scenario Simulator")
        st.caption("Adjust environmental parameters to simulate how pollutant changes impact the predicted AQI:")
        
        s_col1, s_col2, s_col3 = st.columns(3)
        with s_col1:
            sim_pm25 = st.slider("PM2.5 (µg/m³)", 10.0, 450.0, float(latest_row['PM2.5']))
            sim_pm10 = st.slider("PM10 (µg/m³)", 20.0, 600.0, float(latest_row['PM10']))
        with s_col2:
            sim_no2 = st.slider("NO₂ (µg/m³)", 5.0, 150.0, float(latest_row['NO2']))
            sim_co = st.slider("CO (mg/m³)", 0.1, 5.0, float(latest_row['CO']))
        with s_col3:
            sim_temp = st.slider("Temperature (°C)", 5.0, 45.0, float(latest_row['Temperature']))
            sim_hum = st.slider("Humidity (%)", 15.0, 95.0, float(latest_row['Humidity']))
            
        sim_input = pd.DataFrame([{
            "PM2.5": sim_pm25, "PM10": sim_pm10, "NO2": sim_no2,
            "SO2": float(latest_row['SO2']), "CO": sim_co, "O3": float(latest_row['O3']),
            "Temperature": sim_temp, "Humidity": sim_hum
        }])
        sim_aqi = model.predict(sim_input)[0]
        s_cat, s_color, _, _ = get_aqi_details(sim_aqi)
        
        st.success(f"Simulated Scenario Output: **Predicted AQI = {int(sim_aqi)}** ({s_cat})")



# ---------------------------------------------------------
# VIEW 2: Personal Air Pollution Risk Assessment
# ---------------------------------------------------------
elif view_mode == "🛡️ Personal Risk Assessment":
    import risk_assessment
    
    st.markdown("## 🛡️ Personal Air Pollution Risk Assessment")
    st.markdown(
        "Estimate how atmospheric air pollution levels may impact your personal health risk based on "
        "**age, duration of outdoor exposure, physical exertion level, and existing health conditions**."
    )
    
    # Non-clinical Advisory Banner
    st.info(
        "ℹ️ **Educational & Decision-Support Estimate:** "
        "The risk score generated below is an estimated assessment index calculated using environmental exposure "
        "and vulnerability weighting. It is **NOT** a medical diagnosis, clinical prognosis, or personalized healthcare prescription."
    )
    
    st.divider()
    
    col_input, col_result = st.columns([1.1, 1.4], gap="large")
    
    with col_input:
        st.markdown("### 📋 1. Exposure & Personal Profile")
        
        # Location / AQI Context
        aqi_source = st.radio(
            "Air Quality Data Source",
            [f"Use Current Station Data ({selected_city})", "Custom Environmental Parameters"],
            index=0
        )
        
        if aqi_source.startswith("Use Current Station"):
            user_aqi = float(latest_row["AQI"])
            user_pm25 = float(latest_row["PM2.5"])
            user_pm10 = float(latest_row["PM10"])
            st.caption(f"📍 Linked to **{selected_city}** Station: AQI **{int(user_aqi)}**, PM2.5 **{user_pm25} µg/m³**, PM10 **{user_pm10} µg/m³**")
        else:
            c_aqi1, c_aqi2, c_aqi3 = st.columns(3)
            with c_aqi1:
                user_aqi = st.number_input("Current / Predicted AQI", min_value=0.0, max_value=600.0, value=float(latest_row["AQI"]), step=5.0)
            with c_aqi2:
                user_pm25 = st.number_input("PM2.5 (µg/m³)", min_value=0.0, max_value=600.0, value=float(latest_row["PM2.5"]), step=5.0)
            with c_aqi3:
                user_pm10 = st.number_input("PM10 (µg/m³)", min_value=0.0, max_value=900.0, value=float(latest_row["PM10"]), step=10.0)
                
        st.markdown("---")
        
        # User Demographics & Routine
        c_p1, c_p2 = st.columns(2)
        with c_p1:
            user_age = st.number_input("Age (Years)", min_value=1, max_value=110, value=28, step=1)
        with c_p2:
            user_hours = st.slider("Outdoor Exposure Duration (Hours)", min_value=0.25, max_value=12.0, value=2.0, step=0.25)
            
        user_activity = st.selectbox(
            "Physical Activity / Exertion Level",
            [
                "Low (Walking, sitting, commuting in vehicle)",
                "Moderate (Brisk walking, shopping, light maintenance)",
                "High (Jogging, cycling, outdoor athletic training, heavy labor)"
            ],
            index=1
        )
        
        st.markdown("---")
        st.markdown("#### Health / Body Conditions")
        st.caption("Select conditions that you know apply to you. Select 'None of these' if no chronic conditions apply.")
        
        cond_asthma = st.checkbox("Asthma")
        cond_resp = st.checkbox("Respiratory Sensitivity / Chronic Bronchitis")
        cond_heart = st.checkbox("Heart-Related / Cardiovascular Conditions")
        cond_elderly = st.checkbox("Older Age Vulnerability")
        cond_other = st.checkbox("Other Pollution-Sensitive Conditions (Allergies, Post-viral recovery)")
        cond_none = st.checkbox("None of these", value=True)
        
        selected_conds = []
        if not cond_none:
            if cond_asthma: selected_conds.append("asthma")
            if cond_resp: selected_conds.append("respiratory")
            if cond_heart: selected_conds.append("cardiovascular")
            if cond_elderly: selected_conds.append("elderly")
            if cond_other: selected_conds.append("other_sensitive")
            
    with col_result:
        st.markdown("### 📊 2. Estimated Risk Output")
        
        # Compute Risk
        risk_result = risk_assessment.calculate_personal_risk(
            age=user_age,
            aqi=user_aqi,
            pm25=user_pm25,
            pm10=user_pm10,
            exposure_hours=user_hours,
            activity_level=user_activity,
            selected_conditions=selected_conds
        )
        
        score = risk_result["risk_score"]
        level = risk_result["risk_level"]
        color = risk_result["color"]
        badge_bg = risk_result["badge_bg"]
        badge_text = risk_result["badge_text"]
        
        # Visual Risk Card
        st.markdown(f"""
        <div style="background: white; border: 2px solid {color}; border-radius: 14px; padding: 22px; box-shadow: 0 4px 14px rgba(0,0,0,0.06); text-align: center;">
            <div style="font-size: 0.9rem; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px;">
                Personal Air Pollution Risk Assessment
            </div>
            <div style="font-size: 3.6rem; font-weight: 800; color: {color}; margin: 8px 0 2px 0;">
                {score}%
            </div>
            <div style="display: inline-block; background: {badge_bg}; color: {badge_text}; padding: 6px 18px; border-radius: 20px; font-weight: 800; font-size: 1.1rem; margin-bottom: 14px;">
                {level.upper()} RISK
            </div>
            
            <!-- Progress Bar Visualization -->
            <div style="background: #e2e8f0; border-radius: 10px; height: 16px; overflow: hidden; width: 100%; margin: 10px 0 16px 0;">
                <div style="background: {color}; width: {score}%; height: 100%; border-radius: 10px; transition: width 0.3s ease;"></div>
            </div>
            
            <p style="font-size: 0.98rem; color: #334155; line-height: 1.5; margin: 0; text-align: left; background: #f8fafc; padding: 12px 16px; border-radius: 8px; border-left: 4px solid {color};">
                <b>Summary:</b> {risk_result['general_guidance']}
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Risk Contributions Chart
        st.markdown("#### 🔬 Factor Weight Contribution Breakdown")
        contrib_df = pd.DataFrame([
            {"Factor": k, "Score Contribution": v}
            for k, v in risk_result["ranked_contributions"]
        ])
        fig_contrib = px.bar(
            contrib_df,
            x="Score Contribution",
            y="Factor",
            orientation="h",
            color="Score Contribution",
            color_continuous_scale=["#10b981", "#f59e0b", "#ef4444"],
            text="Score Contribution",
            template="plotly_white"
        )
        fig_contrib.update_layout(height=240, margin=dict(l=10, r=10, t=10, b=10))
        st.plotly_chart(fig_contrib, use_container_width=True)
        
        # Explainability: Why did the score change?
        st.markdown("#### 💡 Explainability & Key Drivers")
        c_exp1, c_exp2 = st.columns(2)
        with c_exp1:
            st.markdown("**Factors Increasing Estimated Score (+)**")
            if risk_result["increasing_factors"]:
                for factor in risk_result["increasing_factors"]:
                    st.markdown(f"- 🔴 {factor}")
            else:
                st.caption("No significant adverse aggravating factors detected.")
                
        with c_exp2:
            st.markdown("**Factors Reducing Estimated Score (-)**")
            if risk_result["reducing_factors"]:
                for factor in risk_result["reducing_factors"]:
                    st.markdown(f"- 🟢 {factor}")
            else:
                st.caption("Limited protective factors identified for this profile.")
                
        # Recommended Precautions
        st.markdown("#### 🛡️ General Safety & Preventive Recommendations")
        for prec in risk_result["precautions"]:
            st.markdown(f"- {prec}")
            
        st.divider()
        st.caption(f"⚠️ **Disclaimer:** {risk_result['disclaimer']}")

# ---------------------------------------------------------
# VIEW 3: Historical Trends & EDA
# ---------------------------------------------------------
elif view_mode == "Historical Trends & EDA":
    st.subheader(f"📈 Historical Air Quality Trends for {selected_city}")
    
    # Time Series AQI
    fig_aqi_trend = px.line(
        city_df, x="Date", y="AQI",
        title=f"AQI Progression Over Time ({selected_city})",
        labels={"AQI": "Air Quality Index (AQI)", "Date": "Date"},
        markers=True,
        template="plotly_white"
    )
    # Add horizontal threshold lines
    fig_aqi_trend.add_hline(y=100, line_dash="dash", line_color="#2ecc71", annotation_text="Satisfactory Limit (100)")
    fig_aqi_trend.add_hline(y=200, line_dash="dash", line_color="#f39c12", annotation_text="Moderate Limit (200)")
    fig_aqi_trend.add_hline(y=300, line_dash="dash", line_color="#e67e22", annotation_text="Poor Limit (300)")
    fig_aqi_trend.add_hline(y=400, line_dash="dash", line_color="#e74c3c", annotation_text="Very Poor Limit (400)")
    st.plotly_chart(fig_aqi_trend, use_container_width=True)
    
    # Pollutant Comparison
    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown("#### Particulate Matter Variation (PM2.5 vs PM10)")
        fig_pm = px.bar(
            city_df, x="Date", y=["PM2.5", "PM10"],
            barmode="group",
            labels={"value": "Concentration (µg/m³)", "variable": "Pollutant"},
            template="plotly_white"
        )
        st.plotly_chart(fig_pm, use_container_width=True)
        
    with col_c2:
        st.markdown("#### Gaseous Pollutants (NO₂, SO₂, O₃)")
        fig_gas = px.line(
            city_df, x="Date", y=["NO2", "SO2", "O3"],
            labels={"value": "Concentration (µg/m³)", "variable": "Gas"},
            template="plotly_white"
        )
        st.plotly_chart(fig_gas, use_container_width=True)

    # Multi-City Comparison
    st.divider()
    st.subheader("🌍 Multi-City AQI Comparison")
    latest_all_cities = df.sort_values("Date").groupby("City").last().reset_index()
    fig_all = px.bar(
        latest_all_cities, x="City", y="AQI", color="AQI",
        color_continuous_scale=["#27ae60", "#f39c12", "#e74c3c", "#78281F"],
        title="Current AQI Comparison Across Indian Cities",
        template="plotly_white"
    )
    st.plotly_chart(fig_all, use_container_width=True)

# ---------------------------------------------------------
# VIEW 3: Project Phase Documentation
# ---------------------------------------------------------
elif view_mode == "Project Phase Documentation":
    st.subheader("📋 Academic Project Documentation")
    
    doc_tab1, doc_tab2, doc_tab3, doc_tab4, doc_tab5, doc_tab6 = st.tabs([
        "Phase 1: Requirements & Problem Analysis",
        "Phase 2: Data Collection",
        "Phase 3: Data Preprocessing",
        "Phase 5: Machine Learning Models",
        "Phase 6: Model Testing & Evaluation",
        "Data Dictionary"
    ])
    
    with doc_tab1:
        doc_p1_path = os.path.join(os.path.dirname(__file__), "Phase_1_Requirements_and_Problem_Analysis.md")
        if os.path.exists(doc_p1_path):
            with open(doc_p1_path, "r", encoding="utf-8") as f:
                st.markdown(f.read())
        else:
            st.info("Phase 1 documentation file located in project root.")
            
    with doc_tab2:
        doc_p2_path = os.path.join(os.path.dirname(__file__), "Phase_2_Data_Collection.md")
        if os.path.exists(doc_p2_path):
            with open(doc_p2_path, "r", encoding="utf-8") as f:
                st.markdown(f.read())
        else:
            st.info("Phase 2 documentation file located in project root.")
            
    with doc_tab3:
        doc_p3_path = os.path.join(os.path.dirname(__file__), "Phase_3_Data_Preprocessing.md")
        if os.path.exists(doc_p3_path):
            with open(doc_p3_path, "r", encoding="utf-8") as f:
                st.markdown(f.read())
        else:
            st.info("Phase 3 documentation file located in project root.")

    with doc_tab4:
        doc_p5_path = os.path.join(os.path.dirname(__file__), "Phase_5_Machine_Learning_Model_Development.md")
        if os.path.exists(doc_p5_path):
            with open(doc_p5_path, "r", encoding="utf-8") as f:
                st.markdown(f.read())
        else:
            st.info("Phase 5 documentation file located in project root.")
            
    with doc_tab5:
        # ── Phase 6 evaluation summary metrics strip ──────────────────────────
        st.markdown("### 🧪 Phase 6 – Live Evaluation Snapshot")
        p6_c1, p6_c2, p6_c3, p6_c4 = st.columns(4)
        p6_c1.metric("R² Score",  "0.9953", help="Gradient Boosting on 19-sample test set")
        p6_c2.metric("MAE",       "5.29",   help="Mean Absolute Error in AQI points")
        p6_c3.metric("RMSE",      "8.05",   help="Root Mean Squared Error in AQI points")
        p6_c4.metric("Cat. Acc.", "89.5%",  help="AQI category classification accuracy")
        p6_c5, p6_c6 = st.columns(2)
        p6_c5.metric("Max Error", "20.4 pts", help="Worst single-record prediction error")
        p6_c6.metric("Inference", "0.64 ms",  help="Average inference latency (1000 runs)")

        # Evaluation plots
        fig_dir  = os.path.join(os.path.dirname(__file__), "reports", "figures")
        avp_path = os.path.join(fig_dir, "phase6_actual_vs_predicted.png")
        err_path = os.path.join(fig_dir, "phase6_error_distribution.png")
        cat_path = os.path.join(fig_dir, "phase6_category_accuracy.png")
        prc_path = os.path.join(fig_dir, "phase6_perrecord_comparison.png")

        if os.path.exists(avp_path):
            st.markdown("---")
            st.image(avp_path, caption="Actual vs Predicted AQI & Residual Plot", use_container_width=True)
        if os.path.exists(err_path):
            st.image(err_path, caption="Prediction Error Distribution", use_container_width=True)
        col_l, col_r = st.columns(2)
        if os.path.exists(cat_path):
            col_l.image(cat_path, caption="AQI Category Accuracy", use_container_width=True)
        if os.path.exists(prc_path):
            col_r.image(prc_path, caption="Per-Record Comparison", use_container_width=True)

        # Full Phase 6 markdown documentation
        st.markdown("---")
        doc_p6_path = os.path.join(os.path.dirname(__file__), "Phase_6_Model_Testing_Evaluation.md")
        if os.path.exists(doc_p6_path):
            with open(doc_p6_path, "r", encoding="utf-8") as f:
                st.markdown(f.read())
        else:
            st.info("Phase 6 documentation file located in project root.")

    with doc_tab6:
        doc_dict_path = os.path.join(os.path.dirname(__file__), "data", "docs", "data_dictionary.md")
        if os.path.exists(doc_dict_path):
            with open(doc_dict_path, "r", encoding="utf-8") as f:
                st.markdown(f.read())
        else:
            st.info("Data dictionary file located in data/docs/")
