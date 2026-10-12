"""
AirSense – Interactive Pollution Map Module
Provides Plotly-based interactive map visualization for Indian cities.
Modular add-on component: gracefully handles missing data without raising errors.
"""

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# Geocoordinates for AirSense monitored cities
CITY_COORDINATES = {
    "Delhi": {"lat": 28.6139, "lon": 77.2090, "state": "National Capital Region"},
    "Mumbai": {"lat": 19.0760, "lon": 72.8777, "state": "Maharashtra"},
    "Bengaluru": {"lat": 12.9716, "lon": 77.5946, "state": "Karnataka"},
    "Hyderabad": {"lat": 17.3850, "lon": 78.4867, "state": "Telangana"},
    "Kolkata": {"lat": 22.5726, "lon": 88.3639, "state": "West Bengal"},
    "Chennai": {"lat": 13.0827, "lon": 80.2707, "state": "Tamil Nadu"},
    "Coimbatore": {"lat": 11.0168, "lon": 76.9558, "state": "Tamil Nadu"},
    "Madurai": {"lat": 9.9252, "lon": 78.1198, "state": "Tamil Nadu"},
    "Trichy": {"lat": 10.7905, "lon": 78.7047, "state": "Tamil Nadu"}
}

def get_aqi_category_and_color(aqi_val):
    """Maps AQI numeric value to standard CPCB category, marker color, and text."""
    if pd.isna(aqi_val) or aqi_val is None:
        return "Data unavailable", "#94a3b8"
    val = float(aqi_val)
    if val <= 50:
        return "Good", "#27ae60"  # Green
    elif val <= 100:
        return "Satisfactory", "#2ecc71"  # Light Green
    elif val <= 200:
        return "Moderate", "#f39c12"  # Yellow / Amber
    elif val <= 300:
        return "Poor", "#e67e22"  # Orange
    elif val <= 400:
        return "Very Poor", "#e74c3c"  # Red
    else:
        return "Severe / Hazardous", "#78281f"  # Dark Red / Maroon

def create_pollution_map(df, model=None, feature_cols=None, selected_city=None):
    """
    Creates an interactive Plotly map of India displaying monitored cities.
    Includes current AQI, pollutants, and predicted AQI (if model provided).
    Safe error handling: Returns a go.Figure or None if data processing fails.
    """
    try:
        if df is None or df.empty:
            return None

        # Extract latest record for each city
        latest_records = df.sort_values("Date").groupby("City").last().reset_index()

        map_data = []
        for _, row in latest_records.iterrows():
            city_name = str(row.get("City", ""))
            coords = CITY_COORDINATES.get(city_name)
            if not coords:
                continue

            aqi = row.get("AQI", None)
            cat, color = get_aqi_category_and_color(aqi)

            # Calculate prediction if model & feature columns are available
            pred_aqi_str = "N/A"
            if model is not None and feature_cols is not None:
                try:
                    features_dict = {col: row.get(col, 0.0) for col in feature_cols}
                    pred_df = pd.DataFrame([features_dict])
                    pred_val = float(model.predict(pred_df)[0])
                    pred_aqi_str = f"{int(round(pred_val))}"
                except Exception:
                    pred_aqi_str = "Unavailable"

            # Safe formatting for pollutants
            pm25 = f"{row.get('PM2.5', 'N/A')}" if not pd.isna(row.get('PM2.5')) else "N/A"
            pm10 = f"{row.get('PM10', 'N/A')}" if not pd.isna(row.get('PM10')) else "N/A"
            no2 = f"{row.get('NO2', 'N/A')}" if not pd.isna(row.get('NO2')) else "N/A"
            so2 = f"{row.get('SO2', 'N/A')}" if not pd.isna(row.get('SO2')) else "N/A"
            o3 = f"{row.get('O3', 'N/A')}" if not pd.isna(row.get('O3')) else "N/A"
            co = f"{row.get('CO', 'N/A')}" if not pd.isna(row.get('CO')) else "N/A"
            temp = f"{row.get('Temperature', 'N/A')}" if not pd.isna(row.get('Temperature')) else "N/A"
            hum = f"{row.get('Humidity', 'N/A')}" if not pd.isna(row.get('Humidity')) else "N/A"

            # Marker sizing: highlight currently selected city
            is_active = (city_name == selected_city)
            marker_size = 22 if is_active else 16

            # Rich hover text with all required environmental metrics
            hover_text = (
                f"<b>📍 {city_name}</b> ({coords['state']})<br>"
                f"━━━━━━━━━━━━━━━━━━━━━━━━━<br>"
                f"<b>Current AQI:</b> {int(aqi) if aqi is not None and not pd.isna(aqi) else 'Data unavailable'}<br>"
                f"<b>Category:</b> {cat}<br>"
                f"<b>Predicted 24h AQI:</b> {pred_aqi_str}<br>"
                f"─────────────────────────<br>"
                f"<b>PM2.5:</b> {pm25} µg/m³<br>"
                f"<b>PM10:</b> {pm10} µg/m³<br>"
                f"<b>NO₂:</b> {no2} µg/m³<br>"
                f"<b>SO₂:</b> {so2} µg/m³<br>"
                f"<b>O₃:</b> {o3} µg/m³<br>"
                f"<b>CO:</b> {co} mg/m³<br>"
                f"<b>Temp / Humidity:</b> {temp}°C / {hum}%<br>"
                f"{'⭐ <i>Currently Selected City</i>' if is_active else '<i>Use dropdown to inspect city</i>'}"
            )

            map_data.append({
                "City": city_name,
                "lat": coords["lat"],
                "lon": coords["lon"],
                "AQI": aqi,
                "Category": cat,
                "Color": color,
                "Size": marker_size,
                "HoverText": hover_text,
                "IsActive": is_active
            })

        if not map_data:
            return None

        map_df = pd.DataFrame(map_data)

        # Build figure using Scattergeo for clean, reliable rendering in Streamlit
        fig = go.Figure()

        # Add trace for standard city markers
        fig.add_trace(go.Scattergeo(
            lon=map_df["lon"],
            lat=map_df["lat"],
            text=map_df["City"],
            hoverinfo="text",
            hovertext=map_df["HoverText"],
            mode="markers+text",
            textposition="top center",
            textfont=dict(size=12, family="Plus Jakarta Sans, sans-serif", color="#0f172a"),
            marker=dict(
                size=map_df["Size"],
                color=map_df["Color"],
                opacity=0.92,
                line=dict(
                    width=[3 if act else 1.5 for act in map_df["IsActive"]],
                    color=["#0f172a" if act else "#ffffff" for act in map_df["IsActive"]]
                )
            ),
            showlegend=False
        ))

        # Indian Subcontinent geographic projection configuration
        fig.update_geos(
            scope="asia",
            center=dict(lat=21.5, lon=80.0),
            projection_scale=3.8,
            visible=True,
            showcountries=True,
            countrycolor="#cbd5e1",
            showsubunits=True,
            subunitcolor="#e2e8f0",
            showland=True,
            landcolor="#f8fafc",
            showocean=True,
            oceancolor="#e0f2fe",
            showlakes=True,
            lakecolor="#e0f2fe",
            showcoastlines=True,
            coastlinecolor="#94a3b8",
            fitbounds=False
        )

        fig.update_layout(
            height=460,
            margin=dict(l=0, r=0, t=10, b=0),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            geo=dict(bgcolor="rgba(0,0,0,0)")
        )

        return fig

    except Exception:
        # Graceful fallback: return None so calling code displays friendly message
        return None
