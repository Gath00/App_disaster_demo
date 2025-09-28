import streamlit as st
import pandas as pd
import numpy as np
from streamlit_folium import st_folium
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import requests
import folium
import sqlite3

def init_db():
    conn = sqlite3.connect("geo_alert.db")
    cursor = conn.cursor()
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS risk_assessments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            latitude REAL,
            longitude REAL,
            elevation REAL,
            rainfall REAL,
            soil_moisture REAL,
            risk_probability REAL,
            risk_label TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()

def save_assessment(lat, lon, elev, rain, sm, prob, label):
    conn = sqlite3.connect("geo_alert.db")
    cursor = conn.cursor()
    cursor.execute(
        """
        INSERT INTO risk_assessments
        (latitude, longitude, elevation, rainfall, soil_moisture, risk_probability, risk_label)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (lat, lon, elev, rain, sm, prob, label)
    )
    conn.commit()
    conn.close()

def fetch_real_time_danger_zones():
    url = "https://earthquake.usgs.gov/fdsnws/event/1/query?format=geojson&limit=20"
    response = requests.get(url).json()
    return response["features"]

def add_danger_zones_to_map(m, zones):
    for zone in zones:
        lat, lon, depth = zone["geometry"]["coordinates"]
        magnitude = zone["properties"]["mag"]
        color = "red" if magnitude > 6 else ("orange" if magnitude > 4 else "green")
        folium.CircleMarker(
            location=[lat, lon],
            radius=magnitude * 2,
            color=color,
            fill=True,
            popup=f"Magnitude: {magnitude}"
        ).add_to(m)
    return m
def get_real_elevation(lat, lon):
    try:
        url = f"https://elevation-api.arcgis.com/arcgis/rest/services/elevation-service/v1/elevation/at-many-points={lat},{lon}"
        response = requests.get(url).json()
        return response["results"][0]["elevation"]
    except:
        print("API key doesn't Work")
        return pseudo_elevation(lat, lon)

def pseudo_elevation(lat, lon):
    return 300 + 200 * np.sin(np.radians(lat * 3.3)) * np.cos(np.radians(lon * 1.7))
def pseudo_rainfall(lat, lon):
    base = 1000 * (0.5 + 0.5 * np.cos(np.radians(lat * 2.1)) * np.sin(np.radians(lon * 0.9)))
    return float(np.clip(base, 0, 3000))

def pseudo_soil_moisture(lat, lon):
    return float(0.2 + 0.6 * (np.sin(np.radians(lat * 1.8)) + 1) / 2)

@st.cache_resource
def train_demo_model(random_state=42):
    rng = np.random.RandomState(random_state)
    n = 4000
    lat = rng.uniform(6.0, 36.0, size=n)
    lon = rng.uniform(68.0, 98.0, size=n)
    elev = np.array([pseudo_elevation(a, b) for a, b in zip(lat, lon)])
    rain = np.array([pseudo_rainfall(a, b) for a, b in zip(lat, lon)])
    sm = np.array([pseudo_soil_moisture(a, b) for a, b in zip(lat, lon)])
    risk_score = ((rain > 1200).astype(int) +
                  (elev < 200).astype(int) +
                  (sm > 0.55).astype(int))
    y = (risk_score >= 2).astype(int)
    X = pd.DataFrame({"lat": lat, "lon": lon, "elev": elev, "rain": rain, "soil_moisture": sm})
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)
    model = RandomForestClassifier(n_estimators=100, random_state=random_state)
    model.fit(X_train[["elev", "rain", "soil_moisture"]], y_train)
    return model, X_test, y_test

st.set_page_config(page_title="Geo-Alert — Demo", layout="wide")

st.title("Geo-Alert — Demo application for disaster localization and risk prediction")

st.markdown(
    """
    <style>
        .header {
            background-color: #FF9933;
            padding: 10px;
            color: white;
            font-size: 24px;
            text-align: center;
        }
        .body {
            background-color: #f0f8ff;
            padding: 20px;
        }
        .risk-high {
            background-color: #ff6b6b;
            color: white;
            padding: 5px;
            border-radius: 5px;
        }
        .risk-moderate {
            background-color: #ffd166;
            color: black;
            padding: 5px;
            border-radius: 5px;
        }
        .risk-low {
            background-color: #06d6a0;
            color: white;
            padding: 5px;
            border-radius: 5px;
        }
    </style>
    """,
    unsafe_allow_html=True
)

st.markdown('<div class="header">Geo-Alert — Disaster Risk Prediction</div>', unsafe_allow_html=True)

st.markdown(
    """
    Click on the map to select a point (latitude, longitude).
    The application returns a **risk probability** (e.g., flood risk) calculated by a demo model.
    """
)

model, X_test, y_test = train_demo_model()

left, right = st.columns([1, 2])

with left:
    st.header("Controls")
    st.markdown("**Manually enter values** if you want to override synthetic defaults:")
    lat_input = st.number_input("Latitude", value=20.5937, format="%.6f")
    lon_input = st.number_input("Longitude", value=78.9629, format="%.6f")
    elev_input = st.number_input("Elevation (m) — set 0 to use synthetic", value=0.0)
    rain_input = st.number_input("Rainfall (mm/year) — set 0 to use synthetic", value=0.0)
    sm_input = st.slider("Soil moisture (0-1) — 0 to use synthetic", 0.0, 1.0, 0.0, step=0.01)
    st.write("---")
    st.markdown("**Click on the map** (right) to replace latitude/longitude.")
    st.write("Or use the fields above and click **Evaluate Risk**.")
    if st.button("Evaluate Risk"):
        lat = lat_input
        lon = lon_input
        elev = elev_input if elev_input != 0.0 else pseudo_elevation(lat, lon)
        rain = rain_input if rain_input != 0.0 else pseudo_rainfall(lat, lon)
        sm = sm_input if sm_input != 0.0 else pseudo_soil_moisture(lat, lon)
        X_obs = np.array([[elev, rain, sm]])
        prob = model.predict_proba(X_obs)[0, 1]
        risk_label = "High" if prob > 0.6 else ("Moderate" if prob > 0.3 else "Low")
        st.subheader("Result")
        st.metric("Risk probability (demo)", f"{prob:.2f}")
        st.write(f"Interpretation: **{risk_label}**")
        st.write("Features used:")
        st.write(f"- Elevation (m): {elev:.1f}")
        st.write(f"- Rainfall (mm/year): {rain:.1f}")
        st.write(f"- Soil moisture: {sm:.2f}")
        st.markdown("**Quick advice (demo):**")
        if prob > 0.6:
            st.error("High risk: consider evacuation, monitor alerts, secure belongings.")
        elif prob > 0.3:
            st.warning("Moderate risk: follow forecasts, prepare an emergency kit.")
        else:
            st.success("Low risk: stay informed.")
        st.session_state["last_click"] = {"lat": lat, "lon": lon, "prob": float(prob)}

with right:
    st.header("Interactive Map")
    center = [20.5937, 78.9629]
    m = folium.Map(location=center, zoom_start=5, tiles="OpenStreetMap")
    zones = fetch_real_time_danger_zones()
    m = add_danger_zones_to_map(m, zones)
    folium.Marker(
        location=center,
        popup="Demo center — click anywhere on the map to select a point",
        icon=folium.Icon(color="blue", icon="info-sign")
    ).add_to(m)
    if "last_click" in st.session_state:
        lc = st.session_state["last_click"]
        color = "red" if lc["prob"] > 0.6 else ("orange" if lc["prob"] > 0.3 else "green")
        folium.CircleMarker(
            location=[lc["lat"], lc["lon"]],
            radius=10,
            color=color,
            fill=True,
            fill_opacity=0.7,
            popup=f"Prob (demo): {lc['prob']:.2f}"
        ).add_to(m)
    map_data = st_folium(m, width=900, height=600)
    if map_data and map_data.get("last_clicked"):
        lat_clicked = map_data["last_clicked"]["lat"]
        lon_clicked = map_data["last_clicked"]["lng"]
        st.success(f"Selected point: {lat_clicked:.6f}, {lon_clicked:.6f}")
        elev = pseudo_elevation(lat_clicked, lon_clicked) if elev_input == 0.0 else elev_input
        elevation = elev
        rain = pseudo_rainfall(lat_clicked, lon_clicked) if rain_input == 0.0 else rain_input
        sm = pseudo_soil_moisture(lat_clicked, lon_clicked) if sm_input == 0.0 else sm_input
        X_obs = np.array([[elev, rain, sm]])
        prob = model.predict_proba(X_obs)[0, 1]
        risk_label = "High" if prob > 0.6 else ("Moderate" if prob > 0.3 else "Low")
        st.subheader("Automatic evaluation of clicked point")
        st.write(f"- Latitude, longitude: {lat_clicked:.6f}, {lon_clicked:.6f}")
        st.write(f"- Elevation (synthetic): {elev:.1f} m")
        st.write(f"- Rainfall (synthetic): {rain:.1f} mm/year")
        st.write(f"- Soil moisture (synthetic): {sm:.2f}")
        st.metric("Risk probability (demo)", f"{prob:.2f}")
        if prob > 0.6:
            st.error("High risk: emergency measures may be needed.")
        elif prob > 0.3:
            st.warning("Moderate risk.")
        else:
            st.success("Low risk.")
        folium.CircleMarker(
            location=[lat_clicked, lon_clicked],
            radius=12,
            color="red" if prob > 0.6 else ("orange" if prob > 0.3 else "green"),
            fill=True,
            fill_opacity=0.6,
            popup=f"Prob (demo): {prob:.2f}"
        ).add_to(m)

st.write("---")
st.subheader("Examples (synthetic test set)")
st.dataframe(X_test.sample(min(10, len(X_test))).reset_index(drop=True))

st.write("---")
st.markdown(
    """
    **Notes:** this app is a demo. For production, replace the `pseudo_*` functions with real data:
    - Train model on historical disaster events.
    """
)

if st.button("Save Assessment"):
    save_assessment(lat_input, lon_input, elevation, rain, sm, prob, risk_label)
