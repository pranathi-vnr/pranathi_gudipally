# app.py
# -*- coding: utf-8 -*-
"""
EnviroScan Pollution Source Identifier
Streamlit App
"""

import pandas as pd
import requests
import osmnx as ox
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report
import folium
import streamlit as st


# -------------------------
# Data Collection Functions
# -------------------------

def fetch_openaq_data(city, params):
    """Fetch pollution data from OpenAQ"""
    url = f"https://api.openaq.org/v2/measurements?city={city}"
    response = requests.get(url, params=params)
    data = response.json().get('results', [])
    return pd.DataFrame(data)


def fetch_weather_data(lat, lon, api_key):
    """Fetch weather data from OpenWeatherMap"""
    url = "https://api.openweathermap.org/data/2.5/weather"
    params = {'lat': lat, 'lon': lon, 'appid': api_key}
    response = requests.get(url, params=params)
    return response.json()


def get_location_features(lat, lon, dist=1000):
    """Extract roads and factories near a location using OSMNX"""
    G = ox.graph_from_point((lat, lon), dist=dist, network_type='drive')
    roads = ox.geometries_from_point((lat, lon), tags={'highway': True}, dist=dist)
    factories = ox.geometries_from_point((lat, lon), tags={'landuse': 'industrial'}, dist=dist)
    return {'roads': roads, 'factories': factories}


# -------------------------
# Data Cleaning & Features
# -------------------------

def clean_pollution_data(df):
    df = df.drop_duplicates()
    if 'value' in df.columns:
        df = df.dropna(subset=['value'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
    if 'date' in df.columns and isinstance(df['date'].iloc[0], dict):
        df['timestamp'] = pd.to_datetime(df['date'].apply(lambda x: x.get('utc')))
    return df


def feature_engineering(df):
    if 'value' in df.columns:
        df['value'] = (df['value'] - df['value'].mean()) / df['value'].std()
    if 'timestamp' in df.columns:
        df['hour'] = df['timestamp'].dt.hour
        df['dayofweek'] = df['timestamp'].dt.dayofweek
    return df


# -------------------------
# Source Labeling
# -------------------------

def label_sources(df):
    df['source'] = 'Unknown'
    if 'NO2' in df.columns:
        df.loc[(df.get('near_main_road') == 1) & (df['NO2'] > 40), 'source'] = 'Vehicular'
    if 'SO2' in df.columns:
        df.loc[(df.get('near_factory') == 1) & (df['SO2'] > 20), 'source'] = 'Industrial'
    return df


# -------------------------
# Model Training
# -------------------------

def train_predict_model(df):
    features = ['PM2.5', 'NO2', 'SO2', 'CO', 'roads_proximity',
                'factories_proximity', 'temperature', 'humidity',
                'hour', 'dayofweek']
    df = df.dropna(subset=features + ['source'])
    X = df[features]
    y = df['source']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    clf = RandomForestClassifier()
    param_grid = {'n_estimators': [50, 100], 'max_depth': [5, 10, None]}
    grid = GridSearchCV(clf, param_grid)
    grid.fit(X_train, y_train)

    y_pred = grid.predict(X_test)
    print(classification_report(y_test, y_pred))
    return grid.best_estimator_


# -------------------------
# Visualization
# -------------------------

def plot_heatmap(df):
    if 'latitude' not in df.columns or 'longitude' not in df.columns:
        return None
    m = folium.Map(location=[df['latitude'].mean(), df['longitude'].mean()], zoom_start=12)
    for _, row in df.iterrows():
        folium.Circle(
            location=[row['latitude'], row['longitude']],
            radius=50,
            color="red" if row['source'] == "Industrial" else "blue",
            fill=True
        ).add_to(m)
    return m


# -------------------------
# Streamlit App
# -------------------------

st.set_page_config(page_title="EnviroScan", page_icon="🌍")
st.title("🌍 EnviroScan Pollution Source Identifier")

city = st.text_input("Enter a city name", "Delhi")

if st.button("Analyze"):
    if not city.strip():
        st.warning("Please enter a valid city name.")
    else:
        st.success(f"Analyzing pollution sources for: {city}")
        st.markdown(f"""
        ### AI Analysis Results for **{city}** (Simulated)
        - **Main Pollutants:** PM2.5, NOx, SO2
        - **Likely Sources:**
          - Vehicle emissions
          - Industrial activity
          - Biomass/garbage burning
        - **Air Quality Index (AQI):** 185 (Unhealthy)
        - **Recommendation:** Limit outdoor activity. Use masks. Air purifiers recommended indoors.
        """)
