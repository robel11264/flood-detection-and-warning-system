# ===========================
# STREAMLIT DASHBOARD + HOURLY AUTOMATION + TELEGRAM ALERTS + RAINFALL DATA + SUMMARIES
# ===========================

import os
import logging
import shutil
from dataclasses import dataclass
from datetime import datetime, timedelta
import threading
import time

import numpy as np
import pandas as pd
import rasterio
import richdem as rd
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from matplotlib.patches import Patch
from scipy.interpolate import griddata
import streamlit as st
import schedule
from telegram import Bot
import requests

# ---------------- CONFIGURATION ----------------
@dataclass
class Config:
    dem_path: str = r"C:\Users\gfeka\Desktop\phyton\dire_dawa_clipped_dem.tif"
    output_dir: str = r"C:\Users\gfeka\Desktop\phyton\dashboard_daily"
    hourly_layer_dir: str = r"C:\Users\gfeka\Desktop\phyton\dashboard_hourly_layers"
    telegram_token: str = "8340911108:AAFsikzkUZCQncL173e9AFObATKT68Fa9p8"
    telegram_chat_id: int = -1003045822376
    landcover_path: str = r"C:\Users\gfeka\Desktop\phyton\dire_dawa_landcover_2020.tif"
    soil_path: str = r"C:\Users\gfeka\Desktop\phyton\soil_aligned.tif"
    river_path: str = r"C:\Users\gfeka\Desktop\phyton\dire_dawa_rivers.tif"
    lat: float = 9.03   # Dire Dawa latitude
    lon: float = 38.74  # Dire Dawa longitude

cfg = Config()
os.makedirs(cfg.output_dir, exist_ok=True)
os.makedirs(cfg.hourly_layer_dir, exist_ok=True)
logging.basicConfig(level=logging.INFO)
bot = Bot(token=cfg.telegram_token)

# ---------------- HELPER FUNCTIONS ----------------
def read_raster(path):
    with rasterio.open(path) as src:
        arr = src.read(1, masked=True).astype(np.float32)
        arr = arr.filled(np.nan)
    return arr

def save_image(data, path, cmap_name="terrain", overlay=None, alpha=0.5,
               title="", legend_labels=None, is_continuous=False, cbar_label=None):
    plt.figure(figsize=(8, 6))
    plot_data = np.nan_to_num(data, nan=0.0)
    
    if is_continuous:
        plot_data_norm = (plot_data - np.nanmin(plot_data)) / (np.nanmax(plot_data) - np.nanmin(plot_data) + 1e-6)
        im = plt.imshow(plot_data_norm, cmap=plt.get_cmap(cmap_name))
    else:
        im = plt.imshow(plot_data, cmap=plt.get_cmap(cmap_name), vmin=0, vmax=None)
    
    if overlay is not None:
        overlay_data = np.nan_to_num(overlay, nan=0.0)
        plt.imshow(overlay_data, cmap="coolwarm", alpha=alpha)
    
    plt.text(0.02*data.shape[1], 0.05*data.shape[0], title,
             color="white", fontsize=14, weight='bold',
             bbox=dict(facecolor='black', alpha=0.5, boxstyle='round,pad=0.3'))
    
    if legend_labels is not None and not is_continuous:
        handles = [Patch(color=color, label=label) for label, color in legend_labels.items()]
        plt.legend(handles=handles, loc='lower right', fontsize=10, framealpha=0.7)
    
    if is_continuous:
        cbar = plt.colorbar(im, fraction=0.046, pad=0.04)
        cbar.set_label(cbar_label if cbar_label else "Value", rotation=270, labelpad=15)
    
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(path, bbox_inches="tight", pad_inches=0, dpi=150)
    plt.close()
    logging.info(f"Saved image: {path}")
    return path

def compute_hillshade(dem_arr):
    ls = LightSource(azdeg=315, altdeg=45)
    hillshade = ls.shade(dem_arr, cmap=plt.cm.gray, vert_exag=1, blend_mode="overlay")
    return hillshade

def normalize(arr):
    arr_min, arr_max = np.nanmin(arr), np.nanmax(arr)
    return (arr - arr_min) / (arr_max - arr_min + 1e-6)

def compute_river_proximity(river_arr):
    from scipy.ndimage import distance_transform_edt
    binary_river = (river_arr > 0).astype(int)
    distance = distance_transform_edt(1 - binary_river)
    proximity = 1 - distance / (np.max(distance)+1e-6)
    return proximity

def compute_flood_risk(precip_grid, soil_arr, river_arr, w_precip=0.5, w_soil=0.3, w_river=0.2):
    precip_norm = normalize(precip_grid)
    soil_norm = normalize(soil_arr)
    river_prox = compute_river_proximity(river_arr)
    risk_score = w_precip*precip_norm + w_soil*soil_norm + w_river*river_prox
    return risk_score

def send_risk_alert(risk_score, threshold=0.6):
    avg_risk = np.nanmean(risk_score)
    if avg_risk >= threshold:
        risk_path = os.path.join(cfg.output_dir, "flood_risk_map.png")
        save_image(risk_score, risk_path, cmap_name="Reds", title="Flood Risk Score", 
                   is_continuous=True, cbar_label="Flood Risk Score")
        bot.send_message(cfg.telegram_chat_id,
                         f"⚠️ Flood Risk Alert! Average risk score: {avg_risk:.2f}")
        with open(risk_path, "rb") as img:
            bot.send_photo(cfg.telegram_chat_id, img, caption="Flood Risk Map")

# ---------------- WEATHER DATA FUNCTIONS -----------------
def fetch_hourly_rainfall():
    url = f"https://api.open-meteo.com/v1/forecast?latitude={cfg.lat}&longitude={cfg.lon}&hourly=precipitation&forecast_days=1&timezone=auto"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        df = pd.DataFrame({
            "datetime": pd.to_datetime(data['hourly']['time']),
            "precip": data['hourly']['precipitation']
        })
        return df
    except Exception as e:
        logging.error(f"Failed to fetch hourly rainfall: {e}")
        return pd.DataFrame(columns=['datetime','precip'])

def fetch_historical_rainfall():
    end_date = datetime.utcnow().date()
    start_date = end_date - timedelta(days=14)
    url = f"https://archive-api.open-meteo.com/v1/archive?latitude={cfg.lat}&longitude={cfg.lon}&start_date={start_date}&end_date={end_date}&daily=precipitation_sum&timezone=auto"
    try:
        response = requests.get(url, timeout=10)
        data = response.json()
        df = pd.DataFrame({
            "datetime": pd.to_datetime(data['daily']['time']),
            "precip": data['daily']['precipitation_sum']
        })
        return df
    except Exception as e:
        logging.error(f"Failed to fetch historical rainfall: {e}")
        return pd.DataFrame(columns=['datetime','precip'])

# ---------------- DASHBOARD FUNCTION -----------------
def main_dashboard():
    logging.info(f"Dashboard update started: {datetime.now()}")
    
    dem = read_raster(cfg.dem_path)
    slope = rd.TerrainAttribute(rd.rdarray(dem, no_data=np.nan), attrib="slope_degrees")
    aspect = rd.TerrainAttribute(rd.rdarray(dem, no_data=np.nan), attrib="aspect")
    hillshade = compute_hillshade(dem)

    landcover = read_raster(cfg.landcover_path)
    soil = read_raster(cfg.soil_path)
    river = read_raster(cfg.river_path)

    # Fetch rainfall
    hourly_rainfall = fetch_hourly_rainfall()
    historical_rainfall = fetch_historical_rainfall()

    # Interpolate precipitation grid safely
    if not hourly_rainfall.empty and len(hourly_rainfall) >= 3:
        stations = [(i*10 + np.random.rand()*1e-3, i*5 + np.random.rand()*1e-3) 
                    for i in range(len(hourly_rainfall))]
        values = hourly_rainfall['precip'].values
        xs = np.linspace(0, dem.shape[1]-1, dem.shape[1])
        ys = np.linspace(0, dem.shape[0]-1, dem.shape[0])
        grid_x, grid_y = np.meshgrid(xs, ys)
        precip_grid = griddata(points=stations, values=values, xi=(grid_x, grid_y),
                               method="nearest", fill_value=0)
    else:
        precip_grid = np.zeros(dem.shape)

    # Flood risk
    risk_score = compute_flood_risk(precip_grid, soil, river)
    send_risk_alert(risk_score)

    # Legends
    landcover_legend = {"Urban": "#1f77b4", "Forest": "#2ca02c", "Agriculture": "#ff7f0e", "Water": "#17becf", "Bare": "#d62728"}
    soil_legend = {"Low": "#deebf7", "Medium": "#9ecae1", "High": "#3182bd"}
    river_legend = {"River": "#08519c"}

    # Save maps
    hillshade_path = save_image(hillshade, os.path.join(cfg.output_dir,"hillshade.png"), cmap_name="Greys", title="Hillshade", is_continuous=True, cbar_label="Hillshade")
    slope_path = save_image(slope, os.path.join(cfg.output_dir,"slope.png"), cmap_name="viridis", title="Slope (°)", is_continuous=True, cbar_label="Slope (°)")
    aspect_path = save_image(aspect, os.path.join(cfg.output_dir,"aspect.png"), cmap_name="plasma", title="Aspect (°)", is_continuous=True, cbar_label="Aspect (°)")
    landcover_path = save_image(landcover, os.path.join(cfg.output_dir,"landcover.png"), cmap_name="tab20", title="Landcover", legend_labels=landcover_legend)
    soil_path = save_image(soil, os.path.join(cfg.output_dir,"soil_moisture.png"), cmap_name="Blues", title="Soil Moisture", legend_labels=soil_legend)
    river_path = save_image(river, os.path.join(cfg.output_dir,"river.png"), cmap_name="Blues", title="River Map", legend_labels=river_legend)
    precip_path = save_image(precip_grid, os.path.join(cfg.output_dir,"precipitation_heatmap.png"), cmap_name="terrain", title="Precipitation (mm)", is_continuous=True, cbar_label="Precipitation (mm)")
    risk_path = save_image(risk_score, os.path.join(cfg.output_dir,"flood_risk_score.png"), cmap_name="Reds", title="Flood Risk Score", is_continuous=True, cbar_label="Flood Risk Score")

    # Save hourly snapshot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M")
    layer_copy = os.path.join(cfg.hourly_layer_dir, f"flood_risk_{timestamp}.png")
    shutil.copy(risk_path, layer_copy)

    logging.info(f"Dashboard update finished: {datetime.now()}")
    return hillshade_path, slope_path, aspect_path, landcover_path, soil_path, river_path, precip_path, risk_path, hourly_rainfall, historical_rainfall

# ---------------- AUTOMATION -----------------
def run_hourly_dashboard():
    logging.info(f"Starting scheduled dashboard update: {datetime.now()}")
    main_dashboard()
    logging.info(f"Scheduled dashboard update finished: {datetime.now()}")

def start_scheduler():
    schedule.every().hour.do(run_hourly_dashboard)
    logging.info("Hourly automation scheduler started...")
    while True:
        schedule.run_pending()
        time.sleep(60)

scheduler_thread = threading.Thread(target=start_scheduler, daemon=True)
scheduler_thread.start()

# ---------------- STREAMLIT DASHBOARD -----------------
st.set_page_config(page_title="Hourly Flood Dashboard", page_icon="🌊", layout="wide")
st.title("🌊 Hourly Flood Dashboard for Dire Dawa")
st.markdown("This dashboard shows **terrain, landcover, soil moisture, river maps**, **precipitation & flood risk**, and **rainfall data**.")

# Display maps
hillshade_path, slope_path, aspect_path, landcover_path, soil_path, river_path, precip_path, risk_path, hourly_rainfall, historical_rainfall = main_dashboard()

col1, col2, col3 = st.columns(3)
col1.image(hillshade_path, caption="Hillshade")
col1.image(slope_path, caption="Slope")
col1.image(aspect_path, caption="Aspect")

col2.image(landcover_path, caption="Landcover")
col2.image(soil_path, caption="Soil Moisture")
col2.image(river_path, caption="River Map")

col3.image(precip_path, caption="Precipitation Heatmap")
col3.image(risk_path, caption="Flood Risk Score")

# Hourly rainfall chart
st.subheader("Hourly Rainfall Forecast (Next 24 Hours)")
st.dataframe(hourly_rainfall)
if not hourly_rainfall.empty:
    st.line_chart(hourly_rainfall.set_index('datetime')['precip'])

# Last 14 days rainfall chart
st.subheader("Historical Rainfall (Past 14 Days)")
st.dataframe(historical_rainfall)
if not historical_rainfall.empty:
    st.line_chart(historical_rainfall.set_index('datetime')['precip'])

st.success("✅ Dashboard Updated Successfully!")
