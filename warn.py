# ================= UPDATED DAILY FLOOD DASHBOARD SCRIPT (SOIL MOISTURE + FLOOD RISK + PRECIPITATION) =================

import os
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import rasterio
from rasterio.enums import Resampling
import richdem as rd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
import requests
from PIL import Image

# ---------------- CONFIG ----------------
@dataclass
class Config:
    project_dir: str
    output_dir: str
    dem_file: str
    landcover_file: str
    telegram_bot_token: str
    telegram_chat_id: int
    city: str
    api_key: str
    days_back: int = 14
    scale_factor: float = 0.5

cfg = Config(
    project_dir=r"C:\Users\gfeka\Desktop\phyton",
    output_dir=os.path.join(r"C:\Users\gfeka\Desktop\phyton", "dashboard_daily"),
    dem_file=os.path.join(r"C:\Users\gfeka\Desktop\phyton", "dire_dawa_clipped_dem.tif"),
    landcover_file=os.path.join(r"C:\Users\gfeka\Desktop\phyton", "dire_dawa_landcover_2020.tif"),
    telegram_bot_token="8340911108:AAFsikzkUZCQncL173e9AFObATKT68Fa9p8",
    telegram_chat_id=-1003045822376,
    city="Dire Dawa, Ethiopia",
    api_key="35d23a14d90e4d74ad464140251409"
)
os.makedirs(cfg.output_dir, exist_ok=True)

DAYS_BACK = cfg.days_back
END_DATE = min(datetime.today() - timedelta(days=1), datetime.today())
START_DATE = END_DATE - timedelta(days=DAYS_BACK - 1)

logging.basicConfig(level=logging.INFO)

# ---------------- TELEGRAM ----------------
def send_telegram_photo(photo_path, caption=""):
    try:
        size_mb = os.path.getsize(photo_path) / (1024*1024)
        if size_mb > 9:
            img = Image.open(photo_path)
            img.thumbnail((1024, 1024))
            img.save(photo_path)
            logging.info(f"Image resized to fit Telegram limits: {photo_path}")

        with open(photo_path, "rb") as f:
            response = requests.post(
                f"https://api.telegram.org/bot{cfg.telegram_bot_token}/sendPhoto",
                data={"chat_id": cfg.telegram_chat_id, "caption": caption},
                files={"photo": f}
            )
        if response.status_code != 200:
            logging.warning(f"Telegram API responded with {response.status_code}: {response.text}")
        else:
            logging.info(f"Sent photo to Telegram: {photo_path}")

    except Exception as e:
        logging.warning(f"Failed to send Telegram photo: {e}")

# ---------------- HELPERS ----------------
def normalize_array(arr):
    arr_min = np.nanmin(arr)
    arr_max = np.nanmax(arr)
    if arr_max == arr_min:
        return np.zeros_like(arr)
    return (arr - arr_min) / (arr_max - arr_min)

def compute_hillshade(dem_array, geotransform):
    dem_rd = rd.rdarray(dem_array, no_data=-9999)
    dem_rd.geotransform = geotransform
    slope = rd.TerrainAttribute(dem_rd, attrib='slope_radians')
    aspect = rd.TerrainAttribute(dem_rd, attrib='aspect')
    hs = 255.0 * ((np.cos(slope) * np.cos(np.radians(45))) +
                  (np.sin(slope) * np.sin(np.radians(45)) *
                   np.cos(aspect - np.radians(315))))
    return np.clip(hs, 0, 255)

LC_RUNOFF = {50: 0.9, 40: 0.5, 30: 0.3, 10: 0.2, 80: 1.0}

def compute_flood_risk(lc_crop, slope_array):
    runoff_coeff = np.vectorize(lambda x: LC_RUNOFF.get(x, 0.0))(lc_crop)
    slope_norm = normalize_array(slope_array)
    flood_index = runoff_coeff * (1 + slope_norm)
    flood_norm = normalize_array(flood_index)
    return flood_norm

def fetch_weather(date):
    dt_str = date.strftime("%Y-%m-%d")
    if date >= datetime.today():
        return 0.0, 2.0
    url = f"http://api.weatherapi.com/v1/history.json?key={cfg.api_key}&q={cfg.city}&dt={dt_str}"
    try:
        r = requests.get(url, timeout=12)
        r.raise_for_status()
        data = r.json()
        day = data['forecast']['forecastday'][0]['day']
        total_rain = sum(float(h.get('precip_mm', 0)) for h in data['forecast']['forecastday'][0]['hour'])
        ET = day.get('avgtemp_c', 2) * 0.5
        return total_rain, ET
    except Exception as e:
        logging.warning(f"Weather fetch failed for {dt_str}: {e}")
        return 0.0, 2.0

# ---------------- SOIL MOISTURE ----------------
def simulate_soil_moisture(dem_arr):
    sm = np.zeros_like(dem_arr, dtype=float)
    dates = [START_DATE + timedelta(days=i) for i in range(DAYS_BACK)]
    with ThreadPoolExecutor(max_workers=5) as executor:
        results = list(executor.map(fetch_weather, dates))
    for rain, ET in results:
        sm += rain - ET
        sm = np.clip(sm, 0, None)
    return sm, dates, [r for r, _ in results]

# ---------------- PLOTTING ----------------
def plot_save_image(img_data, cmap, norm=None, title="", output_path="", alpha=1, add_legend=False, colors=None, labels=None):
    fig, ax = plt.subplots(figsize=(8,6))
    ax.imshow(img_data, cmap=cmap, norm=norm, alpha=alpha)
    ax.set_title(title)
    ax.axis('off')
    if add_legend and colors and labels:
        patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(len(labels))]
        ax.legend(handles=patches, loc='lower center', bbox_to_anchor=(0.5, -0.15), ncol=len(labels))
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logging.info(f"{title} saved: {output_path}")

def plot_precipitation(dates, precipitation, output_path):
    fig, ax = plt.subplots(figsize=(10,4))
    ax.bar(dates, precipitation, color='skyblue')
    ax.set_title("Daily Precipitation (mm)")
    ax.set_ylabel("Precipitation (mm)")
    ax.set_xlabel("Date")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    logging.info(f"Precipitation chart saved: {output_path}")

def plot_soil_moisture_flood_precip(sm_array, flood_array, precipitation, dates, output_path):
    import matplotlib.cm as cm
    fig, ax = plt.subplots(figsize=(10,6))

    # Soil moisture heatmap
    sm_norm = normalize_array(sm_array)
    sm_img = cm.get_cmap('Blues')(sm_norm)
    img_sm = ax.imshow(sm_img, alpha=0.7)

    # Flood risk overlay with alpha varying by risk
    flood_norm = normalize_array(flood_array)
    cmap_flood = mcolors.ListedColormap(['#00FF00', '#FFFF00', '#FFA500', '#FF0000'])
    bounds = [0,0.25,0.5,0.75,1]
    norm_flood = mcolors.BoundaryNorm(bounds, cmap_flood.N)
    img_flood = ax.imshow(flood_norm, cmap=cmap_flood, norm=norm_flood, alpha=flood_norm*0.6 + 0.1)

    ax.axis('off')

    # Add soil moisture legend
    cbar_sm = fig.colorbar(img_sm, ax=ax, fraction=0.03, pad=0.02)
    cbar_sm.set_label("Soil Moisture (normalized)", rotation=270, labelpad=15)

    # Add flood risk legend
    cbar_flood = fig.colorbar(img_flood, ax=ax, fraction=0.03, pad=0.08, ticks=[0.125,0.375,0.625,0.875])
    cbar_flood.ax.set_yticklabels(['Low','Moderate','High','Very High'])
    cbar_flood.set_label("Flood Risk Level", rotation=270, labelpad=15)

    # Precipitation bar chart below
    ax2 = fig.add_axes([0.15, -0.15, 0.7, 0.1])
    ax2.bar(dates, precipitation, color='skyblue')
    ax2.set_title("Daily Precipitation (mm)", fontsize=10)
    ax2.set_xticks(dates[::max(1,len(dates)//7)])
    ax2.set_xticklabels([d.strftime("%m-%d") for d in dates[::max(1,len(dates)//7)]], rotation=45, fontsize=8)
    ax2.set_ylabel("Precip (mm)", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logging.info(f"Soil Moisture + Flood Risk + Precipitation heatmap saved: {output_path}")

# ---------------- LOAD DATA ----------------
try:
    with rasterio.open(cfg.dem_file) as src:
        dem = src.read(1).astype(float)
        dem[dem == src.nodata] = np.nan
        dem_geotransform = src.transform.to_gdal()
    with rasterio.open(cfg.landcover_file) as src:
        lc = src.read(1, out_shape=(dem.shape[0], dem.shape[1]), resampling=Resampling.nearest)
        lc[lc == src.nodata] = 0
except Exception as e:
    logging.error(f"Failed to load raster data: {e}")
    raise

dem_rd = rd.rdarray(dem, no_data=-9999)
dem_rd.geotransform = dem_geotransform
slope_array = rd.TerrainAttribute(dem_rd, attrib='slope_degrees')
hs = compute_hillshade(dem, dem_geotransform)
flood_risk = compute_flood_risk(lc, slope_array)
sm, dates, precipitation = simulate_soil_moisture(dem)

# ---------------- COLORS ----------------
lc_colors = ['#006400','#7CFC00','#FFD700','#FF0000','#BEBEBE','#0000FF']
lc_labels = ['Tree','Grass','Cropland','Built-up','Bare','Water']
cmap_lc = mcolors.ListedColormap(lc_colors)
norm_lc = mcolors.BoundaryNorm([0,11,31,41,51,61,81], len(lc_colors))

risk_colors = ['#00FF00','#FFFF00','#FFA500','#FF0000']
risk_labels = ['Low','Moderate','High','Very High']
flood_bins = np.linspace(0,1,len(risk_colors)+1)
norm_risk = mcolors.BoundaryNorm(flood_bins, len(risk_colors))
cmap_risk = mcolors.ListedColormap(risk_colors)

# ---------------- SAVE INDIVIDUAL CHARTS ----------------
hillcover_path = os.path.join(cfg.output_dir,"hillshade_landcover.png")
combined_hill = hs.copy()
plot_save_image(combined_hill, cmap='gray', title="Hillshade", output_path=hillcover_path)
plt.imshow(hs,cmap='gray',alpha=1)
plt.imshow(lc,cmap=cmap_lc,norm=norm_lc,alpha=0.7)
plt.axis('off')
plt.tight_layout()
plt.savefig(hillcover_path,dpi=150)
plt.close()

flood_path = os.path.join(cfg.output_dir,"flood_risk.png")
plot_save_image(flood_risk, cmap=cmap_risk, norm=norm_risk, title="Flood Risk Levels", output_path=flood_path, add_legend=True, colors=risk_colors, labels=risk_labels)

soil_flood_precip_path = os.path.join(cfg.output_dir,"soil_flood_precip_heatmap.png")
plot_soil_moisture_flood_precip(sm, flood_risk, precipitation, dates, soil_flood_precip_path)

stats_path = os.path.join(cfg.output_dir,"stats.png")
fig,ax = plt.subplots(figsize=(6,4))
ax.axis('off')
stats_text = (f"Date: {datetime.today().strftime('%Y-%m-%d')}\n"
              f"Mean DEM: {np.nanmean(dem):.2f}\n"
              f"Slope Avg: {np.nanmean(slope_array):.2f}\n"
              f"Max Flood Risk: {flood_risk.max():.2f}")
ax.text(0.5,0.5,stats_text,fontsize=12,ha='center',va='center')
plt.tight_layout()
plt.savefig(stats_path,dpi=150)
plt.close()

precip_chart_path = os.path.join(cfg.output_dir,"precipitation.png")
plot_precipitation(dates, precipitation, precip_chart_path)

# ---------------- SEND CHARTS VIA TELEGRAM ----------------
send_telegram_photo(hillcover_path, caption="📊 Hillshade + Landcover")
send_telegram_photo(flood_path, caption="📊 Flood Risk Levels")
send_telegram_photo(soil_flood_precip_path, caption="📊 Soil Moisture + Flood Risk + Precipitation (Last 14 Days)")
send_telegram_photo(stats_path, caption="📊 Stats Summary")
send_telegram_photo(precip_chart_path, caption="🌧️ Daily Precipitation (Last 14 Days)")

logging.info("✅ All charts sent as zoomable media via Telegram")
