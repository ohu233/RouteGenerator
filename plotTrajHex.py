"""
Plot each hex grid trajectory individually on Gaode (Amap) tile map.
Generates one PNG image per trajectory under data/traj_plots/.
"""
import pickle
import math
import io
import os
import warnings
import numpy as np
import pandas as pd
import requests
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

warnings.filterwarnings('ignore')

# ── Config ───────────────────────────────────────────────────────────

TILE_URL = "https://webrd0{s}.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}"
TILE_SIZE = 256
OUTPUT_DIR = "data/traj_plots"
MAX_TRAJS = None  # set to a number to limit output, None = all

MODE_COLORS = {'TG': '#e31a1c', 'GG': '#1f78b4', 'GSD': '#fdbf6f', 'TS': '#33a02c'}
MODE_LABELS = {'TG': '高铁 TG', 'GG': '高速 GG', 'GSD': '国省道 GSD', 'TS': '普铁 TS'}

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Gaode tile helpers ──────────────────────────────────────────────

def lonlat_to_tile_xy(lon, lat, zoom):
    n = 2 ** zoom
    x = (lon + 180.0) / 360.0 * n
    lat_rad = math.radians(lat)
    y = (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * n
    return x, y


def fetch_tile(x, y, z):
    for s in ['1', '2', '3', '4']:
        url = TILE_URL.format(s=s, x=x, y=y, z=z)
        try:
            resp = requests.get(url, timeout=10, headers={'User-Agent': 'Mozilla/5.0'})
            if resp.status_code == 200 and len(resp.content) > 100:
                return Image.open(io.BytesIO(resp.content))
        except Exception:
            continue
    return None


def get_tile_background(lon_min, lon_max, lat_min, lat_max, zoom):
    """Fetch and stitch tiles for a bounding box. Returns (PIL.Image, extent_tuple)."""
    x_min, y_top = lonlat_to_tile_xy(lon_min, lat_max, zoom)
    x_max, y_bot = lonlat_to_tile_xy(lon_max, lat_min, zoom)

    tx0 = int(math.floor(min(x_min, x_max)))
    tx1 = int(math.floor(max(x_min, x_max)))
    ty0 = int(math.floor(min(y_top, y_bot)))
    ty1 = int(math.floor(max(y_top, y_bot)))

    ncols = tx1 - tx0 + 1
    nrows = ty1 - ty0 + 1

    if ncols <= 0 or nrows <= 0 or ncols > 10 or nrows > 10:
        return None, None

    canvas = Image.new('RGB', (ncols * TILE_SIZE, nrows * TILE_SIZE), (240, 240, 235))
    for i, tx in enumerate(range(tx0, tx1 + 1)):
        for j, ty in enumerate(range(ty0, ty1 + 1)):
            tile = fetch_tile(tx, ty, zoom)
            if tile:
                canvas.paste(tile, (i * TILE_SIZE, j * TILE_SIZE))

    # Extent in lon/lat
    n = 2 ** zoom
    left_lon = tx0 / n * 360.0 - 180.0
    right_lon = (tx1 + 1) / n * 360.0 - 180.0
    top_lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * ty0 / n))))
    bottom_lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * (ty1 + 1) / n))))

    return canvas, (left_lon, right_lon, bottom_lat, top_lat)


# ── Load data ───────────────────────────────────────────────────────

print("Loading hex_grid.pkl ...")
with open('data/hex_grid.pkl', 'rb') as f:
    hex_grid = pickle.load(f)
print(f"  {len(hex_grid)} hex cells")

coord_to_lonlat = {k: (v['lon'], v['lat']) for k, v in hex_grid.items()}
valid_keys_set = set(coord_to_lonlat.keys())
HEX_OFFSETS_L = [(0,0,0), (1,0,-1), (1,-1,0), (0,-1,1), (-1,0,1), (-1,1,0), (0,1,-1)]

print("Loading trajectory CSV ...")
df = pd.read_csv('data/artificial_traj_mixed_single.csv')
print(f"  {len(df)} trajectory points, {df['ID'].nunique()} trajectories")

# ── Convert cube → lon/lat ──────────────────────────────────────────

q_arr = np.round(df['locx'].values).astype(int)
r_arr = np.round(df['locy'].values).astype(int)
s_arr = -q_arr - r_arr  # enforce q+r+s=0

lons = np.full(len(df), np.nan)
lats = np.full(len(df), np.nan)

for i in range(len(df)):
    key = (q_arr[i], r_arr[i], s_arr[i])
    if key not in valid_keys_set:
        for dq, dr, ds in HEX_OFFSETS_L:
            nkey = (q_arr[i] + dq, r_arr[i] + dr, s_arr[i] + ds)
            if nkey in valid_keys_set:
                key = nkey
                break
        else:
            key = None
    if key is not None:
        lons[i], lats[i] = coord_to_lonlat[key]

df['lon'] = lons
df['lat'] = lats
df_valid = df.dropna(subset=['lon', 'lat']).copy()
print(f"  {len(df_valid)} points mapped to lon/lat")

# ── Plot each trajectory ────────────────────────────────────────────

traj_ids = sorted(df_valid['ID'].unique())
if MAX_TRAJS:
    traj_ids = traj_ids[:MAX_TRAJS]

print(f"\nPlotting {len(traj_ids)} trajectories to {OUTPUT_DIR}/ ...")

for idx, traj_id in enumerate(traj_ids):
    grp = df_valid[df_valid['ID'] == traj_id].sort_values('time')
    mode = grp['mode'].iloc[0]
    color = MODE_COLORS.get(mode, '#000000')
    label = MODE_LABELS.get(mode, mode)

    # Bounding box for this trajectory
    t_lon_min, t_lon_max = grp['lon'].min(), grp['lon'].max()
    t_lat_min, t_lat_max = grp['lat'].min(), grp['lat'].max()

    # Add padding
    lon_pad = max((t_lon_max - t_lon_min) * 0.3, 0.005)
    lat_pad = max((t_lat_max - t_lat_min) * 0.3, 0.005)
    t_lon_min -= lon_pad; t_lon_max += lon_pad
    t_lat_min -= lat_pad; t_lat_max += lat_pad

    # Determine zoom
    extent_deg = max(t_lon_max - t_lon_min, t_lat_max - t_lat_min)
    zoom = int(max(7, min(13, math.floor(math.log2(360 / extent_deg)))))

    # Fetch tiles
    bg_img, extent = get_tile_background(t_lon_min, t_lon_max, t_lat_min, t_lat_max, zoom)

    # Plot — compute aspect ratio to prevent stretch
    mid_lat = (t_lat_min + t_lat_max) / 2.0
    data_aspect = 1.0 / math.cos(math.radians(mid_lat))

    fig_width = 10
    fig_height = fig_width * (t_lat_max - t_lat_min) / (t_lon_max - t_lon_min) * data_aspect
    fig_height = max(4, min(16, fig_height))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    if bg_img and extent:
        ax.imshow(bg_img, extent=extent, aspect='auto', interpolation='bilinear')

    # Plot the trajectory line with direction markers
    lons_t = grp['lon'].values
    lats_t = grp['lat'].values
    ax.plot(lons_t, lats_t, '-o', color=color, linewidth=2, markersize=4,
            markerfacecolor='white', markeredgecolor=color, markeredgewidth=1.5)

    # Mark start and end
    ax.plot(lons_t[0], lats_t[0], 'o', color='green', markersize=8, zorder=5, label='Start')
    ax.plot(lons_t[-1], lats_t[-1], 's', color='red', markersize=8, zorder=5, label='End')

    ax.set_xlim(t_lon_min, t_lon_max)
    ax.set_ylim(t_lat_min, t_lat_max)
    ax.set_aspect(data_aspect)  # correct Mercator stretch: 1°lon = cos(lat)*1°lat
    ax.set_xlabel('Longitude (°E)')
    ax.set_ylabel('Latitude (°N)')
    ax.set_title(f'{label}  |  {traj_id}  |  {len(grp)} points')

    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
    plt.tight_layout()

    fname = os.path.join(OUTPUT_DIR, f'{traj_id}.png')
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close(fig)

    if (idx + 1) % 50 == 0:
        print(f"  {idx + 1}/{len(traj_ids)} done")

print(f"Done. {len(traj_ids)} plots saved to {OUTPUT_DIR}/")
