"""
Plot each hex grid trajectory individually on Gaode (Amap) tile map.
Overlays road network hex cells for the matching mode.
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
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
from PIL import Image
from collections import defaultdict
from hexUtils import hex_round

warnings.filterwarnings('ignore')


# ── Coordinate conversion: WGS-84 → GCJ-02 ──────────────────────────

def wgs84_to_gcj02(lon, lat):
    """Convert WGS-84 to GCJ-02 (Mars coordinates used by Chinese map services)."""
    a = 6378245.0
    ee = 0.00669342162296594323

    def _transform_lat(x, y):
        ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * math.sqrt(abs(x))
        ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
        ret += (20.0 * math.sin(y * math.pi) + 40.0 * math.sin(y / 3.0 * math.pi)) * 2.0 / 3.0
        ret += (160.0 * math.sin(y / 12.0 * math.pi) + 320.0 * math.sin(y * math.pi / 30.0)) * 2.0 / 3.0
        return ret

    def _transform_lon(x, y):
        ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(abs(x))
        ret += (20.0 * math.sin(6.0 * x * math.pi) + 20.0 * math.sin(2.0 * x * math.pi)) * 2.0 / 3.0
        ret += (20.0 * math.sin(x * math.pi) + 40.0 * math.sin(x / 3.0 * math.pi)) * 2.0 / 3.0
        ret += (150.0 * math.sin(x / 12.0 * math.pi) + 300.0 * math.sin(x / 30.0 * math.pi)) * 2.0 / 3.0
        return ret

    dlat = _transform_lat(lon - 105.0, lat - 35.0)
    dlon = _transform_lon(lon - 105.0, lat - 35.0)
    radlat = lat / 180.0 * math.pi
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * math.pi)
    dlon = (dlon * 180.0) / (a / sqrtmagic * math.cos(radlat) * math.pi)
    return lon + dlon, lat + dlat


# Vectorized version for numpy arrays
_wgs84_to_gcj02_vec = np.vectorize(wgs84_to_gcj02, otypes=[np.float64, np.float64])

# ── Config ───────────────────────────────────────────────────────────

TILE_URL = "https://webrd0{s}.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}"
TILE_SIZE = 256
OUTPUT_DIR = "data/traj_plots"
TRAJ_CSV = "data\\artificial_od_all.csv"  # trajectory or OD CSV file
MAX_TRAJS = None  # set to a number to limit output, None = all
MAX_TRAJS_PER_MODE = None  # e.g. 5 to plot only 5 per mode, None = all
PLOT_MODES = ['TG', 'TS']  # set to subset, e.g. ['TG', 'TS']

MODE_COLORS = {'TG': '#e31a1c', 'GG': '#1f78b4', 'GSD': '#fdbf6f', 'TS': '#33a02c'}
MODE_LABELS = {'TG': '高铁 TG', 'GG': '高速 GG', 'GSD': '国省道 GSD', 'TS': '普铁 TS'}
ROAD_ALPHA = 0.5  # transparency of road network overlay
HEX_EDGE_M = 200    # hex edge length in meters
CENTER_DIST_M = math.sqrt(3) * HEX_EDGE_M  # ~346.4m

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

    n = 2 ** zoom
    left_lon = tx0 / n * 360.0 - 180.0
    right_lon = (tx1 + 1) / n * 360.0 - 180.0
    top_lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * ty0 / n))))
    bottom_lat = math.degrees(math.atan(math.sinh(math.pi * (1.0 - 2.0 * (ty1 + 1) / n))))

    return canvas, (left_lon, right_lon, bottom_lat, top_lat)


# ── Road network helpers ─────────────────────────────────────────────

def _allow(code: int, mode: str) -> bool:
    if mode == 'TG':
        return code >> 6 & 1 == 1
    elif mode == 'GG':
        return code >> 4 & 1 == 1 or code >> 3 & 1 == 1
    elif mode == 'GSD':
        return code >> 5 & 1 == 1 or code >> 2 & 1 == 1
    elif mode == 'TS':
        return code >> 1 & 1 == 1
    return False


def build_road_spatial_index(hex_grid, bin_size=0.1):
    """Build a spatial index of road cells binned by lon/lat.
    Returns dict: mode -> {(bin_lon, bin_lat): [(lon, lat), ...]}"""
    index = {m: defaultdict(list) for m in ['TG', 'GG', 'GSD', 'TS']}
    for _, info in hex_grid.items():
        code = info['code']
        lon, lat = info['lon'], info['lat']
        blon = int(lon / bin_size)
        blat = int(lat / bin_size)
        for m in ['TG', 'GG', 'GSD', 'TS']:
            if _allow(code, m):
                index[m][(blon, blat)].append((lon, lat))
    # Convert defaultdict to regular dict and lists to numpy arrays for speed
    out = {}
    for m, d in index.items():
        out[m] = {k: np.array(v) for k, v in d.items()}
        total = sum(len(v) for v in out[m].values())
        print(f"  {m}: {total} road cells in {len(out[m])} spatial bins")
    return out


def make_hex_vertices(cx, cy, mid_lat):
    """Generate 6 vertices of a flat-top hexagon centered at (lon, lat)."""
    # Convert meters to degrees at this latitude
    m_per_deg_lon = 111320.0 * math.cos(math.radians(mid_lat))
    m_per_deg_lat = 111320.0
    # Flat-top hex: vertices at angles 0°, 60°, 120°, 180°, 240°, 300° from center
    # but flattened vertically
    w = HEX_EDGE_M  # half-width in meters (center to right/left vertex)
    h = HEX_EDGE_M * math.sqrt(3) / 2  # half-height in meters
    # Vertices in meters (flat-top: straight edges on top/bottom, points on left/right)
    # Going clockwise from rightmost
    verts_m = np.array([
        [w, 0],
        [w / 2, h],
        [-w / 2, h],
        [-w, 0],
        [-w / 2, -h],
        [w / 2, -h],
    ])
    verts_lon = cx + verts_m[:, 0] / m_per_deg_lon
    verts_lat = cy + verts_m[:, 1] / m_per_deg_lat
    return np.column_stack([verts_lon, verts_lat])


def get_road_cells_in_bbox(road_index, mode, lon_min, lon_max, lat_min, lat_max, bin_size=0.1):
    """Query road cells within a bounding box from the spatial index."""
    b_lon_min = int(lon_min / bin_size)
    b_lon_max = int(lon_max / bin_size)
    b_lat_min = int(lat_min / bin_size)
    b_lat_max = int(lat_max / bin_size)
    results = []
    for blon in range(b_lon_min, b_lon_max + 1):
        for blat in range(b_lat_min, b_lat_max + 1):
            arr = road_index.get(mode, {}).get((blon, blat))
            if arr is not None and len(arr) > 0:
                mask = (arr[:, 0] >= lon_min) & (arr[:, 0] <= lon_max) & \
                       (arr[:, 1] >= lat_min) & (arr[:, 1] <= lat_max)
                if mask.any():
                    results.append(arr[mask])
    if results:
        return np.concatenate(results)
    return np.empty((0, 2))


# ── Load data ───────────────────────────────────────────────────────

print("Loading hex_grid.pkl ...")
with open('data/hex_grid.pkl', 'rb') as f:
    hex_grid = pickle.load(f)
print(f"  {len(hex_grid)} hex cells")

print("Building road spatial index (WGS-84 → GCJ-02) ...")
# Convert hex grid coords to GCJ-02 for Gaode tile matching
hex_grid_gcj = {}
for k, v in hex_grid.items():
    gcj_lon, gcj_lat = wgs84_to_gcj02(v['lon'], v['lat'])
    hex_grid_gcj[k] = {'lon': gcj_lon, 'lat': gcj_lat, 'code': v['code']}
road_index = build_road_spatial_index(hex_grid_gcj)

coord_to_lonlat = {k: (v['lon'], v['lat']) for k, v in hex_grid_gcj.items()}
valid_keys_set = set(coord_to_lonlat.keys())
HEX_OFFSETS_L = [(0, 0, 0), (1, 0, -1), (1, -1, 0), (0, -1, 1), (-1, 0, 1), (-1, 1, 0), (0, 1, -1)]

print(f"Loading CSV: {TRAJ_CSV} ...")
df = pd.read_csv(TRAJ_CSV)
print(f"  {len(df)} rows, {df['ID'].nunique()} unique IDs")

# Detect CSV format: trajectory (locx/locy/locz) vs OD (locxo/locyo/loczo)
IS_OD = 'locxo' in df.columns

if IS_OD:
    print("  Detected OD format (origin-destination pairs)")
else:
    print("  Detected trajectory format (point sequence)")

# ── Convert cube → lon/lat ──────────────────────────────────────────

def cube_to_lonlat(q, r, s):
    """Snap a (possibly jittered) cube coordinate to a valid hex cell and return GCJ-02 lon/lat."""
    # First round to nearest valid cube coordinate (fixes jitter that broke q+r+s=0)
    rq, rr, rs = hex_round(q, r, s)
    key = (rq, rr, rs)
    if key in valid_keys_set:
        return coord_to_lonlat[key]
    # Fallback: search ±1 neighbors
    for dq, dr, ds in HEX_OFFSETS_L:
        nkey = (rq + dq, rr + dr, rs + ds)
        if nkey in valid_keys_set:
            return coord_to_lonlat[nkey]
    return None, None


if IS_OD:
    # OD format: build point sequences from O-D segments
    rows = []
    for traj_id, grp in df.groupby('ID'):
        grp = grp.reset_index(drop=True)  # preserve row order
        mode = grp['mode'].iloc[0]
        for i, (_, row) in enumerate(grp.iterrows()):
            if i == 0:
                # First origin
                qo, ro, so = int(row['locxo']), int(row['locyo']), int(row['loczo'])
                lo, la = cube_to_lonlat(qo, ro, so)
                if lo is not None:
                    rows.append({'ID': traj_id, 'lon': lo, 'lat': la, 'mode': mode, 'time': 0})
            # Destination (every segment)
            qd, rd, sd = int(row['locxd']), int(row['locyd']), int(row['loczd'])
            ld, lad = cube_to_lonlat(qd, rd, sd)
            if ld is not None:
                rows.append({'ID': traj_id, 'lon': ld, 'lat': lad, 'mode': mode, 'time': i + 1})

    df_valid = pd.DataFrame(rows)
else:
    # Trajectory format: each row is a point
    q_arr = np.round(df['locx'].values).astype(int)
    r_arr = np.round(df['locy'].values).astype(int)
    s_arr = -q_arr - r_arr
    lons, lats = np.full(len(df), np.nan), np.full(len(df), np.nan)
    for i in range(len(df)):
        lons[i], lats[i] = cube_to_lonlat(q_arr[i], r_arr[i], s_arr[i])
    df['lon'] = lons
    df['lat'] = lats
    df_valid = df.dropna(subset=['lon', 'lat']).copy()

print(f"  {len(df_valid)} points mapped to lon/lat")

# ── Plot each trajectory ────────────────────────────────────────────

if MAX_TRAJS_PER_MODE is not None:
    # Sample evenly across configured modes
    selected = []
    for m in PLOT_MODES:
        ids = sorted(df_valid[df_valid['mode'] == m]['ID'].unique())
        selected.extend(ids[:MAX_TRAJS_PER_MODE])
    traj_ids = sorted(selected)
else:
    traj_ids = sorted(df_valid[df_valid['mode'].isin(PLOT_MODES)]['ID'].unique())

if MAX_TRAJS:
    traj_ids = traj_ids[:MAX_TRAJS]

print(f"\nPlotting {len(traj_ids)} trajectories to {OUTPUT_DIR}/ ...")
print(f"  Road overlay alpha = {ROAD_ALPHA}")

for idx, traj_id in enumerate(traj_ids):
    grp = df_valid[df_valid['ID'] == traj_id].sort_values('time')
    mode = grp['mode'].iloc[0]
    color = MODE_COLORS.get(mode, '#000000')
    label = MODE_LABELS.get(mode, mode)
    road_color = MODE_COLORS.get(mode, '#888888')

    # Bounding box for this trajectory
    t_lon_min, t_lon_max = grp['lon'].min(), grp['lon'].max()
    t_lat_min, t_lat_max = grp['lat'].min(), grp['lat'].max()
    mid_lat = (t_lat_min + t_lat_max) / 2.0

    # Add padding (expand bbox for road query too)
    lon_pad = max((t_lon_max - t_lon_min) * 0.3, 0.005)
    lat_pad = max((t_lat_max - t_lat_min) * 0.3, 0.005)
    plot_lon_min = t_lon_min - lon_pad
    plot_lon_max = t_lon_max + lon_pad
    plot_lat_min = t_lat_min - lat_pad
    plot_lat_max = t_lat_max + lat_pad

    # Determine zoom
    extent_deg = max(plot_lon_max - plot_lon_min, plot_lat_max - plot_lat_min)
    zoom = int(max(7, min(13, math.floor(math.log2(360 / extent_deg)))))

    # Fetch tiles
    bg_img, extent = get_tile_background(plot_lon_min, plot_lon_max, plot_lat_min, plot_lat_max, zoom)

    # Query road cells for this mode within the bbox
    road_cells = get_road_cells_in_bbox(road_index, mode,
                                        plot_lon_min, plot_lon_max,
                                        plot_lat_min, plot_lat_max)

    # Plot
    mid_lat_plot = (plot_lat_min + plot_lat_max) / 2.0
    data_aspect = 1.0 / math.cos(math.radians(mid_lat_plot))

    fig_width = 10
    fig_height = fig_width * (plot_lat_max - plot_lat_min) / (plot_lon_max - plot_lon_min) * data_aspect
    fig_height = max(4, min(16, fig_height))
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))

    if bg_img and extent:
        ax.imshow(bg_img, extent=extent, aspect='auto', interpolation='bilinear')

    # Draw road network hex cells
    if len(road_cells) > 0:
        patches = []
        for lon, lat in road_cells:
            verts = make_hex_vertices(lon, lat, mid_lat_plot)
            patches.append(Polygon(verts, closed=True))
        pc = PatchCollection(patches, facecolor=road_color, edgecolor='none',
                            alpha=ROAD_ALPHA, zorder=2)
        ax.add_collection(pc)

    # Plot the trajectory line with direction markers
    lons_t = grp['lon'].values
    lats_t = grp['lat'].values
    ax.plot(lons_t, lats_t, '-o', color=color, linewidth=2, markersize=4,
            markerfacecolor='white', markeredgecolor=color, markeredgewidth=1.5,
            zorder=5)

    # Mark start and end
    ax.plot(lons_t[0], lats_t[0], 'o', color='green', markersize=8, zorder=6, label='Start')
    ax.plot(lons_t[-1], lats_t[-1], 's', color='red', markersize=8, zorder=6, label='End')

    ax.set_xlim(plot_lon_min, plot_lon_max)
    ax.set_ylim(plot_lat_min, plot_lat_max)
    ax.set_aspect(data_aspect)
    ax.set_xlabel('Longitude (°E)')
    ax.set_ylabel('Latitude (°N)')

    n_road = len(road_cells)
    ax.set_title(f'{label}  |  {traj_id}  |  {len(grp)} pts  |  {n_road} road cells')

    ax.legend(loc='lower right', fontsize=9, framealpha=0.9)
    plt.tight_layout()

    fname = os.path.join(OUTPUT_DIR, f'{traj_id}.png')
    plt.savefig(fname, dpi=120, bbox_inches='tight')
    plt.close(fig)

    if (idx + 1) % 20 == 0:
        print(f"  {idx + 1}/{len(traj_ids)} done")

print(f"\nDone. {len(traj_ids)} plots saved to {OUTPUT_DIR}/")
