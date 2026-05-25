import pickle
import numpy as np
import pandas as pd
from hexUtils import HEX_NEIGHBORS

"""
Feature extraction from hex-grid generated trajectory data.
Based on 'A Hybrid Method for Intercity Transport Mode Identification
Based on Mobility Features and Sequential Relations Mined from Cellular Signaling Data'
"""

raw_df = pd.read_csv('data/artificial_traj_mixed_single.csv')


# Calculate durations between consecutive records within each trajectory
raw_df['duration'] = raw_df.groupby('ID')['time'].diff().fillna(0)

# Calculate distances between consecutive records
raw_df['dx'] = raw_df.groupby('ID')['locx'].diff().fillna(0)
raw_df['dy'] = raw_df.groupby('ID')['locy'].diff().fillna(0)
raw_df['distance'] = np.sqrt(raw_df['dx'] ** 2 + raw_df['dy'] ** 2)

# Speed in km/h (time is in minutes, distance is in grid units ≈ km)
raw_df['speed'] = 60 * raw_df['distance'] / raw_df['duration'].replace(0, 1)

# Acceleration
raw_df['acc'] = raw_df['speed'].diff().fillna(0) / raw_df['duration'].replace(0, 1)

# Cosine of turning angle between 3 consecutive points
raw_df['cos'] = raw_df.groupby('ID').apply(
    lambda x: (x['locx'].diff().fillna(0) * x['locx'].shift(-1).fillna(0) +
               x['locy'].diff().fillna(0) * x['locy'].shift(-1).fillna(0)) /
              (np.sqrt(x['locx'].diff().fillna(0) ** 2 + x['locy'].diff().fillna(0) ** 2) *
               np.sqrt(x['locx'].shift(-1).fillna(0) ** 2 + x['locy'].shift(-1).fillna(0) ** 2)).replace(0, 1)
).reset_index(drop=True)


def _is_close(code: int, mode: str) -> int:
    """Check if a single hex cell code indicates proximity to the given mode.
    Uses hex_grid.pkl bit encoding:
      bit0=has_xd, bit1=has_pt, bit2=has_sd, bit3=has_gs_sfz,
      bit4=has_gs, bit5=has_gd, bit6=has_gt, bit7=has_hcz
    """
    if mode == 'GG':
        return code >> 4 & 1        # has_gs
    if mode == 'GSD':
        return code >> 5 & 1 or code >> 2 & 1  # has_gd or has_sd
    if mode == 'TG':
        return code >> 6 & 1        # has_gt
    if mode == 'TS':
        return code >> 1 & 1        # has_pt
    return 0


def isClose(hex_grid: dict, coord: tuple, mode: str) -> int:
    """Check if a hex cell or any of its 6 neighbors contains the given mode."""
    if coord not in hex_grid:
        return 0

    # Check self
    if _is_close(hex_grid[coord]['code'], mode):
        return 1

    # Check 6 hex neighbors
    q, r, s = coord
    for dq, dr, ds in HEX_NEIGHBORS:
        nbr = (q + dq, r + dr, s + ds)
        if nbr in hex_grid and _is_close(hex_grid[nbr]['code'], mode):
            return 1
    return 0


def encodeMode(mode: str):
    if mode == 'GG':
        return 1
    if mode == 'GSD':
        return 2
    if mode == 'TG':
        return 3
    if mode == 'TS':
        return 4
    else:
        return 0


with open('data/hex_grid.pkl', 'rb') as f:
    hex_grid = pickle.load(f)

print(f"Loaded hex grid with {len(hex_grid)} cells")

# Build proximity features using hex neighbors
# locx=q, locy=r, locz=s (cube coordinates)
raw_df['TG'] = raw_df.apply(lambda x: isClose(hex_grid, (int(x['locx']), int(x['locy']), int(x['locz'])), 'TG'), axis=1)
raw_df['GSD'] = raw_df.apply(lambda x: isClose(hex_grid, (int(x['locx']), int(x['locy']), int(x['locz'])), 'GSD'), axis=1)
raw_df['GG'] = raw_df.apply(lambda x: isClose(hex_grid, (int(x['locx']), int(x['locy']), int(x['locz'])), 'GG'), axis=1)
raw_df['TS'] = raw_df.apply(lambda x: isClose(hex_grid, (int(x['locx']), int(x['locy']), int(x['locz'])), 'TS'), axis=1)

raw_df['mode'] = raw_df['mode'].apply(lambda x: encodeMode(x))
# Set mode = 0 (static) when speed is 0
raw_df['mode'] = raw_df['mode'] * (raw_df['speed'] != 0)
raw_df.to_csv('data/realWorldMixedFeatures.csv', index=False)

print("Feature extraction complete. Output: data/realWorldMixedFeatures.csv")
