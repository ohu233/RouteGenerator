import random
import pickle
from collections import deque

import numpy as np
import pandas as pd
from hexUtils import HEX_NEIGHBORS, HEX_OFFSETS

HEX_CELL_KM = np.sqrt(3) * 0.2  # hex cell center-to-center distance ≈ 0.346 km

# Cached real-data (dist_km, time_min) pairs grouped by closest mode speed
_real_mode_pairs = None  # {mode: [(dist_km, time_min), ...]}


def _load_real_distribution(csv_path='data/dataset_multicity_with_hex.csv', max_dist_km=50):
    """Load real data and group (dist_km, time_min) pairs by nearest mode velocity."""
    global _real_mode_pairs
    if _real_mode_pairs is not None:
        return _real_mode_pairs

    mode_speeds = {'TG': 300, 'TS': 150, 'GG': 120, 'GSD': 60}
    _real_mode_pairs = {m: [] for m in mode_speeds}

    real = pd.read_csv(csv_path)
    real = real[(real['dist'] > 0) & (real['time'] > 0)]
    real = real[real['dist'] <= max_dist_km * 1000]  # max_dist_km in km, dist in m
    real['vel'] = 3.6 * real['dist'] / real['time']  # km/h

    for _, row in real.iterrows():
        v = row['vel']
        best_mode = min(mode_speeds, key=lambda m: abs(v - mode_speeds[m]))
        _real_mode_pairs[best_mode].append(
            (row['dist'] / 1000, row['time'] / 60)  # (dist_km, time_min)
        )
    return _real_mode_pairs


def _sample_step(mode_pairs, mode):
    """Sample (hex_cells, time_minutes) from the mode's real-data bucket."""
    dist_km, time_min = random.choice(mode_pairs[mode])
    cells = max(1, int(dist_km / HEX_CELL_KM + 0.5))
    return cells, time_min


def _bias(method: str, para1, para2):
    """
    bias function
    :param method: 'uniform' or 'normal'
    :param para1:  if method is 'uniform', para1 is the lower bound; if method is 'normal', para1 is the mean
    :param para2:  if method is 'uniform', para2 is the upper bound; if method is 'normal', para2 is the standard deviation
    :return: return bias value
    """
    if method == 'uniform':
        return random.uniform(para1, para2)
    if method == 'normal':
        return random.normalvariate(para1, para2)


def _change_route(prob: float, current_grid_xy: tuple, current_route):  # TODO UNFINISHED
    """
    change the route according to the probability
    :param prob: the probability of changing route
    :param current_grid: the current grid
    :param current_route: the current route
    :return: the new route
    """
    if random.random() < prob:
        return current_route
    else:
        return current_route


def _allow(code: int, mode: str) -> bool:
    """
    Check if a hex cell with the given code allows travel of the given mode.
    Bit encoding (hex_grid.pkl):
      bit0=has_xd, bit1=has_pt, bit2=has_sd, bit3=has_gs_sfz,
      bit4=has_gs, bit5=has_gd, bit6=has_gt, bit7=has_hcz
    """
    if mode == 'TG' or mode == 'static':
        return code >> 6 & 1 == 1        # has_gt
    elif mode == 'GG':
        return code >> 4 & 1 == 1 or code >> 3 & 1 == 1  # has_gs or has_gs_sfz
    elif mode == 'GSD':
        return code >> 5 & 1 == 1 or code >> 2 & 1 == 1  # has_gd or has_sd
    elif mode == 'TS':
        return code >> 1 & 1 == 1        # has_pt


def _allow_change(code: int, mode: str) -> bool:
    """
    Check if the mode can be changed at the current hex cell.
    """
    if mode == 'TG-GSD' or mode == 'GSD-TG':
        return code >> 7 & 1 == 1 and (code >> 5 & 1 == 1 or code >> 2 & 1 == 1)
        # has_hcz AND (has_gd OR has_sd)
    elif mode == 'GSD-GG' or mode == 'GG-GSD':
        return code >> 3 & 1 == 1 and (code >> 5 & 1 == 1 or code >> 2 & 1 == 1)
        # has_gs_sfz AND (has_gd OR has_sd)
    elif mode == 'TS-TG' or mode == 'TG-TS':
        return code >> 7 & 1 == 1 and code >> 1 & 1 == 1 and code >> 6 & 1 == 1
        # has_hcz AND has_pt AND has_gt
    else:
        return False


def _get_new_coordinates(q, r, s, position) -> tuple:
    """Get the hex cube coordinate at the given offset position (0-6). Position 6 = self."""
    if 0 <= position < len(HEX_OFFSETS):
        dq, dr, ds = HEX_OFFSETS[position]
        return (q + dq, r + dr, s + ds)
    else:
        raise ValueError("Invalid position")


def generate_traj_single(hex_grid: dict, size: int, mode: str, time_interval=6) -> pd.DataFrame:
    """
    Generate single-mode hex cell trajectories using DFS random walk.

    :param hex_grid: dict, {(q,r,s): {'lon', 'lat', 'code'}}
    :param size: number of trajectories to generate for this mode
    :param mode: 'TG', 'GG', 'GSD', 'TS'
    :param time_interval: minutes between samples
    :return: (trajectory_df, route_df)
    """
    df = pd.DataFrame(columns=["ID", "time", "locx", "locy", "locz", "mode"])
    route_df = pd.DataFrame(columns=["ID", "locx", "locy", "locz"])

    # Build candidate start cells once (avoid iterating 6.6M items per trajectory)
    if mode == "TG":
        candidates = [(k, v) for k, v in hex_grid.items()
                      if v['code'] >> 7 & 1 == 1 and v['code'] >> 6 & 1 == 1]
    elif mode == "GG":
        candidates = [(k, v) for k, v in hex_grid.items()
                      if v['code'] >> 3 & 1 == 1]
    elif mode == 'GSD':
        candidates = [(k, v) for k, v in hex_grid.items()
                      if v['code'] >> 5 & 1 == 1 or v['code'] >> 2 & 1 == 1]
    elif mode == 'TS':
        candidates = [(k, v) for k, v in hex_grid.items()
                      if v['code'] >> 7 & 1 == 1]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if not candidates:
        print(f"Warning: No start candidates for mode {mode}")
        return df, route_df

    mode_rtslen_dict = {'TG': 300, 'GG': 200, 'GSD': 200, 'TS': 200}
    mode_len_dict = {'TG': 3, 'GG': 4, 'GSD': 5, 'TS': 4}

    i = 0
    while i < size:
        traj_id = mode + str(i + 1)
        expected_sample_len = mode_rtslen_dict[mode]
        start, _ = random.choice(candidates)

        cnt = 0
        recent = deque(maxlen=20)  # sliding window: only block recently visited cells
        route_list = []

        current_pos = start
        sub_route_list = []  # accumulate (traj_id, cq, cr) tuples
        prev_dir = None
        while cnt < expected_sample_len:
            cq, cr, cs = current_pos
            sub_route_list.append((traj_id, cq, cr, cs))  # q, r, s cube coords

            route_list.append(current_pos)
            recent.append(current_pos)
            cnt += 1

            valid_dirs = []
            for idx in range(6):
                nq, nr, ns = _get_new_coordinates(cq, cr, cs, idx)
                neighbor_coord = (nq, nr, ns)
                if neighbor_coord in hex_grid and neighbor_coord not in recent:
                    neighbor_code = hex_grid[neighbor_coord]['code']
                    if _allow(neighbor_code, mode):
                        valid_dirs.append(idx)

            if not valid_dirs:
                break

            if prev_dir is not None:
                turn_weights = [4, 3, 0, 0, 0, 3]
                weights = [turn_weights[(d - prev_dir) % 6] for d in valid_dirs]
                if sum(weights) == 0:
                    soft_weights = [3, 2, 1, 0, 1, 2]
                    weights = [soft_weights[(d - prev_dir) % 6] for d in valid_dirs]
                    if sum(weights) == 0:
                        weights = [1] * len(valid_dirs)
            else:
                weights = [1] * len(valid_dirs)

            delta = random.choices(valid_dirs, weights=weights, k=1)[0]
            prev_dir = delta
            current_pos = _get_new_coordinates(cq, cr, cs, delta)

        # Sample trajectory: time+distance from real data, grouped by mode velocity
        mode_pairs = _load_real_distribution()

        j = 0
        timestamp = 0
        traj_len = 0

        sub_data = [[traj_id, timestamp * 60, route_list[0][0], route_list[0][1], route_list[0][2], mode]]
        while j < len(route_list):
            spacestep, timestep = _sample_step(mode_pairs, mode)
            timestamp += timestep
            traj_len += 1
            j = int(j + spacestep)
            if j >= len(route_list):
                break
            locx, locy, locz = route_list[j][0], route_list[j][1], route_list[j][2]

            sub_data.append([traj_id,
                             timestamp * 60,
                             locx + int(_bias('uniform', -1.5, 1.5)),
                             locy + int(_bias('uniform', -1.5, 1.5)),
                             locz + int(_bias('uniform', -1.5, 1.5)),
                             mode])

        if traj_len >= mode_len_dict[mode]:
            sub_df = pd.DataFrame(sub_data, columns=["ID", "time", "locx", "locy", "locz", "mode"])
            df = pd.concat([df, sub_df], ignore_index=True)
            sub_route_df = pd.DataFrame(sub_route_list, columns=["ID", "locx", "locy", "locz"])
            route_df = pd.concat([route_df, sub_route_df], ignore_index=True)
            i += 1

    return df, route_df


def generate_traj_mixed(hex_grid: dict, size: int, mode: str, time_interval=6) -> pd.DataFrame:
    """
    Generate mixed-mode hex cell trajectories.
    Supported transitions: TG-GSD, GSD-TG, GSD-GG, GG-GSD, TS-TG, TG-TS

    :param hex_grid: dict, {(q,r,s): {'lon', 'lat', 'code'}}
    :param size: number of trajectories to generate for this mode pair
    :param mode: e.g. 'TG-GSD', 'GSD-GG', 'TS-TG'
    :param time_interval: minutes between samples
    :return: (trajectory_df, route_df)
    """
    df = pd.DataFrame(columns=["ID", "time", "locx", "locy", "locz", "mode"])
    route_df = pd.DataFrame(columns=["ID", "locx", "locy", "locz"])

    # Build candidate start cells once
    if mode.startswith('TG'):
        candidates = [(k, v) for k, v in hex_grid.items()
                      if v['code'] >> 7 & 1 == 1 and v['code'] >> 6 & 1 == 1]
    elif mode.startswith("GSD"):
        candidates = [(k, v) for k, v in hex_grid.items()
                      if v['code'] >> 5 & 1 == 1 or v['code'] >> 2 & 1 == 1]
    elif mode.startswith("GG"):
        candidates = [(k, v) for k, v in hex_grid.items()
                      if v['code'] >> 4 & 1 == 1 and v['code'] >> 3 & 1 == 1]
    elif mode.startswith("TS"):
        candidates = [(k, v) for k, v in hex_grid.items()
                      if v['code'] >> 7 & 1 == 1 and v['code'] >> 1 & 1 == 1]
    else:
        raise ValueError(f"Unknown mode: {mode}")

    if not candidates:
        print(f"Warning: No start candidates for mode {mode}")
        return df, route_df

    i = 0
    while i < size:
        traj_id = mode + str(i + 1)
        start, _ = random.choice(candidates)

        expected_sample_len = 200
        cnt = 0
        recent = deque(maxlen=20)
        route_list = []

        current_pos = start
        changed = False
        next_mode = mode.split('-')[1]
        former_mode = mode.split('-')[0]
        current_mode = mode.split('-')[0]
        change_position = (-1, -1, -1)

        sub_route_list = []  # accumulate (traj_id, cq, cr) tuples
        prev_dir = None
        while cnt < expected_sample_len:
            cq, cr, cs = current_pos
            sub_route_list.append((traj_id, cq, cr, cs))  # q, r, s cube coords
            route_list.append(current_pos)
            recent.append(current_pos)
            cnt += 1
            if not changed and _allow_change(hex_grid[current_pos]['code'], mode) and cnt > 80 and random.random() < 0.8:
                changed = True
                current_mode = next_mode
                change_position = current_pos

            valid_dirs = []
            for idx in range(6):
                nq, nr, ns = _get_new_coordinates(cq, cr, cs, idx)
                neighbor_coord = (nq, nr, ns)
                if neighbor_coord in hex_grid and neighbor_coord not in recent:
                    neighbor_code = hex_grid[neighbor_coord]['code']
                    if _allow(neighbor_code, current_mode):
                        valid_dirs.append(idx)

            if not valid_dirs:
                for idx in range(6):
                    nq, nr, ns = _get_new_coordinates(cq, cr, cs, idx)
                    neighbor_coord = (nq, nr, ns)
                    if neighbor_coord in hex_grid:
                        neighbor_code = hex_grid[neighbor_coord]['code']
                        if _allow(neighbor_code, current_mode):
                            valid_dirs.append(idx)
            if not valid_dirs:
                break

            if prev_dir is not None:
                turn_weights = [4, 3, 0, 0, 0, 3]
                weights = [turn_weights[(d - prev_dir) % 6] for d in valid_dirs]
                if sum(weights) == 0:
                    soft_weights = [3, 2, 1, 0, 1, 2]
                    weights = [soft_weights[(d - prev_dir) % 6] for d in valid_dirs]
                    if sum(weights) == 0:
                        weights = [1] * len(valid_dirs)
            else:
                weights = [1] * len(valid_dirs)

            delta = random.choices(valid_dirs, weights=weights, k=1)[0]
            prev_dir = delta
            current_pos = _get_new_coordinates(cq, cr, cs, delta)

        if change_position == (-1, -1, -1):
            continue
        change_idx = route_list.index(change_position)

        j = 0
        timestamp = 0
        traj_len = 0

        mode_pairs = _load_real_distribution()

        sub_data = []  # accumulate rows
        while j < change_idx:
            spacestep, timestep = _sample_step(mode_pairs, former_mode)
            timestamp += timestep
            traj_len += 1
            j = int(min(j + spacestep, change_idx))
            if j >= len(route_list):
                break
            locx, locy, locz = route_list[j][0], route_list[j][1], route_list[j][2]
            sub_data.append([traj_id,
                             timestamp * 60,
                             locx + int(_bias('uniform', -1.5, 1.5)),
                             locy + int(_bias('uniform', -1.5, 1.5)),
                             locz + int(_bias('uniform', -1.5, 1.5)),
                             former_mode])

        # Station stop: 4 min for TG, 7 min for TS
        timestamp += 4 if former_mode == 'TG' else 7
        sub_data.append([traj_id,
                         timestamp * 60,
                         route_list[change_idx][0] + int(_bias('uniform', -1.5, 1.5)),
                         route_list[change_idx][1] + int(_bias('uniform', -1.5, 1.5)),
                         route_list[change_idx][2] + int(_bias('uniform', -1.5, 1.5)),
                         0])

        # Latter part
        while j < len(route_list) - 1:
            spacestep, timestep = _sample_step(mode_pairs, next_mode)
            timestamp += timestep
            traj_len += 1
            j = int(min(j + spacestep, len(route_list) - 1))
            if j >= len(route_list):
                break
            locx, locy, locz = route_list[j][0], route_list[j][1], route_list[j][2]
            sub_data.append([traj_id,
                             timestamp * 60,
                             locx + int(_bias('uniform', -1.5, 1.5)),
                             locy + int(_bias('uniform', -1.5, 1.5)),
                             locz + int(_bias('uniform', -1.5, 1.5)),
                             next_mode])

        if traj_len >= 6:
            sub_df = pd.DataFrame(sub_data, columns=["ID", "time", "locx", "locy", "locz", "mode"])
            df = pd.concat([df, sub_df], ignore_index=True)
            sub_route_df = pd.DataFrame(sub_route_list, columns=["ID", "locx", "locy", "locz"])
            route_df = pd.concat([route_df, sub_route_df], ignore_index=True)
            i += 1

    return df, route_df


if __name__ == '__main__':
    with open('data/hex_grid.pkl', 'rb') as f:
        hex_grid_data = pickle.load(f)

    print(f"Loaded hex grid with {len(hex_grid_data)} cells")

    traj_GG, rts_GG = generate_traj_single(hex_grid_data, size=200, mode='GG', time_interval=2)
    traj_GSD, rts_GSD = generate_traj_single(hex_grid_data, size=200, mode='GSD', time_interval=2)
    traj_TS, rts_TS = generate_traj_single(hex_grid_data, size=200, mode='TS', time_interval=2)
    traj_TG, rts_TG = generate_traj_single(hex_grid_data, size=200, mode='TG', time_interval=2)

    rts_single = pd.concat([rts_GG, rts_GSD, rts_TS, rts_TG])
    traj_single = pd.concat([traj_GG, traj_GSD, traj_TS, traj_TG])
    traj_single.to_csv('data/artificial_traj_mixed_single.csv', index=False)
    rts_single.to_csv('data/artificial_rts_mixed_single.csv', index=False)

    traj_TG_GSD, rts_TG_GSD = generate_traj_mixed(hex_grid_data, size=50, mode='TG-GSD', time_interval=2)
    traj_GSD_TG, rts_GSD_TG = generate_traj_mixed(hex_grid_data, size=50, mode='GSD-TG', time_interval=2)
    traj_TS_TG, rts_TS_TG = generate_traj_mixed(hex_grid_data, size=50, mode='TS-TG', time_interval=2)
    traj_TG_TS, rts_TG_TS = generate_traj_mixed(hex_grid_data, size=50, mode='TG-TS', time_interval=2)
    traj_GSD_GG, rts_GSD_GG = generate_traj_mixed(hex_grid_data, size=50, mode='GSD-GG', time_interval=2)
    traj_GG_GSD, rts_GG_GSD = generate_traj_mixed(hex_grid_data, size=50, mode='GG-GSD', time_interval=2)

    rts_mult = pd.concat([rts_TG_GSD, rts_GSD_TG, rts_TS_TG, rts_TG_TS, rts_GSD_GG, rts_GG_GSD])
    traj_mult = pd.concat([traj_TG_GSD, traj_GSD_TG, traj_TS_TG, traj_TG_TS, traj_GSD_GG, traj_GG_GSD])
    traj_mult.to_csv('data/artificial_traj_mixed_mult.csv', index=False)
    rts_mult.to_csv('data/artificial_rts_mixed_mult.csv', index=False)

    print("Done generating all trajectories.")
