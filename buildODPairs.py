import numpy as np
import pandas as pd


def build_od_pairs(traj_path: str) -> pd.DataFrame:
    """Convert trajectory point sequences to origin-destination segment pairs."""
    df = pd.read_csv(traj_path)

    rows = []
    for traj_id, group in df.groupby('ID'):
        group = group.sort_values('time').reset_index(drop=True)
        for i in range(len(group) - 1):
            qo = round(group.loc[i, 'locx'])
            ro = round(group.loc[i, 'locy'])
            so = -qo - ro
            qd = round(group.loc[i + 1, 'locx'])
            rd = round(group.loc[i + 1, 'locy'])
            sd = -qd - rd
            rows.append({
                'ID': traj_id,
                'locxo': qo,
                'locyo': ro,
                'loczo': so,
                'locxd': qd,
                'locyd': rd,
                'loczd': sd,
                'mode': group.loc[i + 1, 'mode'],
                'time': group.loc[i + 1, 'time'] - group.loc[i, 'time'],
            })

    out = pd.DataFrame(rows)

    # Hex distance (cube coordinates, in cells)
    out['distance_cells'] = np.maximum(
        np.abs(out['locxd'] - out['locxo']),
        np.abs(out['locyd'] - out['locyo'])
    )
    # Physical distance: hex edge = 200m, center-to-center = sqrt(3) * 200m ≈ 346.4m
    out['distance_m'] = out['distance_cells'] * np.sqrt(3) * 200

    return out


if __name__ == '__main__':
    # Single-mode trajectories
    od_single = build_od_pairs('data/artificial_traj_mixed_single.csv')
    od_single.to_csv('data/artificial_od_single.csv', index=False)
    print(f'Single-mode O-D pairs: {len(od_single)} rows')

    # Mixed-mode trajectories
    od_mult = build_od_pairs('data/artificial_traj_mixed_mult.csv')
    od_mult.to_csv('data/artificial_od_mult.csv', index=False)
    print(f'Mixed-mode O-D pairs:  {len(od_mult)} rows')

    # Combined
    od_all = pd.concat([od_single, od_mult], ignore_index=True)
    od_all.to_csv('data/artificial_od_all.csv', index=False)
    print(f'Total O-D pairs:       {len(od_all)} rows')
    print('\nSample:')
    print(od_all.head())
