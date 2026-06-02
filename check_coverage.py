import pickle
import pandas as pd
from collections import deque
from hexUtils import hex_round, HEX_NEIGHBORS


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


def find_nearest_road(hex_grid, q_raw, r_raw, s_raw, mode, max_dist=5):
    """
    BFS from the given coordinate to find the nearest hex cell that:
    1. Exists in hex_grid
    2. _allow(code, mode) passes

    Returns: (found_q, found_r, found_s, distance)
    Or (None, None, None, None) if not found within max_dist.
    """
    # Start from the hex-rounded coordinate (ensures valid cube coords)
    start = hex_round(q_raw, r_raw, s_raw)

    if start in hex_grid and _allow(hex_grid[start]['code'], mode):
        return start[0], start[1], start[2], 0

    visited = {start}
    q = deque([(start[0], start[1], start[2], 0)])

    while q:
        cq, cr, cs, dist = q.popleft()
        if dist >= max_dist:
            continue

        for dq, dr, ds in HEX_NEIGHBORS:
            nq, nr, ns = cq + dq, cr + dr, cs + ds
            if (nq, nr, ns) in visited:
                continue
            visited.add((nq, nr, ns))

            if (nq, nr, ns) in hex_grid and _allow(hex_grid[(nq, nr, ns)]['code'], mode):
                return nq, nr, ns, dist + 1

            q.append((nq, nr, ns, dist + 1))

    return None, None, None, max_dist + 1  # not found


def analyze_traj(hex_grid, csv_path):
    """分析 trajectory CSV，计算每个点到最近路网的距离分布"""
    df = pd.read_csv(csv_path)
    modes = ['TG', 'GG', 'GSD', 'TS']
    results = {}
    detail = {}  # store per-mode distance lists
    for m in modes:
        sub = df[df['mode'] == m]
        total = len(sub)
        dist_counts = {d: 0 for d in range(7)}  # 0..5, 6+ = not found
        dist_list = []
        for _, row in sub.iterrows():
            q, r, s = int(row['locx']), int(row['locy']), int(row['locz'])
            _, _, _, d = find_nearest_road(hex_grid, q, r, s, m, max_dist=5)
            if d is None:
                d = 6  # not found
            d = min(d, 6)
            dist_counts[d] += 1
            dist_list.append(d)

        results[m] = {
            'total': total,
            'dist_counts': dist_counts,
            'dist_list': dist_list,
        }
    return results


def analyze_route(hex_grid, csv_path):
    """分析 route CSV，计算每个点到最近路网的距离分布"""
    df = pd.read_csv(csv_path)
    modes = ['TG', 'GG', 'GSD', 'TS']
    results = {}
    for m in modes:
        sub = df[df['ID'].str.startswith(m)]
        total = len(sub)
        dist_counts = {d: 0 for d in range(7)}
        for _, row in sub.iterrows():
            q, r, s = int(row['locx']), int(row['locy']), int(row['locz'])
            _, _, _, d = find_nearest_road(hex_grid, q, r, s, m, max_dist=5)
            if d is None:
                d = 6
            d = min(d, 6)
            dist_counts[d] += 1

        results[m] = {
            'total': total,
            'dist_counts': dist_counts,
        }
    return results


def print_results(results, title, hex_km=0.346):
    print(f"\n{'='*100}")
    print(f"  {title}")
    print(f"{'='*100}")
    print(f"  (hex distance unit ≈ {hex_km*1000:.0f}m, estimated from hex edge=200m, center-to-center)")
    print()

    header = (f"{'mode':>6} {'total':>8} {'d=0':>12} {'d≤1':>12} {'d≤2':>12} "
              f"{'d≤3':>12} {'d≤5':>12} {'>5':>8} {'avg_d':>8} {'avg_km':>8}")
    print(header)
    print("-" * 100)

    for m in ['TG', 'GG', 'GSD', 'TS']:
        r = results[m]
        total = r['total']
        dc = r['dist_counts']

        def pct(cnt):
            return f"{cnt:>6} ({cnt/total*100:.1f}%)" if total > 0 else "N/A"

        d0 = pct(dc[0])
        d1 = pct(dc[0] + dc[1])
        d2 = pct(dc[0] + dc[1] + dc[2])
        d3 = pct(dc[0] + dc[1] + dc[2] + dc[3])
        d5 = pct(sum(dc[d] for d in range(6)))
        d6p = f"{dc[6]:>6}"

        avg_d = sum(d * dc[d] for d in range(7)) / total if total > 0 else 0
        avg_km = avg_d * hex_km

        print(f"  {m:>4} {total:>8}  {d0}  {d1}  {d2}  {d3}  {d5} {d6p} {avg_d:>8.2f} {avg_km:>8.2f}")

    print()

    # histogram
    print(f"  Per-mode distance histogram:")
    print(f"  {'mode':>6} {'0':>8} {'1':>8} {'2':>8} {'3':>8} {'4':>8} {'5':>8} {'6+':>8}")
    print(f"  {'-'*60}")
    for m in ['TG', 'GG', 'GSD', 'TS']:
        dc = results[m]['dist_counts']
        total = results[m]['total']
        bars = " ".join(f"{dc[d]:>5}  " for d in range(7))
        print(f"  {m:>4}  {bars}")
        pcts = " ".join(f"{dc[d]/total*100:>4.1f}% " for d in range(7))
        print(f"       {pcts}")
    print()


def extract_origin_mode(traj_id, dst_mode):
    """根据 trajectory ID 和 destination mode 推断 origin 的合理模式"""
    # Single mode: "TG1" -> mode from prefix
    # Mixed mode: "TG-GSD1" -> origin might be either TG or GSD
    if '-' in traj_id:
        modes_in_id = traj_id.split('-')
        former = modes_in_id[0]  # e.g., "TG"
        # latter is modes_in_id[1] but may have digits: "GSD1"
        latter = ''.join(c for c in modes_in_id[1] if c.isalpha())  # e.g., "GSD"
        # If destination mode matches latter, origin is also from latter part (after switch)
        if dst_mode == latter:
            return latter
        else:
            return former
    else:
        # Single mode: extract mode prefix
        return ''.join(c for c in traj_id if c.isalpha())


def analyze_od(hex_grid, csv_path):
    """分析 OD CSV，O 和 D 各用正确的模式检查"""
    df = pd.read_csv(csv_path)
    modes = ['TG', 'GG', 'GSD', 'TS']
    results = {}
    mismatch_ids = []  # collect rows where O/D modes differ
    for m in modes:
        sub = df[df['mode'] == m]
        total = len(sub)
        dist_counts = {d: 0 for d in range(7)}
        for _, row in sub.iterrows():
            traj_id = row['ID']
            dst_mode = row['mode']
            orig_mode = extract_origin_mode(traj_id, dst_mode)

            # origin — use inferred origin mode
            qo = round(row['locxo'])
            ro = round(row['locyo'])
            so = -qo - ro
            _, _, _, do = find_nearest_road(hex_grid, qo, ro, so, orig_mode, max_dist=5)
            do = min(do if do is not None else 6, 6)
            dist_counts[do] += 1

            # destination — use the row's mode column
            qd = round(row['locxd'])
            rd = round(row['locyd'])
            sd = -qd - rd
            _, _, _, dd = find_nearest_road(hex_grid, qd, rd, sd, dst_mode, max_dist=5)
            dd = min(dd if dd is not None else 6, 6)
            dist_counts[dd] += 1

            if orig_mode != dst_mode and (do >= 6 or dd >= 6):
                mismatch_ids.append((traj_id, orig_mode, dst_mode, do, dd))

        results[m] = {
            'total': total * 2,  # O + D
            'dist_counts': dist_counts,
        }

    # Print mismatch examples
    if mismatch_ids:
        print(f"\n  OD segments with mode-mismatch (origin mode ≠ destination mode) that have d≥6:")
        for tid, om, dm, do, dd in mismatch_ids[:10]:
            print(f"    ID={tid}  orig_mode={om}  dst_mode={dm}  dist_o={do}  dist_d={dd}")
        if len(mismatch_ids) > 10:
            print(f"    ... and {len(mismatch_ids) - 10} more")
    return results


def main():
    print("Loading hex_grid.pkl ...")
    with open('data/hex_grid.pkl', 'rb') as f:
        hex_grid = pickle.load(f)
    print(f"Loaded {len(hex_grid):,} hex cells.")

    hex_km = 0.346  # sqrt(3) * 0.2 km, hex center-to-center distance

    # --- Route ---
    print("\n>>> Analyzing Route (pre-jitter walk path) ...")
    route = analyze_route(hex_grid, 'data/artificial_rts_mixed_single.csv')
    print_results(route, "Route Distance to Nearest Road (artificial_rts_mixed_single.csv)", hex_km)

    # --- Trajectory ---
    print(">>> Analyzing Trajectory (jittered signal data) ...")
    traj = analyze_traj(hex_grid, 'data/artificial_traj_mixed_single.csv')
    print_results(traj, "Trajectory Distance to Nearest Road (artificial_traj_mixed_single.csv)", hex_km)

    # --- OD All ---
    print(">>> Analyzing OD All (origin+destination points) ...")
    od = analyze_od(hex_grid, 'data/artificial_od_all.csv')
    print_results(od, "OD Points Distance to Nearest Road (artificial_od_all.csv)", hex_km)


if __name__ == '__main__':
    main()
