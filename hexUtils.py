"""
Hex grid utilities for flat-top hexagons using cube coordinates (q, r, s) where q + r + s = 0.
"""

# 6 neighbor directions for flat-top hexagons in cube coordinates
# Order: NE, E, SE, SW, W, NW
HEX_NEIGHBORS = [
    (1, 0, -1),   # NE
    (1, -1, 0),   # E
    (0, -1, 1),   # SE
    (-1, 0, 1),   # SW
    (-1, 1, 0),   # W
    (0, 1, -1),   # NW
]

# 7 offsets: self + 6 neighbors (matching the old 9-offset pattern, index 4 = self)
#      |z/s
#      |
#      |
#      /\
#     /  \
#    /    \
#   y/r    x/q
HEX_OFFSETS = [
    (1, 0, -1),   # 0: NE
    (1, -1, 0),   # 1: E
    (0, -1, 1),   # 2: SE
    (-1, 0, 1),   # 3: SW
    (-1, 1, 0),   # 4: W
    (0, 1, -1),   # 5: NW
    (0, 0, 0),    # 6: self
]


def get_neighbor_coord(q, r, s, direction):
    """Get the cube coordinate of the neighbor in the given direction (0-5).
    获取给定方向（0-5）邻居的立方体坐标。"""
    dq, dr, ds = HEX_NEIGHBORS[direction]
    return (q + dq, r + dr, s + ds)


def hex_distance(q1, r1, s1, q2, r2, s2):
    """Distance between two hex cells in cube coordinates.
    计算两个六边形单元格之间的距离，使用立方体坐标系。"""
    return max(abs(q1 - q2), abs(r1 - r2), abs(s1 - s2))


def hex_line(start, end):
    """Hex grid line interpolation using cube coordinates (replaces Bresenham for square grids).
    使用立方体坐标进行六边形网格线插值（替代方形网格的Bresenham算法）。"""
    q1, r1, s1 = start
    q2, r2, s2 = end
    n = hex_distance(q1, r1, s1, q2, r2, s2)
    results = []
    for i in range(n + 1):
        t = i / max(n, 1)
        q = q1 + (q2 - q1) * t
        r = r1 + (r2 - r1) * t
        s = s1 + (s2 - s1) * t
        results.append(hex_round(q, r, s))
    return results


def hex_round(q, r, s):
    """Round a fractional cube coordinate to the nearest hex cell.
    将一个分数立方体坐标四舍五入到最近的六边形单元格。"""
    rq = round(q)
    rr = round(r)
    rs = round(s)
    dq = abs(rq - q)
    dr = abs(rr - r)
    ds = abs(rs - s)
    if dq > dr and dq > ds:
        rq = -rr - rs
    elif dr > ds:
        rr = -rq - rs
    else:
        rs = -rq - rr
    return (rq, rr, rs)
