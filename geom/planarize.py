from typing import List, Tuple, Dict, Set
import math
from geom.graph import Graph
from geom.grid import UniformGrid

Point = Tuple[float, float]


# --- быстрые геометрические утилиты
from collections import defaultdict

Point = Tuple[float, float]

AABB_PAD      = 1e-9   # подушка к bbox
MAX_CELL_LOAD = 48     # допустимая загрузка ячейки
MIN_SPLIT_LEN = 0.25   # минимальная доля eps для смысла реза

def _dot(ax, ay, bx, by): return ax*bx + ay*by
def _cross(ax, ay, bx, by): return ax*by - ay*bx
def _seg_len(a: Point, b: Point) -> float: return math.hypot(b[0]-a[0], b[1]-a[1])

def _orient(a: Point, b: Point, c: Point) -> float:
    return _cross(b[0]-a[0], b[1]-a[1], c[0]-a[0], c[1]-a[1])

def _line_dist(a: Point, b: Point, p: Point) -> float:
    L = _seg_len(a, b)
    if L <= 1e-18: return float('inf')
    return abs(_orient(a, b, p)) / L

def _between(a: Point, b: Point, c: Point, eps: float) -> bool:
    if _line_dist(a, b, c) > eps: return False
    minx, maxx = (a[0], b[0]) if a[0] <= b[0] else (b[0], a[0])
    miny, maxy = (a[1], b[1]) if a[1] <= b[1] else (b[1], a[1])
    return (minx - eps <= c[0] <= maxx + eps) and (miny - eps <= c[1] <= maxy + eps)

def _unit(vx: float, vy: float) -> Tuple[float, float]:
    L = math.hypot(vx, vy)
    if L <= 1e-18: return (0.0, 0.0)
    return (vx / L, vy / L)

def _canon_normal(dx: float, dy: float) -> Tuple[float, float]:
    # нормаль к прямой, каноническая (nx > 0 или nx==0, ny>0)
    nx, ny = _unit(dy, -dx)
    if nx < 0 or (nx == 0 and ny < 0):
        nx, ny = -nx, -ny
    return nx, ny

def _line_key_int(a: Point, b: Point, eps: float) -> Tuple[int, int, int]:
    # Ключ "прямая": (nx, ny, d), квантованные в целые
    nx, ny = _canon_normal(b[0]-a[0], b[1]-a[1])
    d = nx*a[0] + ny*a[1]   # смещение
    Qn = 10**6              # квант нормали (1e-6)
    Qd = max(int(round(1.0/eps)), 1)   # квант смещения по eps
    return (int(round(nx*Qn)), int(round(ny*Qn)), int(round(d*Qd)))

def _precut_collinear_overlaps_fast(H, eps: float, cut_points: Dict[int, List[Point]]) -> None:
    """
    Группируем рёбра по прямым (nx,ny,d) и режем все коллинеарные перекрытия
    одним проходом по линейной проекции. В cut_points накапливаем точки реза.
    """
    # 1) сгруппируем ребра по ключу прямой
    groups: Dict[Tuple[int,int,int], List[int]] = defaultdict(list)
    for eid, (u, v) in enumerate(H.edges):
        if u == -1 or v == -1: 
            continue
        a = H.nodes[u]; b = H.nodes[v]
        if _seg_len(a, b) <= 1e-18:
            continue
        key = _line_key_int(a, b, eps)
        groups[key].append(eid)

    # 2) для каждой прямой — общая ось t (единичный тангент)
    for key, eids in groups.items():
        if len(eids) < 2:
            continue
        # базовая ось: возьмем первый сегмент
        e0 = eids[0]
        u0, v0 = H.edges[e0]
        p0 = H.nodes[u0]; p1 = H.nodes[v0]
        tx, ty = _unit(p1[0]-p0[0], p1[1]-p0[1])
        if tx == 0 and ty == 0:
            continue

        # выберем общий origin по минимальному t всех концов
        t_all = []
        ends_by_eid = {}
        for eid in eids:
            u, v = H.edges[eid]
            a = H.nodes[u]; b = H.nodes[v]
            ta = (a[0]-p0[0])*tx + (a[1]-p0[1])*ty
            tb = (b[0]-p0[0])*tx + (b[1]-p0[1])*ty
            if tb < ta: ta, tb = tb, ta
            ends_by_eid[eid] = (ta, tb, a, b)
            t_all.append(ta); t_all.append(tb)

        # уникальные точки разбиения по линии (с допуском)
        t_all.sort()
        uniq_t = []
        last = None
        tol_t = eps * 0.5  # порог слияния по оси
        for t in t_all:
            if last is None or abs(t - last) > tol_t:
                uniq_t.append(t)
                last = t

        # 3) для каждого сегмента — добавим все внутренние точки из uniq_t
        for eid in eids:
            ta, tb, a, b = ends_by_eid[eid]
            if tb - ta <= 1e-12:
                continue
            # восстановление точки по t: P = p0 + (t)*(tangent)
            def P_of(t):
                return (p0[0] + t*tx, p0[1] + t*ty)

            # границы (с небольшим запасом внутрь)
            lo = ta + tol_t
            hi = tb - tol_t
            if hi <= lo:
                continue

            pts_local = []
            for t in uniq_t:
                if lo <= t <= hi:
                    pts_local.append(P_of(t))
            if pts_local:
                cut_points.setdefault(eid, []).extend(pts_local)

def _collect_tjunction_cuts(H, grid, eps: float, cut_points: Dict[int, List[Point]]) -> None:
    """
    Для каждого КОНЦА ребра проверяем, не лежит ли он на интерьере других рёбер.
    Используем grидовый поиск соседних сегментов по точке.
    """
    for eid, (u, v) in enumerate(H.edges):
        if u == -1 or v == -1:
            continue
        for nid in (u, v):
            P = H.nodes[nid]
            cand = grid.nearby_segments_by_point(P[0], P[1], radius=max(4*eps, 1e-6))
            for ej in cand:
                if ej == eid: 
                    continue
                a, b = H.edges[ej]
                if a == -1 or b == -1:
                    continue
                pa, pb = H.nodes[a], H.nodes[b]
                # «внутри» чужого ребра?
                if _between(pa, pb, P, eps):
                    cut_points.setdefault(ej, []).append(P)



def _dot(ax, ay, bx, by): return ax*bx + ay*by
def _cross(ax, ay, bx, by): return ax*by - ay*bx

def _orient(a: Point, b: Point, c: Point) -> float:
    # подписанная площадь *2 = cross(AB, AC)
    return _cross(b[0]-a[0], b[1]-a[1], c[0]-a[0], c[1]-a[1])

def _seg_len(a: Point, b: Point) -> float:
    return math.hypot(b[0]-a[0], b[1]-a[1])

def _line_dist(a: Point, b: Point, p: Point) -> float:
    # расстояние от p до прямой (ab)
    L = _seg_len(a, b)
    if L <= 1e-18: return float('inf')
    return abs(_orient(a, b, p)) / L

def _between(a: Point, b: Point, c: Point, eps: float) -> bool:
    # c лежит на отрезке ab (с допуском по ширине eps)
    if _line_dist(a, b, c) > eps: 
        return False
    minx, maxx = (a[0], b[0]) if a[0] <= b[0] else (b[0], a[0])
    miny, maxy = (a[1], b[1]) if a[1] <= b[1] else (b[1], a[1])
    return (minx - eps <= c[0] <= maxx + eps) and (miny - eps <= c[1] <= maxy + eps)

def _project_t(a: Point, b: Point, p: Point) -> float:
    # параметр p на отрезке ab: a + t*(b-a)
    ax, ay = a; bx, by = b
    vx, vy = (bx-ax, by-ay)
    L2 = vx*vx + vy*vy
    if L2 <= 1e-18: return 0.0
    return _dot(p[0]-ax, p[1]-ay, vx, vy) / L2

def _seg_intersection(a: Point, b: Point, c: Point, d: Point, eps: float):
    """
    ('proper', P)  — пересечение в интерьерах обоих
    ('t_on_cd', P) — a или b лежит на cd
    ('u_on_ab', P) — c или d лежит на ab
    ('none', None) — иначе
    Коллинеарные перекрытия здесь не обрабатываем (их лучше схлопнуть на подготовке).
    """
    # bbox-тест
    if (max(a[0], b[0]) + eps < min(c[0], d[0]) or
        max(c[0], d[0]) + eps < min(a[0], b[0]) or
        max(a[1], b[1]) + eps < min(c[1], d[1]) or
        max(c[1], d[1]) + eps < min(a[1], b[1])):
        return ('none', None)

    o1 = _orient(a, b, c)
    o2 = _orient(a, b, d)
    o3 = _orient(c, d, a)
    o4 = _orient(c, d, b)

    # proper cross — строгие разнонаправленные знаки
    if (o1 * o2 < 0) and (o3 * o4 < 0):
        ax, ay = a; bx, by = b; cx, cy = c; dx, dy = d
        r_x, r_y = bx-ax, by-ay
        s_x, s_y = dx-cx, dy-cy
        denom = _cross(r_x, r_y, s_x, s_y)
        if abs(denom) < 1e-18:
            return ('none', None)
        t = _cross(cx-ax, cy-ay, s_x, s_y) / denom
        P = (ax + t*r_x, ay + t*r_y)
        return ('proper', P)

    # T-стыки (конец одного на интерьере другого)
    for p in (a, b):
        if _between(c, d, p, eps):
            return ('t_on_cd', p)
    for p in (c, d):
        if _between(a, b, p, eps):
            return ('u_on_ab', p)

    return ('none', None)

from geom.grid import UniformGrid
from geom.graph import Graph

def planarize_graph(
    G: Graph, 
    eps: float = 0.002,
    grid_cell: float = None,
    max_passes: int | None = 2,   # обычно хватит 1–2 прохода
    handle_collinear: bool = True,
    endpoint_snap: bool = True
) -> Graph:
    """
    Быстрая планаризация PSLG:
      1) Группы прямых → разрез всех коллинеарных перекрытий (O(k log k) в группе).
      2) Грид: T-junction через «конец→ребро» (без парного перебора).
      3) Грид: proper-cross (X) пары + AABB reject.
      4) Сплит с отсечкой микрорезов; при необходимости — второй короткий проход.
    """
    H = G.clone()

    def _build_grid_adaptive():
        cell = grid_cell if grid_cell is not None else max(eps * 50.0, 0.02)  # ~2см или больше
        for _ in range(6):
            grid = UniformGrid(cell)
            alive = []
            for ei, (u, v) in enumerate(H.edges):
                if u == -1 or v == -1: continue
                a, b = H.nodes[u], H.nodes[v]
                grid.insert_segment(ei, a, b, pad=0.0)
                alive.append(ei)
            max_load = max((len(eids) for eids in grid.cells.values()), default=0)
            if max_load <= MAX_CELL_LOAD:
                return grid, set(alive)
            cell *= 0.5
        return grid, set(alive)

    safe_limit = max_passes if max_passes is not None else 2

    # --- ПРЕ-ПРОХОД: коллинеарные перекрытия (сильно разгружает пары)
    if handle_collinear:
        cut_points: Dict[int, List[Point]] = {}
        _precut_collinear_overlaps_fast(H, eps, cut_points)
        # применим сплиты из pre-cut
        for ei, pts in cut_points.items():
            u, v = H.edges[ei]
            if u == -1 or v == -1: continue
            A = H.nodes[u]; B = H.nodes[v]
            vx, vy = (B[0]-A[0], B[1]-A[1])
            L2 = vx*vx + vy*vy
            if L2 <= 1e-18: continue
            def _t(p): return _dot(p[0]-A[0], p[1]-A[1], vx, vy) / L2
            pts_sorted = sorted(pts, key=_t)
            current_eid = ei
            for P in pts_sorted:
                if current_eid >= len(H.edges): break
                uu, vv = H.edges[current_eid]
                if uu == -1 or vv == -1: break
                Au, Bu = H.nodes[uu], H.nodes[vv]
                du = math.hypot(Au[0]-P[0], Au[1]-P[1])
                dv = math.hypot(Bu[0]-P[0], Bu[1]-P[1])
                min_len = max(MIN_SPLIT_LEN * eps, 1e-9)
                # «снэп» к концам + фильтр микрорезов
                P_use = Au if du <= eps else (Bu if dv <= eps else P)
                if du <= min_len or dv <= min_len:
                    continue
                _, _, right = H.split_edge(current_eid, P_use)
                current_eid = right

    # --- ОСНОВНЫЕ ПРОХОДЫ
    for _pass in range(safe_limit):
        grid, alive_set = _build_grid_adaptive()
        cut_points: Dict[int, List[Point]] = {}
        changed = False

        # (A) T-стыки: «конец → ребро»
        _collect_tjunction_cuts(H, grid, eps, cut_points)

        # (B) proper crosses (X) — только пары с AABB-пересечением
        seen_pairs: Set[Tuple[int,int]] = set()
        for _, eids in grid.cells.items():
            m = len(eids)
            if m < 2: continue
            for i in range(m):
                ei = eids[i]
                if ei not in alive_set: continue
                u1, v1 = H.edges[ei]
                if u1 == -1 or v1 == -1: continue
                a = H.nodes[u1]; b = H.nodes[v1]
                # AABB ei
                minxa = min(a[0], b[0]) - AABB_PAD; maxxa = max(a[0], b[0]) + AABB_PAD
                minya = min(a[1], b[1]) - AABB_PAD; maxya = max(a[1], b[1]) + AABB_PAD

                for j in range(i+1, m):
                    ej = eids[j]
                    if ej not in alive_set or ei == ej: continue
                    pair = (min(ei, ej), max(ei, ej))
                    if pair in seen_pairs: continue
                    seen_pairs.add(pair)

                    u2, v2 = H.edges[ej]
                    if u2 == -1 or v2 == -1: continue
                    c = H.nodes[u2]; d = H.nodes[v2]

                    # быстрый AABB reject
                    minxb = min(c[0], d[0]) - AABB_PAD; maxxb = max(c[0], d[0]) + AABB_PAD
                    minyb = min(c[1], d[1]) - AABB_PAD; maxyb = max(c[1], d[1]) + AABB_PAD
                    if maxxa < minxb or maxxb < minxa or maxya < minyb or maxyb < minya:
                        continue

                    # proper cross?
                    o1 = _orient(a, b, c)
                    o2 = _orient(a, b, d)
                    o3 = _orient(c, d, a)
                    o4 = _orient(c, d, b)
                    if (o1 * o2 < 0) and (o3 * o4 < 0):
                        ax, ay = a; bx, by = b; cx, cy = c; dx, dy = d
                        r_x, r_y = bx-ax, by-ay
                        s_x, s_y = dx-cx, dy-cy
                        denom = _cross(r_x, r_y, s_x, s_y)
                        if abs(denom) > 1e-18:
                            t = _cross(cx-ax, cy-ay, s_x, s_y) / denom
                            P = (ax + t*r_x, ay + t*r_y)
                            cut_points.setdefault(ei, []).append(P)
                            cut_points.setdefault(ej, []).append(P)

        # применяем сплиты
        for ei, pts in list(cut_points.items()):
            if ei >= len(H.edges): continue
            u, v = H.edges[ei]
            if u == -1 or v == -1: continue

            A = H.nodes[u]; B = H.nodes[v]
            vx, vy = (B[0]-A[0], B[1]-A[1])
            L2 = vx*vx + vy*vy
            if L2 <= 1e-18: continue
            def _t(p): return _dot(p[0]-A[0], p[1]-A[1], vx, vy) / L2

            # оставим только точки на ребре и отсортируем
            pts_on = [p for p in pts if _between(A, B, p, eps)]
            if not pts_on: 
                continue

            # дедуп
            kq = max(eps * 0.25, 1e-9)
            seen = set(); uniq = []
            for p in pts_on:
                key = (round(p[0]/kq), round(p[1]/kq))
                if key in seen: continue
                seen.add(key); uniq.append(p)

            pts_sorted = sorted(uniq, key=_t)
            current_eid = ei
            for P in pts_sorted:
                if current_eid >= len(H.edges): break
                uu, vv = H.edges[current_eid]
                if uu == -1 or vv == -1: break
                Au, Bu = H.nodes[uu], H.nodes[vv]
                du = math.hypot(Au[0]-P[0], Au[1]-P[1])
                dv = math.hypot(Bu[0]-P[0], Bu[1]-P[1])
                # «снэп» к концам + фильтр микрорезов
                P_use = Au if (endpoint_snap and du <= eps) else (Bu if (endpoint_snap and dv <= eps) else P)
                min_len = max(MIN_SPLIT_LEN * eps, 1e-9)
                if du <= min_len or dv <= min_len:
                    continue
                _, _, right = H.split_edge(current_eid, P_use)
                current_eid = right
                changed = True

        if not changed:
            break

    return H
