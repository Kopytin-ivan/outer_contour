# geom/extend.py
import math
from geom.templates import make_template
from geom.cycles import find_planar_faces

def _len2(ax, ay): return ax*ax + ay*ay
def _len(ax, ay): return math.hypot(ax, ay)
def _dot(ax, ay, bx, by): return ax*bx + ay*by
def _cross(ax, ay, bx, by): return ax*by - ay*bx






# =======================  CLOSED-ISLAND → HOST (двойные мостики)  =======================
# Идея: для каждой замкнутой компоненты (все degree==2) находим пары параллельных рёбер,
# считаем "цену доводки" (минимальная дистанция до первого хита по оси ребра на host),
# выбираем пару с МИНИМАЛЬНОЙ суммой и добавляем ДВА коротких коннектора к host.

import math

def _edge_len_nodes(a, b): 
    return math.hypot(b[0]-a[0], b[1]-a[1])

def _edge_len_by_eid(G, eid):
    u, v = G.edges[eid]
    if u == -1 or v == -1: 
        return 0.0
    return _edge_len_nodes(G.nodes[u], G.nodes[v])

def _comp_edges_local(G, comp_nodes):
    comp = set(comp_nodes)
    out = []
    for eid, (u, v) in enumerate(G.edges):
        if u == -1 or v == -1: 
            continue
        if u in comp and v in comp:
            out.append(eid)
    return out

def _components_and_host(G):
    # используем твои функции из cycles.py
    from geom.cycles import connected_components, is_degree2_component
    comps = connected_components(G)
    # выбираем host по суммарной длине рёбер
    best_i, best_L = -1, -1.0
    comp_info = []
    for i, comp in enumerate(comps):
        eids = _comp_edges_local(G, comp)
        Lsum = sum(_edge_len_by_eid(G, eid) for eid in eids)
        deg2 = True
        for nid in comp:
            if G.degree(nid) != 2:
                deg2 = False; break
        comp_info.append({"nodes": comp, "eids": eids, "deg2": deg2, "Lsum": Lsum})
        if Lsum > best_L:
            best_L, best_i = Lsum, i
    host = comp_info[best_i] if best_i >= 0 else None
    return comp_info, best_i, host

def _angle_mod_pi(dx, dy):
    # угол в градусах в [0, 180)
    ang = math.degrees(math.atan2(dy, dx)) % 180.0
    if ang < 0: ang += 180.0
    return ang

def _group_edges_by_orientation(G, eids, ang_tol_deg):
    """
    Кластеризация по направлению (0..180) с шагом ang_tol_deg.
    Возвращает список групп: [{"key":k, "eids":[...], "dir":(ux,uy)}].
    """
    bins = {}
    for eid in eids:
        u, v = G.edges[eid]
        ax, ay = G.nodes[u]; bx, by = G.nodes[v]
        dx, dy = (bx-ax, by-ay)
        L = math.hypot(dx, dy)
        if L <= 1e-12:
            continue
        ang = _angle_mod_pi(dx, dy)
        key = int(round(ang / float(ang_tol_deg)))  # дискретизируем по толерансу
        if key not in bins:
            bins[key] = {"eids": [], "vects": []}
        bins[key]["eids"].append(eid)
        bins[key]["vects"].append((dx/L, dy/L))
    groups = []
    for k, rec in bins.items():
        # усреднённый unit-вектор направления
        sx = sum(x for x, y in rec["vects"]); sy = sum(y for x, y in rec["vects"])
        L = math.hypot(sx, sy)
        ux, uy = (sx/L, sy/L) if L > 1e-12 else (1.0, 0.0)
        groups.append({"key": k, "eids": rec["eids"], "dir": (ux, uy)})
    return groups

def _first_host_hit_along_ray(G, grid, params, host_eids_set, startP, D, ignore_eids):
    """
    Первый валидный хит лучом startP + t*D по рёбрам HOST c проверкой видимости.
    Возвращает (t, eid_target, u_on_target, X) или None.
    ВАЖНО: радиус поиска и лимит t берутся из CLOSED_BRIDGE_MAX, а не из R_QUERY.
    """
    from geom.extend import _ray_segment_intersection, _has_blocking_hit  # уже есть в файле

    max_d       = float(params.get("CLOSED_BRIDGE_MAX", 0.20))  # например 0.20 м (200 мм)
    ang_tol     = float(params.get("ANGLE_TOL", 2.0))
    near_perp   = float(params.get("NEAR_PERP_MAX", 0.01))
    eps         = float(params.get("EPS_SNAP", 0.002))

    # Радиус поиска: фиксированный BRIDGE_SEARCH_R ИЛИ множитель от max_d
    if params.get("BRIDGE_SEARCH_R") is not None:
        search_r = float(params["BRIDGE_SEARCH_R"])
    else:
        mult = float(params.get("BRIDGE_SEARCH_R_MULT", 1.5))
        search_r = max_d * mult

    cand = None
    for eid in grid.nearby_segments_by_point(startP[0], startP[1], search_r):
        if eid not in host_eids_set:
            continue
        u, v = G.edges[eid]
        if u == -1 or v == -1:
            continue
        A = G.nodes[u]; B = G.nodes[v]
        hit = _ray_segment_intersection(startP, D, A, B,
                                        ang_tol_deg=ang_tol, col_perp_tol=near_perp)
        if not hit:
            continue
        t, ut, X = hit
        if t < 1e-9 or t > max_d:           # ← лимит по CLOSED_BRIDGE_MAX
            continue
        # видимость только на отрезке startP→X
        if _has_blocking_hit(G, grid, startP, X, ignore_eids=ignore_eids | {eid}, eps=eps):
            continue
        if (cand is None) or (t < cand[0]):
            cand = (t, eid, ut, X)
    return cand


def _best_extension_for_island_edge(G, grid, params, host_eids_set, island_eid, island_eids_set):
    """
    Для ребра острова берём два полу-луча: из B по +d и из A по -d.
    Возвращает лучшую попытку: dict{start_nid, t, target_eid, u, X, dir}
    или None, если валидных хитов нет.
    """
    u, v = G.edges[island_eid]
    A = G.nodes[u]; B = G.nodes[v]
    dx, dy = (B[0]-A[0], B[1]-A[1])
    L = math.hypot(dx, dy)
    if L <= 1e-12:
        return None
    d = (dx/L, dy/L)
    ignore = set(island_eids_set)  # игнорируем собственный многоугольник при видимости

    # 1) из B по +d
    hit1 = _first_host_hit_along_ray(G, grid, params, host_eids_set, B, d, ignore)
    # 2) из A по -d
    hit2 = _first_host_hit_along_ray(G, grid, params, host_eids_set, A, (-d[0], -d[1]), ignore)

    best = None
    if hit1:
        t, eid_t, u_on, X = hit1
        best = {"start_nid": v, "t": t, "target_eid": eid_t, "u": u_on, "X": X, "dir": +1}
    if hit2:
        t, eid_t, u_on, X = hit2
        if (best is None) or (t < best["t"]):
            best = {"start_nid": u, "t": t, "target_eid": eid_t, "u": u_on, "X": X, "dir": -1}
    return best

def _project_on_normal_of_group(G, eid, gdir):
    # нормаль к среднему направлению группы
    nx, ny = (-gdir[1], gdir[0])
    a, b = G.edges[eid]
    ax, ay = G.nodes[a]; bx, by = G.nodes[b]
    mx, my = (0.5*(ax+bx), 0.5*(ay+by))  # центр ребра
    return mx*nx + my*ny

def _apply_connector_to_host(G, best, params=None):
    """
    Создаёт КОРОТКИЙ коннектор start_nid → target.
    Если целевое ребро уже расколото ранее, пытается найти готовый узел в точке X
    и использовать его, а не выполнять split повторно.
    """
    eps = (params or {}).get("EPS_SNAP", 2.0)

    def _find_node_at(X, tol):
        tx, ty = X
        tol2 = tol * tol
        for nid, (x, y) in enumerate(G.nodes):
            if (x - tx) * (x - tx) + (y - ty) * (y - ty) <= tol2:
                return nid
        return None

    start   = best["start_nid"]
    eid_t   = best["target_eid"]
    u       = best["u"]
    X       = best["X"]

    target = None

    # Попадание в середину исходного ребра
    if 1e-9 < u < 1.0 - 1e-9:
        # Если ребро ещё «живое» — колем его.
        if 0 <= eid_t < len(G.edges):
            a, b = G.edges[eid_t]
        else:
            a = b = -1
        if a != -1 and b != -1:
            mid, e1, e2 = G.split_edge(eid_t, X)
            target = mid
        else:
            # Ребро уже расколото ранее — попробуем найти готовый узел в точке X
            nid = _find_node_at(X, tol=max(eps, 1e-9))
            if nid is None:
                return None  # ничего лучше не остаётся — безопасно пропустить
            target = nid
    else:
        # Попадание в конец: берём соответствующую вершину (если ребро ещё существует),
        # иначе пытаемся найти узел возле X.
        if 0 <= eid_t < len(G.edges):
            a, b = G.edges[eid_t]
        else:
            a = b = -1
        if a != -1 and b != -1:
            target = b if u >= 0.5 else a
        else:
            nid = _find_node_at(X, tol=max(eps, 1e-9))
            if nid is None:
                return None
            target = nid

    eid_new = G.add_edge(start, target)
    return {"mode": "island-bridge", "connector_eid": eid_new, "from": start, "to": target, "t": best["t"]}

def _live_host_sets(G, host_seed_nodes):
    """
    Возвращает (host_eids_set, host_nodes_set) для ТЕКУЩЕГО графа.
    Ищем компоненту, которая содержит хотя бы часть исходных host_seed_nodes.
    Если не нашли (крайний случай) — берём компоненту с максимальной суммой длин рёбер.
    """
    from geom.cycles import connected_components
    comps = connected_components(G)

    # Быстрый индекс узлов -> id компоненты
    nid2cid = {}
    for cid, comp in enumerate(comps):
        for nid in comp:
            nid2cid[nid] = cid

    # Выберем «лучшую» компоненту: максимальный пересекающийся объём с seed'ами
    best_cid = None
    best_overlap = -1
    seed = set(host_seed_nodes)
    for cid, comp in enumerate(comps):
        ov = len(seed.intersection(comp))
        if ov > best_overlap:
            best_overlap = ov
            best_cid = cid

    # Если seed-узлы вдруг все выпали, fallback — берём компоненту с макс. длиной рёбер
    if best_overlap <= 0:
        comp_info, _, host = _components_and_host(G)
        host_nodes = set(host["nodes"]) if host else set()
    else:
        host_nodes = set(comps[best_cid])

    host_eids = set()
    for eid, (u, v) in enumerate(G.edges):
        if u == -1 or v == -1:
            continue
        if (u in host_nodes) and (v in host_nodes):
            host_eids.add(eid)
    return host_eids, host_nodes



def bridge_closed_island_min_parallel(G, grid, params, host_seed_nodes, island_nodes) -> list:
    """
    Для замкнутого островка выбираем пару ПАРАЛЛЕЛЬНЫХ рёбер с минимальной суммой доводок
    и добавляем ДВА коротких коннектора к хосту. Рёбра хоста считаются «живыми».
    """
    eids_island = _comp_edges_local(G, island_nodes)
    if len(eids_island) < 3:
        return []

    groups = _group_edges_by_orientation(G, eids_island, params["ANGLE_TOL"])
    if not groups:
        return []

    island_set = set(eids_island)

    # Посчитаем «лучшую» доводку для каждого ребра на текущий (живой) хост
    host_eids_set, _ = _live_host_sets(G, host_seed_nodes)
    best_for_edge = {}
    for eid in eids_island:
        best = _best_extension_for_island_edge(G, grid, params, host_eids_set, eid, island_set)
        if best:
            best_for_edge[eid] = best

    # Выбираем пару параллельных рёбер с минимальной суммарной t и разнесённостью по нормали
    EPS = params.get("EPS_SNAP", 2.0)
    best_pair = None
    best_group = None
    for g in groups:
        eids = [eid for eid in g["eids"] if eid in best_for_edge]
        if len(eids) < 2:
            continue
        eids.sort(key=lambda e: _project_on_normal_of_group(G, e, g["dir"]))
        for i in range(len(eids)-1):
            for j in range(i+1, len(eids)):
                e1, e2 = eids[i], eids[j]
                s1 = _project_on_normal_of_group(G, e1, g["dir"])
                s2 = _project_on_normal_of_group(G, e2, g["dir"])
                if abs(s2 - s1) < 2*EPS:
                    continue
                t_sum = best_for_edge[e1]["t"] + best_for_edge[e2]["t"]
                if (best_pair is None) or (t_sum < best_pair[0]):
                    best_pair = (t_sum, e1, e2)
                    best_group = g
    if not best_pair:
        return []

    _, e1, e2 = best_pair
    ops = []

    # 1) Ставим первый мостик
    op1 = _apply_connector_to_host(G, best_for_edge[e1])
    if op1:
        ops.append(op1)
        # сетка изменилась — перестроим
        from geom.grid import build_grid_from_graph
        grid = build_grid_from_graph(G, grid.cell, pad=params.get("EPS_SNAP", 0.0))

    # 2) Для второй стороны — ПЕРЕСЧИТЫВАЕМ лучшую доводку на "живом" хосте
    host_eids_set, _ = _live_host_sets(G, host_seed_nodes)
    best2 = _best_extension_for_island_edge(G, grid, params, host_eids_set, e2, island_set)
    if best2:
        op2 = _apply_connector_to_host(G, best2)
        if op2:
            ops.append(op2)
            try:
                from geom.grid import build_grid_from_graph
                grid = build_grid_from_graph(G, grid.cell, pad=params.get("EPS_SNAP", 0.0))
            except Exception:
                pass

    return ops


def connect_closed_islands_to_host(G, grid, params, rebuild_grid_each=True):
    """
    Подключает ВСЕ замкнутые островки к хост-компоненте двумя коннекторами (параллельная пара).
    """
    from geom.cycles import is_degree2_component
    comp_info, host_idx, host = _components_and_host(G)
    if host is None:
        return []

    host_seed_nodes = list(host["nodes"])
    all_ops = []

    for i, info in enumerate(comp_info):
        if i == host_idx:
            continue
        if not is_degree2_component(G, info["nodes"]):
            continue  # только замкнутые острова

        ops = bridge_closed_island_min_parallel(G, grid, params, host_seed_nodes, info["nodes"])
        if ops:
            all_ops.extend(ops)
            if rebuild_grid_each:
                from geom.grid import build_grid_from_graph
                grid = build_grid_from_graph(G, grid.cell, pad=params.get("EPS_SNAP", 0.0))
                # после изменений пересчитаем seed'ы хоста: текущая host-компонента содержит старые seed-узлы
                _, host_nodes_now = _live_host_sets(G, host_seed_nodes)
                host_seed_nodes = list(host_nodes_now)

    return all_ops

# =======================  END CLOSED-ISLAND FEATURE  =======================

def _unit(dx, dy):
    L = _len(dx, dy)
    return (0.0, 0.0) if L <= 1e-18 else (dx/L, dy/L)

def _ray_segment_intersection(P, D, A, B, ang_tol_deg=2.0, col_perp_tol=10.0, eps=1e-12):
    """
    Пересечение луча P + t*D (t>=0) с отрезком [A,B].
    Возвращает (t, u, X) или None.
    Коллинеарный случай: если [A,B] почти на оси луча (поперечное отклонение <= col_perp_tol),
    берём ближайшую вперёд точку (A/B).
    """
    rx, ry = D
    sx, sy = (B[0]-A[0], B[1]-A[1])
    denom = _cross(rx, ry, sx, sy)

    if abs(denom) > eps:
        APx, APy = (A[0]-P[0], A[1]-P[1])
        t = _cross(APx, APy, sx, sy) / denom
        u = _cross(APx, APy, rx, ry) / denom
        if t < 0.0 or u < 0.0 or u > 1.0:
            return None
        X = (P[0] + t*rx, P[1] + t*ry)
        return t, u, X

    # почти параллельно → смотрим поперечное отклонение концов сегмента от оси луча
    def _perp(Z):
        vx, vy = (Z[0]-P[0], Z[1]-P[1])
        s = _dot(vx, vy, rx, ry)
        per2 = max(0.0, vx*vx + vy*vy - s*s)
        return s, math.sqrt(per2)

    sA, perA = _perp(A)
    sB, perB = _perp(B)
    if perA <= col_perp_tol and perB <= col_perp_tol:
        cand = [s for s in (sA, sB) if s >= 0.0]
        if not cand:
            return None
        t = min(cand)
        X = (P[0] + t*rx, P[1] + t*ry)
        vv = sx*sx + sy*sy
        u = 0.0 if vv <= eps else max(0.0, min(1.0, _dot(X[0]-A[0], X[1]-A[1], sx, sy) / vv))
        return t, u, X

    return None

def _is_isolated_tail(G, tail) -> bool:
    """Истинно, если рассматриваемый хвост принадлежит отрезку, у которого оба конца degree=1."""
    eid = tail["eid"]
    u, v = G.edges[eid]
    if u == -1 or v == -1:
        return False
    return (G.degree(u) == 1) and (G.degree(v) == 1)



def _nearest_point_on_segment(P, A, B):
    vx, vy = B[0]-A[0], B[1]-A[1]
    wx, wy = P[0]-A[0], P[1]-A[1]
    vv = vx*vx + vy*vy
    if vv <= 1e-18: return _len(P[0]-A[0], P[1]-A[1]), A, 0.0
    u = max(0.0, min(1.0, (wx*vx + wy*vy)/vv))
    X = (A[0] + u*vx, A[1] + u*vy)
    return _len(P[0]-X[0], P[1]-X[1]), X, u

def _face_contains_directed_edge(face_nodes, a, b):
    k = len(face_nodes)
    for i in range(k):
        if face_nodes[i] == a and face_nodes[(i+1)%k] == b:
            return True
    return False

def _decompose_along(P, D, X):
    """PX = s*D + n. Возвращает (s — вдоль, per — поперёк оси луча)."""
    vx, vy = X[0]-P[0], X[1]-P[1]
    s = _dot(vx, vy, D[0], D[1])
    per2 = max(0.0, vx*vx + vy*vy - s*s)
    return s, math.sqrt(per2)

import os, json  # если хочешь лог

def _bbox(P, Q):
    x1, y1 = P; x2, y2 = Q
    return (min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2))

def _seg_intersection_proper(P, Q, A, B, eps=1e-9):
    def orient(a,b,c): return (b[0]-a[0])*(c[1]-a[1]) - (b[1]-a[1])*(c[0]-a[0])
    o1 = orient(P,Q,A); o2 = orient(P,Q,B)
    o3 = orient(A,B,P); o4 = orient(A,B,Q)
    minx1, miny1, maxx1, maxy1 = _bbox(P,Q)
    minx2, miny2, maxx2, maxy2 = _bbox(A,B)
    if maxx1+eps < minx2 or maxx2+eps < minx1 or maxy1+eps < miny2 or maxy2+eps < miny1:
        return False
    return (o1*o2 < -eps) and (o3*o4 < -eps)


def _has_blocking_hit(G, grid, A, B, ignore_eids, eps):
    # проверяем пересечения на всём отрезке A–B
    def orient(p,q,r): return (q[0]-p[0])*(r[1]-p[1]) - (q[1]-p[1])*(r[0]-p[0])
    def bbox(p,q): return (min(p[0],q[0]), min(p[1],q[1]), max(p[0],q[0]), max(p[1],q[1]))
    minx1,miny1,maxx1,maxy1 = bbox(A,B)
    cx, cy = (A[0]+B[0])*0.5, (A[1]+B[1])*0.5
    r = 0.5*math.hypot(B[0]-A[0], B[1]-A[1]) + 5*eps
    for eid in grid.nearby_segments_by_point(cx, cy, r):
        if eid in ignore_eids: 
            continue
        u,v = G.edges[eid]
        if u == -1 or v == -1: 
            continue
        C, D = G.nodes[u], G.nodes[v]
        minx2,miny2,maxx2,maxy2 = bbox(C,D)
        if maxx1+eps < minx2 or maxx2+eps < minx1 or maxy1+eps < miny2 or maxy2+eps < miny1:
            continue
        o1 = orient(A,B,C); o2 = orient(A,B,D)
        o3 = orient(C,D,A); o4 = orient(C,D,B)
        if o1*o2 < -eps and o3*o4 < -eps:
            return True
    return False



def _gather_candidates(G, grid, tail, params):
    P = tail["P"]; D = tail["D"]
    prev = tail["prev"]; end = tail["end"]

    max_extend = params["MAX_EXTEND"]
    iso_mult   = params.get("ISOLATED_EXT_MULT", 2.0)
    # если хвост на изолированном сегменте — разрешим до 2×MAX_EXTEND
    ext_limit  = max_extend * (iso_mult if _is_isolated_tail(G, tail) else 1.0)

    ang_tol    = params["ANGLE_TOL"]
    near_perp_max    = params.get("NEAR_PERP_MAX", 10.0)
    near_forward_min = params.get("NEAR_FORWARD_MIN", 0.0)
    eps = params.get("EPS_SNAP", 2.0)

    # радиус поиска берём по расширенному лимиту
    search_r = params.get("SEARCH_RADIUS") or (ext_limit * 1.25)
    eids = [e for e in grid.nearby_segments_by_point(P[0], P[1], search_r)
            if e != tail["eid"] and G.edges[e][0] != -1]

    cand = []

    # --- RAY (по направлению хвоста) ---
    best_ray = None
    for eid in eids:
        A = G.nodes[G.edges[eid][0]]; B = G.nodes[G.edges[eid][1]]
        hit = _ray_segment_intersection(P, D, A, B, ang_tol_deg=ang_tol, col_perp_tol=near_perp_max)
        if not hit:
            continue
        t, u, X = hit
        if t < 1e-9 or t > ext_limit:     # ← ИСПОЛЬЗУЕМ ext_limit
            continue
        # видимость только для новой добавки P→T
        target = G.edges[eid][1] if (u >= 0.5 or abs(u-1.0) < 1e-9) else G.edges[eid][0]
        T = X if (1e-9 < u < 1-1e-9) else G.nodes[target]
        if _has_blocking_hit(G, grid, P, T, ignore_eids={tail["eid"], eid}, eps=eps):
            continue
        if (best_ray is None) or (t < best_ray["dist"]):
            best_ray = {"mode":"ray","target_eid":eid,"u":u,"X":X,"dist":t}
    if best_ray:
        cand.append(best_ray)

    # --- NEAREST (узкий коридор вдоль D) ---
    best_near = None
    for eid in eids:
        A = G.nodes[G.edges[eid][0]]; B = G.nodes[G.edges[eid][1]]
        _, X, u = _nearest_point_on_segment(P, A, B)
        s, per = _decompose_along(P, D, X)
        if s < near_forward_min or s > ext_limit or per > near_perp_max:   # ← ext_limit
            continue

        # оценим излом для выбора (как раньше)
        vx, vy = X[0]-P[0], X[1]-P[1]
        Lv = math.hypot(vx, vy) or 1.0
        ux, uy = vx/Lv, vy/Lv
        dot = max(-1.0, min(1.0, D[0]*ux + D[1]*uy))
        kink_deg = math.degrees(math.acos(dot))

        T = X
        if _has_blocking_hit(G, grid, P, T, ignore_eids={tail["eid"], eid}, eps=eps):
            continue
        if (best_near is None) or (s < best_near["dist"]):
            best_near = {"mode":"nearest","target_eid":eid,"u":u,"X":X,"dist":s, "kink_deg":kink_deg}
    if best_near:
        cand.append(best_near)

    return cand



def _make_tail(P, prev):
    """Направление 'наружу' из конца prev->P."""
    return _unit(P[0]-prev[0], P[1]-prev[1])

def find_tails(G):
    tails = []
    for eid, (u, v) in enumerate(G.edges):
        if u == -1: continue
        # конец в u
        if G.degree(u) == 1:
            P = G.nodes[u]; prev = v
            tails.append({"eid":eid, "end":u, "prev":prev, "P":P, "D":_make_tail(P, G.nodes[prev])})
        if G.degree(v) == 1:
            P = G.nodes[v]; prev = u
            tails.append({"eid":eid, "end":v, "prev":prev, "P":P, "D":_make_tail(P, G.nodes[prev])})
    return tails

def _try_template_for_choice(G, end_nid, choice, tdb, params):
    """Проверяем: если добавить это ребро, получается ли грань, похожая на шаблон из базы?"""
    # 1) клон графа и временное добавление
    G2 = G.clone()
    eid = choice["target_eid"]; u = choice["u"]; X = choice["X"]
    a, b = G2.edges[eid]
    if u > 1e-9 and u < 1-1e-9:
        mid, e1, e2 = G2.split_edge(eid, X)
        target = mid
    else:
        target = b if u >= 0.5 else a
    # подключаем конец
    G2.add_edge(end_nid, target)

    # 2) находим грани и выбираем те, где есть directed edge end_nid->target
    faces = find_planar_faces(G2, include_outer=False, right_hand=True)
    faces_hit = [cyc for cyc in faces if _face_contains_directed_edge(cyc, end_nid, target)]
    if not faces_hit:
        return False

    # 3) строим шаблоны и спрашиваем память
    ang_tol = params.get("TEMPLATE_ANG_TOL", 2)
    len_tol = params.get("TEMPLATE_LEN_TOL", 5)
    for cyc in faces_hit:
        tpl = make_template(G2, cyc)
        if not tpl: 
            continue
        if tdb.has_similar(tpl, ang_tol=ang_tol, len_tol=len_tol):
            return True
    return False

def choose_by_templates(G, grid, tail, tdb, params):
    """Сначала пытаемся найти вариант доводки, который даёт 'похожую' грань из памяти."""
    cand = _gather_candidates(G, grid, tail, params)
    for choice in cand:
        if _try_template_for_choice(G, tail["end"], choice, tdb, params):
            return choice  # этот вариант порождает грань, похожую на известную
    return None

def choose_rule_based(G, grid, tail, params):
    """Fallback: правило 'ray vs nearest' c коэффициентом и расширенным лимитом для изолированных хвостов."""
    cand = _gather_candidates(G, grid, tail, params)
    if not cand:
        return None

    max_extend = params["MAX_EXTEND"]
    iso_mult   = params.get("ISOLATED_EXT_MULT", 2.0)
    ext_limit  = max_extend * (iso_mult if _is_isolated_tail(G, tail) else 1.0)

    ratio        = params.get("DIR_TO_NEAR_RATIO", 2.0)
    near_kinkmax = params.get("NEAR_MAX_KINK_DEG", 12)

    ray  = next((c for c in cand if c["mode"] == "ray"), None)
    near = next((c for c in cand if c["mode"] == "nearest"), None)

    ray_ok  = bool(ray  and ray["dist"]  <= ext_limit)
    near_ok = bool(near and near["dist"] <= ext_limit)

    if ray_ok and not near_ok:
        return ray
    if near_ok and not ray_ok:
        return near  # луча нет/слишком далеко — берём ближайший

    if not ray_ok and not near_ok:
        return None

    # Оба валидны: если nearest делает заметный излом — выбираем луч
    if near.get("kink_deg", 0.0) > near_kinkmax:
        return ray

    # Иначе стандартное сравнение по ratio
    if ray["dist"] <= ratio * near["dist"]:
        return ray
    return near


def apply_choice(G, grid, params, tail, choice):
    eid_target = choice["target_eid"]
    u          = choice["u"]
    X          = choice["X"]
    mode       = choice["mode"]
    dist       = choice["dist"]

    a, b = G.edges[eid_target]
    if 1e-9 < u < 1.0 - 1e-9:
        mid, e1, e2 = G.split_edge(eid_target, X)
        target = mid
    else:
        target = b if u >= 0.5 else a

    prev = tail["prev"]; end = tail["end"]
    P = G.nodes[end]
    T = G.nodes[target]
    eps =  params.get("EPS_SNAP", 2.0)

    if mode == "ray":
        # безопасно «дотянуть по прямой»: новый отрезок prev→target коллинеарен старому
        # (дополнительно убеждаемся, что по пути нет блокеров)
        if _has_blocking_hit(G, grid, P, T, ignore_eids={tail["eid"], eid_target}, eps=eps):
            return None
        G.remove_edge(tail["eid"])
        eid_new = G.add_edge(prev, target)
        return {"mode": mode, "tail_eid_old": tail["eid"], "tail_eid_new": eid_new,
                "target_nid": target, "dist": dist}

    else:  # mode == "nearest"
        # НЕ поворачиваем всю длинную грань! Добавляем КОРОТКИЙ коннектор end→target.
        if _has_blocking_hit(G, grid, P, T, ignore_eids={tail["eid"], eid_target}, eps=eps):
            return None  # даже короткий отрезок чем-то перекрыт
        # оставляем prev→end как есть и добавляем end→target
        eid_conn = G.add_edge(end, target)
        return {"mode": mode, "connector_eid": eid_conn, "end": end,
                "target_nid": target, "dist": dist}


def close_tails_smart(G, grid, params, templates_db, iter_max=5):
    """
    Итеративно доводит хвосты.
    1) Сначала пробует варианты, которые дают грань «похожую» на шаблон из базы.
    2) Если не нашлось — правило «луч vs ближайший».
    Возвращает СПИСОК операций (каждая — dict из apply_choice).
    """
    ops = []
    for _ in range(iter_max):
        tails = find_tails(G)
        did = 0
        for t in tails:
            choice = choose_by_templates(G, grid, t, templates_db, params)
            if not choice:
                choice = choose_rule_based(G, grid, t, params)
            if not choice:
                continue
            op = apply_choice(G, grid, params, t, choice)
            if not op:
                continue
            ops.append(op)
            did += 1

        if did == 0:
            break
        # После серии изменений обновим сетку, чтобы следующие итерации видели новые рёбра
        try:
            from geom.grid import build_grid_from_graph
            grid = build_grid_from_graph(G, grid.cell, pad=params.get("EPS_SNAP", 0.0))
        except Exception:
            # Если по какой-то причине обновить сетку не удалось, продолжаем со старой
            pass
    return ops


def _neighbors(G, nid):
    for eid in G.adj.get(nid, []):
        u, v = G.edges[eid]
        if u == -1 or v == -1:
            continue
        yield v if u == nid else u


def _connected_components(G):
    visited = set()
    comps = []
    for nid in range(len(G.nodes)):
        if nid in visited:
            continue
        # узел может быть полностью изолирован без рёбер
        if not G.adj.get(nid):
            visited.add(nid)
            comps.append({nid})
            continue
        if any(G.edges[eid][0] != -1 for eid in G.adj.get(nid, [])):
            # BFS по существующим рёбрам
            stack = [nid]
            comp = set()
            visited.add(nid)
            while stack:
                x = stack.pop()
                comp.add(x)
                for y in _neighbors(G, x):
                    if y not in visited:
                        visited.add(y)
                        stack.append(y)
            comps.append(comp)
        else:
            visited.add(nid)
            comps.append({nid})
    return comps


def connect_islands_to_nearby(G, grid, params, max_iters=2):
    """Продлевает ХВОСТЫ островков вдоль их направления (ray) и подключает к главной компоненте.
    Только пересечение лучом с сегментами основной компоненты, расстояние ≤ BRIDGE_DIST_MAX (по умолчанию 2×MAX_EXTEND).
    Возвращает число выполненных продлений.
    """
    from geom.grid import build_grid_from_graph

    bridge_max = params.get("BRIDGE_DIST_MAX")
    if bridge_max is None:
        bridge_max = 2.0 * params.get("MAX_EXTEND", 0.05)
    ang_tol = params.get("ANGLE_TOL", 2.0)
    eps = params.get("EPS_SNAP", 0.002)

    added = 0
    for _ in range(max_iters):
        comps = _connected_components(G)
        if not comps:
            break
        comps_sorted = sorted(comps, key=lambda c: len(c), reverse=True)
        main = comps_sorted[0]
        others = comps_sorted[1:]
        if not others:
            break

        did = 0
        # хвосты только внутри островков
        all_tails = find_tails(G)
        for comp in others:
            comp_tails = [t for t in all_tails if t["end"] in comp and t["prev"] in comp]
            for tail in comp_tails:
                P = tail["P"]
                D = tail["D"]
                best = None  # (t, eid, u, X)
                for eid in grid.nearby_segments_by_point(P[0], P[1], bridge_max * 1.1):
                    a0, b0 = G.edges[eid]
                    if a0 == -1 or b0 == -1:
                        continue
                    # только сегменты НЕ из этой компоненты (то есть из главной)
                    if a0 in comp or b0 in comp:
                        continue
                    A = G.nodes[a0]; B = G.nodes[b0]
                    hit = _ray_segment_intersection(P, D, A, B, ang_tol_deg=ang_tol, col_perp_tol=params.get("NEAR_PERP_MAX", 10.0))
                    if not hit:
                        continue
                    t, u, X = hit
                    if t < 0.0 or t > bridge_max:
                        continue
                    if _has_blocking_hit(G, grid, P, X, ignore_eids={tail["eid"], eid}, eps=eps):
                        continue
                    if (best is None) or (t < best[0]):
                        best = (t, eid, u, X)
                if best is None:
                    # Нет валидных хвостов: продлеваем КРАЯ сегментов островка строго к ближайшему
                    # целевому сегменту главной компоненты и только по направлению, близкому к нормали.
                    # 1) выбираем ближайший сегмент в главной компоненте
                    nearest = None  # (dist, target_eid, u_hit, X_hit, nid_from)
                    # оценим по ближайшей паре (узел островка -> точка на любом сегменте main)
                    for nid_probe in comp:
                        Pp = G.nodes[nid_probe]
                        for eidM in grid.nearby_segments_by_point(Pp[0], Pp[1], bridge_max * 1.1):
                            aM, bM = G.edges[eidM]
                            if aM == -1 or bM == -1:
                                continue
                            if aM in comp or bM in comp:
                                continue
                            AM = G.nodes[aM]; BM = G.nodes[bM]
                            d_probe, Xp, up = _nearest_point_on_segment(Pp, AM, BM)
                            if (nearest is None) or (d_probe < nearest[0]):
                                nearest = (d_probe, eidM, up, Xp, nid_probe)
                    if nearest is None:
                        continue
                    _, target_eid, _, _, _ = nearest
                    aT, bT = G.edges[target_eid]
                    AT = G.nodes[aT]; BT = G.nodes[bT]
                    Tx, Ty = (BT[0]-AT[0], BT[1]-AT[1])
                    Lt = math.hypot(Tx, Ty) or 1.0
                    Ux, Uy = (Tx/Lt, Ty/Lt)
                    # нормали к целевому сегменту
                    Nx1, Ny1 = (-Uy, Ux)
                    Nx2, Ny2 = (Uy, -Ux)
                    align_tol = math.radians(params.get("ISLAND_ALIGN_TOL", 20))

                    # 2) собираем кандидатов: направления концов сегментов островка, близкие к нормали целевого
                    cand2 = []  # (t, end_nid, u_on_target, X_hit)
                    comp_eids = [eid2 for eid2,(u2,v2) in enumerate(G.edges) if u2 != -1 and v2 != -1 and (u2 in comp and v2 in comp)]
                    for eid2 in comp_eids:
                        u2, v2 = G.edges[eid2]
                        P1 = G.nodes[u2]; P2 = G.nodes[v2]
                        for end_nid, other_xy in ((u2, P2), (v2, P1)):
                            P_end = G.nodes[end_nid]
                            Dx, Dy = (P_end[0]-other_xy[0], P_end[1]-other_xy[1])
                            Ld = math.hypot(Dx, Dy)
                            if Ld <= 1e-18:
                                continue
                            Udx, Udy = (Dx/Ld, Dy/Ld)
                            # проверяем выравнивание с нормалью целевого сегмента
                            dot1 = abs(Udx*Nx1 + Udy*Ny1)
                            dot2 = abs(Udx*Nx2 + Udy*Ny2)
                            if max(dot1, dot2) < math.cos(align_tol):
                                continue
                            hit2 = _ray_segment_intersection(P_end, (Udx, Udy), AT, BT, ang_tol_deg=ang_tol, col_perp_tol=params.get("NEAR_PERP_MAX", 10.0))
                            if not hit2:
                                continue
                            t2, u2p, X2 = hit2
                            if t2 < 0.0 or t2 > bridge_max:
                                continue
                            if _has_blocking_hit(G, grid, P_end, X2, ignore_eids={eid2, target_eid}, eps=eps):
                                continue
                            cand2.append((t2, end_nid, u2p, X2))
                    if not cand2:
                        continue
                    # выберем до двух наименьших по t с разными концами
                    cand2.sort(key=lambda r: r[0])
                    used_ends = set()
                    for t2, end_nid, u2p, X2 in cand2:
                        if end_nid in used_ends:
                            continue
                        used_ends.add(end_nid)
                        # подготовить точку на целевом сегменте
                        aM, bM = G.edges[target_eid]
                        if 1e-9 < u2p < 1.0 - 1e-9:
                            mid, e1, e2 = G.split_edge(target_eid, X2)
                            target = mid
                            # целевой сегмент мог измениться; обновим его id на ближайший к X2
                            target_eid = e1  # неважно какой; для следующих кандидатов всё равно перещепим
                        else:
                            target = bM if u2p >= 0.5 else aM
                        # добавляем прямое продление: коннектор от конца сегмента островка
                        if not _has_blocking_hit(G, grid, G.nodes[end_nid], G.nodes[target], ignore_eids=set(), eps=eps):
                            G.add_edge(end_nid, target)
                            did += 1
                            added += 1
                        if len(used_ends) >= 2:
                            break
                    continue

                t, eid, u, X = best
                # подготовить цель: разрез или конец
                a0, b0 = G.edges[eid]
                if 1e-9 < u < 1.0 - 1e-9:
                    mid, e1, e2 = G.split_edge(eid, X)
                    target = mid
                else:
                    target = b0 if u >= 0.5 else a0

                # продлить сам хвост (как для режима ray): переподключение prev→target
                prev = tail["prev"]
                eid_tail = tail["eid"]
                G.remove_edge(eid_tail)
                G.add_edge(prev, target)
                did += 1
                added += 1

        if did == 0:
            break
        grid = build_grid_from_graph(G, grid.cell, pad=eps)

    return added
