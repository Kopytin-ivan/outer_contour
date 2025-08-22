# geom/cycles.py
from collections import deque
import math


def connected_components(G):
    seen = set()
    comps = []
    for start in range(len(G.nodes)):
        if start in seen:
            continue
        if start not in G.adj:
            seen.add(start)
            comps.append([start])
            continue
        q = deque([start])
        seen.add(start)
        comp = [start]
        while q:
            u = q.popleft()
            for eid in G.adj.get(u, []):
                a, b = G.edge_nodes(eid)
                v = b if a == u else a
                if v not in seen:
                    seen.add(v)
                    q.append(v)
                    comp.append(v)
        comps.append(comp)
    return comps

def is_degree2_component(G, comp_nodes):
    for nid in comp_nodes:
        if G.degree(nid) != 2:
            return False
    return True

def _component_edges(G, comp_nodes):
    comp = set(comp_nodes)
    eids = []
    for nid in comp_nodes:
        for eid in G.adj.get(nid, []):
            a, b = G.edge_nodes(eid)
            if a in comp and b in comp:
                eids.append(eid)
    return eids

def _next_neighbor(G, prev_node, curr_node):
    """В degree=2 у узла ровно два соседа; берём тот, который != prev."""
    out = []
    for eid in G.adj[curr_node]:
        a, b = G.edge_nodes(eid)
        v = b if a == curr_node else a
        out.append((eid, v))
    # out = [(eid1, v1), (eid2, v2)]
    if not out:
        return None
    if len(out) == 1:
        return out[0]  # деградация, но обычно не бывает в degree=2
    (e1, v1), (e2, v2) = out
    if v1 != prev_node:
        return (e1, v1)
    else:
        return (e2, v2)

# geom/cycles.py
from collections import defaultdict, deque

def find_simple_cycles(G):
    """
    Возвращает список простых циклов как списки node_id в порядке обхода.
    Работает на общем неориентированном графе (узлы могут иметь любую степень).
    Алгоритм: строим остов (DFS/BFS) по компонентам; для каждого нетростового ребра (u,v)
    восстанавливаем путь u→v по остову и получаем фундаментальный цикл.
    Дедупликуем циклы канонизацией (минимальная ротация + выбор направления).
    """
    n = len(G.nodes)
    adj = defaultdict(list)
    for eid, (u, v) in enumerate(G.edges):
        adj[u].append(v)
        adj[v].append(u)

    seen_cycles = set()
    out_cycles = []

    seen_node = set()
    for root in range(n):
        if root in seen_node or root not in G.adj:
            continue

        # BFS-остов для компоненты
        parent = {root: None}
        depth  = {root: 0}
        order  = []
        q = deque([root])
        seen_node.add(root)

        while q:
            u = q.popleft()
            order.append(u)
            for v in adj[u]:
                if v not in parent:
                    parent[v] = u
                    depth[v]  = depth[u] + 1
                    seen_node.add(v)
                    q.append(v)

        # множество остовных рёбер (для быстрого теста)
        tree_edges = set()
        for v in parent:
            u = parent[v]
            if u is not None:
                a, b = (u, v) if u < v else (v, u)
                tree_edges.add((a, b))

        # хелпер: путь между двумя вершинами по остову
        def path_between(a, b):
            pa, pb = [a], [b]
            ua, ub = a, b
            # выровнять глубины
            while depth[ua] > depth[ub]:
                ua = parent[ua]; pa.append(ua)
            while depth[ub] > depth[ua]:
                ub = parent[ub]; pb.append(ub)
            # подниматься до LCA
            while ua != ub:
                ua = parent[ua]; pa.append(ua)
                ub = parent[ub]; pb.append(ub)
            lca = ua
            # цикл: a..LCA + reverse(b..LCA без повторения LCA)
            return pa + list(reversed(pb[:-1]))

        # обойти нетростовые рёбра и собрать фундаментальные циклы
        visited_back = set()
        for u in parent.keys():
            for v in adj[u]:
                a, b = (u, v) if u < v else (v, u)
                if (a, b) in tree_edges:
                    continue  # остовное ребро
                if (a, b) in visited_back:
                    continue
                visited_back.add((a, b))

                # соберём цикл
                cyc = path_between(u, v)
                if len(cyc) < 3:
                    continue

                # канонизация для дедупликации
                # — сдвиг до минимального id
                k = len(cyc)
                min_pos = min(range(k), key=lambda i: cyc[i])
                rot1 = cyc[min_pos:] + cyc[:min_pos]
                # — и обратное направление
                rcyc = list(reversed(cyc))
                min_pos_r = min(range(k), key=lambda i: rcyc[i])
                rot2 = rcyc[min_pos_r:] + rcyc[:min_pos_r]
                canon = tuple(rot2) if tuple(rot2) < tuple(rot1) else tuple(rot1)

                if canon not in seen_cycles:
                    seen_cycles.add(canon)
                    out_cycles.append(list(canon))

    return out_cycles

# --- ВСТАВЬ НИЖЕ В КОНЕЦ geom/cycles.py ---
import math
from collections import defaultdict

def _angle(p, q):
    return math.atan2(q[1]-p[1], q[0]-p[0])


def _angle_left_turn(ax, ay, bx, by):
    """Подписанный поворот A→B: atan2(cross, dot) ∈ (-pi, pi]."""
    cross = ax*by - ay*bx
    dot   = ax*bx + ay*by
    return math.atan2(cross, dot)

import math
from collections import defaultdict

def _ang(p, q):
    return math.atan2(q[1]-p[1], q[0]-p[0])

def find_planar_faces(G, include_outer=False, right_hand=False):
    """
    Half-edge обход граней на PSLG:
      - строит исходящие полурёбра ИЗ СПИСКА РЁБЕР (оба направления для каждого eid),
      - сортирует их по углу CCW,
      - метит посещение по ориентированному полурёбру (u,v,eid),
      - next — минимальный левый поворот с отсечкой 'почти прямо'.
    Возвращает список циклов (списки id вершин) CCW; по умолчанию без внешней (unbounded) грани.
    """
    # 1) исходящие полурёбра: u -> [(v, eid, theta)] (оба направления для каждого ребра)
    out = defaultdict(list)
    for eid, (u, v) in enumerate(G.edges):
        if u == -1 or v == -1:
            continue
        Pu = G.nodes[u]; Pv = G.nodes[v]
        out[u].append((v, eid, _ang(Pu, Pv)))
        out[v].append((u, eid, _ang(Pv, Pu)))
    if not out:
        return []

    for u in out:
        out[u].sort(key=lambda t: t[2])  # CCW

    visited = set()  # ориентированные полурёбра (u, v, eid)
    EPS_ANG = 1e-9   # отсечка 'почти прямо', можно повысить до 1e-7 при шуме

    def next_left(u, v, eid):
        cand = out.get(v)
        if not cand:
            return None
        Vx, Vy = G.nodes[v]
        Ux, Uy = G.nodes[u]
        ax, ay = (Ux - Vx, Uy - Vy)  # вход: вектор v->u

        best = None
        best_delta = None
        for (w, eid2, _) in cand:
            Wx, Wy = G.nodes[w]
            bx, by = (Wx - Vx, Wy - Vy)  # выход: вектор v->w
            # поворот A->B: (-pi, pi] -> (0, 2pi]
            cross = ax*by - ay*bx
            dot   = ax*bx + ay*by
            delta = math.atan2(cross, dot)
            if delta <= 0.0:
                delta += 2.0 * math.pi
            # запрещаем 'прямо' (иначе «срезает» зуб)
            if delta < EPS_ANG:
                continue
            if best_delta is None or delta < best_delta - 1e-12:
                best_delta = delta
                best = (v, w, eid2)
        return best

    def walk(u0, v0, e0):
        u, v, eid = u0, v0, e0
        cyc = [u]
        while True:
            visited.add((u, v, eid))
            cyc.append(v)
            nxt = next_left(u, v, eid) if not right_hand else None
            if nxt is None:
                return None
            u, v, eid = nxt
            if (u, v, eid) == (u0, v0, e0):
                break
            if (u, v, eid) in visited:
                return None
        if len(cyc) >= 3 and cyc[0] == cyc[-1]:
            cyc = cyc[:-1]
        return cyc if len(cyc) >= 3 else None

    faces = []
    for u, L in out.items():
        for (v, eid, _) in L:
            if (u, v, eid) in visited:
                continue
            cyc = walk(u, v, eid)
            if cyc:
                faces.append(cyc)

    if not faces:
        return []

    # площадь
    def area(cyc):
        s = 0.0
        k = len(cyc)
        for i in range(k):
            x1, y1 = G.nodes[cyc[i]]
            x2, y2 = G.nodes[cyc[(i+1) % k]]
            s += x1*y2 - x2*y1
        return 0.5*s

    # убрать внешнюю грань, если не просили включать
    if not include_outer:
        k = max(range(len(faces)), key=lambda i: abs(area(faces[i])))
        faces.pop(k)

    # нормализуем CCW
    out_faces = []
    for cyc in faces:
        if area(cyc) < 0:
            cyc = list(reversed(cyc))
        out_faces.append(cyc)
    return out_faces

