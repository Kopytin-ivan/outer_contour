# snap.py
from collections import defaultdict
import math

def _bucket(p, cell):
    return (int(math.floor(p[0]/cell)), int(math.floor(p[1]/cell)))

def snap_points(segments, eps):
    """Возвращает (nodes, edges), где:
       nodes: dict[node_id] = (x,y) — опорные узлы
       edges: list[(u,v)]    — индексы узлов
    """
    buckets = defaultdict(list)  # (ix,iy) -> [node_id]
    nodes = []                   # id -> (x,y)
    node_of = {}                 # точка после снапа: кеширование по исходной точке (опционально)

    def get_node_id(p):
        ix,iy = _bucket(p, eps)
        # ищем ближайший существующий узел в соседних 3x3 ведрах
        best_id, best_d = None, eps
        for dx in (-1,0,1):
            for dy in (-1,0,1):
                for nid in buckets[(ix+dx, iy+dy)]:
                    x,y = nodes[nid]
                    d = math.hypot(x-p[0], y-p[1])
                    if d <= best_d:
                        best_d, best_id = d, nid
        if best_id is not None:
            return best_id
        # создаём новый узел
        nid = len(nodes)
        nodes.append(p)
        buckets[(ix,iy)].append(nid)
        return nid

    edges = []
    for a,b in segments:
        u = get_node_id(a)
        v = get_node_id(b)
        if u != v:
            edges.append((u,v))
        # если u==v: сегмент нулевой длины — можно игнорировать
    return nodes, edges
