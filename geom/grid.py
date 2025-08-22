# grid.py (подбор)
import math
from statistics import median
from collections import defaultdict
import math

class UniformGrid:
    def __init__(self, cell):
        self.cell = float(cell)
        self.cells = defaultdict(list)   # (ix,iy) -> [segment_id]

    def _cell_of(self, x, y):
        return (int(math.floor(x/self.cell)), int(math.floor(y/self.cell)))

    def _cells_for_bbox(self, xmin, ymin, xmax, ymax):
        ix0, iy0 = self._cell_of(xmin, ymin)
        ix1, iy1 = self._cell_of(xmax, ymax)
        for ix in range(ix0, ix1+1):
            for iy in range(iy0, iy1+1):
                yield (ix, iy)

    def insert_segment(self, seg_id, a, b, pad=0.0):
        xmin = min(a[0], b[0]) - pad
        xmax = max(a[0], b[0]) + pad
        ymin = min(a[1], b[1]) - pad
        ymax = max(a[1], b[1]) + pad
        for key in self._cells_for_bbox(xmin, ymin, xmax, ymax):
            self.cells[key].append(seg_id)


    def nearby_segments_by_point(self, x, y, radius):
        """Вернёт set id сегментов из ячеек, покрывающих круг радиуса radius вокруг (x,y)."""
        r = float(radius)
        cx, cy = self._cell_of(x, y)
        kr = int(math.ceil(r / self.cell))  # сколько «колец» по клеткам
        out = set()
        for dx in range(-kr, kr+1):
            for dy in range(-kr, kr+1):
                key = (cx+dx, cy+dy)
                if key in self.cells:
                    out.update(self.cells[key])
        return out
    
    
def build_grid_from_graph(graph, cell, pad=0.0):
    grid = UniformGrid(cell)
    for eid, (u, v) in enumerate(graph.edges):
        # Пропускаем удалённые рёбра (помечены как -1)
        if u == -1 or v == -1:
            continue
        a = graph.nodes[u]
        b = graph.nodes[v]
        grid.insert_segment(eid, a, b, pad=pad)
    return grid




def _nn_distances(endpoints, sample_max=5000):
    pts = endpoints[:sample_max]
    out = []
    for i,(x1,y1) in enumerate(pts):
        best = float("inf")
        for j,(x2,y2) in enumerate(pts):
            if i==j: continue
            d = math.hypot(x1-x2, y1-y2)
            if d < best: best = d
        if best < float("inf"): out.append(best)
    return out

def pick_grid_params(nodes_xy, eps_snap):
    """Подбор GRID_CELL и R_QUERY из данных (устойчиво и быстро)."""
    if not nodes_xy:
        return 1.0, 3.0
    xs = sorted(p[0] for p in nodes_xy)
    ys = sorted(p[1] for p in nodes_xy)
    dxs = [abs(xs[i+1]-xs[i]) for i in range(len(xs)-1)]
    dys = [abs(ys[i+1]-ys[i]) for i in range(len(ys)-1)]
    base = median([d for d in (dxs+dys) if d > 0]) if (dxs or dys) else eps_snap*5
    base = max(base, eps_snap*5)
    cell = min(5.0, max(0.02, 3.0*base))   # 2 см … 5 м
    rqry = 3.0*cell
    return cell, rqry
