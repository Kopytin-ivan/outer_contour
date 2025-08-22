# io.py
import json

import json

def _as_point(item):
    # ожидаем [x, y]
    if (isinstance(item, (list, tuple)) and len(item) == 2 and
        all(isinstance(v, (int, float)) for v in item)):
        return (float(item[0]), float(item[1]))
    return None

def _is_point_list(lst):
    # список из точек [x, y]
    return (isinstance(lst, list) and len(lst) >= 2 and
            _as_point(lst[0]) is not None and _as_point(lst[1]) is not None)

def _is_segment_list(lst):
    # список из сегментов [[p1],[p2]]
    if not isinstance(lst, list) or not lst:
        return False
    a = lst[0]
    return (isinstance(a, (list, tuple)) and len(a) == 2 and
            _as_point(a[0]) is not None and _as_point(a[1]) is not None)

def load_segments(path):
    """
    Возвращает список сегментов вида [((x1,y1),(x2,y2)), ...]
    Поддерживает:
      - {"segments": [ [[x,y],[x,y]], ... ]}   (вариант А)
      - [ [x,y], [x,y], ... ]                  (вариант Б: полилиния)
      - [ [[x,y],[x,y]], ... ]                 (прямой список сегментов)
    """
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Вариант А: словарь с ключом "segments"
    if isinstance(data, dict) and "segments" in data:
        segs = data["segments"]
        if not _is_segment_list(segs):
            raise ValueError("Ожидался массив сегментов в data['segments'].")
        return [ (tuple(_as_point(s[0])), tuple(_as_point(s[1]))) for s in segs ]

    # Вариант Б1: чистая полилиния — массив точек
    if _is_point_list(data):
        pts = [ _as_point(p) for p in data ]
        # превращаем в сегменты, соединяя соседей
        segs = []
        for i in range(len(pts) - 1):
            segs.append((pts[i], pts[i+1]))
        return segs

    # Вариант Б2: прямой список сегментов без обёртки
    if _is_segment_list(data):
        return [ (tuple(_as_point(s[0])), tuple(_as_point(s[1]))) for s in data ]

    raise ValueError("Не распознан формат входного JSON. "
                     "Ожидался {'segments': [...]}, или [[x,y],...], или [[[x,y],[x,y]], ...].")


def save_segments(path, segments, params=None):
    payload = {"segments": [ [list(a), list(b)] for (a,b) in segments ] }
    if params:
        payload["params"] = params
    json.dump(payload, open(path, "w", encoding="utf-8"), ensure_ascii=False, indent=2)


# --- DXF -> segments (meters)
def load_segments_dxf(path: str,
                      arc_max_deg: float = 10.0,
                      arc_chord: float = 0.01,
                      include_blocks: bool = True,
                      spline_segments: int = 100,
                      max_virt: int = 300_000):
    """
    DXF -> сегменты в МЕТРАХ.
    - ЛИНИИ/ПОЛИЛИНИИ -> отрезки 1:1
    - ДУГИ/ОКРУЖНОСТИ -> ломаная (кап на число сегментов)
    - СПЛАЙНЫ -> approximate(segments=spline_segments)
    - INSERT: либо разворачиваем (virtual_entities), либо пропускаем
      (include_blocks=False).
    - max_virt: верхний лимит на развёрнутые примитивы (safety).
    """
    import ezdxf, math
    doc = ezdxf.readfile(path)
    msp = doc.modelspace()

    units_code = int(doc.header.get("$INSUNITS", 0))
    u2m = {0: 1.0, 1: 0.0254, 2: 0.3048, 3: 0.9144, 4: 0.001, 5: 0.01, 6: 0.1, 7: 1.0}
    scale = u2m.get(units_code, 1.0)

    segs = []
    count = 0

    def add(p, q):
        nonlocal count
        segs.append([[p[0] * scale, p[1] * scale], [q[0] * scale, q[1] * scale]])
        count += 1

    def poly_to_segs(pts, closed=False):
        for i in range(len(pts) - 1):
            add(pts[i], pts[i + 1])
        if closed and len(pts) > 1:
            add(pts[-1], pts[0])

    def arc_to_segs(center, r, a1_deg, a2_deg):
        a1 = math.radians(a1_deg); a2 = math.radians(a2_deg)
        da = (a2 - a1) % (2 * math.pi)
        step1 = math.radians(max(1e-3, arc_max_deg))
        step2 = 2 * math.acos(max(0.0, 1 - (arc_chord / max(r, 1e-9))))
        step = max(1e-3, min(step1, step2))
        n = max(2, int(math.ceil(da / step)))
        n = min(n, 720)  # кап: не больше 720 сегментов на дугу
        prev = (center[0] + r * math.cos(a1), center[1] + r * math.sin(a1))
        for k in range(1, n + 1):
            t = a1 + da * k / n
            cur = (center[0] + r * math.cos(t), center[1] + r * math.sin(t))
            add(prev, cur)
            prev = cur

    def handle_entity(v):
        t = v.dxftype()
        if t == "LINE":
            a = v.dxf.start; b = v.dxf.end
            add((a.x, a.y), (b.x, b.y))
        elif t in ("LWPOLYLINE", "POLYLINE"):
            pts = list(v.get_points("xy")) if t == "LWPOLYLINE" else [(p.x, p.y) for p in v.points()]
            poly_to_segs(pts, bool(getattr(v, "closed", False)))
        elif t == "ARC":
            c = v.dxf.center
            arc_to_segs((c.x, c.y), v.dxf.radius, v.dxf.start_angle, v.dxf.end_angle)
        elif t == "CIRCLE":
            c = v.dxf.center
            arc_to_segs((c.x, c.y), v.dxf.radius, 0.0, 360.0)
        elif t == "SPLINE":
            try:
                pts = v.approximate(segments=max(20, int(spline_segments)))
                pts2d = []
                for p in pts:
                    # ezdxf возвращает Vec3 или кортеж
                    if hasattr(p, "x"):                     # Vec3
                        pts2d.append((float(p.x), float(p.y)))
                    else:                                   # tuple/list
                        if len(p) >= 2:
                            pts2d.append((float(p[0]), float(p[1])))
                if len(pts2d) >= 2:
                    poly_to_segs(pts2d, False)
            except Exception:
                pass


    # Быстрый путь: без разворота блоков
    if not include_blocks:
        for v in msp.query("LINE LWPOLYLINE POLYLINE ARC CIRCLE SPLINE"):
            handle_entity(v)
        return segs

    # Полный путь: разворачивание INSERT (может быть тяжёлым)
    for e in msp:
        ents = e.virtual_entities() if hasattr(e, "virtual_entities") else [e]
        for v in ents:
            handle_entity(v)
            if count >= max_virt:
                return segs  # ранний выход, чтобы не зависнуть

    return segs
