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
    payload = {"segments": [[list(a), list(b)] for (a, b) in segments]}
    if params:
        payload["params"] = params
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)