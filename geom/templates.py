# geom/templates.py
import json, math
from typing import List, Dict, Any

def _dist(a, b):
    return math.hypot(b[0]-a[0], b[1]-a[1])

def _unit(dx, dy):
    L = math.hypot(dx, dy)
    if L == 0.0: return (0.0, 0.0)
    return (dx/L, dy/L)

def _largest_remainder_percent(lengths):
    """
    Преобразует длины рёбер в целые проценты, сумма = 100.
    Метод наибольших остатков + защита от нулей для ненулевых рёбер.
    """
    P = sum(lengths)
    if P <= 0:
        return [0] * len(lengths)

    raw = [100.0 * L / P for L in lengths]      # плавающие проценты
    floor = [int(math.floor(x)) for x in raw]   # «пол»
    rema  = [(raw[i] - floor[i], i) for i in range(len(raw))]
    need = 100 - sum(floor)

    # раздать по +1% тем, у кого остаток больше
    rema.sort(reverse=True)
    perc = floor[:]
    for _, i in rema[:max(0, need)]:
        perc[i] += 1

    # если у ненулевого ребра получилось 0% — отнимем 1% у крупнейшего и отдадим ему
    for i, L in enumerate(lengths):
        if L > 0 and perc[i] == 0:
            j = max(range(len(perc)), key=lambda k: perc[k])
            if perc[j] > 1:
                perc[j] -= 1
                perc[i] += 1

    return perc

def _angles_internal(pts):
    """Внутренние углы 0..180 (целые), по направлению рёбер."""
    k = len(pts)
    # предвычислим единичные направления рёбер
    dirs = []
    for i in range(k):
        a = pts[i]
        b = pts[(i+1) % k]
        dirs.append(_unit(b[0]-a[0], b[1]-a[1]))
    # угол в вершине i — между -dirs[i-1] и dirs[i]
    out = []
    for i in range(k):
        ux, uy = dirs[i-1]
        vx, vy = dirs[i]
        mu, mv = -ux, -uy
        dot = mu*vx + mv*vy
        dot = max(-1.0, min(1.0, dot))
        ang = math.degrees(math.acos(dot))
        if ang > 180.0: ang = 360.0 - ang
        a = int(round(ang))
        if a < 0: a = 0
        if a > 180: a = 180
        out.append(a)
    return out

def _booth_min_rotation(seq):
    """
    Минимальная циклическая ротация за O(n) (алгоритм Бутта).
    Работает для сравнимых элементов (кортежи (angle, percent)).
    Возвращает индекс старта минимальной ротации.
    """
    s = seq + seq
    n = len(seq)
    i, j, k = 0, 1, 0
    while i < n and j < n and k < n:
        a = s[i+k]
        b = s[j+k]
        if a == b:
            k += 1
            continue
        if a > b:
            i = i + k + 1
            if i <= j:
                i = j + 1
        else:
            j = j + k + 1
            if j <= i:
                j = i + 1
        k = 0
    start = min(i, j)
    return start

def _canonical_pairs(pairs):
    """Канонизация: минимальная ротация (Booth) для прямой и обратной; берём лучшую."""
    if not pairs:
        return []
    # прямая
    s0 = _booth_min_rotation(pairs)
    cand_fwd = pairs[s0:] + pairs[:s0]
    # обратная
    rev = list(reversed(pairs))
    s1 = _booth_min_rotation(rev)
    cand_rev = rev[s1:] + rev[:s1]
    return cand_rev if cand_rev < cand_fwd else cand_fwd

def make_template(G, cycle_nodes: List[int]) -> Dict[str, Any] | None:
    """
    Формирует шаблон циклической фигуры:
      - углы (целые, внутренние);
      - длины рёбер в процентах (целые, сумма=100, метод наибольших остатков);
      - канонизация пар (угол, %) по минимальной ротации и реверсу.
    """
    k = len(cycle_nodes)
    if k < 3:
        return None

    pts = [G.nodes[nid] for nid in cycle_nodes]

    lengths = [_dist(pts[i], pts[(i+1) % k]) for i in range(k)]
    if not any(L > 0 for L in lengths):
        return None

    perc = _largest_remainder_percent(lengths)
    ang  = _angles_internal(pts)

    pairs = list(zip(ang, perc))
    canon = _canonical_pairs(pairs)

    A  = ",".join(str(a) for a, _ in canon)
    Ls = ",".join(str(p) for _, p in canon)
    K  = f"{k}|A={A}|L={Ls}"
    text = f"Форма #{k}: углы [{A.replace(',', '°, ')}°], длины [{Ls.replace(',', '%, ')}%]"

    return {
        "n": k,
        "angles": [a for a, _ in canon],
        "percents": [p for _, p in canon],
        "pairs": canon,
        "K": K,
        "text": text
    }

class TemplatesDB:
    def __init__(self):
        self._byK = {}

    def add(self, tpl: Dict[str, Any] | None):
        if not tpl:
            return
        K = tpl["K"]
        rec = self._byK.get(K)
        if rec is None:
            tpl = dict(tpl)
            tpl["count"] = 1
            self._byK[K] = tpl
        else:
            rec["count"] += 1

    def to_list(self):
        return list(self._byK.values())

    def size(self):
        return len(self._byK)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_list(), f, ensure_ascii=False, indent=2)

    def load(self, path):
        """Подгружает ранее сохранённые шаблоны (если файл есть). Возвращает их число."""
        import os
        if not os.path.exists(path):
            return 0
        with open(path, "r", encoding="utf-8") as f:
            arr = json.load(f)
        for rec in arr:
            self._byK[rec["K"]] = dict(rec)
        return len(arr)

    @staticmethod
    def _seq_close_rot(A1, L1, A2, L2, ang_tol: int, len_tol: int) -> bool:
        """
        Проверка 'похожести' с допусками, с учётом циклического сдвига и реверса.
        Все списки должны быть длины n.
        """
        n = len(A1)
        if n == 0 or len(L1) != n or len(A2) != n or len(L2) != n:
            return False

        # прямое направление
        for s in range(n):
            ok = True
            for i in range(n):
                if abs(A1[i] - A2[(i + s) % n]) > ang_tol or abs(L1[i] - L2[(i + s) % n]) > len_tol:
                    ok = False
                    break
            if ok:
                return True

        # обратное направление
        Ar = list(reversed(A2))
        Lr = list(reversed(L2))
        for s in range(n):
            ok = True
            for i in range(n):
                if abs(A1[i] - Ar[(i + s) % n]) > ang_tol or abs(L1[i] - Lr[(i + s) % n]) > len_tol:
                    ok = False
                    break
            if ok:
                return True

        return False

    def has_similar(self, tpl: Dict[str, Any], ang_tol: int = 2, len_tol: int = 5) -> bool:
        """Есть ли в базе 'похожий' шаблон (по допускам)."""
        A1 = tpl.get("angles", [])
        L1 = tpl.get("percents", [])
        n = len(A1)
        if n < 3 or len(L1) != n:
            return False

        for rec in self._byK.values():
            if rec.get("n") != n:
                continue
            A2 = rec.get("angles", [])
            L2 = rec.get("percents", [])
            if len(A2) != n or len(L2) != n:
                continue
            if TemplatesDB._seq_close_rot(A1, L1, A2, L2, ang_tol, len_tol):
                return True
        return False
