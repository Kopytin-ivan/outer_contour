
# units.py
import math

def ensure_meters(segments, scale=1.0):
    """Если данные пришли не в метрах, укажи scale (например, миллиметры -> scale=0.001)."""
    if scale == 1.0: 
        return segments
    def s(p): return (p[0]*scale, p[1]*scale)
    return [(s(a), s(b)) for a,b in segments]

def quantize_mm_round(segments):
    """Округлить координаты до ближайшего миллиметра."""
    def q(p): return (round(p[0], 3), round(p[1], 3))
    return [(q(a), q(b)) for a, b in segments]

def quantize_mm_trunc(segments):
    """Отбросить всё после миллиметра (жёсткое усечение)."""
    def t(x): return math.floor(x * 1000.0) / 1000.0
    def q(p): return (t(p[0]), t(p[1]))
    return [(q(a), q(b)) for a, b in segments]
