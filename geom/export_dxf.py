# geom/export_dxf.py
# Экспорт отрезков в DXF. Если ezdxf установлен — используем его;
# иначе пишем минимальный ASCII-DXF вручную.
from typing import List, Tuple

Point = Tuple[float, float]
Segment = Tuple[Point, Point]

_INSUNITS = {
    "Unitless": 0, "Inches": 1, "Feet": 2, "Miles": 3, "Millimeters": 4,
    "Centimeters": 5, "Meters": 6, "Kilometers": 7
}

def save_dxf_lines(
    segments: List[Segment],
    path: str,
    layer: str = "OUTLINE",
    color: int = 7,            # 7=белый/чёрный
    lineweight: int = 25,      # 0.25 мм → код 25 (DXF group 370)
    insunits: str = "Millimeters"   # твои координаты — в метрах
) -> None:
    try:
        import ezdxf  # type: ignore
        _save_with_ezdxf(segments, path, layer, color, lineweight, insunits)
    except Exception:
        _save_plain_ascii_dxf(segments, path, layer, color, lineweight, insunits)


def _save_with_ezdxf(
    segments: List[Segment], path: str, layer: str, color: int, lw: int, insunits: str
) -> None:
    import ezdxf  # type: ignore

    # Код единиц DXF (см. _INSUNITS)
    u = _INSUNITS.get(insunits, 6)  # по умолчанию Meters

    # Создаём документ и задаём единицы
    doc = ezdxf.new(setup=True)  # AC1027
    doc.header["$INSUNITS"] = u
    # 1 = metric, 0 = imperial
    doc.header["$MEASUREMENT"] = 1 if u in (4, 5, 6, 7) else 0
    # Для совместимости с ezdxf API (если доступно)
    try:
        doc.units = u
    except Exception:
        pass

    # Слой
    if layer not in doc.layers:
        try:
            doc.layers.add(name=layer, color=color, lineweight=lw)
        except Exception:
            # на старых версиях ezdxf lineweight может не поддерживаться при добавлении
            doc.layers.add(name=layer, color=color)

    # Линии
    msp = doc.modelspace()
    attrs = {"layer": layer, "color": color, "lineweight": lw}
    for (p1, p2) in segments:
        msp.add_line(
            (float(p1[0]), float(p1[1]), 0.0),
            (float(p2[0]), float(p2[1]), 0.0),
            dxfattribs=attrs,
        )

    # Сохранить
    doc.saveas(path)


def _save_plain_ascii_dxf(segments, path, layer, color, lw, insunits):
    iu = _INSUNITS.get(insunits, 6)

    lines = []
    push = lines.append

    # HEADER
    push("0"); push("SECTION")
    push("2"); push("HEADER")
    push("9"); push("$INSUNITS");   push("70"); push(str(iu))
    push("9"); push("$MEASUREMENT");push("70"); push("1")    # 1 = metric
    push("0"); push("ENDSEC")

    # TABLES (минимум)
    push("0"); push("SECTION")
    push("2"); push("TABLES")
    push("0"); push("ENDSEC")

    # ENTITIES
    push("0"); push("SECTION")
    push("2"); push("ENTITIES")
    for (p1, p2) in segments:
        push("0"); push("LINE")
        push("8"); push(layer)
        push("62"); push(str(color))
        push("370"); push(str(lw))
        push("10"); push(f"{p1[0]:.12f}")
        push("20"); push(f"{p1[1]:.12f}")
        push("30"); push("0.0")
        push("11"); push(f"{p2[0]:.12f}")
        push("21"); push(f"{p2[1]:.12f}")
        push("31"); push("0.0")
    push("0"); push("ENDSEC")
    push("0"); push("EOF")

    with open(path, "w", encoding="ascii", newline="\n") as f:
        f.write("\n".join(lines))


# --- В КОНЕЦ geom/export_dxf.py ---

from typing import List, Tuple


def save_dxf_polyline(
    points: list[tuple[float, float]],
    path: str,
    layer: str = "OUTER",
    color: int = 7,
    lineweight: int = 25,
    insunits: str = "Meters",   # В МЕТРАХ, как и остальной пайплайн
    closed: bool = True,
) -> None:
    try:
        import ezdxf  # type: ignore

        u = _INSUNITS.get(insunits, 6)  # 6 = Meters
        doc = ezdxf.new(setup=True)
        doc.header["$INSUNITS"] = u
        doc.header["$MEASUREMENT"] = 1 if u in (4,5,6,7) else 0
        try:
            doc.units = u
        except Exception:
            pass

        if layer not in doc.layers:
            try:
                doc.layers.add(name=layer, color=color, lineweight=lineweight)
            except Exception:
                doc.layers.add(name=layer, color=color)

        msp = doc.modelspace()
        pl = msp.add_lwpolyline(points, dxfattribs={"layer": layer, "color": color, "lineweight": lineweight})
        if closed:
            try:
                pl.closed = True
            except Exception:
                # старые версии: установить флаг через DXF атрибуты
                pl.set_dxf_attrib("flags", 1)  # bit 1 = closed

        doc.saveas(path)

    except Exception:
        # ASCII fallback
        iu = _INSUNITS.get(insunits, 6)
        lines = []
        push = lines.append
        # HEADER
        push("0"); push("SECTION"); push("2"); push("HEADER")
        push("9"); push("$INSUNITS");    push("70"); push(str(iu))
        push("9"); push("$MEASUREMENT"); push("70"); push("1")
        push("0"); push("ENDSEC")
        # TABLES (минимум)
        push("0"); push("SECTION"); push("2"); push("TABLES"); push("0"); push("ENDSEC")
        # ENTITIES
        push("0"); push("SECTION"); push("2"); push("ENTITIES")
        push("0"); push("LWPOLYLINE")
        push("8"); push(layer)
        push("62"); push(str(color))
        push("370"); push(str(lineweight))
        push("90"); push(str(len(points)))
        push("70"); push("1" if closed else "0")  # closed flag
        for x, y in points:
            push("10"); push(f"{float(x):.12f}")
            push("20"); push(f"{float(y):.12f}")
        push("0"); push("ENDSEC"); push("0"); push("EOF")
        with open(path, "w", encoding="ascii", newline="\n") as f:
            f.write("\n".join(lines))

