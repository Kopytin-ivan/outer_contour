# main.py — CLI: возвращаем абсолютное смещение в выходные файлы
import json
import os
from pathlib import Path

from geom.io import load_segments
from geom.units import ensure_meters, quantize_mm_round
from geom.snap import snap_points
from geom.graph import Graph
from geom.grid import pick_grid_params, build_grid_from_graph
from geom.config import PARAMS
from geom.prof import Prof

from geom.extend import close_tails_smart, connect_closed_islands_to_host
from geom.export_dxf import save_dxf_lines
from geom.outer import save_outer_via_faces_union

prof = Prof(enabled=True)


def _bbox_offset_meters(segs_m):
    """Возвращает (xmin, ymin) для списка отрезков в МЕТРАХ."""
    if not segs_m:
        return 0.0, 0.0
    xs, ys = [], []
    for a, b in segs_m:
        xs.extend([a[0], b[0]])
        ys.extend([a[1], b[1]])
    return float(min(xs)), float(min(ys))


def _add_offset_to_segments(segs, dx, dy):
    """[[[x1,y1],[x2,y2]], ...] -> сдвиг на (dx,dy)."""
    if (dx == 0.0 and dy == 0.0) or not segs:
        return segs
    out = []
    for a, b in segs:
        out.append([[a[0] + dx, a[1] + dy], [b[0] + dx, b[1] + dy]])
    return out


def stage0_1_init(path_in, unit_scale=0.001):
    """
    0) загрузка → перевод единиц (мм→м при unit_scale=0.001)
    1) округление до мм
    2) снап → граф
    3) сетка
    Возвращает (G, grid, params, src_dx, src_dy), где src_dx/dy — левый‑нижний угол bbox исходника в МЕТРАХ.
    """
    # 0) загрузка и масштаб
    with prof.section("stage0: load+scale"):
        segs_in = load_segments(path_in)             # исходные единицы
        segs_m  = ensure_meters(segs_in, unit_scale) # всегда в метрах

    # смещение считаем ДО любых топологических операций
    src_dx, src_dy = _bbox_offset_meters(segs_m)

    # 0.1) округление до мм (ускоряет снап/грид и схлопывает микрозазоры)
    with prof.section("stage0: quantize_mm"):
        segs_m = quantize_mm_round(segs_m)

    # 1) снап вершин
    with prof.section("stage1: snap_points"):
        nodes, edges = snap_points(segs_m, PARAMS["EPS_SNAP"])
        G = Graph(nodes, edges)

    # 2) параметры сетки
    with prof.section("stage1: pick_grid_params"):
        grid_cell, r_query = pick_grid_params(G.nodes, PARAMS["EPS_SNAP"])
        if PARAMS["GRID_CELL"] is None:
            PARAMS["GRID_CELL"] = grid_cell
        if PARAMS["R_QUERY"] is None:
            PARAMS["R_QUERY"] = r_query

    # 3) построение сетки
    with prof.section("stage1: build_grid"):
        grid = build_grid_from_graph(G, PARAMS["GRID_CELL"], pad=PARAMS["EPS_SNAP"])

    params = {
        "EPS_SNAP":   PARAMS["EPS_SNAP"],
        "GRID_CELL":  PARAMS["GRID_CELL"],
        "R_QUERY":    PARAMS["R_QUERY"],
        "MAX_EXTEND": PARAMS["MAX_EXTEND"],
        "ANGLE_TOL":  PARAMS["ANGLE_TOL"],
        "LEN_TOL":    PARAMS["LEN_TOL"],
    }
    return G, grid, params, src_dx, src_dy


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)

    # путь к входу
    in_path = r"data\\Test_Sminex_Avtozavodskaya_2.json"
    # 0–1) загрузка → снап → сетка (мм → м: unit_scale=0.001)
    G, grid, params, src_dx, src_dy = stage0_1_init(in_path, unit_scale=0.001)

    # 2) умная доводка хвостов
    try:
        ops1 = close_tails_smart(G, grid, PARAMS, templates_db=None, iter_max=5)
    except Exception:
        ops1 = []
    if ops1:
        grid = build_grid_from_graph(G, PARAMS["GRID_CELL"], pad=PARAMS["EPS_SNAP"])

    # 3) мостики для замкнутых островов → host
    try:
        ops2 = connect_closed_islands_to_host(G, grid, PARAMS, rebuild_grid_each=True)
    except Exception:
        ops2 = []
    if ops2:
        grid = build_grid_from_graph(G, PARAMS["GRID_CELL"], pad=PARAMS["EPS_SNAP"])

    # 4) внешний контур (локально, в метрах)
    out_json = "output/outer.json"
    out_dxf  = "output/outer.dxf"
    save_outer_via_faces_union(
        G,
        out_json,
        out_dxf,
        eps_snap_m=0.002  # 2 мм; при необходимости можно повысить до 0.003–0.005
    )

    # 5) ДОБАВЛЯЕМ ОБРАТНО ИСХОДНОЕ СМЕЩЕНИЕ (в метрах) И ПЕРЕСОБИРАЕМ DXF
    try:
        with open(out_json, "r", encoding="utf-8") as f:
            data = json.load(f)
        segs_local = data.get("segments", [])

        # вернуть абсолют
        segs_abs = _add_offset_to_segments(segs_local, src_dx, src_dy)
        data["segments"] = segs_abs

        # перезаписать JSON
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        # перезаписать DXF этими же координатами (в метрах)
        dxf_lines = [((float(a[0]), float(a[1])), (float(b[0]), float(b[1])))
                     for a, b in segs_abs]
        save_dxf_lines(
            dxf_lines,
            out_dxf,
            layer="OUTER",
            color=1,
            lineweight=25,
            insunits="Meters"
        )
    except Exception as e:
        print("WARN: failed to reapply source offset:", e)

    print("saved:", out_json)
    print("saved:", out_dxf)
