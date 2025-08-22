# main.py
import json
import os

from geom.io import load_segments
from geom.units import ensure_meters, quantize_mm_round
from geom.snap import snap_points
from geom.graph import Graph
from geom.grid import pick_grid_params, build_grid_from_graph
from geom.config import PARAMS
from geom.prof import Prof

from geom.templates import TemplatesDB, make_template     # для close_tails_smart (можно пустую память)
from geom.cycles import find_planar_faces  # только чтобы засеять память из уже замкнутых
from geom.extend import close_tails_smart, connect_closed_islands_to_host
from geom.export_dxf import save_dxf_lines

from geom.outer import save_outer_via_faces_union

from pathlib import Path
from geom.outer import save_outer_from_graph
from geom.outer import save_outer_best
from geom.outer import save_outer_by_rightmost_on_H
from geom.outer import save_outer_clockwise

prof = Prof(enabled=True)

def stage0_1_init(path_in, unit_scale=0.001):
    # 0) загрузка и масштаб (0.001 = мм → м)
    with prof.section("stage0: load+scale"):
        segs_in = load_segments(path_in)
        segs = ensure_meters(segs_in, scale=unit_scale)

    # 0.1) округление до мм (ускоряет снап/грид)
    with prof.section("stage0: quantize_mm"):
        segs = quantize_mm_round(segs)

    # 1) снап вершин
    with prof.section("stage1: snap_points"):
        nodes, edges = snap_points(segs, PARAMS["EPS_SNAP"])
        G = Graph(nodes, edges)

    # 2) параметры сетки
    with prof.section("stage1: pick_grid_params"):
        grid_cell, r_query = pick_grid_params(G.nodes, PARAMS["EPS_SNAP"])
        if PARAMS["GRID_CELL"] is None:
            PARAMS["GRID_CELL"] = grid_cell
        if PARAMS["R_QUERY"] is None:
            PARAMS["R_QUERY"] = r_query
        # при желании зафиксируй вручную:
        # PARAMS["GRID_CELL"] = 20.0
        # PARAMS["R_QUERY"]   = 60.0

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
    return G, grid, params


if __name__ == "__main__":
    os.makedirs("output", exist_ok=True)

    # 0–1) загрузка → снап → сетка (мм → м: unit_scale=0.001)
    G, grid, params = stage0_1_init(r"data\\Test_Sminex_Avtozavodskaya_2.json", unit_scale=0.001)

    # 2) умная доводка хвостов (без «памяти» — допускается None)
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

    # 4) внешний контур (планаризация + объединение лиц) → один JSON и один DXF
    out_json = "output/outer.json"
    out_dxf  = "output/outer.dxf"

    from geom.outer import save_outer_via_faces_union
    save_outer_via_faces_union(
        G,
        out_json,
        out_dxf,
        eps_snap_m=0.002  # 2 мм; при необходимости можно повысить до 0.003–0.005
    )

    print("saved:", out_json)
    print("saved:", out_dxf)
