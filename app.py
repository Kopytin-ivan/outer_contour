import os, io, json, time, math
from pathlib import Path

import streamlit as st

# === импорты из твоего проекта (важно запускать из корня репо) ===
from geom.units import ensure_meters, quantize_mm_round
from geom.snap import snap_points
from geom.graph import Graph
from geom.grid import pick_grid_params, build_grid_from_graph
from geom.config import PARAMS
from geom.extend import close_tails_smart, connect_closed_islands_to_host
from geom.export_dxf import save_dxf_lines
from geom.outer import save_outer_via_faces_union  # можно заменить на save_outer_smart
from geom.io import load_segments, load_segments_dxf


# ---------- утилиты ----------
def _run_pipeline(input_path: str,
                  unit_scale: float,
                  do_extend: bool,
                  do_bridge: bool,
                  use_round_mm: bool = True):
    """
    Возвращает (G, out_all_json_path, out_all_dxf_path).
    """
    # 0) загрузка + масштаб
    segs_in = load_segments(input_path)
    segs = ensure_meters(segs_in, scale=unit_scale)

    # 0.1) округление до мм (ускоряет снап/грид)
    if use_round_mm:
        segs = quantize_mm_round(segs)

    # 1) снап -> граф
    nodes, edges = snap_points(segs, PARAMS["EPS_SNAP"])
    G = Graph(nodes, edges)

    # 2) сетка
    grid_cell, r_query = pick_grid_params(G.nodes, PARAMS["EPS_SNAP"])
    if PARAMS["GRID_CELL"] is None:
        PARAMS["GRID_CELL"] = grid_cell
    if PARAMS["R_QUERY"] is None:
        PARAMS["R_QUERY"] = r_query
    grid = build_grid_from_graph(G, PARAMS["GRID_CELL"], pad=PARAMS["EPS_SNAP"])

    # 3) доводка/мостики (по желанию)
    if do_extend:
        close_tails_smart(G, grid, PARAMS, templates_db=None, iter_max=5)
        grid = build_grid_from_graph(G, PARAMS["GRID_CELL"], pad=PARAMS["EPS_SNAP"])

    if do_bridge:
        connect_closed_islands_to_host(G, grid, PARAMS, rebuild_grid_each=True)
        grid = build_grid_from_graph(G, PARAMS["GRID_CELL"], pad=PARAMS["EPS_SNAP"])

    # 4) экспорт всех сегментов (для отладки/проверки)
    outdir = Path("output"); outdir.mkdir(exist_ok=True)
    stem = Path(input_path).stem
    all_json = outdir / f"{stem}_all.json"
    all_dxf  = outdir / f"{stem}_all.dxf"

    out_segs = []
    for (u,v) in G.edges:
        if u == -1 or v == -1: 
            continue
        x1,y1 = G.nodes[u]; x2,y2 = G.nodes[v]
        out_segs.append([[round(x1,5), round(y1,5)], [round(x2,5), round(y2,5)]])

    with open(all_json, "w", encoding="utf-8") as f:
        json.dump({"segments": out_segs}, f, ensure_ascii=False, indent=2)

    dxf_segs = [((float(a[0]), float(a[1])), (float(b[0]), float(b[1]))) for a,b in out_segs]
    save_dxf_lines(dxf_segs, str(all_dxf), layer="ALL", color=7, lineweight=25, insunits="Meters")

    return G, str(all_json), str(all_dxf), stem

# ---------- UI ----------
st.set_page_config(page_title="Outer Builder", layout="wide")
st.title("🧭 Внешний контур")

with st.sidebar:
    st.header("⚙️ Параметры")

    unit = st.selectbox("Единицы входных координат", ["мм", "метры"], index=0)
    unit_scale = 0.001 if unit == "мм" else 1.0

    # базовые
    eps_snap_mm = st.number_input("EPS_SNAP (мм)", value=2.0, min_value=0.1, step=0.1)
    max_extend  = st.number_input("MAX_EXTEND (м)", value=float(PARAMS.get("MAX_EXTEND", 0.1)), min_value=0.0, step=0.05)
    angle_tol   = st.number_input("ANGLE_TOL (°)", value=float(PARAMS.get("ANGLE_TOL", 2)), min_value=0.0, step=0.5)
    len_tol     = st.number_input("LEN_TOL", value=float(PARAMS.get("LEN_TOL", 5)), min_value=0.0, step=0.5)

    st.markdown("---")
    st.subheader("DXF: аппроксимация дуг")
    arc_deg   = st.number_input("Макс. угол сегмента (°)", value=10.0, min_value=1.0, step=1.0)
    arc_ch_mm = st.number_input("Макс. хорда (мм)", value=10.0, min_value=0.5, step=0.5)

    flatten_blocks = st.checkbox("Разворачивать блоки (INSERT)", value=True)
    spline_seg     = st.number_input("Сплайн: сегментов на кривую", value=100, min_value=20, step=10)
    virt_limit     = st.number_input("Лимит развёрнутых примитивов", value=300000, min_value=50000, step=50000)



    st.markdown("---")
    st.subheader("Доводка")
    do_extend = st.checkbox("Доводить хвосты (smart extend)", value=True)
    do_bridge = st.checkbox("Мостики замкнутых островов", value=True)

    st.markdown("---")
    st.subheader("Анти-микромостики (опционально)")
    min_extend = st.number_input("MIN_EXTEND (м)", value=float(PARAMS.get("MIN_EXTEND", 0.0)), min_value=0.0, step=0.02)
    near_deg2  = st.number_input("NEAR_DEG2_THRESHOLD (м)", value=float(PARAMS.get("NEAR_DEG2_THRESHOLD", 0.0)), min_value=0.0, step=0.02)
    skip_to_deg= st.checkbox("Не тянуть к узлам (deg>1) вблизи", value=bool(PARAMS.get("SKIP_EXTEND_TO_DEG_GT1", False)))

    st.markdown("---")
    st.subheader("Метод внешки")
    method = st.selectbox("Способ", ["faces_union (надёжный)", "smart (быстрый+fallback)"], index=0)
    eps_outer_mm = st.number_input("eps для внешки (мм)", value=2.0, min_value=0.5, step=0.5)

    st.markdown("---")
    st.caption("Файлы сохраняются в папку ./output")

# применяем параметры к глобальному PARAMS
PARAMS["EPS_SNAP"]   = eps_snap_mm / 1000.0
PARAMS["MAX_EXTEND"] = max_extend
PARAMS["ANGLE_TOL"]  = angle_tol
PARAMS["LEN_TOL"]    = len_tol
PARAMS["MIN_EXTEND"] = min_extend
PARAMS["NEAR_DEG2_THRESHOLD"] = near_deg2
PARAMS["SKIP_EXTEND_TO_DEG_GT1"] = skip_to_deg
# обнулим автоподбор (будет выбран дальше)
PARAMS["GRID_CELL"] = None
PARAMS["R_QUERY"]   = None

uploaded = st.file_uploader("Загрузите JSON или DXF", type=["json", "dxf"])
run = st.button("▶️ Запустить")

if run:
    if not uploaded:
        st.error("Загрузите JSON или DXF файл.")
        st.stop()

    up_dir = Path("uploads"); up_dir.mkdir(exist_ok=True)
    tmp_path = up_dir / f"in_{int(time.time()*1000)}_{uploaded.name}"
    tmp_path.write_bytes(uploaded.getvalue())

    # Подготовим вход: JSON как есть; DXF → конвертируем в временный JSON (метры)
    input_path = str(tmp_path)
    unit_scale_eff = unit_scale

    if tmp_path.suffix.lower() == ".dxf":
        try:
            segs_m = load_segments_dxf(
                str(tmp_path),
                arc_max_deg=float(arc_deg),
                arc_chord=float(arc_ch_mm) / 1000.0,
                include_blocks=bool(flatten_blocks),
                spline_segments=int(spline_seg),
                max_virt=int(virt_limit),
            )
        except Exception as e:
            st.exception(e)
            st.stop()

        if not segs_m:
            st.error("DXF прочитан, но сегменты не найдены (возможно, вся геометрия в блоках/слоях?). Попробуйте включить «Разворачивать блоки (INSERT)».")
            st.stop()

        conv_json = up_dir / f"{tmp_path.stem}_converted.json"
        with open(conv_json, "w", encoding="utf-8") as f:
            json.dump({"segments": segs_m}, f, ensure_ascii=False, indent=2)

        input_path = str(conv_json)
        unit_scale_eff = 1.0  # уже метры

    with st.spinner("Обработка..."):
        t0 = time.time()
        try:
            G, all_json, all_dxf, stem = _run_pipeline(input_path, unit_scale_eff, do_extend, do_bridge)
            outdir = Path("output")
            outer_json = outdir / f"{stem}_outer.json"
            outer_dxf  = outdir / f"{stem}_outer.dxf"

            eps_outer = eps_outer_mm / 1000.0
            if method.startswith("smart"):
                meta = save_outer_via_faces_union(G, str(outer_json), str(outer_dxf), eps_snap_m=eps_outer)
            else:
                meta = save_outer_via_faces_union(G, str(outer_json), str(outer_dxf), eps_snap_m=eps_outer)

        except Exception as e:
            st.exception(e)
            st.stop()

        dt = time.time() - t0

    st.success(f"Готово за {dt:.2f} c")
    st.json({"outer_meta": meta})

    # кнопки скачивания
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "⬇️ Скачать внешний контур (JSON)",
            data=open(outer_json, "rb").read(),
            file_name=outer_json.name,
            mime="application/json",
        )
    with c2:
        st.download_button(
            "⬇️ Скачать внешний контур (DXF)",
            data=open(outer_dxf, "rb").read(),
            file_name=outer_dxf.name,
            mime="application/dxf",
        )

    st.markdown("—" * 30)
    with st.expander("Все сегменты после доводки (для проверки)"):
        st.download_button(
            "⬇️ Скачать все сегменты (JSON)",
            data=open(all_json, "rb").read(),
            file_name=Path(all_json).name,
            mime="application/json",
        )
        st.download_button(
            "⬇️ Скачать все сегменты (DXF)",
            data=open(all_dxf, "rb").read(),
            file_name=Path(all_dxf).name,
            mime="application/dxf",
        )
