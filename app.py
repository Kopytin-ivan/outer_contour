# app.py — JSON-only, превью исходника и результата, возврат исходного смещения

import json, time
from pathlib import Path
from io import BytesIO

import streamlit as st
import matplotlib.pyplot as plt

# === импорты из твоего проекта (запускай из корня репо) ===
from geom.units import ensure_meters, quantize_mm_round
from geom.snap import snap_points
from geom.graph import Graph
from geom.grid import pick_grid_params, build_grid_from_graph
from geom.config import PARAMS
from geom.extend import close_tails_smart, connect_closed_islands_to_host
from geom.export_dxf import save_dxf_lines
from geom.outer import save_outer_via_faces_union
from geom.io import load_segments
from pathlib import Path
import shutil

# ---------------------- хелперы парсинга/смещения/отрисовки ----------------------

def _segments_from_json_bytes(file_bytes):
    """
    Возвращает (segments, polyline):
      segments: список [[x1,y1],[x2,y2]]
      polyline: список [[x,y], ...] если исходник — полилиния (иначе None)

    Поддерживаемые форматы:
      1) {"segments": [[[x1,y1],[x2,y2]], ...]}
      2) [[[x1,y1],[x2,y2]], ...]      # список сегментов без ключа
      3) [[x,y], [x,y], ...]           # полилиния
    """
    try:
        data = json.loads(file_bytes.decode("utf-8"))
    except Exception:
        return [], None


    # --- 1) dict с ключом "segments"
    if isinstance(data, dict) and isinstance(data.get("segments"), list):
        segs = []
        for it in data["segments"]:
            if (isinstance(it, list) and len(it) == 2
                and isinstance(it[0], list) and isinstance(it[1], list)
                and len(it[0]) >= 2 and len(it[1]) >= 2):
                a, b = it
                segs.append([[float(a[0]), float(a[1])],
                             [float(b[0]), float(b[1])]])
        return segs, None

    # --- 2) list: определяем, это polyline или список сегментов
    if isinstance(data, list) and len(data) >= 1 and isinstance(data[0], list):
        first = data[0]

        # 2a) polyline [[x,y], ...]
        if (len(first) >= 2 and
            isinstance(first[0], (int, float)) and isinstance(first[1], (int, float))):
            poly = []
            for p in data:
                if isinstance(p, list) and len(p) >= 2:
                    poly.append([float(p[0]), float(p[1])])
            segs = [[poly[i], poly[i+1]] for i in range(len(poly)-1)]
            if len(poly) >= 3 and poly[0] != poly[-1]:
                segs.append([poly[-1], poly[0]])  # визуально замкнём
            return segs, poly

        # 2b) список сегментов [[[x1,y1],[x2,y2]], ...]
        if (len(first) == 2 and isinstance(first[0], list) and isinstance(first[1], list)):
            segs = []
            for a, b in data:
                if (isinstance(a, list) and isinstance(b, list)
                    and len(a) >= 2 and len(b) >= 2):
                    segs.append([[float(a[0]), float(a[1])],
                                 [float(b[0]), float(b[1])]])
            return segs, None

    # не распознали формат
    return [], None

def clean_output(dirpath: str, keep_last_n: int = 0):
    p = Path(dirpath)
    if not p.exists():
        return
    if keep_last_n <= 0:
        # удалить всё
        for child in p.iterdir():
            if child.is_file():
                child.unlink(missing_ok=True)
            else:
                shutil.rmtree(child, ignore_errors=True)


def _compute_source_offset_from_uploaded(uploaded_bytes, unit_scale: float):
    """
    Возвращает (dx, dy) — смещение исходника, вычисленное как левый‑нижний угол (xmin,ymin)
    входных данных (после пересчёта единиц).
    Поддерживает и формат segments, и полилинию.
    Если не удаётся распарсить — вернёт (0.0, 0.0).
    """
    segs, poly = _segments_from_json_bytes(uploaded_bytes)
    pts = []
    if segs:
        for a, b in segs:
            pts.append(a); pts.append(b)
    elif poly:
        pts.extend(poly)
    else:
        return 0.0, 0.0

    # применим масштаб в метры, т.к. пайплайн работает в метрах
    xs = [p[0] * unit_scale for p in pts]
    ys = [p[1] * unit_scale for p in pts]
    if not xs or not ys:
        return 0.0, 0.0
    return float(min(xs)), float(min(ys))


def _shift_segments(segs, dx, dy):
    """[[[x1,y1],[x2,y2]], ...] -> сдвиг на (dx,dy)"""
    return [ [[a[0]+dx, a[1]+dy], [b[0]+dx, b[1]+dy]] for a,b in segs ]

def _scale_segments(segs, scale: float, ndigits: int):
    """
    Масштабирует список отрезков [[[x1,y1],[x2,y2]], ...] и округляет координаты.
    """
    if scale == 1.0:
        return segs
    out = []
    for a, b in segs:
        out.append([
            [round(a[0]*scale, ndigits), round(a[1]*scale, ndigits)],
            [round(b[0]*scale, ndigits), round(b[1]*scale, ndigits)],
        ])
    return out

def _scale_polyline(points, scale: float, ndigits: int):
    """
    Масштабирует полилинию [[x,y], ...] и округляет координаты.
    """
    if scale == 1.0:
        return points
    return [[round(x*scale, ndigits), round(y*scale, ndigits)] for x, y in points]




def _segments_from_graph(G):
    """Собрать отрезки [[x1,y1],[x2,y2]] из графа (без служебных рёбер)."""
    segs = []
    for (u, v) in G.edges:
        if u == -1 or v == -1:
            continue
        x1, y1 = G.nodes[u]; x2, y2 = G.nodes[v]
        segs.append([[x1, y1], [x2, y2]])
    return segs


def _plot_preview(segments, outer_poly=None, title=None, lw_all=0.8, lw_outer=2.0):
    """
    Рисует картинку и возвращает байты PNG (для показа в st.image и скачивания).
      segments  — список [[x1,y1],[x2,y2]] (в метрах)
      outer_poly — список точек внешнего контура [[x,y], ...] (опционально)
    """
    fig, ax = plt.subplots(figsize=(7, 7))
    # все сегменты
    for (a, b) in segments:
        ax.plot([a[0], b[0]], [a[1], b[1]], linewidth=lw_all)

    # внешняя полилиния (если есть)
    if outer_poly and len(outer_poly) >= 2:
        xs = [p[0] for p in outer_poly] + [outer_poly[0][0]]
        ys = [p[1] for p in outer_poly] + [outer_poly[0][1]]
        ax.plot(xs, ys, linewidth=lw_outer)

    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.2)
    if title:
        ax.set_title(title)

    ax.margins(0.05)

    buf = BytesIO()
    plt.tight_layout()
    fig.savefig(buf, format="png", dpi=150)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# ---------------------- основной пайплайн ----------------------

def _run_pipeline(input_path: str,
                  unit_scale: float,
                  do_extend: bool,
                  do_bridge: bool,
                  use_round_mm: bool = True,
                  retain_absolute: bool = True,
                  src_dx: float = 0.0,
                  src_dy: float = 0.0,
                  out_scale: float = 1.0,
                  out_ndigits: int = 5,
                  out_dxf_units: str = "Meters"):
    """
    Возвращает (G, out_all_json_path, out_all_dxf_path, stem).
    Если retain_absolute=True, экспортируемые сегменты будут сдвинуты на (src_dx, src_dy) в метрах.
    """
    # 0) загрузка + масштаб
    segs_in = load_segments(input_path)
    segs = ensure_meters(segs_in, scale=unit_scale)

    # 0.1) округление до мм (ускоряет снап/грид и схлопывает микрозазоры)
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

    # 4) экспорт всех сегментов (для проверки)
    outdir = Path("output"); outdir.mkdir(exist_ok=True)
    stem = Path(input_path).stem
    all_json = outdir / f"{stem}_all.json"
    all_dxf  = outdir / f"{stem}_all.dxf"

    out_segs = []
    for (u, v) in G.edges:
        if u == -1 or v == -1:
            continue
        x1, y1 = G.nodes[u]; x2, y2 = G.nodes[v]
        out_segs.append([[round(x1, 5), round(y1, 5)],
                         [round(x2, 5), round(y2, 5)]])

    # === ЕДИНЫЙ МАСШТАБ ДЛЯ ВЫХОДОВ ===
    # 1) вернуть смещение в абсолют при необходимости (в метрах)
    if retain_absolute and (src_dx != 0.0 or src_dy != 0.0):
        out_segs = _shift_segments(out_segs, src_dx, src_dy)

    # 2) масштабировать под выбранный выход (мм = *1000, м = *1)
    out_segs_scaled = _scale_segments(out_segs, out_scale, out_ndigits)

    # 3) ALL.json – уже масштабированный
    with open(all_json, "w", encoding="utf-8") as f:
        json.dump({"segments": out_segs_scaled}, f, ensure_ascii=False, indent=2)

    # 4) ALL.dxf – координаты в тех же единицах, что и JSON, и правильные INSUNITS
    dxf_segs_scaled = [((float(a[0]), float(a[1])), (float(b[0]), float(b[1])))
                    for a, b in out_segs_scaled]
    save_dxf_lines(dxf_segs_scaled, str(all_dxf),
                layer="ALL", color=7, lineweight=25,
                insunits=out_dxf_units)  # 'Millimeters' или 'Meters'

    return G, str(all_json), str(all_dxf), stem

# ---------------------- UI ----------------------

st.set_page_config(page_title="Outer Builder", layout="wide")
st.title("🧭 Внешний контур")

with st.sidebar:
    st.header("⚙️ Основные настройки")

    unit = st.selectbox(
        "Единицы входных координат",
        ["мм", "метры"],
        index=0,
        help="Во что перевести входные координаты перед обработкой.\n\n"
             "• Если ваши исходные данные в миллиметрах — выберите «мм». Мы автоматически переведём их в метры.\n"
             "• Если уже в метрах — выберите «метры»."
    )
    unit_scale = 0.001 if unit == "мм" else 1.0

    eps_snap_mm = st.number_input(
        "Точность склейки точек — EPS_SNAP (мм)",
        value=2.0, min_value=0.1, step=0.1,
        help="Насколько близко точки должны быть друг к другу, чтобы считаться одной.\n\n"
             "• Меньше значение — точнее, но могут остаться микрозазоры.\n"
             "• Больше значение — сильнее склейка, но можно «слипнуть» лишнее.\n\n"
             "Обычно 1–3 мм работает хорошо."
    )
    max_extend  = st.number_input(
        "Максимальное продление хвостов — MAX_EXTEND (м)",
        value=float(PARAMS.get("MAX_EXTEND", 0.1)), min_value=0.0, step=0.05,
        help="Насколько далеко можно автоматически «дотянуть» незамкнутую линию до соседней.\n\n"
             "Если не хотите автопродлений — поставьте 0."
    )
    angle_tol   = st.number_input(
        "Допустимое отклонение по направлению — ANGLE_TOL (°)",
        value=float(PARAMS.get("ANGLE_TOL", 2)), min_value=0.0, step=0.5,
        help="Насколько линии могут отличаться по направлению, чтобы система считала их продолжением друг друга.\n\n"
             "Например, 2–5° означает «почти одна линия»."
    )
    len_tol     = st.number_input(
        "Допуск по длине — LEN_TOL",
        value=float(PARAMS.get("LEN_TOL", 5)), min_value=0.0, step=0.5,
        help="Чувствительность к длинам при сопоставлении линий (как в вашем ядре).\n\n"
             "Обычно менять не нужно."
    )

    st.markdown("---")
    st.subheader("🔧 Автодоводка (по желанию)")
    do_extend = st.checkbox(
        "Автоматически дотягивать короткие «хвосты»",
        value=True,
        help="Попробовать аккуратно замкнуть разрывы: если конец линии рядом с другой, их соединят. "
             "Это удобно, когда в исходных данных есть маленькие зазоры."
    )
    do_bridge = st.checkbox(
        "Подключать замкнутые «острова» к основному контуру",
        value=True,
        help="Если на чертеже есть отдельные замкнутые куски (острова), программа попробует подключить их "
             "к основной части двумя короткими «мостиками»."
    )

    st.markdown("---")
    st.subheader("🚫 Защита от «микромостиков»")
    min_extend = st.number_input(
        "Не тянуть, если продление короче (м)",
        value=float(PARAMS.get("MIN_EXTEND", 0.0)), min_value=0.0, step=0.02,
        help="Минимальная длина, на которую разрешено «дотягивать» линию. Всё, что короче — игнорируем. "
             "Например, поставьте 0.05 (5 см), чтобы не создавать совсем мелкие перемычки."
    )
    near_deg2  = st.number_input(
        "Не тянуть к «узлам» ближе, чем (м)",
        value=float(PARAMS.get("NEAR_DEG2_THRESHOLD", 0.0)), min_value=0.0, step=0.02,
        help="«Узлом» считаем точку, где сходится несколько линий. Если такая точка слишком близко, "
             "короткую доводку к ней лучше не делать, чтобы не усложнять контур. Здесь задайте эту безопасную дистанцию."
    )
    skip_to_deg= st.checkbox(
        "Запретить короткие доводки к «узлам»",
        value=bool(PARAMS.get("SKIP_EXTEND_TO_DEG_GT1", False)),
        help="Включите, чтобы программа не делала короткие перемычки к местам, где сходится несколько линий. "
             "Это помогает избежать «зашумления» контура мелкими вставками."
    )

    st.markdown("---")
    st.subheader("🧩 Построение внешнего контура")
    method = st.selectbox(
        "Метод",
        ["faces_union (надёжный)", "smart (быстрый + подстраховка)"],
        index=0,
        help="Как искать внешнюю границу:\n\n"
             "• «Надёжный»: аккуратно собираем все внутренние области и берём их внешнюю оболочку. "
             "Лучше работает на сложных чертежах.\n"
             "• «Быстрый»: старается идти по границе быстрее и переключается на надёжный, если что-то не сходится."
    )
    eps_outer_mm = st.number_input(
        "Точность для поиска внешнего контура (мм)",
        value=2.0, min_value=0.5, step=0.5,
        help="Ещё одно значение точности, но только для шага, где ищется внешняя граница. "
             "Иногда полезно сделать его немного больше, чем «Точность склейки точек», чтобы убрать крошечные "
             "зазоры именно на этом этапе."
    )

    st.markdown("---")
    st.caption("Все результаты сохраняются в папку ./output")

    show_src_preview = st.checkbox("Показывать превью исходника", value=True,
                                   help="Сразу после загрузки файла отобразить его содержимое (до обработки).")

    st.markdown("---")
    st.subheader("👀 Превью на странице")
    show_preview = st.checkbox("Показывать изображение результата", value=True,
                               help="Отрисовывает картинку с получившимися линиями и внешним контуром.")
    lw_all = st.slider("Толщина линий чертежа", 0.3, 2.5, 0.8, 0.1)
    lw_outer = st.slider("Толщина внешнего контура", 1.0, 4.0, 2.0, 0.1)

    st.markdown("---")
    st.subheader("📍 Координаты")
    retain_absolute = st.checkbox(
        "Возвращать исходное смещение (абсолютные координаты)",
        value=True,
        help="При экспорте добавлять назад смещение исходника, чтобы планы этажей оставались на своих местах и не накладывались."
    )
    # --- ЕДИНИЦЫ ВЫВОДА (общие для всех файлов) ---
    out_units = st.selectbox(
        "Единицы вывода (все файлы)",
        ["мм", "метры"],
        index=0,
        help="Единицы для сохранения: *_all.json, *_outer.json и DXF будут в одном формате."
    )
    OUT_SCALE = 1000.0 if out_units == "мм" else 1.0  # мм = метры * 1000
    OUT_NDIGITS = 3 if OUT_SCALE == 1000.0 else 5      # для mm округляем до тысячных мм
    OUT_DXF_UNITS = "Millimeters" if OUT_SCALE == 1000.0 else "Meters"



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

# инициализация состояния для смещения
if "src_dx" not in st.session_state:
    st.session_state.src_dx = 0.0
if "src_dy" not in st.session_state:
    st.session_state.src_dy = 0.0

uploaded = st.file_uploader(
    "Загрузите JSON",
    type=["json"],
    help="Поддерживаются форматы:\n"
         "• JSON вида {\"segments\": [[[x1,y1],[x2,y2]], ...]}\n"
         "• Либо список отрезков [[[x1,y1],[x2,y2]], ...]\n"
         "• Либо полилиния [[x,y], ...]\n"
)

run = st.button(
    "▶️ Запустить",
    help="Старт: склейка точек → построение сетки → (по желанию) доводка и мостики → внешний контур."
)

# мгновенное превью исходного файла + вычисление смещения
if uploaded:
    try:
        src_dx, src_dy = _compute_source_offset_from_uploaded(uploaded.getvalue(), unit_scale)
        st.session_state.src_dx = src_dx
        st.session_state.src_dy = src_dy
    except Exception:
        st.session_state.src_dx = 0.0
        st.session_state.src_dy = 0.0

if uploaded and show_src_preview:
    segs_src, poly_src = _segments_from_json_bytes(uploaded.getvalue())
    if segs_src or poly_src:
        png_src = _plot_preview(
            segments=segs_src,
            outer_poly=poly_src,  # если файл полилиния, покажем её жирнее
            title="Исходный файл (как загружен)",
            lw_all=lw_all,
            lw_outer=lw_outer
        )
        st.image(png_src, caption="Превью исходника", use_container_width=True)
        st.download_button("⬇️ Скачать превью исходника (PNG)",
                           data=png_src,
                           file_name=f"source_preview.png",
                           mime="image/png")
    else:
        st.info("Не удалось распознать формат исходника. Поддерживаются {'segments': ...} или [[x,y], ...].")

# запуск пайплайна
if run:
    if not uploaded:
        st.error("Загрузите JSON файл.")
        st.stop()

    up_dir = Path("uploads"); up_dir.mkdir(exist_ok=True)
    tmp_path = up_dir / f"in_{int(time.time()*1000)}_{uploaded.name}"
    tmp_path.write_bytes(uploaded.getvalue())

    input_path = str(tmp_path)
    unit_scale_eff = unit_scale

    outdir = Path("output")
    outdir.mkdir(exist_ok=True)
    clean_output(outdir, keep_last_n=0)


    with st.spinner("Обработка..."):
        t0 = time.time()
        try:
            # основной расчёт и экспорт ALL (с возвратом смещения при необходимости)
            G, all_json, all_dxf, stem = _run_pipeline(
                input_path, unit_scale_eff, do_extend, do_bridge,
                retain_absolute=retain_absolute,
                src_dx=st.session_state.get("src_dx", 0.0),
                src_dy=st.session_state.get("src_dy", 0.0),
                out_scale=OUT_SCALE,
                out_ndigits=OUT_NDIGITS,
                out_dxf_units=OUT_DXF_UNITS,
            )

            # === ВНЕШНИЙ КОНТУР (с возвратом абсолютного смещения и единицами как у ALL) ===
            outer_json = outdir / f"{stem}_outer.json"
            outer_dxf  = outdir / f"{stem}_outer.dxf"

            # 1) считаем внешний контур в ЛОКАЛЬНЫХ МЕТРАХ (как раньше)
            meta_outer = save_outer_via_faces_union(
                G,
                str(outer_json),
                str(outer_dxf),
                eps_snap_m=PARAMS["EPS_SNAP"]
            )

            # 2) читаем его обратно, возвращаем абсолют (если включено), масштабируем и перезаписываем
            try:
                with open(outer_json, "r", encoding="utf-8") as f:
                    data_outer = json.load(f)
                segs_outer = data_outer.get("segments", [])

                sdx = st.session_state.get("src_dx", 0.0)
                sdy = st.session_state.get("src_dy", 0.0)

                # вернуть исходное смещение в МЕТРАХ (как для ALL)
                if retain_absolute and (sdx != 0.0 or sdy != 0.0):
                    segs_outer = _shift_segments(segs_outer, sdx, sdy)

                # привести к выбранным единицам вывода (мм/м) с тем же округлением
                segs_outer = _scale_segments(segs_outer, OUT_SCALE, OUT_NDIGITS)

                # перезаписать OUTER.json «правильными» координатами
                data_outer["segments"] = segs_outer
                data_outer.setdefault("params", {})["meta"] = meta_outer
                with open(outer_json, "w", encoding="utf-8") as f:
                    json.dump(data_outer, f, ensure_ascii=False, indent=2)

                # и OUTER.dxf — теми же координатами и INSUNITS
                dxf_lines_outer = [((float(a[0]), float(a[1])), (float(b[0]), float(b[1])))
                                   for a, b in segs_outer]
                save_dxf_lines(
                    dxf_lines_outer,
                    str(outer_dxf),
                    layer="OUTER",
                    color=1,
                    lineweight=25,
                    insunits=OUT_DXF_UNITS
                )
            except Exception as e:
                st.warning(f"Не удалось применить смещение/масштаб к OUTER: {e}")

        except Exception as e:
            st.exception(e)
            st.stop()

    dt = time.time() - t0
    st.success(f"Готово за {dt:.2f} с")
    st.write({"outer_meta": meta_outer})  # <-- выводим корректную мету

    # кнопки скачивания
    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "⬇️ Скачать внешний контур (JSON)",
            data=open(outer_json, "rb").read(),
            file_name=outer_json.name,
            mime="application/json",
            help=f"Внешний контур как список отрезков (в {out_units})."
        )
    with c2:
        st.download_button(
            "⬇️ Скачать внешний контур (DXF)",
            data=open(outer_dxf, "rb").read(),
            file_name=outer_dxf.name,
            mime="application/dxf",
            help=f"DXF с полилинией внешнего контура (в {out_units})."
        )

    # --- превью результата (те же единицы, что и файлы) ---
    all_segs_m = _segments_from_graph(G)  # локальные метры
    sdx = st.session_state.get("src_dx", 0.0)
    sdy = st.session_state.get("src_dy", 0.0)
    segs_for_plot_m = _shift_segments(all_segs_m, sdx, sdy) if retain_absolute else all_segs_m
    segs_for_plot   = _scale_segments(segs_for_plot_m, OUT_SCALE, OUT_NDIGITS)

    # внешний контур для превью берём из outer_json (он уже «абсолютный» + масштабированный)
    outer_poly = None
    try:
        data_plot = json.load(open(outer_json, "r", encoding="utf-8"))
        segs_outer_plot = data_plot.get("segments", [])
        if segs_outer_plot:
            poly = [segs_outer_plot[0][0], segs_outer_plot[0][1]]
            used = {0}
            changed = True
            while changed and len(used) < len(segs_outer_plot):
                changed = False
                tail = poly[-1]
                for i, (a, b) in enumerate(segs_outer_plot):
                    if i in used:
                        continue
                    if abs(a[0]-tail[0]) < 1e-9 and abs(a[1]-tail[1]) < 1e-9:
                        poly.append(b); used.add(i); changed = True; break
                    if abs(b[0]-tail[0]) < 1e-9 and abs(b[1]-tail[1]) < 1e-9:
                        poly.append(a); used.add(i); changed = True; break
            if len(poly) >= 2:
                outer_poly = poly
    except Exception:
        outer_poly = None

    if show_preview:
        png_bytes = _plot_preview(
            segments=segs_for_plot,
            outer_poly=outer_poly,
            title="Результат (с учётом исходного смещения)" if retain_absolute else "Результат (локальные координаты)",
            lw_all=lw_all,
            lw_outer=lw_outer
        )
        st.image(png_bytes, caption="Превью результата", use_container_width=True)
        st.download_button("⬇️ Скачать превью (PNG)", data=png_bytes,
                           file_name=f"{stem}_preview.png", mime="image/png")

    st.markdown("—" * 30)
    with st.expander("Все сегменты после обработки (для проверки)"):
        st.download_button(
            "⬇️ Скачать все сегменты (JSON)",
            data=open(all_json, "rb").read(),
            file_name=Path(all_json).name,
            mime="application/json",
            help="Все линии после склейки точек и (при включённых опциях) автоматических доводок."
        )
        st.download_button(
            "⬇️ Скачать все сегменты (DXF)",
            data=open(all_dxf, "rb").read(),
            file_name=Path(all_dxf).name,
            mime="application/dxf",
            help="DXF с набором всех линий после обработки — удобно сравнить с исходником."
        )
