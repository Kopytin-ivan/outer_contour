PARAMS = {
    "EPS_SNAP": 0.010,         # 2 мм
    "MAX_EXTEND": 0.30,        # 100 мм
    "ANGLE_TOL": 2,
    "LEN_TOL": 5,

    "DIR_TO_NEAR_RATIO": 2.0,
    "SEARCH_RADIUS": None,
    "TEMPLATE_ANG_TOL": 2,
    "TEMPLATE_LEN_TOL": 5,

    "GRID_CELL": None,
    "R_QUERY": None,

    "NEAR_PERP_MAX": 0.010,    # 10 мм
    "NEAR_FORWARD_MIN": 0.0,
    "NEAR_MAX_KINK_DEG": 12,
    "ISOLATED_EXT_MULT": 2.0,
    "CLOSED_BRIDGE_MAX": 0.20,      # максимальная дальность доводки от островка до host (внутр. единицы)
    "BRIDGE_SEARCH_R_MULT": 1.5,
    "BRIDGE_SEARCH_R": None,
}



def tune_spatial_grid(nodes_xy, params):
    # Оценим характерный шаг узлов: медиана расстояния до ближайшего соседа
    import numpy as np
    if len(nodes_xy) < 2:
        cell = 1.0; r = 3.0
    else:
        # Быстро и без тяжёлого KD-дерева: возьмём подвыборку
        sample = np.array(nodes_xy[::max(1, len(nodes_xy)//5000)])
        # «Грубый» шаг как медиана по осям (дешёво, но стабильно)
        dx = np.median(np.abs(np.diff(np.sort(sample[:,0])))) or 0.05
        dy = np.median(np.abs(np.diff(np.sort(sample[:,1])))) or 0.05
        dnn = max(min(dx, dy), params["EPS_SNAP"]*5)  # не меньше 5×EPS
        cell = max(0.02, min(5.0, 3.0*dnn))          # 2 см…5 м безопасные рамки
        r    = 3.0*cell
    params["GRID_CELL"] = cell
    params["R_QUERY"]   = r
