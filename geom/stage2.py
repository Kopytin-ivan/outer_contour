# geom/stage2.py
import json
from typing import Optional

from geom.cycles import find_planar_faces as find_cycles
from .templates import make_template, TemplatesDB

def stage2_build_and_save_templates(
    G,
    out_unique_path: str = "output/closed_templates.json",
    out_all_path: Optional[str] = None,    # если задан — пишем ВСЕ формы построчно (NDJSON)
    prof=None,
    flush_every: int = 1000
):
    # 1) найти циклы
    if prof: prof.start("stage2: find_cycles")
    cycles = find_cycles(G)
    if prof: prof.stop("stage2: find_cycles")

    # 2) делать шаблоны + дедуп по ключу (в памяти)
    db = TemplatesDB()
    written_all = 0
    f = None
    if out_all_path:
        f = open(out_all_path, "w", encoding="utf-8")

    if prof: prof.start("stage2: make_templates")
    try:
        for cyc in cycles:
            tpl = make_template(G, cyc)
            if not tpl:
                continue
            db.add(tpl)                      # учёт уникальных
            if f:                            # и одновременно «стримим» все формы
                f.write(json.dumps(tpl, ensure_ascii=False) + "\n")
                written_all += 1
                if written_all % flush_every == 0:
                    f.flush()
    finally:
        if f:
            f.close()
    if prof: prof.stop("stage2: make_templates")

    # 3) сохранить уникальные
    if prof: prof.start("stage2: save_templates")
    db.save(out_unique_path)
    if prof: prof.stop("stage2: save_templates")

    return len(cycles), db.size(), written_all
