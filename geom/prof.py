from time import perf_counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import csv
import os

@dataclass
class Prof:
    enabled: bool = True
    events: List[Tuple[str, float]] = field(default_factory=list)
    accum: Dict[str, float] = field(default_factory=dict)
    start_times: Dict[str, float] = field(default_factory=dict)
    out_csv: Optional[str] = None  # e.g. "timings.csv"

    def start(self, name: str):
        if not self.enabled: return
        self.start_times[name] = perf_counter()

    def stop(self, name: str):
        if not self.enabled: return
        t0 = self.start_times.pop(name, None)
        if t0 is None: return
        dt = perf_counter() - t0
        self.events.append((name, dt))
        self.accum[name] = self.accum.get(name, 0.0) + dt

    @contextmanager
    def section(self, name: str):
        self.start(name)
        try:
            yield
        finally:
            self.stop(name)

    def report_console(self):
        if not self.enabled: return
        # сгруппированная сводка
        print("\n=== TIME REPORT (aggregated) ===")
        for k, v in sorted(self.accum.items(), key=lambda kv: -kv[1]):
            print(f"{k:<40s} {v:8.3f} s")
        # последовательные события
        print("\n--- timeline ---")
        for name, dt in self.events:
            print(f"{name:<40s} {dt:8.3f} s")

    def dump_csv(self, path: Optional[str] = None):
        if not self.enabled: return
        p = path or self.out_csv
        if not p: return
        os.makedirs(os.path.dirname(p) or ".", exist_ok=True)
        with open(p, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["name", "dt_sec"])
            for name, dt in self.events:
                w.writerow([name, f"{dt:.6f}"])




# --- добавь в конец geom/prof.py ---

# Глобальный экземпляр профайлера, чтобы им мог пользоваться декоратор
PROF = Prof()

def prof_step(name: str):
    """
    Контекст-менеджер короткой записи:
        with prof_step("stage2:templates"):
            ...
    """
    return PROF.section(name)

def stage_timer(name: str):
    """
    Декоратор для измерения времени выполнения функции.
        @stage_timer("stage0_1_init")
        def stage0_1_init(...):
            ...
    """
    def deco(func):
        def wrapper(*args, **kwargs):
            with PROF.section(name):
                return func(*args, **kwargs)
        return wrapper
    return deco

def prof_report(csv_path: str = None):
    """
    Удобный вызов в конце программы.
    """
    PROF.report_console()
    if csv_path:
        PROF.dump_csv(csv_path)
