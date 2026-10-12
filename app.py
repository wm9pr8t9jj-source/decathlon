# ЗАПУСК:  streamlit run decathlon_app.py     (нужны: streamlit numpy scipy pandas; больше никаких файлов)
# Это ОДИН файл: модель, план по Грузенкину, Монте-Карло и интерфейс (раздел 11 внизу). Данные сохраняются (см. DEPLOY.md).
# Командный режим тоже есть: python decathlon_app.py --mode selftest
# -*- coding: utf-8 -*-
"""
decathlon_suite.py — единая система планирования и прогнозирования для десятиборья (один файл).

Объединяет: (A) иерархическую байесовскую модель Fitness–Fatigue с эмпирическими априорами мировых десятиборцев,
(B) гранулярную физиологию нагрузки (тензор [CNS, MET, MECH], три пула утомления с периодами полураспада),
(C) типизированную библиотеку упражнений и сессий, (D) биологические шлюзы, (E) дискретный планировщик по дням,
(F) объяснение и визуализацию плана («почему именно эта сессия в этот день»), (G) Монте-Карло турнира.

Зависимости: numpy, scipy, pandas. Запуск (из терминала):
    python decathlon_suite.py                                 # полный прогон для профиля Самандыка Д. (холодный старт)
    python decathlon_suite.py --start 2026-10-05 --horizon 21 --html plan.html
    python decathlon_suite.py --why S_HILL_SL                 # почему эта сессия стоит именно там, а не раньше
    python decathlon_suite.py --why-day 2                     # объяснение конкретного дня плана
    python decathlon_suite.py --journal j.csv --results r.csv # калибровка на ваших данных → персональный план
    python decathlon_suite.py --mode library | selftest
Форматы CSV: журнал  date,module,minutes,rpe  (module = ключ сессии из библиотеки, minutes/rpe необязательны);
             результаты  date,obs,value  (obs: 100m, LJ, SP, HJ, 400m, 110mH, DT, PV, JT, 1500m, T30f, TSLJ, TSHF, T300, T1000;
             value — секунды или метры).

СТРУКТУРА (резать по заголовкам «═══» на подмодули):
  1 scoring   — очки World Athletics
  2 tensor    — вектор стресса [CNS, MET, MECH], пулы утомления с периодами полураспада (24–48 / 24–72 / 72–96 ч)
  3 library   — упражнения → сессии, проверка профилей LOW/MID/HIGH/EXTREME
  4 gates     — CNS Gate, Interference Gate, Tendon Protection (с причиной и временем снятия блока)
  5 prior     — ковариация видов, байесовская усадка профиля, априоры τ1 и T½ пулов, штрафы 2-го дня
  6 model     — Fitness–Fatigue (каналы + пулы), forward(), журнал → тензоры, MAP-калибровка, rolling CV
  7 solver    — дискретный планировщик: beam search + локальный поиск, оптимизатор дня
  8 explain   — объяснение «почему», текстовый и HTML-вывод плана
  9 montecarlo— вероятностный турнир: медиана, P05–P95, вероятности рубежей, чувствительность
  10 demo     — профиль атлета, синтетика, самопроверка, CLI

ЧЕСТНО ОБ ИСТОЧНИКАХ ЧИСЕЛ
  • Корреляции [USER] взяты из ТЗ и не проверялись по первоисточнику; [EXPERT] — экспертное заполнение
    (заменить на estimate_corr_from_protocols()).
  • Векторы стресса упражнений, веса пулов W, дозы соревнований, штрафы 2-го дня, σ видов, пороги шлюзов — стартовые
    экспертные значения (не измеренные константы); все вынесены в словари/dataclass.
  • Без журнала параметры атлета = мода априора (холодный старт): план — сценарий, а не индивидуальный прогноз.
    Разница между сессиями при априорах мала (единицы очков), поэтому состав плана определяют шлюзы и баланс.
"""
from __future__ import annotations

import argparse
import dataclasses
import os
import sys
import html as _html
import itertools
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from types import SimpleNamespace
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.signal import lfilter
from scipy.stats import chi2

WD = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]


# ═════════════════════════════ 1. SCORING — очки World Athletics (десятиборье, мужчины) ═════════════════════════════
# P = A·(B−T)^C (бег, T в с) | P = A·(D−B)^C (поле: прыжки в см, метания в м). Результаты в модуле: секунды и метры.
WA: Dict[str, Tuple[float, float, float, str, float]] = {  # A, B, C, тип, множитель единиц (м→см для прыжков)
    "100m": (25.4347, 18.0, 1.81, "T", 1.0), "LJ": (0.14354, 220.0, 1.40, "D", 100.0),
    "SP": (51.39, 1.5, 1.05, "D", 1.0), "HJ": (0.8465, 75.0, 1.42, "D", 100.0),
    "400m": (1.53775, 82.0, 1.81, "T", 1.0), "110mH": (5.74352, 28.5, 1.92, "T", 1.0),
    "DT": (12.91, 4.0, 1.10, "D", 1.0), "PV": (0.2797, 100.0, 1.35, "D", 100.0),
    "JT": (10.14, 7.0, 1.08, "D", 1.0), "1500m": (0.03768, 480.0, 1.85, "T", 1.0),
}
EVENTS = ("100m", "LJ", "SP", "HJ", "400m", "110mH", "DT", "PV", "JT", "1500m")
DAY1, DAY2 = EVENTS[:5], EVENTS[5:]
IS_TIME = {e: WA[e][3] == "T" for e in EVENTS}
SIGN = {e: (-1.0 if IS_TIME[e] else 1.0) for e in EVENTS}  # знак: время — меньше лучше


def wa_points(event: str, result, floor: bool = False):
    """Очки WA за результат. floor=True — как в протоколе; для оптимизации берём непрерывные очки."""
    A, B, C, kind, unit = WA[event]
    x = np.asarray(result, float) * unit
    p = A * np.maximum((B - x) if kind == "T" else (x - B), 0.0) ** C
    return np.floor(p) if floor else p


def wa_result(event: str, points):
    """Обратная формула: результат (с или м), соответствующий очкам."""
    A, B, C, kind, unit = WA[event]
    s = (np.asarray(points, float) / A) ** (1.0 / C)
    return ((B - s) if kind == "T" else (B + s)) / unit



# ═════════════════════════════ 2. TENSOR — вектор стресса [CNS, MET, MECH] и пулы утомления с периодами полураспада ═════════════════════════════
POOLS: Tuple[str, str, str] = ("CNS", "MET", "MECH")
HALF_LIFE_RANGE_H: Dict[str, Tuple[float, float]] = {"CNS": (24.0, 48.0), "MET": (24.0, 72.0), "MECH": (72.0, 96.0)}


class Level(Enum):
    """Качественный уровень стресса пула (границы по значению в [0, 1])."""
    LOW = (0.0, 0.35)
    MID = (0.35, 0.70)
    HIGH = (0.70, 0.95)
    EXTREME = (0.95, 1.0001)

    @classmethod
    def of(cls, x: float) -> "Level":
        return next(l for l in cls if l.value[0] <= x < l.value[1])


@dataclass(frozen=True)
class StressVector:
    """Вектор стресса одного упражнения/сессии при стандартном объёме (volume = 1)."""
    cns: float
    met: float
    mech: float

    def __post_init__(self):
        for name, v in zip(POOLS, self.as_tuple()):
            if not 0.0 <= v <= 1.0:
                raise ValueError(f"{name}={v}: стресс должен лежать в [0, 1]")

    def as_tuple(self) -> Tuple[float, float, float]:
        return (self.cns, self.met, self.mech)

    def array(self) -> np.ndarray:
        return np.array(self.as_tuple(), float)

    def levels(self) -> Tuple[Level, Level, Level]:
        return tuple(Level.of(v) for v in self.as_tuple())  # type: ignore[return-value]

    def __str__(self) -> str:
        return "/".join(f"{p}:{l.name}" for p, l in zip(POOLS, self.levels()))


@dataclass(frozen=True)
class PoolConstants:
    """Периоды полураспада утомления по пулам, часы. По умолчанию — середины диапазонов из ТЗ.
    Позже уточняются по данным атлета (MAP): prior_log() даёт логнормальный априор в тех же диапазонах."""
    half_life_h: Tuple[float, float, float] = (36.0, 48.0, 84.0)
    strict: bool = True   # False — разрешить значения вне диапазона ТЗ (результат калибровки по данным атлета)

    def __post_init__(self):
        for p, h in zip(POOLS, self.half_life_h):
            lo, hi = HALF_LIFE_RANGE_H[p]
            if self.strict and not lo <= h <= hi:
                raise ValueError(f"T½_{p}={h} ч вне физиологического диапазона {lo}–{hi} ч")

    def half(self) -> np.ndarray:
        return np.array(self.half_life_h, float)

    def tau_e(self) -> np.ndarray:
        """e-постоянная времени τ = T½ / ln2 (часы)."""
        return self.half() / np.log(2.0)

    def decay_factor(self, hours: float) -> np.ndarray:
        """Доля утомления, остающаяся через `hours` часов, по трём пулам."""
        return 0.5 ** (hours / self.half())

    @staticmethod
    def prior_log() -> Tuple[np.ndarray, np.ndarray]:
        """(μ, σ) логнормального априора ln T½: диапазон из ТЗ трактуется как ±1 SD вокруг геометрической середины."""
        lo = np.array([HALF_LIFE_RANGE_H[p][0] for p in POOLS])
        hi = np.array([HALF_LIFE_RANGE_H[p][1] for p in POOLS])
        return 0.5 * np.log(lo * hi), 0.5 * np.log(hi / lo)


class PoolState:
    """Текущий вектор утомления [CNS, MET, MECH] и часы. Единицы пула = «стандартные сессии» (доза 1.0 = стресс 1.0·объём 1)."""

    def __init__(self, const: PoolConstants = PoolConstants(), values: Sequence[float] = (0.0, 0.0, 0.0), clock_h: float = 0.0):
        self.const, self.P, self.clock_h = const, np.array(values, float), float(clock_h)

    def copy(self) -> "PoolState":
        return PoolState(self.const, self.P.copy(), self.clock_h)

    def advance(self, hours: float) -> "PoolState":
        """Свободное восстановление на `hours` часов."""
        if hours < 0:
            raise ValueError("hours ≥ 0")
        self.P = self.P * self.const.decay_factor(hours)
        self.clock_h += hours
        return self

    def apply(self, stress: StressVector, volume: float = 1.0) -> "PoolState":
        """Добавляет дозу сессии: P += stress · volume (volume 1.0 = стандартный объём; 0.5 — половина)."""
        if volume < 0:
            raise ValueError("volume ≥ 0")
        self.P = self.P + stress.array() * volume
        return self

    def hours_until(self, pool: str, threshold: float) -> float:
        """Через сколько часов пул опустится до threshold (0, если уже ниже). Точная формула T½·log2(P/thr)."""
        p = self.P[POOLS.index(pool)]
        if p <= threshold:
            return 0.0
        return float(self.const.half()[POOLS.index(pool)] * np.log2(p / threshold))

    def as_dict(self) -> Dict[str, float]:
        return {p: round(float(v), 4) for p, v in zip(POOLS, self.P)}


def simulate(events: Sequence[Tuple[float, StressVector, float]], horizon_h: float, step_h: float = 6.0,
             const: PoolConstants = PoolConstants()) -> pd.DataFrame:
    """Временной ряд пулов. events: [(час от начала, StressVector, volume)]. Каждое событие применяется ровно в свой час,
    затем идёт распад; точки ряда — каждые step_h часов (значение ПОСЛЕ событий этого момента)."""
    ev = sorted(events, key=lambda e: e[0])
    st, rows, t, i = PoolState(const), [], 0.0, 0
    while t <= horizon_h + 1e-9:
        while i < len(ev) and ev[i][0] <= t + 1e-9:
            st.advance(max(ev[i][0] - st.clock_h, 0.0)).apply(ev[i][1], ev[i][2])
            i += 1
        st.advance(t - st.clock_h)
        rows.append({"hour": t, **{p: v for p, v in zip(POOLS, st.P)}})
        t += step_h
    return pd.DataFrame(rows)


def recovery_table(const: PoolConstants = PoolConstants(), hours: Sequence[float] = (12, 24, 48, 72, 96)) -> pd.DataFrame:
    """Сколько % дозы остаётся в каждом пуле через заданное число часов."""
    return pd.DataFrame({f"{int(h)} ч": (const.decay_factor(h) * 100).round(1) for h in hours}, index=list(POOLS))


# ═════════════════════════════ 3. LIBRARY — упражнения → дискретные сессии ═════════════════════════════
CHANNELS: Tuple[str, ...] = ("SPR", "JMP", "THR", "SE", "END")   # спринт/барьеры, прыжки, метания, скоростная вынослив., аэробика
AU_REF = 400.0   # 400 AU (минуты × RPE) = 1.0 нормированная единица нагрузки


class Category(Enum):
    SPRINT = "спринт/ускорение"
    GLYCOLYTIC = "гликолиз"
    PLYO = "плиометрика/механика"
    TECHNIQUE = "техника"
    STRENGTH = "силовая"
    THROWS = "метания"
    AEROBIC = "аэробика"
    RECOVERY = "восстановление"


@dataclass(frozen=True)
class Exercise:
    """Микро-упражнение. stress — на стандартную рабочую единицу (серия типового объёма);
    chan — какие фитнес-каналы развивает (сумма ≈ 1; пусто — профилактика)."""
    key: str
    title: str
    stress: StressVector
    chan: Tuple[Tuple[str, float], ...] = ()

    def __post_init__(self):
        bad = [c for c, _ in self.chan if c not in CHANNELS]
        if bad:
            raise ValueError(f"{self.key}: неизвестные каналы {bad}")


def _ex(key, title, cns, met, mech, **chan) -> Exercise:
    return Exercise(key, title, StressVector(cns, met, mech), tuple(chan.items()))


_EXERCISES: List[Exercise] = [
    # спринт / нейромышечные
    _ex("fly30", "Спринт 30 м с ходу (макс.)", .92, .08, .25, SPR=1.0),
    _ex("acc20", "Старты/ускорения 20 м", .85, .15, .33, SPR=1.0),
    # гликолитические / аэробные
    _ex("sub100", "100 м субмакс (85–90 %), короткий отдых", .55, .92, .50, SE=.75, SPR=.25),
    _ex("rep300", "300 м повторно", .60, .93, .45, SE=.85, END=.15),
    _ex("int1000", "Интервалы 1000 м", .20, .85, .20, END=.8, SE=.2),
    _ex("cross30", "Аэробный кросс 30 мин", .05, .45, .15, END=1.0),
    _ex("tempo_run", "Восстановительный темповый бег", .08, .40, .18, END=.8, SPR=.2),
    # ударные / механические
    _ex("hill_sl", "Прыжки на одной ноге в крутую горку", .55, .20, .98, JMP=.8, SPR=.2),
    _ex("drop_jump", "Спрыгивания 40–60 см", .62, .10, .92, JMP=.85, SPR=.15),
    _ex("hurdle_plyo", "Барьерная плиометрика", .72, .15, .88, JMP=.6, SPR=.4),
    _ex("low_bounds", "Низкие подскоки / скакалка", .35, .10, .40, JMP=.7, SPR=.3),
    # техника
    _ex("hj_short", "Высота: короткий разбег, техника", .62, .10, .28, JMP=1.0),
    _ex("hj_full", "Высота: полный разбег", .88, .12, .28, JMP=.8, SPR=.2),
    _ex("pv_short", "Шест: короткий разбег, технические звенья", .65, .12, .30, JMP=.7, SPR=.3),
    _ex("pv_full", "Шест: полный разбег", .90, .15, .30, JMP=.7, SPR=.3),
    _ex("hurdle_rhythm", "Барьеры: ритм 3–5 барьеров", .72, .15, .30, SPR=.7, JMP=.3),
    # силовые
    _ex("squat90", "Приседания 3×3 на 90 %", .88, .18, .94, THR=.5, JMP=.3, SPR=.2),
    _ex("snatch", "Рывок 3×2–3", .92, .10, .62, THR=.4, JMP=.4, SPR=.2),
    _ex("gym_base", "Силовая база (объём, 70–80 %)", .40, .30, .62, THR=.5, JMP=.25, SPR=.25),
    # метания
    _ex("throw_full", "Метания: полные броски", .82, .12, .55, THR=1.0),
    _ex("medball", "Взрывные броски ядра/мяча", .78, .15, .45, THR=.9, JMP=.1),
    _ex("throw_tech", "Метания: техника 70 %", .55, .10, .33, THR=1.0),
    # профилактика
    _ex("mobility", "Мобильность / профилактика", .03, .03, .05),
    _ex("core", "Корпус / стабилизация", .15, .10, .20, THR=.3, JMP=.2),
]
EXERCISES: Dict[str, Exercise] = {e.key: e for e in _EXERCISES}
assert len(EXERCISES) == len(_EXERCISES), "дублирующиеся ключи упражнений"


@dataclass(frozen=True)
class Session:
    """Дискретная сессия. parts — ((ключ упражнения, доля рабочего времени), ...), доли нормируются.
    profile — заявленный уровень (CNS, MET, MECH). coord — тонкая координация (шест, высота, барьеры):
    флаг, который читает Interference Gate (пункт 3)."""
    key: str
    title: str
    category: Category
    parts: Tuple[Tuple[str, float], ...]
    minutes: float
    rpe: float
    profile: Tuple[Level, Level, Level]
    coord: bool = False
    why: str = ""

    def __post_init__(self):
        for k, s in self.parts:
            if k not in EXERCISES:
                raise KeyError(f"{self.key}: нет упражнения «{k}»")
            if s <= 0:
                raise ValueError(f"{self.key}: доля {k} должна быть > 0")

    @property
    def stress(self) -> StressVector:
        if not self.parts:
            return StressVector(0.0, 0.0, 0.0)
        tot = sum(s for _, s in self.parts)
        v = sum(s * EXERCISES[k].stress.array() for k, s in self.parts) / tot
        return StressVector(*(float(np.clip(x, 0.0, 1.0)) for x in v))

    @property
    def chan(self) -> Dict[str, float]:
        tot = sum(s for _, s in self.parts) or 1.0
        out: Dict[str, float] = {}
        for k, s in self.parts:
            for c, w in EXERCISES[k].chan:
                out[c] = out.get(c, 0.0) + s * w / tot
        return out

    @property
    def AU(self) -> float:
        return self.minutes * self.rpe

    def channel_load(self) -> np.ndarray:
        """Нагрузка по фитнес-каналам в единицах AU/AU_REF (для модели Fitness–Fatigue)."""
        return (self.AU / AU_REF) * np.array([self.chan.get(c, 0.0) for c in CHANNELS])

    def levels(self) -> Tuple[Level, Level, Level]:
        return self.stress.levels()


_L, _M, _H, _X = Level.LOW, Level.MID, Level.HIGH, Level.EXTREME
_C = Category
_SESSIONS: List[Session] = [
    # ── шесть модулей из ТЗ (+ их ближайшие варианты) ──
    Session("S_FLY30", "4×30 м с ходу", _C.SPRINT, (("fly30", 1),), 55, 7, (_H, _L, _L), False,
            "алактатная максимальная скорость: ЦНС↑, лактата почти нет, связки не перегружены"),
    Session("S_GLYCO100", "4×100 м submax (гликолиз)", _C.GLYCOLYTIC, (("sub100", 1),), 70, 8, (_M, _H, _M), False,
            "лактатная мощность и устойчивость к закислению (400 м)"),
    Session("S_HILL_SL", "Прыжки на 1 ноге в горку", _C.PLYO, (("hill_sl", 1),), 50, 7, (_M, _L, _X), False,
            "реактивная сила и жёсткость сухожилий (барьеры, шест, длина)"),
    Session("S_DROP_JUMPS", "Спрыгивания + низкие подскоки", _C.PLYO, (("drop_jump", .8), ("low_bounds", .2)), 45, 7,
            (_M, _L, _H), False, "реактивная сила при более мягкой механике, чем горка"),
    Session("S_HJ_FULL", "Высота: полный разбег (техника)", _C.TECHNIQUE, (("hj_full", 1),), 90, 6, (_H, _L, _L), True,
            "соревновательная техника высоты на скорости"),
    Session("S_PV_FULL", "Шест: полный разбег (техника)", _C.TECHNIQUE, (("pv_full", 1),), 90, 6, (_H, _L, _L), True,
            "соревновательная техника шеста на скорости"),
    Session("S_EXPLOSIVE", "Взрывная сила: присед 3×3 90 % + рывок", _C.STRENGTH,
            (("squat90", .6), ("snatch", .4)), 75, 8, (_H, _L, _H), False,
            "максимальная и взрывная сила ног (метания, прыжки, спринт)"),
    Session("S_CROSS30", "Аэробный кросс 30 мин", _C.AEROBIC, (("cross30", 1),), 30, 5, (_L, _M, _L), False,
            "аэробная база и ускорение восстановления"),
    Session("S_TEMPO_RUN", "Восстановительный темповый бег", _C.AEROBIC, (("tempo_run", 1),), 35, 4, (_L, _M, _L), False,
            "активное восстановление с умеренной метаболической нагрузкой"),
    # ── варианты и дополнительные модули (для метаний, барьеров, 400/1500 м) ──
    Session("S_HJ_SHORT", "Высота: короткий разбег (техника)", _C.TECHNIQUE, (("hj_short", 1),), 75, 5, (_M, _L, _L), True,
            "техника высоты без высокого CNS-стресса"),
    Session("S_PV_SHORT", "Шест: короткий разбег (техника)", _C.TECHNIQUE, (("pv_short", 1),), 75, 5, (_M, _L, _L), True,
            "технические звенья шеста без высокого CNS-стресса"),
    Session("S_ACCEL20", "Старты 6×20 м", _C.SPRINT, (("acc20", 1),), 50, 7, (_H, _L, _L), False,
            "стартовое ускорение для 100 м и 400 м"),
    Session("S_SPEEDEND300", "3–4×300 м", _C.GLYCOLYTIC, (("rep300", 1),), 65, 8, (_M, _H, _M), False,
            "скоростная выносливость для 400 м"),
    Session("S_INT1000", "Интервалы 4–5×1000 м", _C.AEROBIC, (("int1000", 1),), 60, 7, (_L, _H, _L), False,
            "аэробная мощность для 1500 м"),
    Session("S_HURDLE_PLYO", "Барьерная плиометрика", _C.PLYO, (("hurdle_plyo", 1),), 50, 7, (_H, _L, _H), False,
            "ударная работа для барьеров и прыжков"),
    Session("S_HURDLE_RHYTHM", "Барьеры: ритм", _C.TECHNIQUE, (("hurdle_rhythm", 1),), 70, 6, (_H, _L, _L), True,
            "ритм и техника 110 м с/б"),
    Session("S_STRENGTH_BASE", "Силовая база", _C.STRENGTH, (("gym_base", 1),), 70, 7, (_M, _L, _M), False,
            "объём силовой работы без пикового CNS"),
    Session("S_THROW_POWER", "Метания: полные броски + ядро", _C.THROWS, (("throw_full", .7), ("medball", .3)), 80, 7,
            (_H, _L, _M), False, "дальность метаний"),
    Session("S_THROW_TECH", "Метания: техника 70 %", _C.THROWS, (("throw_tech", 1),), 70, 5, (_M, _L, _L), False,
            "техника метаний при низкой цене восстановления"),
    Session("S_MOBILITY", "Мобильность и профилактика", _C.RECOVERY, (("mobility", .7), ("core", .3)), 30, 3, (_L, _L, _L), False,
            "профилактика, почти нулевой стресс"),
    Session("S_REST", "Отдых", _C.RECOVERY, (), 0, 0, (_L, _L, _L), False, "полное восстановление"),
]


class SessionLibrary:
    """Реестр сессий: доступ по ключу, фильтры, таблица, валидация, доза для PoolState."""

    def __init__(self, sessions: Iterable[Session] = _SESSIONS):
        self._s: Dict[str, Session] = {}
        for s in sessions:
            if s.key in self._s:
                raise ValueError(f"дублирующийся ключ сессии {s.key}")
            self._s[s.key] = s
        self.validate()

    def __getitem__(self, key: str) -> Session:
        return self._s[key]

    def __iter__(self):
        return iter(self._s.values())

    def __len__(self) -> int:
        return len(self._s)

    def keys(self) -> List[str]:
        return list(self._s)

    def validate(self) -> None:
        """Расчётный вектор стресса каждой сессии должен давать заявленный профиль Level."""
        bad = [(s.key, [l.name for l in s.levels()], [l.name for l in s.profile]) for s in self if s.levels() != s.profile]
        if bad:
            raise AssertionError(f"профиль ≠ расчёт: {bad}")

    def filter(self, category: Optional[Category] = None, cns: Optional[Level] = None, met: Optional[Level] = None,
               mech: Optional[Level] = None, coord: Optional[bool] = None) -> List[Session]:
        out = []
        for s in self:
            lv = s.levels()
            if ((category is None or s.category == category) and (cns is None or lv[0] == cns)
                    and (met is None or lv[1] == met) and (mech is None or lv[2] == mech)
                    and (coord is None or s.coord == coord)):
                out.append(s)
        return out

    def table(self) -> pd.DataFrame:
        rows = []
        for s in self:
            st = s.stress
            rows.append(dict(key=s.key, title=s.title, category=s.category.value, CNS=round(st.cns, 2), MET=round(st.met, 2),
                             MECH=round(st.mech, 2), profile="/".join(l.name[0] for l in s.levels()), AU=s.AU, coord=s.coord))
        return pd.DataFrame(rows)

    def apply(self, state: PoolState, key: str, volume: float = 1.0) -> PoolState:
        """Добавляет в PoolState (пункт 1) дозу сессии key при объёме volume."""
        return state.apply(self[key].stress, volume)


LIBRARY = SessionLibrary()

# Шесть обязательных модулей из ТЗ с их профилями (проверяется в самопроверке)
REQUIRED = {"S_FLY30": (_H, _L, _L), "S_GLYCO100": (_M, _H, _M), "S_HILL_SL": (_M, _L, _X), "S_HJ_FULL": (_H, _L, _L),
            "S_PV_FULL": (_H, _L, _L), "S_EXPLOSIVE": (_H, _L, _H), "S_CROSS30": (_L, _M, _L), "S_TEMPO_RUN": (_L, _M, _L)}


R = len(CHANNELS)


# ═════════════════════════════ 4. GATES — биологические шлюзы ═════════════════════════════
@dataclass(frozen=True)
class GateConfig:
    cns_stress: float = 0.8            # сессии с CNS ≥ этого значения проверяются CNS Gate
    cns_ref_dose: float = 0.70         # эталонная доза для порога свежести
    cns_fresh_after_h: float = 24.0    # через сколько часов остаток эталона = порог
    cns_thr_override: Optional[float] = None   # задать порог напрямую (иначе выводится из T½_CNS)
    met_stress: float = 0.8            # лактатная сессия: MET ≥ этого значения
    interference_h: float = 48.0       # блок тонкой техники после лактатной сессии
    mech_stress: float = 0.8           # тяжёлая механическая сессия: MECH ≥ этого значения
    mech_extreme: float = 0.95         # экстремальная механика (горка)
    tendon_h: float = 72.0             # минимальный интервал между тяжёлыми механическими сессиями
    tendon_extreme_h: float = 96.0     # ... если хотя бы одна из двух экстремальная

    def cns_threshold(self, half_life_cns_h: float) -> float:
        if self.cns_thr_override is not None:
            return self.cns_thr_override
        return self.cns_ref_dose * 0.5 ** (self.cns_fresh_after_h / half_life_cns_h)


@dataclass(frozen=True)
class Block:
    gate: str            # "CNS" | "INTERFERENCE" | "TENDON"
    message: str         # причина по-русски с числами
    free_in_h: float     # через сколько часов этот шлюз откроется


@dataclass(frozen=True)
class Verdict:
    session: str
    hour: float
    blocks: Tuple[Block, ...]

    @property
    def allowed(self) -> bool:
        return not self.blocks

    @property
    def free_in_h(self) -> float:
        """Через сколько часов допустимы ВСЕ шлюзы (0 — можно сейчас). Точно: пулы распадаются независимо от решений."""
        return max((b.free_in_h for b in self.blocks), default=0.0)

    def text(self) -> str:
        return "допустимо" if self.allowed else "; ".join(f"{b.gate}: {b.message}" for b in self.blocks)


class AthleteState:
    """Состояние для шлюзов: три пула (п.1) + часы последней лактатной и тяжёлой механической сессий."""

    def __init__(self, const: PoolConstants = PoolConstants(), cfg: GateConfig = GateConfig()):
        self.const, self.cfg = const, cfg
        self.pools = PoolState(const)
        self.last_lactic_h = -np.inf
        self.last_mech_h = -np.inf
        self.last_mech_extreme = False
        self.log: List[Tuple[float, str, float]] = []

    def pools_at(self, hour: float) -> PoolState:
        """Пулы на момент hour без изменения состояния."""
        if hour < self.pools.clock_h - 1e-9:
            raise ValueError("нельзя смотреть в прошлое относительно последней записанной сессии")
        return self.pools.copy().advance(hour - self.pools.clock_h)

    def record(self, session: Session, hour: float, volume: float = 1.0) -> None:
        """Фиксирует выполненную сессию: доза в пулы, обновление счётчиков шлюзов."""
        self.pools.advance(hour - self.pools.clock_h).apply(session.stress, volume)
        st, c = session.stress, self.cfg
        if st.met >= c.met_stress:
            self.last_lactic_h = hour
        if st.mech >= c.mech_stress:
            self.last_mech_h, self.last_mech_extreme = hour, st.mech >= c.mech_extreme
        self.log.append((hour, session.key, volume))

    def copy(self) -> "AthleteState":
        s = AthleteState(self.const, self.cfg)
        s.pools, s.last_lactic_h, s.last_mech_h = self.pools.copy(), self.last_lactic_h, self.last_mech_h
        s.last_mech_extreme, s.log = self.last_mech_extreme, list(self.log)
        return s


class BioGates:
    def __init__(self, cfg: GateConfig = GateConfig(), const: PoolConstants = PoolConstants()):
        self.cfg, self.const = cfg, const
        self.cns_thr = cfg.cns_threshold(const.half_life_h[0])

    def check(self, state: AthleteState, session: Session, hour: float) -> Verdict:
        c, st, blocks = self.cfg, session.stress, []
        pools = state.pools_at(hour)
        # 1) CNS Gate
        if st.cns >= c.cns_stress and pools.P[0] > self.cns_thr:
            blocks.append(Block("CNS", f"пул ЦНС {pools.P[0]:.2f} > порога {self.cns_thr:.2f} "
                                       f"(сессия с CNS {st.cns:.2f}); освободится через {pools.hours_until('CNS', self.cns_thr):.0f} ч",
                                pools.hours_until("CNS", self.cns_thr)))
        # 2) Interference Gate
        if session.coord:
            dt = hour - state.last_lactic_h
            if dt < c.interference_h:
                blocks.append(Block("INTERFERENCE", f"лактатная сессия была {dt:.0f} ч назад, тонкая техника — не раньше чем "
                                                    f"через {c.interference_h:.0f} ч; осталось {c.interference_h - dt:.0f} ч",
                                    c.interference_h - dt))
        # 3) Tendon Protection
        if st.mech >= c.mech_stress:
            dt = hour - state.last_mech_h
            need = c.tendon_extreme_h if (st.mech >= c.mech_extreme or state.last_mech_extreme) else c.tendon_h
            if dt < need:
                blocks.append(Block("TENDON", f"тяжёлая механика была {dt:.0f} ч назад, нужен интервал ≥{need:.0f} ч; "
                                              f"осталось {need - dt:.0f} ч", need - dt))
        return Verdict(session.key, hour, tuple(blocks))

    def check_all(self, state: AthleteState, hour: float, library: SessionLibrary = LIBRARY) -> pd.DataFrame:
        """Допустимость каждой сессии библиотеки в момент hour."""
        rows = []
        for s in library:
            v = self.check(state, s, hour)
            rows.append(dict(key=s.key, title=s.title, allowed=v.allowed,
                             gates=",".join(b.gate for b in v.blocks), free_in_h=round(v.free_in_h, 1), reason=v.text()))
        return pd.DataFrame(rows).sort_values(["allowed", "key"], ascending=[False, True]).reset_index(drop=True)

    def earliest_allowed(self, state: AthleteState, session: Session, from_hour: float) -> float:
        """Ближайший момент ≥ from_hour, когда сессия допустима (точно, если до него не добавлять новых сессий)."""
        return from_hour + self.check(state, session, from_hour).free_in_h

    def validate_schedule(self, events: Iterable[Tuple[float, str, float]], library: SessionLibrary = LIBRARY) -> pd.DataFrame:
        """Проверка готового расписания [(час, ключ сессии, объём)]: каждая сессия проверяется на состоянии,
        накопленном предыдущими (нарушители тоже записываются, чтобы видеть последствия)."""
        state, rows = AthleteState(self.const, self.cfg), []
        for hour, key, vol in sorted(events, key=lambda e: e[0]):
            v = self.check(state, library[key], hour)
            rows.append(dict(hour=hour, key=key, allowed=v.allowed, reason=v.text()))
            state.record(library[key], hour, vol)
        return pd.DataFrame(rows)


def hour_label(h: float) -> str:
    return f"{WD[int(h // 24) % 7]} {int(h % 24):02d}:00"


# ═════════════════════════════ 5. PRIOR — популяционные априоры (ковариация видов, кинетика, штрафы 2-го дня) ═════════════════════════════
# ---- 2.1 Ковариация (корреляция) видов ----------------------------------------------------------------------
def nearest_corr(C: np.ndarray, eps: float = 1e-3) -> np.ndarray:
    """Проекция на ближайшую положительно-определённую корреляционную матрицу (обрезка собственных чисел)."""
    C = (C + C.T) / 2
    w, V = np.linalg.eigh(C)
    C = (V * np.maximum(w, eps)) @ V.T
    d = np.sqrt(np.diag(C))
    C = C / np.outer(d, d)
    np.fill_diagonal(C, 1.0)
    return C


def build_event_corr(overrides: Optional[Dict[Tuple[str, str], float]] = None, source: str = "synth") -> np.ndarray:
    """Корреляции очков между 10 видами (популяция десятиборцев). Порядок строк/столбцов — EVENTS."""
    idx = {e: i for i, e in enumerate(EVENTS)}
    C = np.full((10, 10), 0.20)  # фон: общий «уровень подготовленности» даёт слабую положительную связь

    def put(a, b, r):
        C[idx[a], idx[b]] = C[idx[b], idx[a]] = r

    # [USER] — из ТЗ (не проверялись по первоисточнику)
    put("100m", "LJ", .83); put("SP", "DT", .78); put("100m", "110mH", .75)
    put("SP", "1500m", -.35); put("HJ", "SP", -.05)
    # [EXPERT] — экспертное заполнение остальных связок (заменить эмпирической оценкой)
    for a, b, r in [("100m", "400m", .70), ("LJ", "110mH", .65), ("400m", "110mH", .60), ("HJ", "LJ", .55),
                    ("HJ", "110mH", .50), ("HJ", "PV", .40), ("PV", "110mH", .45), ("PV", "LJ", .50),
                    ("PV", "100m", .45), ("JT", "SP", .60), ("JT", "DT", .55), ("1500m", "400m", .45),
                    ("1500m", "100m", -.05), ("1500m", "LJ", -.05), ("1500m", "DT", -.25), ("1500m", "JT", -.15),
                    ("SP", "100m", .10), ("SP", "LJ", .10), ("DT", "100m", .05), ("DT", "HJ", -.05),
                    ("JT", "HJ", 0.05), ("PV", "SP", .10), ("PV", "JT", .15), ("1500m", "HJ", -.10)]:
        put(a, b, r)
    for k, v in (overrides or {}).items():
        put(k[0], k[1], v)
    np.fill_diagonal(C, 1.0)
    C = nearest_corr(C)
    if source == "prior":
        return C
    if source == "synth":
        return synth_corr()
    if source == "new":
        return nearest_corr(lit2_corr_points())
    if source == "old":
        return nearest_corr(lit_corr_points())
    if source == "blend":      # опубликованная матрица (Олимпиады 1952–60) и экспертно-ТЗ-матрица смешиваются в z-пространстве Фишера 50/50
        return fisher_blend([lit_corr_points(), C], [1.0, 1.0])
    if source == "lit":
        return nearest_corr(lit_corr_points())
    return C


def estimate_corr_from_protocols(points: pd.DataFrame, shrink: float = 0.3,
                                 prior: Optional[np.ndarray] = None) -> np.ndarray:
    """Эмпирическая корреляция из базы протоколов: points — DataFrame [атлет × 10 видов] (очки, колонки = EVENTS).
    Усадка к prior (shrink) стабилизирует оценку при малой выборке."""
    S = np.corrcoef(points[list(EVENTS)].to_numpy(float), rowvar=False)
    P = build_event_corr() if prior is None else prior
    return nearest_corr((1 - shrink) * S + shrink * P)


@dataclass
class ProfilePrior:
    """Гауссов априор профиля очков атлета: p ~ N(level·1, Σ), Σ = D·C·D.
    level — «уровень» атлета (иерархический параметр), D — SD очков вида вокруг уровня внутри атлета (заглушка)."""
    C: np.ndarray = field(default_factory=build_event_corr)
    sd_pop: Dict[str, float] = field(default_factory=lambda: {
        "100m": 70, "LJ": 80, "SP": 110, "HJ": 80, "400m": 70, "110mH": 80, "DT": 110, "PV": 110, "JT": 110, "1500m": 110})

    def shrink(self, pts: Dict[str, float], noise_sd: Optional[Dict[str, float]] = None, alpha: float = 0.05):
        """Байесовская усадка наблюдаемого профиля к ковариации популяции.
        Постериор: (Σ⁻¹+N⁻¹)⁻¹(Σ⁻¹μ+N⁻¹p). noise_sd — неопределённость очков каждого вида (давность/условия замера):
        чем она меньше, тем сильнее данные атлета «пробивают» популяционный регуляризатор.
        Возвращает (таблица, d², p-value совместимости профиля с популяцией)."""
        p = np.array([pts[e] for e in EVENTS], float)
        D = np.diag([self.sd_pop[e] for e in EVENTS])
        Sig = D @ self.C @ D
        Nsd = np.array([(noise_sd or {}).get(e, 60.0) for e in EVENTS], float)
        N = np.diag(Nsd ** 2)
        mu = np.full(10, p.mean())
        Si, Ni = np.linalg.inv(Sig), np.linalg.inv(N)
        post = np.linalg.solve(Si + Ni, Si @ mu + Ni @ p)
        d = p - mu
        d2 = float(d @ np.linalg.solve(Sig + N, d))  # расстояние Махаланобиса профиля до популяции (при данном уровне)
        pval = float(1 - chi2.cdf(d2, df=9))  # 10 видов − 1 (уровень оценён по данным)
        out = pd.DataFrame({"event": EVENTS, "pts_obs": p.round(0), "pts_post": post.round(0)})
        out["delta"] = (out.pts_post - out.pts_obs).round(0)
        out["flag"] = np.where(np.abs(out.delta) > 1.5 * Nsd * 0.5, "← сдвинут", "")
        return out, d2, pval


# ---- 2.2 Популяционные априоры кинетики (empirical Bayes) ---------------------------------------------------
# Постоянная времени ФИТНЕСА τ1 (дни) — диапазоны из ТЗ; JMP/THR/SE — интерполяция [EXPERT]
TAU1_RANGE = {"SPR": (32, 40), "JMP": (32, 40), "THR": (35, 45), "SE": (36, 46), "END": (42, 50)}
HALF_LIFE_H = {"CNS": (24, 48), "MET": (24, 72), "MECH": (72, 96)}


@dataclass
class KineticPrior:
    """Логнормальные априоры ln τ1 (по каналам), ln T½ (по трём пулам), ln K. Диапазон трактуется как ≈ ±1 SD.
    strength>1 — жёстче (доверие популяции), <1 — мягче. Это и есть Ridge/MAP-регуляризация."""
    mu1: np.ndarray = field(default_factory=lambda: np.array([0.5 * np.log(np.prod(TAU1_RANGE[c])) for c in CHANNELS]))
    sd1: np.ndarray = field(default_factory=lambda: np.array([0.5 * np.log(TAU1_RANGE[c][1] / TAU1_RANGE[c][0]) for c in CHANNELS]))
    muH: np.ndarray = field(default_factory=lambda: np.array([0.5 * np.log(np.prod(HALF_LIFE_H[p])) for p in POOLS]))
    sdH: np.ndarray = field(default_factory=lambda: np.array([0.5 * np.log(HALF_LIFE_H[p][1] / HALF_LIFE_H[p][0]) for p in POOLS]))
    K1_median: float = 3.0   # % к опорному результату на единицу нормированной нагрузки (фитнес) [заглушка]
    K2_median: float = 2.0   # % на единицу утомления канала (h_r = Σ W·P) [заглушка]
    K_logsd: float = 0.6
    strength: float = 1.0

    def update_from_fits(self, thetas: List[np.ndarray], n0: float = 3.0) -> "KineticPrior":
        """Иерархический шаг (empirical Bayes): по θ нескольких откалиброванных атлетов уточняем гиперпараметры
        популяции (метод моментов с псевдо-наблюдениями n0 — вес исходного литературного априора)."""
        lay = Layout(ModelConfig())
        P = [lay.unpack(t) for t in thetas]
        n = len(P)
        l1, lh = np.log([p.tau1 for p in P]), np.log([p.half_h for p in P])
        new = KineticPrior(strength=self.strength, K1_median=self.K1_median, K2_median=self.K2_median, K_logsd=self.K_logsd)
        new.mu1, new.muH = (n0 * self.mu1 + n * l1.mean(0)) / (n0 + n), (n0 * self.muH + n * lh.mean(0)) / (n0 + n)
        v1 = l1.var(0, ddof=1) if n > 1 else self.sd1 ** 2
        vh = lh.var(0, ddof=1) if n > 1 else self.sdH ** 2
        new.sd1 = np.sqrt((n0 * self.sd1 ** 2 + (n - 1) * v1) / (n0 + n - 1))
        new.sdH = np.sqrt((n0 * self.sdH ** 2 + (n - 1) * vh) / (n0 + n - 1))
        return new


# ---- 2.3 Эмпирические штрафы 2-го дня ----------------------------------------------------------------------
@dataclass
class Day2Penalty:
    """Ухудшение результата 2-го дня относительно «свежего» теста, % (время — дольше, высота/дальность — меньше).
    Усталость 1-го дня F1 и общая F_tot — индексы «стоимости» дня: Σ w_e·(очки_e/800)/Σ w_e (=1 при 800 очков в
    каждом виде: чем выше результат, тем дороже он обошёлся). ВСЕ коэффициенты — стартовые [заглушки]:
    замените регрессией по протоколам (день-2 результат ~ свежий результат, F1, масса)."""
    w1: Dict[str, float] = field(default_factory=lambda: {"100m": .6, "LJ": .9, "SP": .7, "HJ": .9, "400m": 1.2})
    w2: Dict[str, float] = field(default_factory=lambda: {"110mH": .8, "DT": .6, "PV": .9, "JT": .6})
    base: Dict[str, float] = field(default_factory=lambda: {"110mH": 1.2, "PV": 3.0, "DT": 1.5, "JT": 2.0, "1500m": 4.5})
    k_f1: Dict[str, float] = field(default_factory=lambda: {"110mH": 3.0, "PV": 8.0})  # % на +1 к F1
    mass_ref: float = 75.0
    k_mass_1500: float = 0.12   # % к времени 1500 м на каждый кг массы выше mass_ref
    k_ftot_1500: float = 6.0    # % на +1 к F_tot (спринт/тоннаж за 2 дня)

    def indices(self, results: Dict[str, float]) -> Tuple[float, float]:
        eff = {e: w * wa_points(e, results[e]) / 800.0 for e, w in {**self.w1, **self.w2}.items()}
        f1 = sum(eff[e] for e in self.w1) / sum(self.w1.values())
        ft = sum(eff.values()) / (sum(self.w1.values()) + sum(self.w2.values()))
        return float(f1), float(ft)

    def pct(self, results: Dict[str, float], mass: float) -> Dict[str, float]:
        f1, ft = self.indices(results)
        p = {e: self.base[e] for e in self.base}
        for e, k in self.k_f1.items():
            p[e] += k * (f1 - 1.0)
        p["1500m"] += self.k_mass_1500 * (mass - self.mass_ref) + self.k_ftot_1500 * (ft - 1.0)
        return {e: max(v, 0.0) for e, v in p.items()}

    def multipliers(self, results: Dict[str, float], mass: float, skip=()) -> Dict[str, float]:
        """Множители к «свежему» результату. skip — виды, чьи данные уже получены во 2-й день (не штрафуем дважды)."""
        pc = self.pct(results, mass)
        return {e: (1.0 if e not in pc or e in skip else (1 + pc[e] / 100 if IS_TIME[e] else 1 - pc[e] / 100))
                for e in EVENTS}


# ═════════════════════════════ 5b. LITERATURE — данные из журнала «Лёгкая атлетика» (фото IMG_8048–8068) ═════════════════════════════
# Всё ниже переписано с фото; ячейки, в которых я не уверен, помечены в LIT_UNSURE. Это ПОПУЛЯЦИОННЫЕ данные 1952–86 гг.
# (таблица очков 1962 г. для 1952–84; 1985 г. — для 1986): они калибруют/проверяют модель, но не заменяют данные атлета.
TIME_EV = ("100m", "400m", "110mH", "1500m")
# Табл. 1 «Математика и десятиборье» (Олимпиады 1952/56/60): корреляции РЕЗУЛЬТАТОВ (время: меньше = лучше). Порядок строк = EVENTS.
_T = ["100m", "LJ", "SP", "HJ", "400m", "110mH", "DT", "PV", "JT", "1500m"]
LIT_CORR_RESULTS_ROWS = {  # строка: значения против предыдущих видов в порядке _T
    "LJ": [-.60], "SP": [-.45, .51], "HJ": [-.46, .47, .40], "400m": [.69, -.45, -.28, -.42],
    "110mH": [.56, -.51, -.56, -.52, .45], "DT": [-.39, .44, .77, .23, -.23, -.49],
    "PV": [-.08, .30, .32, .19, -.06, -.19, .35], "JT": [-.39, .47, .61, .30, -.33, -.29, .47, .56],
    "1500m": [-.05, -.20, -.16, -.02, .42, .07, -.13, -.17, -.03]}
LIT_CORR_SUM = {"100m": -.71, "LJ": .75, "SP": .74, "HJ": .62, "400m": -.73, "110mH": -.71, "DT": .61, "PV": .53, "JT": .72, "1500m": -.19}
LIT_UNSURE = {("JT", "100m"): "цифра −0,39 или −0,89; принято −0,39 (0,89 невозможно при r(сумма,100м)=−0,71)",
              "Narimanov bench max": "в таблице напечатано «50»; по крайним значениям КД = 150",
              "Filin speed endurance": "II/III разряд 2,15/2,11 — немонотонно, как напечатано",
              "Soviet 100m 1972 (Kudu T1 p.12)": "10,30 или 10,80 — не использовано"}
LIT_NONSIG = {("DT", "HJ"), ("DT", "400m"), ("PV", "HJ"), ("PV", "400m"), ("PV", "110mH")} | {("1500m", e) for e in ("100m", "LJ", "SP", "HJ", "110mH", "DT", "PV", "JT")}
LIT_MULTI_R = {"SP+400m+100m": .918, "SP+400m+DT": .917, "SP+400m+LJ": .923, "SP+400m+110mH": .930, "SP+400m+HJ": .932,
               "SP+400m+1500m": .919, "SP+400m+JT": .944, "SP+400m+PV": .971}   # две лучшие пары: 400м+ядро r(сумма)=0,916


def lit_corr_points() -> np.ndarray:
    """Корреляция ОЧКОВ: у времён знак результата противоположен знаку очков, поэтому пары «время–поле» меняют знак."""
    ix = {e: i for i, e in enumerate(EVENTS)}
    C = np.eye(10)
    for ev, vals in LIT_CORR_RESULTS_ROWS.items():
        for other, r in zip(_T, vals):
            flip = (ev in TIME_EV) != (other in TIME_EV)
            C[ix[ev], ix[other]] = C[ix[other], ix[ev]] = -r if flip else r
    return C


def fisher_blend(Cs: Sequence[np.ndarray], w: Sequence[float]) -> np.ndarray:
    z = sum(wi * np.arctanh(np.clip(C - np.eye(10) * 1e-9, -.999, .999)) for wi, C in zip(w, Cs)) / sum(w)
    B = np.tanh(z); np.fill_diagonal(B, 1.0)
    return nearest_corr(B)


# ═════════════════════ 5c. LITERATURE 2 — Юшкевич Ю., Романов Н. «Мир спорта» № 3 (48), 2012 (фото экрана) ═════════════════════
# Переписано с фото. Табл. 1 — корреляции РЕЗУЛЬТАТОВ (время: меньше = лучше) у сильнейших десятиборцев мира; Табл. 2/3 — множественные R с суммой очков
# для сочетаний 2 и 3 видов; рис. 1 — рост результата в прыжках от часов подготовки. «Достоверны только коэффициенты выше 0,21» ⇒ выборка ≈ 90.
_ORD2 = ["100m", "LJ", "SP", "HJ", "400m", "110mH", "DT", "PV", "JT", "1500m"]       # порядок столбцов в таблице
LIT2_ROWS = {   # строка → значения против ПОСЛЕДУЮЩИХ видов в порядке _ORD2
    "100m": [-.46, -.06, -.13, .40, .54, -.07, .14, .24, -.01], "LJ": [.03, .11, -.13, -.24, -.06, -.12, -.02, -.15],
    "SP": [.19, .14, .08, .61, -.18, .11, -.08], "HJ": [.12, .00, .08, -.24, -.07, -.12], "400m": [.23, .22, .11, .16, .05],
    "110mH": [.03, -.13, .11, -.08], "DT": [-.25, .12, -.07], "PV": [-.10, .15], "JT": [-.21]}
LIT2_SUM = {"100m": -.48, "LJ": .49, "SP": .43, "HJ": .24, "400m": -.28, "110mH": -.44, "DT": .35, "PV": .08, "JT": .31, "1500m": -.22}
LIT2_PAIRS = {("110mH", "SP"): .641, ("100m", "SP"): .628, ("100m", "DT"): .580, ("110mH", "DT"): .573, ("110mH", "JT"): .566, ("400m", "SP"): .545}
LIT2_TRIPLES = {("110mH", "SP", "LJ"): .745, ("110mH", "SP", "HJ"): .661, ("110mH", "SP", "PV"): .649, ("400m", "SP", "PV"): .586, ("400m", "SP", "HJ"): .581,
                ("400m", "HJ", "PV"): .568}
LIT2_N, LIT_N_OLD, LIT_N_PRIOR = 90, 57, 12     # n: из порога значимости (0,21 → ≈90; у старой таблицы порог между 0,23 и 0,29 → ≈57); у экспертной матрицы — вес «12 наблюдений»
LIT_REL_OLD = 0.25                               # пригодность Олимпиад 1952–60 (МОЁ допущение, но обоснованное: у них R сочетаний с суммой 0,77–0,98 против 0,55–0,75 в 2012 — гораздо шире разброс уровня,
                                                 # а для разброса результатов ОДНОГО спортсмена в турнире нужна корреляция при узком диапазоне уровня; плюс иная таблица очков и бамбуковый шест)
LIT2_UNSURE = {("400m", "HJ", "PV"): "по Табл. 1 это сочетание даёт R=0,44, а не 0,568; зато «400 м – высота – диск» даёт 0,566 — вероятна ошибка в подписи (шест ↔ диск) или в таблице; "
                                     "для плана не используется до проверки по оригиналу",
               "photo": "фото кривое: клетки Табл. 1, не участвующие в Табл. 2/3 (например 100м–длина −0,46, ядро–диск 0,61, диск–шест −0,25, копьё–1500 −0,21), не перепроверены"}
_TIME_EV = {"100m", "400m", "110mH", "1500m"}


def _sgn(ev: Sequence[str]) -> np.ndarray:
    return np.array([-1.0 if e in _TIME_EV else 1.0 for e in ev])


def lit2_corr_results() -> np.ndarray:
    """Табл. 1 как матрица корреляций РЕЗУЛЬТАТОВ в порядке EVENTS."""
    ix = {e: i for i, e in enumerate(EVENTS)}
    C = np.eye(10)
    for ev, vals in LIT2_ROWS.items():
        for other, r in zip(_ORD2[_ORD2.index(ev) + 1:], vals):
            C[ix[ev], ix[other]] = C[ix[other], ix[ev]] = r
    return C


def lit2_corr_points() -> np.ndarray:
    """Корреляции ОЧКОВ (для времён знак меняется) по Табл. 1."""
    s = _sgn(EVENTS)
    return lit2_corr_results() * np.outer(s, s)


def sum_corr_points(src: str) -> np.ndarray:
    """Корреляции очков каждого вида с суммой очков (в порядке EVENTS): src = 'new' (2012) или 'old' (1952–60)."""
    d = LIT2_SUM if src == "new" else LIT_CORR_SUM
    return np.array([d[e] for e in EVENTS]) * _sgn(EVENTS)


def corr_prior_matrix() -> np.ndarray:
    """Матрица из ТЗ и экспертных оценок (прежний prior) — вес малый."""
    return build_event_corr(source="prior")


def synth_corr(w_new: float = 1.0, w_old: float = 1.0, w_prior: float = 1.0, return_weights: bool = False):
    """СИНТЕЗ трёх источников в z-пространстве Фишера. Вес источника = (n − 3) × пригодность × множитель пользователя:
    2012 (сильнейшие десятиборцы, n≈90, пригодность 1,0), Олимпиады 1952–60 (n≈57, пригодность 0,25), ТЗ/экспертная матрица (вес 12−3).
    Дисперсия z = 1/(n−3), поэтому это обратно-дисперсионное усреднение; пригодность старой таблицы — моё допущение."""
    W = np.array([w_new * (LIT2_N - 3), w_old * (LIT_N_OLD - 3) * LIT_REL_OLD, w_prior * (LIT_N_PRIOR - 3)], float)
    Cs = [lit2_corr_points(), lit_corr_points(), corr_prior_matrix()]
    z = sum(w * np.arctanh(np.clip(C - np.eye(10) * 1e-9, -.999, .999)) for w, C in zip(W, Cs)) / W.sum()
    B = np.tanh(z)
    np.fill_diagonal(B, 1.0)
    C = nearest_corr(B)
    return (C, W / W.sum()) if return_weights else C


def synth_sum_corr(w_new: float = 1.0, w_old: float = 1.0) -> np.ndarray:
    W = np.array([w_new * (LIT2_N - 3), w_old * (LIT_N_OLD - 3) * LIT_REL_OLD])
    z = (W[0] * np.arctanh(np.clip(sum_corr_points("new"), -.99, .99)) + W[1] * np.arctanh(np.clip(sum_corr_points("old"), -.99, .99))) / W.sum()
    return np.tanh(z)


def multiple_R(C: np.ndarray, s: np.ndarray, events: Sequence[str]) -> float:
    """Множественный R суммы очков с набором видов: √(r' C⁻¹ r), где r — корреляции видов с суммой (по очкам)."""
    k = [EVENTS.index(e) for e in events]
    r = s[k]
    return float(np.sqrt(max(0.0, r @ np.linalg.solve(C[np.ix_(k, k)], r))))


def corr_report() -> pd.DataFrame:
    """Сверка: опубликованные R для сочетаний из Табл. 2/3 против значений, воспроизведённых из матрицы (Табл. 1 / старая / синтез)."""
    Cn, sn = lit2_corr_points(), sum_corr_points("new")
    Co, so = lit_corr_points(), sum_corr_points("old")
    Cs, ss = synth_corr(), synth_sum_corr()
    rows = []
    for ev, R in {**LIT2_PAIRS, **LIT2_TRIPLES}.items():
        rows.append(dict(event=ev, Сочетание=" + ".join(ev), Опубликовано=R, **{"Табл. 1 (2012)": round(multiple_R(Cn, sn, ev), 3), "Старая (1952–60)": round(multiple_R(Co, so, ev), 3),
                                                                           "Синтез": round(multiple_R(Cs, ss, ev), 3)}))
    return pd.DataFrame(rows)


def corr_compare() -> pd.DataFrame:
    """Все 45 пар: 2012, 1952–60, ТЗ/эксперт и синтез (корреляции очков) — чтобы видеть, где источники расходятся."""
    A, B, P, S = lit2_corr_points(), lit_corr_points(), corr_prior_matrix(), synth_corr()
    rows = []
    for i in range(10):
        for j in range(i + 1, 10):
            rows.append(dict(a=EVENTS[i], b=EVENTS[j], Пара=f"{EVENTS[i]} – {EVENTS[j]}", **{"2012": round(A[i, j], 2), "1952–60": round(B[i, j], 2), "ТЗ/эксперт": round(P[i, j], 2),
                                                                                         "Синтез": round(S[i, j], 2), "Разброс": round(max(A[i, j], B[i, j], P[i, j]) - min(A[i, j], B[i, j], P[i, j]), 2)}))
    return pd.DataFrame(rows).sort_values("Разброс", ascending=False).reset_index(drop=True)


# Рис. 1 той же статьи: прирост результата в прыжках (очки) от часов тренировочного времени в этом виде: y = α·ᵏ√t, на графике подписано y = 60·⁵√(t²) = 60·t^0,4.
# НО: подписанная формула расходится с самой статьёй: по ней 50 ч → 287 очков, тогда как в тексте «первые 50 часов — около 250 очков, следующие 50 — около 100»,
# и точки на графике (снял с фото) лежат ниже кривой при малых t. Поэтому по умолчанию — подгонка по тексту и точкам (a≈38, p≈0,48); подписанная формула оставлена для сравнения.
HOURS_PUBLISHED = dict(a=60.0, p=0.4)
HOURS_FIG_PTS = [(15, 145), (30, 200), (45, 250), (63, 300), (82, 335), (105, 375), (155, 440), (208, 490)]   # точки с рисунка (±10 очков)
HOURS_TEXT = [(50, 250.0), (100, 350.0)]                                                                    # из текста: 50 ч → 250; ещё 50 ч → +100


def _fit_hours() -> Dict[str, float]:
    t = np.array([q[0] for q in HOURS_FIG_PTS] + [q[0] for q in HOURS_TEXT], float)
    y = np.array([q[1] for q in HOURS_FIG_PTS] + [q[1] for q in HOURS_TEXT], float)
    w = np.array([1.0] * len(HOURS_FIG_PTS) + [3.0] * len(HOURS_TEXT))          # текст надёжнее моей оцифровки
    p_, ln_a = np.polyfit(np.log(t), np.log(y), 1, w=np.sqrt(w))
    return dict(a=float(np.exp(ln_a)), p=float(p_))


HOURS_LAW = _fit_hours()                  # a — «крутизна»; для других видов (не прыжков) в статье не приведена → допущение: тот же закон


def hours_gain(t, a: float = HOURS_LAW["a"], p: float = HOURS_LAW["p"]):
    """Ожидаемый прирост очков за t часов подготовки в виде (закон убывающей отдачи)."""
    return a * np.power(np.maximum(t, 0.0), p)


def hours_marginal(t, a: float = HOURS_LAW["a"], p: float = HOURS_LAW["p"]):
    """Прирост очков от одного дополнительного часа на уровне t часов: a·p·t^(p−1)."""
    return a * p * np.power(np.maximum(t, 1.0), p - 1.0)


def hours_fit_check() -> pd.DataFrame:
    """Сверка двух вариантов закона с точками рисунка и текстом статьи."""
    rows = []
    for nm, par in (("подписанная формула 60·t^0,4", HOURS_PUBLISHED), ("подгонка (по умолчанию)", HOURS_LAW)):
        pr = [float(hours_gain(t, **par)) for t, _ in HOURS_FIG_PTS]
        rm = float(np.sqrt(np.mean([(a - b[1]) ** 2 for a, b in zip(pr, HOURS_FIG_PTS)])))
        rows.append({"Вариант": nm, "a": round(par["a"], 1), "p": round(par["p"], 3), "50 ч": round(float(hours_gain(50.0, **par))), "следующие 50 ч": round(float(hours_gain(100.0, **par) - hours_gain(50.0, **par))),
                     "200 ч": round(float(hours_gain(200.0, **par))), "СКО по точкам рисунка": round(rm, 1)})
    rows.append({"Вариант": "в тексте статьи", "a": None, "p": None, "50 ч": 250, "следующие 50 ч": 100, "200 ч": None, "СКО по точкам рисунка": None})
    return pd.DataFrame(rows)


# Heinla (Таллин, 1986/87): нормы роста и потерь
LIT_HEINLA = dict(
    cap_two_seasons=[(7300, (600, 650)), (7600, (450, 500)), (7800, (300, 300))],   # (до класса, допустимый прирост за 2 сезона)
    normal_loss=(295.0, 90.0),                      # ΣЛР по видам − реальная сумма; обычно 200–400
    loss_range=(200, 400), big_gap=500,
    growth_groupV={2: (626, 187), 3: (832, 208), 4: (995, 172)},     # сезонов «семитысячника» → прирост очков (M, SD)
    growth_usa34={2: 653, 3: 811, 4: 934}, usa_start=6829,
    age_at_level_B=((7000, 20.3), (7500, 21.9), (7800, 23.1)), age_at_level_V=((7000, 19.9), (7500, 20.9), (7950, 21.1)),
    r_sum_vs_sum_of_PR=.89, r_day1_sum=.62, r_day2_sum=.79, r_400m_loss=.38, r_weight_SP=.84,
    to_8000_seasons=(3, 4))


def heinla_cap(points: float, seasons: float = 2.0) -> Tuple[float, float]:
    """Допустимый прирост очков за `seasons` сезонов для атлета с данной суммой (линейно пересчитано от 2-сезонных норм)."""
    for lim, (lo, hi) in LIT_HEINLA["cap_two_seasons"]:
        if points < lim:
            return lo * seasons / 2, hi * seasons / 2
    return 0.0, 300.0 * seasons / 2


# Нариманов (ВНИИФК): тесты по уровням суммы очков и критерии КД1/КД2
NAR_LEVELS = (4000, 5000, 6000, 7000, 7500, 8000, 8500, 9000)
NAR = {  # тест: (значения по уровням, теор. min, теор. max)
    "vmax": ([8.8, 9.2, 9.7, 10.0, 10.2, 10.5, 10.7, 11.0], 6.0, 12.0),
    "vo2": ([3.00, 3.50, 4.30, 4.60, 4.80, 5.00, 5.10, 5.20], 2.0, 7.0),
    "hxm": ([122.5, 135.0, 144.0, 155.7, 166.5, 166.5, 168.3, 171.0], 80.0, 200.0),
    "triple_sj": ([7.80, 8.10, 8.70, 9.00, 9.29, 9.40, 9.60, 10.00], 5.0, 11.0),
    "shot_bwd": ([10.0, 12.0, 14.0, 15.0, 15.5, 16.0, 16.5, 17.0], 5.0, 20.0),
    "bench": ([50, 60, 70, 90, 100, 110, 120, 130], 30.0, 150.0),
    "t1000": ([190, 185, 180, 175, 170, 170, 165, 160], 400.0, 140.0),
    "KD1": ([37.8, 52.6, 55.8, 62.0, 64.0, 64.5, 67.9, 69.2], 28.0, 82.0),
    "KD2": ([22.3, 36.4, 38.0, 43.0, 45.5, 46.5, 49.4, 50.0], 4.0, 61.0)}


def kd1(vmax, vo2, height_m, mass_kg, triple_m, shot_m, bench_kg):
    return vmax + 2 * vo2 + 0.07 * height_m * mass_kg + 1.2 * triple_m + 0.7 * shot_m + 0.1 * bench_kg


def kd2(vmax, height_m, mass_kg, triple_m, shot_m, bench_kg, t1000_s):
    return vmax + 0.07 * height_m * mass_kg + 1.2 * triple_m + 0.7 * shot_m + 0.1 * bench_kg - 0.05 * t1000_s


def nar_level(test: str, value: float) -> float:
    """Уровень суммы очков, которому соответствует значение теста (линейная интерполяция по таблице, вне — с зажимом)."""
    v = np.array(NAR[test][0], float); L = np.array(NAR_LEVELS, float)
    if v[-1] < v[0]:
        v, L = v[::-1], L[::-1]
    return float(np.interp(value, v, L))


# Филин: тесты по разрядам (МС, КМС, I, II, III)
FILIN_RANKS = ("МС", "КМС", "I", "II", "III")
FILIN = {"fly20_s": [1.84, 1.93, 1.98, 2.06, 2.18], "start30_s": [3.91, 3.97, 4.03, 4.16, 4.40],
         "HJ_standing_cm": [73.9, 68.4, 67.7, 65.2, 58.8], "TJ_standing_cm": [901.5, 892, 863.4, 837, 773.7],
         "shot_fwd_m": [14.03, 13.71, 12.84, 11.54, 10.14], "shot_bwd_m": [14.94, 14.73, 13.58, 12.28, 10.54],
         "reactivity": [3.36, 2.83, 2.63, 2.39, 2.15], "general_endurance_s": [7.26, 7.87, 7.95, 8.36, 8.58]}

# Средние результаты десятиборья: 3 сильнейших зарубежных / 3 советских в олимпийских циклах (Куду, стр.12 табл.1; ТАБЛИЦА 1962)
ELITE_YEARS = (1960, 1964, 1968, 1972, 1976, 1980, 1984)
ELITE_ABROAD = {"100m": [11.04, 11.10, 10.71, 11.00, 10.94, 11.07, 10.78], "LJ": [7.25, 7.14, 7.72, 7.19, 7.37, 7.78, 7.79],
                "SP": [14.53, 14.40, 14.54, 14.31, 14.96, 14.61, 15.85], "HJ": [1.83, 1.91, 1.92, 1.97, 2.06, 2.11, 2.08],
                "400m": [49.05, 48.98, 47.69, 48.66, 47.95, 48.65, 48.31], "110mH": [15.20, 15.04, 15.14, 14.58, 14.54, 14.78, 14.41],
                "DT": [46.28, 44.46, 44.66, 49.94, 47.11, 41.63, 48.52], "PV": [4.10, 4.13, 4.36, 4.48, 4.61, 4.73, 4.86],
                "JT": [69.72, 59.77, 69.94, 60.18, 65.70, 63.18, 62.96]}
# Сводка 10 сильнейших 1976–84 (стр. 8052): (мир, СССР) по годам 1976,80,81,82,83,84
ELITE10_YEARS = (1976, 1980, 1981, 1982, 1983, 1984)
ELITE10_TOTAL = {"world": [8203, 8319, 8217, 8370, 8468, 8477], "ussr": [7964, 8208, 8101, 8200, 8290, 8430]}
# Олимпийцы по таблице 1962 (Куду, стр.14): средний итог
OLYMPIC_MEAN_1962 = {1952: 7122.83, 1956: 7310.16, 1960: 7597.66, 1964: 7777.33, 1968: 7976.33, 1972: 8057.83, 1976: 8296.16}
OLYMPIC_MEAN_EVENTS = {  # Куду, стр.14 табл.2 (средние по Олимпиадам)
    1960: dict(zip(EVENTS, [11.1, 7.10, 13.92, 1.82, 50.0, 15.0, 47.05, 4.02, 65.64, 288.1])),
    1964: dict(zip(EVENTS, [11.0, 7.03, 14.11, 1.88, 49.1, 15.2, 43.71, 4.21, 61.50, 271.8])),
    1968: dict(zip(EVENTS, [10.7, 7.63, 14.65, 1.97, 49.0, 15.0, 45.21, 4.31, 65.11, 308.5])),   # 1968: 1500 м и длина прочитаны как напечатано (5.08,5 / 7,63) — подозрительно
    1972: dict(zip(EVENTS, [10.92, 7.26, 14.11, 1.96, 48.6, 14.87, 44.43, 4.51, 58.19, 264.7])),
    1976: dict(zip(EVENTS, [11.02, 7.23, 14.88, 2.00, 48.1, 14.97, 46.07, 4.62, 68.48, 268.2]))}
# Куду, табл.3: 10 сильнейших Монреаля (рост, вес, год 1-го выступления, число стартов, лучший результат)
KUDU_MONTREAL = dict(height_cm=187.6, mass_kg=88.3, years_to_peak=7, starts=26, best_pts=8129)
# Ушаков/Верхошанский: балансы дня 1 / дня 2 (без 1500 м) и % (стр. 8057, табл.2)
DAY_BALANCE = {"USSR_mean": (4154, 3298, 79.45), "World_mean": (4535, 3270, 72.14)}
# Ушаков, 1977: структура года и нагрузки
LIT_PERIODIZATION = dict(prep_weeks=31, prep_autumn_winter=21, prep_spring=10, competition_weeks=18, transition_weeks=3,
                         mesocycle_weeks=3, sessions_per_event_year=(80, 110), sprint_hurdle_sessions=(90, 120),
                         annual_volume_growth=(.10, .15), intense_zone_growth=(.15, .20),
                         starts_decathlon_year=(4, 5), starts_pentathlon_summer=(5, 7), winter_reduced_multi=(2, 3),
                         hours_elite=(1000, 1300), sessions_elite=(550, 650),
                         jenner_1975=[7665, 8138, 8058, 8524, 8045], tumi_1969_range=(8075, 8417))
# Недельные микроциклы (Ушаков/Верхошанский): порядок вида в тренировочном дне (1 = первым). Вс — отдых.
WEEK_MODEL = {
    "prep": {"Пн": ["LJ", "SP", "HJ", "JT", "run"], "Вт": ["110mH", "DT", "PV", "weights", "run"], "Ср": ["DT", "run", "JT", "jump_drills"],
             "Чт": ["LJ", "SP", "PV"], "Пт": ["DT", "HJ", "110mH", "JT", "run"], "Сб": ["110mH", "DT", "weights", "run"]},
    "precomp": {"Пн": ["sprint", "LJ", "SP", "HJ"], "Вт": ["110mH", "SP", "PV", "run"], "Ср": ["throw*", "weights", "run"],
                "Чт": ["LJ", "sprint", "run", "jump_drills", "DT"], "Пт": ["110mH", "PV", "DT", "run"], "Сб": ["throw*", "jump_drills"]}}
# Грузенкин: «цена 10 см/1 м» в таблице 1985 — проверочные значения
GRUZENKIN = dict(PV_per10cm=(29, 32), LJ_1000pts_cm=776, DT_900pts_m=51.40, LJ_7p10_8p10_per10cm=(24, 26), DT_39_51_per_m=(20, 21))
# Портной (1986): модель мирового рекорда следующего поколения (>8900)
PORTNOY_RECORD = dict(zip(EVENTS, [10.60, 7.85, 16.00, 2.10, 47.40, 14.00, 48.60, 4.90, 66.00, 261.0]))
# Зациорский–Годик: x1 = 3,1·x2 − 218·x3 + 10·x4 + 10060 (x2 ядро, см; x3 400 м, с; x4 шест, см), ошибка ≤225 в 12 из 18 случаев
def zatsiorsky_sum(sp_m: float, t400_s: float, pv_m: float) -> float:
    return 3.1 * sp_m * 100 - 218 * t400_s + 10 * pv_m * 100 + 10060


# ---- Таблица очков 1962 (реконструкция по двум опорным точкам; ошибка вида ≈ −15…−30 очков суммы) ----
P62 = {"100m": (15.7, 10.2), "LJ": (359, 790), "SP": (4.70, 18.75), "HJ": (113, 217), "400m": (78.10, 46.00),
       "110mH": (32.0, 13.70), "DT": (12.81, 57.50), "PV": (149, 478), "JT": (14.02, 81.00), "1500m": (399.7, 220.2)}
_DIST = {"100m": 100.0, "400m": 400.0, "110mH": 110.0, "1500m": 1500.0}


def pts62(event: str, x: float, floor: bool = True) -> float:
    """Очки по таблице 1962 (бег — линейно по скорости, поле — линейно по √результата). x: с / м (электронное время)."""
    x0, x1 = P62[event]
    if event in _DIST:
        v, v0, v1 = _DIST[event] / x, _DIST[event] / x0, _DIST[event] / x1
        p = 1000 * (v - v0) / (v1 - v0)
    else:
        u = x * (100.0 if event in ("LJ", "HJ", "PV") else 1.0)
        p = 1000 * (np.sqrt(u) - np.sqrt(x0)) / (np.sqrt(x1) - np.sqrt(x0))
    p = max(float(p), 0.0)
    return float(int(p)) if floor else p


def total62(res: Dict[str, float], floor: bool = False) -> float:
    return sum(pts62(e, res[e], floor) for e in EVENTS)


def total85(res: Dict[str, float], floor: bool = True) -> float:
    return float(sum(wa_points(e, res[e], floor=floor) for e in EVENTS))


def growth_check(forecast: float, baseline: float, horizon_days: int) -> dict:
    """Согласованность прогноза с нормами Хейнлы: прирост относительно суммы личных рекордов и норма потерь 295±90."""
    lo, hi = heinla_cap(baseline, 2.0)
    gain = forecast - baseline
    scale = horizon_days / 365.0 * 2 / 2.0     # доля 2-сезонного лимита, «разрешённая» длиной горизонта при равномерном росте
    mu, sd = LIT_HEINLA["normal_loss"]
    return dict(baseline=baseline, forecast=forecast, gain=gain, cap_2seasons=(lo, hi),
                cap_for_horizon=(lo * horizon_days / 730.0, hi * horizon_days / 730.0),
                implied_PR_sum=forecast + mu, implied_PR_growth=forecast + mu - baseline,
                normal_realised=(baseline - mu, sd))


def lit_selftest():
    """Проверки корректности переписанных данных (не доказательство, но ловят опечатки в сотнях цифр)."""
    C = lit_corr_points()
    assert np.allclose(C, C.T) and np.all(np.abs(C) <= 1)
    # 1) формулы КД против крайних значений таблицы
    k1min = kd1(6, 2, 1, 80, 5, 5, 30) ; k1max = kd1(12, 7, 1.0, 200, 11, 20, 150)
    assert abs(k1min - 28.0) < 1 and abs(k1max - 82.0) < 1, (k1min, k1max)
    k2min = kd2(6, 1, 80, 5, 5, 30, 400); k2max = kd2(12, 1, 200, 11, 20, 150, 140)
    assert abs(k2min - 4.0) < 1 and abs(k2max - 61.0) < 1.5, (k2min, k2max)
    # 2) таблица Грузенкина против формул 1985 г.
    pv = [wa_points("PV", (4.20 + .10 * i + .10) ) - wa_points("PV", 4.20 + .10 * i) for i in range(12)]
    assert all(28 <= d <= 33 for d in pv), pv
    assert abs(wa_points("LJ", 7.76) - 1000) < 8 and abs(wa_points("DT", 51.40) - 900) < 6
    # 3) олимпийские средние (табл.2) через реконструкцию 1962 против табл.1 (кроме 1968: подозрительные ячейки)
    for y, tot in OLYMPIC_MEAN_1962.items():
        if y in OLYMPIC_MEAN_EVENTS and y != 1968:
            d = total62(OLYMPIC_MEAN_EVENTS[y]) - tot
            assert abs(d) < 45, (y, d)
    # 4) 1985: проверка Эдмонтона 2001
    r = dict(zip(EVENTS, [10.62, 7.73, 15.60, 2.03, 47.17, 14.46, 46.89, 5.10, 72.70, 273.0]))
    assert total85({**r}) > 8400
    print("литературные данные: внутренние проверки пройдены (КД, Грузенкин, олимпийские средние 1960–76)")


def run_benchmarks(a=None):
    """Сопоставление текущего профиля атлета с литературой: обе таблицы, классы, нормы Хейнлы, Нариманов, Филин, элита."""
    pd.set_option("display.width", 170)
    ath = samandyk()
    cur = ath.current
    rows = [dict(event=e, result=cur[e], pts1985=wa_points(e, cur[e], floor=True), pts1962=pts62(e, cur[e]),
                 elite1984_abroad=ELITE_ABROAD[e][-1] if e in ELITE_ABROAD else np.nan,
                 portnoy_record=PORTNOY_RECORD[e]) for e in EVENTS]
    df = pd.DataFrame(rows)
    df["gap_to_elite_pts1985"] = [wa_points(e, ELITE_ABROAD[e][-1], floor=True) - p if e in ELITE_ABROAD else np.nan
                                  for e, p in zip(EVENTS, df.pts1985)]
    print(f"ПРОФИЛЬ {ath.name}: сумма текущих результатов — по таблице 1985: {df.pts1985.sum():.0f}, по реконструкции 1962: {df.pts1962.sum():.0f}")
    print(df.to_string(index=False))
    d1 = sum(df.pts1985[:5]); d2 = sum(df.pts1985[5:9])
    print(f"\nБаланс дней (1985): день 1 = {d1:.0f}, день 2 без 1500 м = {d2:.0f} → {100 * d2 / d1:.1f} %  "
          f"(мир {DAY_BALANCE['World_mean'][2]} %, СССР {DAY_BALANCE['USSR_mean'][2]} %)")
    base = float(df.pts1985.sum())
    lo, hi = heinla_cap(base, 2.0)
    print(f"Хейнла: класс {base:.0f} → допустимый прирост за 2 сезона {lo:.0f}–{hi:.0f}; нормальная потеря от ΣЛР {LIT_HEINLA['normal_loss'][0]:.0f}±{LIT_HEINLA['normal_loss'][1]:.0f}")
    t1000 = 167.0
    print(f"Нариманов: 1000 м 2:47 → уровень ≈ {nar_level('t1000', t1000):.0f} очков (по таблице тестов; остальные тесты — нужны: vmax, МПК, тройной с места, ядро назад, жим, рост×масса)")
    print(f"Зациорский–Годик (модель 1952–60, ошибка ≤225 в 2/3 случаев): ядро {cur['SP']:.2f}, 400 м {cur['400m']:.2f}, шест {cur['PV']:.2f} → "
          f"{zatsiorsky_sum(cur['SP'], cur['400m'], cur['PV']):.0f} очков (по таблице 1962; ориентир, не прогноз)")
    return df


# ═════════════════════════════ 6. MODEL — Fitness–Fatigue (каналы + пулы), журнал, MAP-калибровка ═════════════════════════════
# g(t+1) = a1·g + (1−a1)·u(t)            u = (AU/AU_ref)·вес_канала / CHAN_SCALE        (фитнес, τ1 по каналам)
# P(t+1) = d·(P + dose)                  dose = стресс_сессии·(AU/AU_сессии), d = 0.5^(24/T½)  (3 пула, пункт 2)
# h_r = Σ_p W[r,p]·P_p                   утомление канала
# y_r = K1·ψ(g) − K2·h_r,  ψ(g)=G(1−e^{−g/G});   z_o = a_o + Σ L_or·y_r   (результат/тест, % к опорному)
# Чувствительность канала к пулу (строки = SPR,JMP,THR,SE,END; столбцы = CNS,MET,MECH; сумма строки = 1) [EXPERT]
POOL_W = np.array([[.55, .10, .35], [.40, .10, .50], [.45, .10, .45], [.25, .60, .15], [.10, .65, .25]])
TYPICAL_WEEK = ["S_FLY30", "S_THROW_POWER", "S_CROSS30", "S_EXPLOSIVE", "S_PV_FULL", "S_GLYCO100", "S_REST"]  # Пн–Вс, «как раньше» [заглушка]
CHAN_SCALE = np.mean([LIBRARY[k].channel_load() for k in TYPICAL_WEEK], axis=0)   # типовая неделя = 1.0 на канал


@dataclass
class Params:
    K1: np.ndarray       # (R) усиление фитнеса, % на единицу
    K2: np.ndarray       # (R) усиление утомления, % на единицу
    tau1: np.ndarray     # (R) постоянная фитнеса, сутки
    half_h: np.ndarray   # (3) T½ пулов CNS, MET, MECH, часы

    def to_model(self, sat_G: float = 2.0) -> "PerfModel":
        return PerfModel(K1=self.K1.copy(), K2=self.K2.copy(), tau1=self.tau1.copy(), sat_G=sat_G,
                         const=PoolConstants(tuple(float(x) for x in self.half_h), strict=False))

    @property
    def decay(self) -> np.ndarray:
        return 0.5 ** (24.0 / self.half_h)

    def classical(self) -> pd.DataFrame:
        return pd.DataFrame({"K1": self.K1, "K2": self.K2, "tau1": self.tau1}, index=list(CHANNELS)).round(3)

    def pools_table(self) -> pd.DataFrame:
        return pd.DataFrame({"half_life_h": self.half_h, "decay_per_day": self.decay}, index=list(POOLS)).round(3)


@dataclass
class ModelConfig:
    sat_G: float = 2.0            # насыщение фитнеса ψ (без него оптимум планировщика вырождается)
    warm_days: int = 28
    K_bounds: Tuple[float, float] = (0.05, 40.0)
    tau1_bounds: Tuple[float, float] = (8.0, 120.0)
    half_widen: Tuple[float, float] = (0.75, 1.25)   # границы T½ при калибровке: диапазон из ТЗ, расширенный на ±25 %


class Layout:
    """θ = [ln K1(R), ln K2(R), ln τ1(R), ln T½(3)] — гладкая положительная параметризация."""
    def __init__(self, cfg: ModelConfig):
        self.cfg, self.n = cfg, 3 * R + 3

    def unpack(self, th) -> Params:
        return Params(np.exp(th[:R]), np.exp(th[R:2 * R]), np.exp(th[2 * R:3 * R]), np.exp(th[3 * R:]))

    def pack(self, p: Params) -> np.ndarray:
        return np.concatenate([np.log(p.K1), np.log(p.K2), np.log(p.tau1), np.log(p.half_h)])

    def bounds(self):
        c = self.cfg
        hl = np.log([c.half_widen[0] * HALF_LIFE_RANGE_H[p][0] for p in POOLS])
        hh = np.log([c.half_widen[1] * HALF_LIFE_RANGE_H[p][1] for p in POOLS])
        lo = np.r_[np.full(2 * R, np.log(c.K_bounds[0])), np.full(R, np.log(c.tau1_bounds[0])), hl]
        hi = np.r_[np.full(2 * R, np.log(c.K_bounds[1])), np.full(R, np.log(c.tau1_bounds[1])), hh]
        return lo, hi


def psi(g, cfg: ModelConfig):
    return g if not cfg.sat_G else cfg.sat_G * (1.0 - np.exp(-g / cfg.sat_G))


def form(g, pools, p: Params, cfg: ModelConfig):
    """Форма каналов y = K1·ψ(g) − K2·(P @ Wᵀ), % к опорному результату. g: (...,R), pools: (...,3)."""
    return p.K1 * psi(g, cfg) - p.K2 * (pools @ POOL_W.T)


def forward(U: np.ndarray, Pd: np.ndarray, p: Params, cfg: ModelConfig, carry=None):
    """Состояния g (T,R) и пулов (T,3) на НАЧАЛО суток; обе рекурсии — БИХ-фильтры 1-го порядка (lfilter).
    Без carry стартуем в устойчивом состоянии по средней нагрузке первых warm_days."""
    T = len(U)
    a1, d = np.exp(-1 / p.tau1), p.decay
    w = max(1, min(cfg.warm_days, T))
    g0, p0 = (U[:w].mean(0), d * Pd[:w].mean(0) / (1 - d)) if carry is None else carry
    g, pools, gT, pT = np.empty((T, R)), np.empty((T, 3)), np.empty(R), np.empty(3)
    for r in range(R):
        yg = lfilter([1 - a1[r]], [1, -a1[r]], U[:, r], zi=[a1[r] * g0[r]])[0]
        g[0, r], g[1:, r], gT[r] = g0[r], yg[:-1], yg[-1]
    for k in range(3):
        yp = lfilter([d[k]], [1, -d[k]], Pd[:, k], zi=[d[k] * p0[k]])[0]    # P(t+1)=d·P(t)+d·dose(t)
        pools[0, k], pools[1:, k], pT[k] = p0[k], yp[:-1], yp[-1]
    return g, pools, (gT, pT)


def _session_rows(sessions: pd.DataFrame, library: SessionLibrary):
    s = sessions.copy()
    s["date"] = pd.to_datetime(s["date"]).dt.normalize()
    s = s.sort_values("date", kind="stable").reset_index(drop=True)
    mods = [library[m] for m in s["module"]]
    mn = pd.to_numeric(s["minutes"], errors="coerce") if "minutes" in s else pd.Series(np.nan, index=s.index)
    rp = pd.to_numeric(s["rpe"], errors="coerce") if "rpe" in s else pd.Series(np.nan, index=s.index)
    au = (mn.fillna(pd.Series([m.minutes for m in mods], index=s.index)).to_numpy() *
          rp.fillna(pd.Series([m.rpe for m in mods], index=s.index)).to_numpy())
    return s, mods, au


def build_daily(sessions: pd.DataFrame, library: SessionLibrary = LIBRARY) -> dict:
    """Журнал сессий → суточные тензоры (нагрузка по каналам в нормированных единицах и дозы пулов).
    Несколько сессий в день суммируются. Пропущенные дни = отдых (0)."""
    s, mods, au = _session_rows(sessions, library)
    dates = pd.date_range(s["date"].min(), s["date"].max(), freq="D")
    day = (s["date"] - dates[0]).dt.days.to_numpy()
    U, P = np.zeros((len(dates), R)), np.zeros((len(dates), 3))
    for k, m in enumerate(mods):
        vol = au[k] / m.AU if m.AU > 0 else 0.0           # объём относительно стандартной сессии
        U[day[k]] += (au[k] / AU_REF) * np.array([m.chan.get(c, 0.0) for c in CHANNELS]) / CHAN_SCALE
        P[day[k]] += vol * m.stress.array()
    return dict(dates=dates, U=U, P=P)


# ---- 6.2 CALIB — MAP-калибровка с иерархическими априорами ----
@dataclass
class ObsSpec:
    name: str
    loadings: Dict[str, float]   # вклад каналов в результат (строка L_o)
    higher_is_better: bool
    sigma: float = 1.5           # SD шума измерения, %
    ref: Optional[float] = None  # опорный результат; None → медиана наблюдений


_LOAD = {"100m": {"SPR": 1.0}, "LJ": {"JMP": .7, "SPR": .3}, "SP": {"THR": 1.0}, "HJ": {"JMP": .8, "SPR": .2},
         "400m": {"SE": .7, "SPR": .3}, "110mH": {"SPR": .7, "JMP": .3}, "DT": {"THR": 1.0},
         "PV": {"JMP": .6, "SPR": .4}, "JT": {"THR": .8, "SPR": .2}, "1500m": {"END": 1.0}}
_SIG = {"100m": 1.4, "LJ": 2.5, "SP": 2.5, "HJ": 2.4, "400m": 1.7, "110mH": 1.7, "DT": 4.0, "PV": 3.0, "JT": 4.5, "1500m": 1.9}


def default_specs() -> Dict[str, ObsSpec]:
    return {e: ObsSpec(e, _LOAD[e], not IS_TIME[e], _SIG[e]) for e in EVENTS}


def test_specs() -> Dict[str, ObsSpec]:
    """Стандартизованные тесты — главный источник идентифицируемости (чаще и «чище», чем старты)."""
    return {"T30f": ObsSpec("T30f", {"SPR": 1.0}, False, 1.0), "TSLJ": ObsSpec("TSLJ", {"JMP": 1.0}, True, 1.5),
            "TSHF": ObsSpec("TSHF", {"THR": 1.0}, True, 2.5), "T300": ObsSpec("T300", {"SE": 1.0}, False, 1.2),
            "T1000": ObsSpec("T1000", {"END": 1.0}, False, 1.5)}


@dataclass
class FitResult:
    theta: np.ndarray; params: Params; intercepts: Dict[str, float]; n_obs: int
    rmse: float; cost: float; post_sd: np.ndarray; runs: List[Tuple[float, np.ndarray]]


class Calibrator:
    """MAP: остатки данных (интерсепты профилируются аналитически, веса 1/σ_o) + априорные остатки из KineticPrior.
    Мало данных → решение ≈ априор (Ridge); много данных → данные атлета доминируют (иерархическая усадка)."""

    def __init__(self, daily: dict, obs: pd.DataFrame, specs: Dict[str, ObsSpec], prior: Optional[KineticPrior] = None,
                 cfg: Optional[ModelConfig] = None, burn_in: int = 42, min_obs: int = 4):
        self.U, self.P, self.cfg = daily["U"], daily["P"], cfg or ModelConfig()
        self.prior, self.lay = prior or KineticPrior(), Layout(self.cfg)
        o = obs.copy()
        o["date"] = pd.to_datetime(o["date"]).dt.normalize()
        o = o[o["obs"].isin(list(specs))]
        day = (o["date"] - daily["dates"][0]).dt.days
        o = o[(day >= burn_in) & (day < len(self.U))]
        cnt = o["obs"].value_counts()
        self.types = [n for n in specs if cnt.get(n, 0) >= min_obs]
        o = o[o["obs"].isin(self.types)].reset_index(drop=True)
        if o.empty:
            raise ValueError("нужно ≥ min_obs наблюдений на тип после burn_in")
        ti = {n: i for i, n in enumerate(self.types)}
        ref = np.array([specs[n].ref or float(o.loc[o.obs == n, "value"].median()) for n in self.types])
        sgn = np.array([1.0 if specs[n].higher_is_better else -1.0 for n in self.types])
        Lt = np.array([[specs[n].loadings.get(c, 0.0) for c in CHANNELS] for n in self.types])
        self.sgn, self.ref, self.specs = sgn, ref, specs
        self.d = SimpleNamespace(
            day=(o["date"] - daily["dates"][0]).dt.days.to_numpy(), type=o["obs"].map(ti).to_numpy(),
            sig=np.array([specs[n].sigma for n in self.types])[o["obs"].map(ti).to_numpy()])
        self.d.z = sgn[self.d.type] * (o["value"].to_numpy(float) / ref[self.d.type] - 1.0) * 100.0
        self.d.L = Lt[self.d.type]
        self.d.obs = o

    def _sub(self, mask):
        s = SimpleNamespace(**{k: getattr(self.d, k)[mask] for k in ("day", "type", "z", "sig", "L")})
        return s

    def _resid(self, th, d):
        p = self.lay.unpack(th)
        g, pl, _ = forward(self.U, self.P, p, self.cfg)
        pred = (form(g, pl, p, self.cfg)[d.day] * d.L).sum(1)
        e, w, nt = d.z - pred, 1.0 / d.sig ** 2, len(self.types)
        den = np.bincount(d.type, w, nt)
        a = np.where(den > 0, np.bincount(d.type, w * e, nt) / np.maximum(den, 1e-12), np.nan)  # профилирование a_o
        return (e - a[d.type]) / d.sig, a

    def _prior_resid(self, th):
        pr, p, s = self.prior, self.lay.unpack(th), self.prior.strength
        return np.concatenate([(np.log(p.tau1) - pr.mu1) / (pr.sd1 / s), (np.log(p.half_h) - pr.muH) / (pr.sdH / s),
                               (np.log(p.K1) - np.log(pr.K1_median)) / (pr.K_logsd / s),
                               (np.log(p.K2) - np.log(pr.K2_median)) / (pr.K_logsd / s)])

    def prior_theta(self) -> np.ndarray:
        """Мода априора (холодный старт без данных)."""
        pr = self.prior
        return self.lay.pack(Params(np.full(R, pr.K1_median), np.full(R, pr.K2_median), np.exp(pr.mu1), np.exp(pr.muH)))

    def fit(self, mask=None, n_starts: int = 6, seed: int = 0, theta0=None, max_nfev: int = 80) -> FitResult:
        d = self.d if mask is None else self._sub(mask)
        lo, hi = self.lay.bounds()
        rng = np.random.default_rng(seed)
        base = self.prior_theta() if theta0 is None else theta0
        starts = [np.clip(base, lo + 1e-6, hi - 1e-6)]
        while len(starts) < n_starts:  # мультистарт вокруг моды априора (в масштабе апостериорных SD)
            jit = np.r_[rng.normal(0, .5, 2 * R), rng.normal(0, .3, R), rng.normal(0, .3, 3)]
            starts.append(np.clip(base + jit, lo + 1e-6, hi - 1e-6))
        fun = lambda th: np.concatenate([self._resid(th, d)[0], self._prior_resid(th)])
        runs = []
        for t0 in starts:
            try:
                runs.append(least_squares(fun, t0, bounds=(lo, hi), loss="soft_l1", f_scale=2.0, x_scale="jac", max_nfev=max_nfev))
            except (ValueError, FloatingPointError, np.linalg.LinAlgError):
                continue
        runs.sort(key=lambda s: s.cost)
        b = runs[0]
        r, a = self._resid(b.x, d)
        free = (b.x > lo + 1e-6) & (b.x < hi - 1e-6)
        J = b.jac[:, free]
        sd = np.full(len(b.x), np.nan)
        sd[free] = np.sqrt(np.clip(np.diag(np.linalg.pinv(J.T @ J)), 0, None))  # Лаплас: апостериорная SD в θ-пространстве
        return FitResult(b.x, self.lay.unpack(b.x), {n: float(a[i]) for i, n in enumerate(self.types)}, len(r),
                         float(np.sqrt(np.mean((r * d.sig) ** 2))), float(b.cost), sd, [(s.cost, s.x) for s in runs])

    def rolling_cv(self, res: FitResult, n_splits: int = 3, horizon: int = 42, first_cut: float = 0.5) -> pd.DataFrame:
        """Прогноз вперёд: переобучаемся на данных до среза и предсказываем следующие horizon дней.
        skill = 1 − MSE_model/MSE_baseline(среднее по типу); skill > 0 → модель лучше наивного базлайна."""
        T = len(self.U)
        rows = []
        for c in np.unique(np.linspace(first_cut * T, T - horizon, n_splits).astype(int)):
            tr, te = self.d.day < c, (self.d.day >= c) & (self.d.day < c + horizon)
            if te.sum() < 3 or tr.sum() < 20:
                continue
            r = self.fit(tr, n_starts=2, theta0=res.theta, max_nfev=40)
            d, t = self._sub(tr), self._sub(te)
            p = r.params
            g, pl, _ = forward(self.U, self.P, p, self.cfg)
            a = np.array([r.intercepts.get(n, np.nan) for n in self.types])
            pred = (form(g, pl, p, self.cfg)[t.day] * t.L).sum(1) + a[t.type]
            nt = len(self.types)
            zbar = np.bincount(d.type, d.z, nt) / np.maximum(np.bincount(d.type, minlength=nt), 1)
            ok = np.isfinite(pred)
            if ok.sum():
                em, eb = t.z[ok] - pred[ok], t.z[ok] - zbar[t.type][ok]
                rows.append(dict(cut_day=int(c), n_test=int(ok.sum()), rmse_model=np.sqrt(np.mean(em ** 2)),
                                 rmse_base=np.sqrt(np.mean(eb ** 2)), skill=1 - np.mean(em ** 2) / np.mean(eb ** 2)))
        return pd.DataFrame(rows).round(3)




# ═════════════════════════════ 7. SOLVER — дискретный планировщик: beam search + локальный поиск + оптимизатор дня ═════════════════════════════
COMP_U = (np.array([1.5, 1.5, 1.5, 1.5, .5]), np.array([1.5, 1.5, 1.5, .5, 1.5]))   # фитнес-стимул в D1/D2 (норм. единицы)
COMP_DOSE = (np.array([1.2, .6, .6]), np.array([1.05, .675, .75]))                    # дозы пулов в D1/D2 [EXPERT]


@dataclass
class Athlete:
    name: str
    mass: float
    current: Dict[str, float]            # «свежие» текущие результаты по 10 видам (с / м)
    already_day2: Tuple[str, ...] = ()   # виды, чьи значения получены во 2-й день (штраф не накладываем повторно)
    noise_sd: Dict[str, float] = field(default_factory=dict)   # неопределённость очков вида (давность замера)


@dataclass
class PerfModel:
    """Параметры Fitness–Fatigue. По умолчанию — холодный старт (середины диапазонов); K — заглушки до калибровки."""
    K1: np.ndarray = field(default_factory=lambda: np.full(R, 3.0))
    K2: np.ndarray = field(default_factory=lambda: np.full(R, 2.0))
    tau1: np.ndarray = field(default_factory=lambda: np.array([36., 36., 40., 41., 46.]))   # сутки, SPR JMP THR SE END
    sat_G: float = 2.0
    const: PoolConstants = field(default_factory=PoolConstants)

    @classmethod
    def from_prior(cls, prior: Optional["KineticPrior"] = None, sat_G: float = 2.0) -> "PerfModel":
        """Холодный старт: мода популяционного априора (τ1 и T½ из диапазонов ТЗ, K — заглушки)."""
        pr = prior or KineticPrior()
        return cls(K1=np.full(R, pr.K1_median), K2=np.full(R, pr.K2_median), tau1=np.exp(pr.mu1), sat_G=sat_G,
                   const=PoolConstants(tuple(float(x) for x in np.exp(pr.muH))))

    def a1(self) -> np.ndarray:
        return np.exp(-1.0 / self.tau1)

    def form(self, g, pools):
        psi = self.sat_G * (1.0 - np.exp(-g / self.sat_G))
        return self.K1 * psi - self.K2 * (pools @ POOL_W.T)


@dataclass
class PlanLimits:
    """Ограничители здравого смысла сверх шлюзов."""
    ramp: float = 1.15               # недельный AU ≤ ramp × AU типовой недели
    h_cap_mult: float = 1.15         # усталость каналов h_r ≤ mult × её пик в типовой неделе
    max_same_per_week: int = 2       # одна и та же сессия ≤ N раз за 7 дней плана (было 3: план сводился к 1000 м, силовой и барьерам)
    max_rest_per_week: int = 2       # дней отдыха ≤ N за 7 дней плана (вне тейпера)
    taper_days: int = 2              # в последние дни лимит отдыха не действует
    min_channel_frac: float = 0.5    # недельный стимул каждого канала должен быть ≥ frac × типовой недели (баланс)
    balance_weight: float = 15.0     # штраф, очков за единицу недобора (1 ед. = средняя суточная нагрузка канала)
    cov_weight: float = 80.0         # штраф, очков за каждую недостающую за неделю сессию группы видов (разнообразие)
    group_need: Dict[str, int] = field(default_factory=lambda: dict(GROUP_NEED))


# Группы видов: каждую неделю должны быть сессии каждой группы («не реже раза в неделю в каждом виде», Ушаков 1977;
# 80–110 занятий на вид в год). Метания — дважды: диск и копьё самые слабые виды профиля. Ключи — сессии библиотеки.
SESSION_GROUPS = {
    "СПРИНТ": ("S_FLY30", "S_ACCEL20"), "БАРЬЕРЫ": ("S_HURDLE_RHYTHM", "S_HURDLE_PLYO"),
    "ПРЫЖКИ (высота)": ("S_HJ_FULL", "S_HJ_SHORT"), "ШЕСТ": ("S_PV_FULL", "S_PV_SHORT"),
    "МЕТАНИЯ": ("S_THROW_POWER", "S_THROW_TECH"), "СИЛА": ("S_STRENGTH_BASE",),
    "ВЗРЫВНАЯ СИЛА": ("S_EXPLOSIVE", "S_DROP_JUMPS", "S_HILL_SL"),
    "ВЫНОСЛИВОСТЬ": ("S_INT1000", "S_TEMPO_RUN", "S_CROSS30", "S_GLYCO100", "S_SPEEDEND300")}
GROUP_NEED = {"СПРИНТ": 1, "БАРЬЕРЫ": 1, "ПРЫЖКИ (высота)": 1, "ШЕСТ": 1, "МЕТАНИЯ": 2, "СИЛА": 1, "ВЗРЫВНАЯ СИЛА": 0.5,
              "ВЫНОСЛИВОСТЬ": 1}


@dataclass
class SolverState:
    g: np.ndarray            # фитнес по каналам на день 0
    ast: AthleteState        # пулы + память шлюзов (часы)
    t0: float                # час сессии дня 0 (абсолютные часы того же календаря, что и у ast)
    h_peak: np.ndarray       # пик усталости каналов в типовой неделе (для лимита)
    week_au: float           # AU типовой недели


def warm_start(model: PerfModel, library: SessionLibrary = LIBRARY, gcfg: GateConfig = GateConfig(),
               week: Sequence[str] = TYPICAL_WEEK, days: int = 56, session_hour: float = 17.0) -> SolverState:
    """Холодный старт без журнала: «разогрев» типовой неделей. План начинается сразу после последней записанной сессии."""
    ast, a1 = AthleteState(model.const, gcfg), model.a1()
    g = np.mean([library[k].channel_load() / CHAN_SCALE for k in week], axis=0)
    hmax = np.zeros(R)
    for i in range(days):
        s, hour = library[week[i % 7]], 24.0 * i + session_hour
        if i >= days - 28:
            hmax = np.maximum(hmax, ast.pools_at(hour).P @ POOL_W.T)
        ast.record(s, hour)
        g = a1 * g + (1 - a1) * s.channel_load() / CHAN_SCALE
    return SolverState(g, ast, 24.0 * days + session_hour, hmax, float(np.mean([library[k].AU for k in week]) * 7))


class DailySolver:
    def __init__(self, model: PerfModel, athlete: Athlete, state: SolverState, library: SessionLibrary = LIBRARY,
                 gcfg: GateConfig = GateConfig(), limits: PlanLimits = PlanLimits(), ref: Optional[SolverState] = None):
        """state — с чего начинается проигрывание плана (по журналу или типовая неделя). ref — состояние, в котором «выставлены» личные результаты
        (точка отсчёта формы y0). По умолчанию совпадает со state (холодный старт). С журналом ref ДОЛЖЕН быть эталонным: иначе усталость от
        вчерашней тяжёлой тренировки занижает «стартовую форму» и прогноз растёт ни с чего (рекорды ставились не в усталом состоянии)."""
        self.m, self.ath, self.s0, self.lib, self.gc, self.lim = model, athlete, state, library, gcfg, limits
        self.gates = BioGates(gcfg, model.const)
        self.names = [s.key for s in library]
        S = [library[n] for n in self.names]
        self.idx = {n: i for i, n in enumerate(self.names)}
        self.stress = np.array([s.stress.array() for s in S])
        self.u = np.array([s.channel_load() / CHAN_SCALE for s in S])
        self.au = np.array([s.AU for s in S])
        self.cns_gate = self.stress[:, 0] >= gcfg.cns_stress
        self.lactic = self.stress[:, 1] >= gcfg.met_stress
        self.mech_heavy = self.stress[:, 2] >= gcfg.mech_stress
        self.extreme = self.stress[:, 2] >= gcfg.mech_extreme
        self.coord = np.array([s.coord for s in S])
        self.is_rest = np.array([n == "S_REST" for n in self.names])
        self.ridx = self.names.index("S_REST")
        self.a1, self.dp = model.a1(), model.const.decay_factor(24.0)
        self.thr = self.gates.cns_thr
        self.au_cap = limits.ramp * state.week_au
        self.h_cap = limits.h_cap_mult * state.h_peak
        self.L = np.array([[_LOAD[e].get(c, 0.0) for c in CHANNELS] for e in EVENTS])
        self.cur = athlete.current
        self.mult = Day2Penalty().multipliers(self.cur, athlete.mass, skip=athlete.already_day2)   # штрафы 2-го дня (слой 1)
        self.need = limits.min_channel_frac * 7.0 * np.ones(R)     # недельная потребность по каналам (норм. единицы)
        self.bw = limits.balance_weight
        gn = list(limits.group_need)
        self.G = np.array([[float(n in SESSION_GROUPS.get(g, ())) for g in gn] for n in self.names])
        self.gneed = np.array([limits.group_need[g] if self.G[:, i].any() else 0 for i, g in enumerate(gn)], float)
        self.cw = limits.cov_weight
        self.cap = np.full(len(self.names), limits.max_same_per_week)   # группы с нормой ≥2 в неделю: каждая сессия ≤1 раза → ротация
        for i, g in enumerate(gn):
            if limits.group_need[g] >= 2:
                self.cap[self.G[:, i] > 0] = 1
        rf = ref if ref is not None else state
        self.y0 = model.form(rf.g, rf.ast.pools_at(rf.t0).P)

    # ── прогноз ──
    def results(self, g, pools) -> np.ndarray:
        """g (N,R), pools (N,3) — состояние в момент D1 → результаты (N,10) на D1/D2."""
        y1 = self.m.form(g, pools)
        y2 = self.m.form(self.a1 * g + (1 - self.a1) * COMP_U[0], self.dp * (pools + COMP_DOSE[0]))
        out = np.empty((len(g), 10))
        for i, e in enumerate(EVENTS):
            dz = (y1 if e in DAY1 else y2) @ self.L[i] - self.y0 @ self.L[i]
            out[:, i] = self.cur[e] * (1 + SIGN[e] * dz / 100.0) * self.mult[e]
        return out

    def points(self, res: np.ndarray) -> np.ndarray:
        return sum(wa_points(e, res[:, i]) for i, e in enumerate(EVENTS))

    def _value(self, g, pools, k: int) -> np.ndarray:
        return self.points(self.results(g * self.a1 ** k, pools * self.dp ** k))

    # ── векторная динамика и шлюзы ──
    def _batch(self) -> SimpleNamespace:
        a, t0 = self.s0.ast, self.s0.t0
        return SimpleNamespace(g=self.s0.g[None, :].copy(), pools=a.pools_at(t0).P[None, :].copy(),
                               dsl=np.array([t0 - a.last_lactic_h]), dsm=np.array([t0 - a.last_mech_h]),
                               ext=np.array([a.last_mech_extreme]), auw=np.zeros(1), restw=np.zeros(1),
                               cnt=np.zeros((1, len(self.names))), wk=np.zeros((1, R)), pen=np.zeros(1))

    def _week_pen(self, wk, k: int, cnt=None) -> np.ndarray:
        """Штраф недобора недельного стимула каналов + недобора сессий по группам видов (пропорционально к дню k недели)."""
        pen = self.bw * np.maximum(0.0, self.need * k / 7.0 - wk).sum(-1)
        if cnt is not None:
            pen = pen + self.cw * np.maximum(0.0, self.gneed * k / 7.0 - cnt @ self.G).sum(-1)
        return pen

    def _week_reset(self, S, d: int):
        if d % 7:
            return S
        pen = S.pen + (self._week_pen(S.wk, 7, S.cnt) if d > 0 else 0.0)     # закрываем предыдущую неделю
        return SimpleNamespace(**{**vars(S), "auw": np.zeros_like(S.auw), "restw": np.zeros_like(S.restw),
                                  "cnt": np.zeros_like(S.cnt), "wk": np.zeros_like(S.wk), "pen": pen})

    def _reasons(self, S, d: int, H: int):
        c, lim = self.gc, self.lim
        gall = self.a1 * S.g[:, None, :] + (1 - self.a1) * self.u[None]
        pall = self.dp * (S.pools[:, None, :] + self.stress[None])
        need = np.where(self.extreme[None, :] | S.ext[:, None], c.tendon_extreme_h, c.tendon_h)
        rs = {"CNS": self.cns_gate[None, :] & (S.pools[:, [0]] > self.thr),
              "INTERFERENCE": self.coord[None, :] & (S.dsl[:, None] < c.interference_h),
              "TENDON": self.mech_heavy[None, :] & (S.dsm[:, None] < need),
              "AU": (S.auw[:, None] + self.au[None]) > self.au_cap,
              "FATIGUE": (pall @ POOL_W.T > self.h_cap).any(2),
              "REPEAT": (S.cnt >= self.cap[None]) & ~self.is_rest[None],
              "REST": self.is_rest[None] & (S.restw[:, None] >= lim.max_rest_per_week) & (d < H - lim.taper_days)}
        return rs, gall, pall

    def _ok(self, rs) -> np.ndarray:
        ok = ~np.any(list(rs.values()), axis=0)
        ok[~ok.any(1), self.ridx] = True     # отдых допустим всегда как «выход»
        return ok

    def _step(self, S, b, m):
        cnt = S.cnt[b].copy()
        cnt[np.arange(len(m)), m] += 1
        mh = self.mech_heavy[m]
        return SimpleNamespace(
            g=self.a1 * S.g[b] + (1 - self.a1) * self.u[m], pools=self.dp * (S.pools[b] + self.stress[m]),
            dsl=np.where(self.lactic[m], 24.0, S.dsl[b] + 24.0), dsm=np.where(mh, 24.0, S.dsm[b] + 24.0),
            ext=np.where(mh, self.extreme[m], S.ext[b]), auw=S.auw[b] + self.au[m],
            restw=S.restw[b] + self.is_rest[m], cnt=cnt, wk=S.wk[b] + self.u[m], pen=S.pen[b])

    # ── поиск ──
    def solve(self, horizon: int = 21, beam: int = 300, top: int = 3, locked: Optional[Dict[int, str]] = None,
              refine: bool = True) -> List[dict]:
        """Лучшие планы [{seq, points}]. locked — {день с 0: ключ сессии}: принудительная сессия (тест, старт, отдых)."""
        locked = {k: self.idx[v] for k, v in (locked or {}).items()}
        S, hist = self._batch(), np.zeros((1, horizon), dtype=np.int16)
        for d in range(horizon):
            S = self._week_reset(S, d)
            rs, gall, pall = self._reasons(S, d, horizon)
            ok = self._ok(rs)
            if d in locked:
                ok[:] = False
                ok[:, locked[d]] = True
            b, m = np.nonzero(ok)
            k = d % 7 + 1
            last = d == horizon - 1
            f = (1.0 if k >= 4 else 0.0) if last else 0.5       # на последнем дне — точный штраф, раньше — «темповая» эвристика
            score = (self._value(gall[b, m], pall[b, m], horizon - d - 1) - S.pen[b]
                     - f * self._week_pen(S.wk[b] + self.u[m], k, S.cnt[b] + np.eye(len(self.names))[m]))
            keep = np.argsort(-score)[:beam]
            b, m, score = b[keep], m[keep], score[keep]
            S, hist = self._step(S, b, m), hist[b]
            hist[:, d] = m
        out = []
        for o in np.argsort(-score)[:top]:
            seq = [self.names[j] for j in hist[o]]
            if refine:
                seq = self.local_search(seq, set(locked))
            ok_, obj, _ = self.evaluate(seq)
            br = self.breakdown(seq)
            out.append(dict(seq=seq, objective=obj, points=br["points"], penalty=br["penalty"], feasible=ok_))
        return sorted(out, key=lambda r: -r["objective"])

    def evaluate(self, seq: Sequence[str], trace: bool = False):
        """Проигрывает план. Возвращает (допустим, ЦЕЛЬ = очки − штраф баланса, результаты[, trace по дням])."""
        feas, pts, pen, resd, tr = self._run(seq, trace)
        return (feas, pts - pen, resd, tr) if trace else (feas, pts - pen, resd)

    def breakdown(self, seq: Sequence[str]) -> Dict[str, float]:
        feas, pts, pen, _, _ = self._run(seq, False)
        return dict(points=pts, penalty=pen, objective=pts - pen)

    def _run(self, seq: Sequence[str], trace: bool):
        H, S, feas, tr = len(seq), self._batch(), True, []
        for d, n in enumerate(seq):
            S = self._week_reset(S, d)
            rs, _, _ = self._reasons(S, d, H)
            ok, j = self._ok(rs), self.idx[n]
            feas &= bool(ok[0, j])
            if trace:
                tr.append(dict(pools=S.pools[0].copy(), reasons={k: v[0].copy() for k, v in rs.items()}, ok=ok[0].copy(),
                               bad=[k for k, v in rs.items() if v[0, j]]))
            S = self._step(S, np.array([0]), np.array([j]))
        res = self.results(S.g, S.pools)
        pts, resd = float(self.points(res)[0]), {e: float(res[0, i]) for i, e in enumerate(EVENTS)}
        k = (H - 1) % 7 + 1
        pen = float(S.pen[0] + (self._week_pen(S.wk, k, S.cnt)[0] if k >= 4 else 0.0))
        return feas, pts, pen, resd, tr

    def local_search(self, seq: List[str], locked=frozenset(), passes: int = 3) -> List[str]:
        """Замены одного дня и обмены пар дней (окно 7 суток) по точной цели; принимаются только допустимые планы."""
        cur = list(seq)
        f, best, _ = self.evaluate(cur)
        best = best if f else -np.inf
        for _ in range(passes):
            improved = False
            cands = ((d, n) for d in range(len(cur)) if d not in locked for n in self.names)
            for d, n in cands:
                if n == cur[d]:
                    continue
                new = cur[:d] + [n] + cur[d + 1:]
                f, p, _ = self.evaluate(new)
                if f and p > best + 1e-6:
                    cur, best, improved = new, p, True
            for d1 in range(len(cur)):
                for d2 in range(d1 + 1, min(d1 + 7, len(cur))):
                    if d1 in locked or d2 in locked or cur[d1] == cur[d2]:
                        continue
                    new = list(cur)
                    new[d1], new[d2] = new[d2], new[d1]
                    f, p, _ = self.evaluate(new)
                    if f and p > best + 1e-6:
                        cur, best, improved = new, p, True
            if not improved:
                break
        return cur

    def baseline(self, horizon: int, week: Sequence[str] = TYPICAL_WEEK) -> Tuple[bool, Dict[str, float]]:
        seq = [week[i % 7] for i in range(horizon)]
        return self.evaluate(seq)[0], self.breakdown(seq)

    # ── точная проверка эталонным BioGates ──
    def verify(self, seq: Sequence[str]) -> pd.DataFrame:
        """Проигрывает план через эталонный bio_gates (с причинами и временем снятия блока)."""
        st, rows = self.s0.ast.copy(), []
        for d, n in enumerate(seq):
            hour = self.s0.t0 + 24.0 * d
            v = self.gates.check(st, self.lib[n], hour)
            rows.append(dict(day=d + 1, session=n, allowed=v.allowed, reason=v.text()))
            st.record(self.lib[n], hour)
        return pd.DataFrame(rows)

    # ── оптимизатор дня ──
    def recommend_today(self, horizon: int = 21, beam: int = 60) -> pd.DataFrame:
        """Для каждой сессии: допустима ли сегодня (эталонные шлюзы), через сколько часов откроется, и итог плана
        (очки на старте), если поставить её сегодня и оптимально продолжить."""
        st, rows = self.s0.ast, []
        for n in self.names:
            v = self.gates.check(st, self.lib[n], self.s0.t0)
            pts = np.nan
            if v.allowed:
                pts = self.solve(horizon, beam=beam, top=1, locked={0: n}, refine=False)[0]["objective"]
            rows.append(dict(session=n, title=self.lib[n].title, allowed=v.allowed, free_in_h=round(v.free_in_h, 1),
                             plan_points=round(pts, 1), reason=v.text()))
        df = pd.DataFrame(rows)
        best = df.plan_points.max()
        df["delta_vs_best"] = (df.plan_points - best).round(1)
        return df.sort_values(["allowed", "plan_points"], ascending=[False, False]).reset_index(drop=True)

    def plan_table(self, seq: Sequence[str], start: pd.Timestamp) -> pd.DataFrame:
        """План по дням: сессия, нагрузка, пулы перед сессией, вклад дня (очки vs замена дня на отдых), проверка шлюзов."""
        _, base, _, tr = self.evaluate(seq, trace=True)
        ver = self.verify(seq)
        rows = []
        for d, n in enumerate(seq):
            rest_alt = list(seq[:d]) + ["S_REST"] + list(seq[d + 1:])
            date = start + pd.Timedelta(days=d)
            rows.append(dict(day=d + 1, date=f"{date:%d.%m}", wd=WD[date.weekday()], session=n, title=self.lib[n].title,
                             AU=self.lib[n].AU, CNS_vs_thr=f"{tr[d]['pools'][0] / self.thr:.0%}", MET=round(tr[d]["pools"][1], 2),
                             MECH=round(tr[d]["pools"][2], 2), gain_vs_rest=round(base - self.evaluate(rest_alt)[1], 1),   # вклад дня в цель (очки − штраф баланса)
                             gates_ok=bool(ver.allowed[d])))
        return pd.DataFrame(rows)




def state_from_journal(sessions: pd.DataFrame, model: "PerfModel", library: SessionLibrary = LIBRARY,
                       gcfg: GateConfig = GateConfig(), session_hour: float = 17.0) -> SolverState:
    """Состояние атлета по журналу (sessions: date, module, [minutes], [rpe]): фитнес, пулы и память шлюзов на момент
    после последней записи; план начинается на следующий день. Лимиты (недельный AU, пик усталости) — по последним 4 неделям."""
    s, mods, au = _session_rows(sessions, library)
    start = s["date"].min()
    T = int((s["date"].max() - start).days) + 1
    day = (s["date"] - start).dt.days.to_numpy()
    U, AUd = np.zeros((T, R)), np.zeros(T)
    for k, m in enumerate(mods):
        U[day[k]] += (au[k] / AU_REF) * np.array([m.chan.get(c, 0.0) for c in CHANNELS]) / CHAN_SCALE
        AUd[day[k]] += au[k]
    a1, ast = model.a1(), AthleteState(model.const, gcfg)
    g, hmax = U[:min(28, T)].mean(0), np.zeros(R)
    by_day: Dict[int, List[int]] = {}
    for k, d in enumerate(day):
        by_day.setdefault(int(d), []).append(k)
    for d in range(T):
        hour = 24.0 * d + session_hour
        if d >= T - 28:
            hmax = np.maximum(hmax, ast.pools_at(hour).P @ POOL_W.T)
        for k in by_day.get(d, []):
            m = mods[k]
            ast.record(m, hour, au[k] / m.AU if m.AU > 0 else 0.0)
        g = a1 * g + (1 - a1) * U[d]
    n_w = max(min(28, T) / 7.0, 1.0)
    return SolverState(g, ast, 24.0 * T + session_hour, hmax, float(AUd[-28:].sum() / n_w))


# ═════════════════════════════ 8. EXPLAIN — объяснение плана и визуализация ═════════════════════════════
REASON_RU = {"CNS": "CNS Gate", "INTERFERENCE": "Interference Gate", "TENDON": "Tendon Gate",
             "AU": "недельный потолок AU", "FATIGUE": "потолок усталости каналов", "REPEAT": "лимит повторов за неделю",
             "REST": "лимит дней отдыха"}
GATE_NAMES = ("CNS", "INTERFERENCE", "TENDON")
CH_RU = {"SPR": "спринт/барьеры", "JMP": "прыжки", "THR": "метания", "SE": "скоростная выносливость", "END": "аэробика"}
POOL_COLOR = {"CNS": "#2f6fdb", "MET": "#d64545", "MECH": "#e08a1e", "REST": "#9aa0a6"}


def _short(title: str, n: int = 30) -> str:
    return title if len(title) <= n else title[:n - 1] + "…"


class PlanExplainer:
    """Объяснение плана. Всё считается теми же правилами, что и в солвере (шлюзы — эталонный BioGates), поэтому объяснение
    не может разойтись с планом. Для каждого дня: состояние пулов → что запрещено и почему → почему выбрана именно эта
    сессия (вклад в цель, баланс недели, цена замены) → почему не в соседние дни (для «тяжёлых» сессий)."""

    def __init__(self, solver: "DailySolver"):
        self.sol = solver

    def replay(self, seq: Sequence[str]):
        """Один проход: цель, по дням — вердикты шлюзов для ВСЕХ сессий (на состоянии до сессии дня), след солвера, накопленный стимул недели."""
        sol = self.sol
        _, obj, _, tr = sol.evaluate(seq, trace=True)
        st, wk, days = sol.s0.ast.copy(), np.zeros(R), []
        for d, n in enumerate(seq):
            hour = sol.s0.t0 + 24.0 * d
            if d % 7 == 0:
                wk = np.zeros(R)
            days.append(dict(hour=hour, verdicts={k: sol.gates.check(st, sol.lib[k], hour) for k in sol.names},
                             wk=wk.copy(), tr=tr[d]))
            st.record(sol.lib[n], hour)
            wk = wk + sol.u[sol.idx[n]]
        return obj, days

    def _blocked(self, verdicts, n: str, cns_pct: float) -> str:
        groups: Dict[str, List[Tuple[str, float]]] = {}
        for k, v in verdicts.items():
            if k != n and not v.allowed:
                for b in v.blocks:
                    groups.setdefault(b.gate, []).append((k, b.free_in_h))
        parts = []
        for g in GATE_NAMES:
            if g in groups:
                it = groups[g]
                names = ", ".join(_short(self.sol.lib[k].title, 22) for k, _ in it[:4]) + ("…" if len(it) > 4 else "")
                lo, hi = min(f for _, f in it), max(f for _, f in it)
                extra = f"пул ЦНС {cns_pct:.0%} порога; " if g == "CNS" else ""
                rng_txt = f"{lo:.0f}" if round(lo) == round(hi) else f"{lo:.0f}–{hi:.0f}"
                parts.append(f"{REASON_RU[g]} ({extra}откроется через {rng_txt} ч): {names}")
        return " | ".join(parts)

    def _neighbors(self, seq, days, d: int, obj: float, start: pd.Timestamp, key: str, lo: int, hi: int) -> str:
        sol, items = self.sol, []
        j = sol.idx[key]
        for d2 in range(lo, hi):
            if d2 == d or not 0 <= d2 < len(seq) or seq[d2] == key:
                continue
            wd = f"{WD[(start + pd.Timedelta(days=d2)).weekday()]} {(start + pd.Timedelta(days=d2)):%d.%m}"
            v = days[d2]["verdicts"][key]
            if not v.allowed:
                g = v.blocks[0]
                items.append(f"{wd}: {REASON_RU[g.gate]} (ещё {g.free_in_h:.0f} ч)")
                continue
            lim = [REASON_RU[k] for k in ("AU", "REPEAT", "FATIGUE") if days[d2]["tr"]["reasons"][k][j]]
            if lim:
                items.append(f"{wd}: {lim[0]}")
                continue
            sw = list(seq)
            sw[d], sw[d2] = sw[d2], sw[d]
            f3, o3, _ = sol.evaluate(sw)
            items.append(f"{wd}: обмен {o3 - obj:+.1f} очк." + ("" if f3 else ", нарушает шлюзы"))
        return "; ".join(items)

    def explain(self, seq: Sequence[str], start: pd.Timestamp, alternatives: int = 2) -> pd.DataFrame:
        sol = self.sol
        obj, days = self.replay(seq)
        rows = []
        for d, n in enumerate(seq):
            t, sess, j = days[d], sol.lib[n], sol.idx[n]
            date = start + pd.Timedelta(days=d)
            pools = t["tr"]["pools"]
            cns_pct = float(pools[0] / sol.thr)
            # 1) состояние
            state_txt = (f"ЦНС {cns_pct:.0%} порога ({'свеж' if cns_pct <= 1 else 'утомлён'}) · MET {pools[1]:.2f} · MECH {pools[2]:.2f}")
            # 2) вклад дня и цена замены
            rest_obj = sol.evaluate(list(seq[:d]) + ["S_REST"] + list(seq[d + 1:]))[1]
            alts = []
            for m, nm in enumerate(sol.names):
                if nm == n or not t["tr"]["ok"][m] or not t["verdicts"][nm].allowed:
                    continue
                f2, o2, _ = sol.evaluate(list(seq[:d]) + [nm] + list(seq[d + 1:]))
                alts.append((o2 - obj, nm, f2))
            alts.sort(reverse=True)
            alt_txt = "; ".join(f"«{_short(sol.lib[nm].title)}» {dp:+.1f} очк." + ("" if f2 else " (ломает шлюзы позже)")
                                for dp, nm, f2 in alts[:alternatives]) or "альтернатив нет"
            tie = bool(alts) and alts[0][0] > -2.0
            # 3) баланс недели
            k = d % 7 + 1
            pace = sol.need * k / 7.0
            deficit = [c for i, c in enumerate(CHANNELS) if pace[i] - t["wk"][i] > 0.2 and sol.u[j][i] >= 0.25]
            bal = ("закрывает отставание недели по каналам: " + ", ".join(
                f"{CH_RU[c]} ({t['wk'][CHANNELS.index(c)]:.1f} из {pace[CHANNELS.index(c)]:.1f})" for c in deficit)) if deficit else ""
            # 4) соседние дни для «тяжёлых» сессий
            heavy = sol.cns_gate[j] or sol.mech_heavy[j] or sol.coord[j] or sol.lactic[j]
            shift = self._neighbors(seq, days, d, obj, start, n, d - 4, d + 3) if heavy else ""
            # 5) почему допустима/недопустима по ЦНС — короткая формула
            if n == "S_REST":
                role_why = "отдых: пулы падают, форма к старту растёт"
            elif sol.cns_gate[j]:
                role_why = f"высокий CNS допустим: ЦНС {cns_pct:.0%} ≤ порога"
            else:
                role_why = sess.why
            dom = "REST" if n == "S_REST" else POOLS[int(np.argmax(sess.stress.array()))]
            rows.append(dict(day=d + 1, date=date, weekday=WD[date.weekday()], session=n, title=sess.title, AU=sess.AU,
                             state=state_txt, role=sess.why, why_ok=role_why,
                             blocked_today=self._blocked(t["verdicts"], n, cns_pct), gain_vs_rest=round(obj - rest_obj, 1),
                             alternatives=alt_txt + (" — по модели почти равноценно, решают шлюзы/баланс" if tie else ""),
                             balance=bal, neighbors=shift, dom=dom, cns=float(pools[0]), thr=sol.thr, met=float(pools[1]),
                             mech=float(pools[2])))
        return pd.DataFrame(rows)

    def explain_session(self, seq: Sequence[str], start: pd.Timestamp, key: str) -> str:
        """Почему сессия key стоит там, где стоит (первое вхождение), а не раньше — главный ответ на «почему сдвинула на пятницу»."""
        sol = self.sol
        if key not in seq:
            return f"{key} в плане нет."
        d = list(seq).index(key)
        obj, days = self.replay(seq)
        date = start + pd.Timedelta(days=d)
        head = f"«{sol.lib[key].title}» стоит на {WD[date.weekday()]} {date:%d.%m} (день {d + 1})."
        if d == 0:
            return head + " Это первый день плана."
        body = self._neighbors(seq, days, d, obj, start, key, max(0, d - 6), d)
        later = self._neighbors(seq, days, d, obj, start, key, d + 1, d + 3)
        return head + "\nПочему не раньше → " + (body or "—") + ("\nПочему не позже → " + later if later else "")

    @staticmethod
    def explain_day(df: pd.DataFrame, day: int) -> str:
        r = df[df.day == day].iloc[0]
        return render_text(df[df.day == day], full=True) if len(r) else ""


def render_text(df: pd.DataFrame, full: bool = True) -> str:
    """Текстовый календарь плана с объяснением каждого дня."""
    out = []
    for _, r in df.iterrows():
        out.append(f"{r.weekday} {r.date:%d.%m}  [{r.session}]  {r.title}  ({r.AU:.0f} AU)  вклад дня {r.gain_vs_rest:+.1f}")
        if full:
            out.append(f"   состояние: {r.state}")
            out.append(f"   почему: {r.why_ok}" + (f"; {r.balance}" if r.balance else ""))
            if r.blocked_today:
                out.append(f"   запрещено сегодня: {r.blocked_today}")
            out.append(f"   цена замены: {r.alternatives}")
            if r.neighbors:
                out.append(f"   почему не в соседние дни: {r.neighbors}")
    return "\n".join(out)


def plan_to_html(df: pd.DataFrame, forecast: Optional[pd.DataFrame], title: str, path: str, notes: str = "") -> str:
    """Самодостаточный HTML-отчёт: SVG-график пулов с порогом CNS Gate и столбцами нагрузки, календарь (цвет = доминирующий
    пул), раскрывающиеся объяснения по дням, прогноз турнира. Без внешних зависимостей, адаптивный, светлая/тёмная тема."""
    esc = _html.escape
    W_, H_, pad, n = 760, 250, 34, len(df)
    ymax = max(df[["cns", "met", "mech"]].to_numpy().max(), float(df.thr.iloc[0])) * 1.12
    X = lambda i: pad + i * (W_ - 2 * pad) / max(n - 1, 1)
    Y = lambda v: H_ - pad - v / ymax * (H_ - 2 * pad)
    aumax = max(df.AU.max(), 1.0)
    svg = [f'<svg viewBox="0 0 {W_} {H_}" width="100%" role="img" aria-label="Пулы утомления по дням">']
    for i, r in df.iterrows():   # столбцы нагрузки дня (AU) фоном
        h = r.AU / aumax * 40
        svg.append(f'<rect x="{X(i) - 5:.1f}" y="{H_ - pad - h:.1f}" width="10" height="{h:.1f}" fill="{POOL_COLOR[r.dom]}" opacity=".28"/>')
    thr = float(df.thr.iloc[0])
    svg.append(f'<line x1="{pad}" y1="{Y(thr):.1f}" x2="{W_ - pad}" y2="{Y(thr):.1f}" stroke="{POOL_COLOR["CNS"]}" stroke-dasharray="5 4"/>')
    svg.append(f'<text x="{W_ - pad}" y="{Y(thr) - 4:.1f}" font-size="11" text-anchor="end" fill="{POOL_COLOR["CNS"]}">порог CNS Gate</text>')
    for li, (nm, col) in enumerate((("cns", "CNS"), ("met", "MET"), ("mech", "MECH"))):
        pts = " ".join(f"{X(i):.1f},{Y(v):.1f}" for i, v in enumerate(df[nm]))
        svg.append(f'<polyline fill="none" stroke="{POOL_COLOR[col]}" stroke-width="2" points="{pts}"/>')
        svg.append(f'<text x="{pad}" y="{14 + 13 * li}" font-size="11" fill="{POOL_COLOR[col]}">{col}</text>')
    for i, r in df.iterrows():
        svg.append(f'<text x="{X(i):.1f}" y="{H_ - 10}" font-size="9" text-anchor="middle" fill="currentColor">{r.weekday}</text>')
    svg.append("</svg>")
    rows = "".join(f'<tr><td>{r.weekday} {r.date:%d.%m}</td><td><span class="chip" style="background:{POOL_COLOR[r.dom]}"></span>'
                   f'{esc(r.title)}</td><td>{r.AU:.0f}</td><td>{r.gain_vs_rest:+.1f}</td></tr>' for _, r in df.iterrows())
    cards = []
    for _, r in df.iterrows():
        extra = "".join(f"<p><b>{k}:</b> {esc(str(v))}</p>" for k, v in (
            ("Запрещено сегодня", r.blocked_today), ("Цена замены", r.alternatives), ("Почему не в соседние дни", r.neighbors)) if v)
        cards.append(f'<details><summary>{r.weekday} {r.date:%d.%m} — {esc(r.title)}</summary><p><b>Состояние:</b> {esc(r.state)}</p>'
                     f'<p><b>Почему:</b> {esc(r.why_ok)}{("; " + esc(r.balance)) if r.balance else ""}</p>{extra}</details>')
    fc = "" if forecast is None else forecast.to_html(index=False, border=0, classes="fc")
    doc = f"""<!doctype html><html lang="ru"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{esc(title)}</title><style>
:root{{--bg:#fff;--fg:#1d2330;--mut:#667;--line:#dde1e8}}@media(prefers-color-scheme:dark){{:root{{--bg:#14171f;--fg:#e6e9f0;--mut:#9aa3b5;--line:#2b3140}}}}
body{{font:15px/1.5 system-ui,sans-serif;background:var(--bg);color:var(--fg);max-width:820px;margin:0 auto;padding:16px}}
table{{border-collapse:collapse;width:100%}}td,th{{padding:6px 8px;border-bottom:1px solid var(--line);text-align:left}}
.chip{{display:inline-block;width:10px;height:10px;border-radius:50%;margin-right:8px}}details{{border-bottom:1px solid var(--line);padding:6px 0}}
summary{{cursor:pointer;font-weight:600}}p{{margin:4px 0}}.mut{{color:var(--mut)}}.fc{{font-size:13px}}
</style></head><body><h1>{esc(title)}</h1><p class="mut">{esc(notes)}</p><h2>Пулы утомления</h2>{''.join(svg)}
<h2>Календарь</h2><table><tr><th>День</th><th>Сессия</th><th>AU</th><th>Вклад</th></tr>{rows}</table>
<h2>Почему так</h2>{''.join(cards)}<h2>Прогноз турнира</h2>{fc}</body></html>"""
    with open(path, "w", encoding="utf-8") as f:
        f.write(doc)
    return path


# ═════════════════════════════ 9. MONTECARLO — СЛОЙ 3: вероятностный турнир ═════════════════════════════
# Дисперсия «формы дня» по видам, % (низкая: 100 м, 110 м с/б; высокая: шест, диск, копьё, длина, высота) [заглушки]
DAY_SIGMA = {"100m": .9, "LJ": 1.8, "SP": 2.0, "HJ": 1.5, "400m": 1.2, "110mH": 1.1, "DT": 4.0, "PV": 2.5, "JT": 4.0, "1500m": 1.8}
ATT_SD = {"LJ": 2.0, "SP": 1.5, "DT": 4.0, "JT": 4.0}          # разброс между попытками внутри соревнования, %
P_FOUL = {"LJ": .10, "SP": .03, "DT": .12, "JT": .05}          # вероятность заступа/недействительной попытки
BAR = {"HJ": dict(step=3.0, s=2.0, open_off=15.0, p_nh=.01),   # см: шаг планки, масштаб логистики, разница опорного/заявочной, P(нет высоты) не по умению
       "PV": dict(step=10.0, s=6.0, open_off=40.0, p_nh=.02)}
P_HURDLE_CRASH, P_1500_DNF = .02, .005


@dataclass
class MCResult:
    pts: pd.DataFrame; res: pd.DataFrame; total: np.ndarray; thresholds: Tuple[int, ...]

    def summary(self) -> pd.DataFrame:
        rows = []
        for e in EVENTS:
            rows.append(dict(event=e, median_result=np.nanmedian(self.res[e]), median_pts=self.pts[e].median(),
                             p05=self.pts[e].quantile(.05), p95=self.pts[e].quantile(.95), P_zero=(self.pts[e] == 0).mean()))
        return pd.DataFrame(rows).round(3)

    def total_stats(self) -> dict:
        t = self.total
        out = dict(median=float(np.median(t)), sum_of_medians=float(self.pts.median().sum()),
                   p05=float(np.percentile(t, 5)), p95=float(np.percentile(t, 95)), mean=float(t.mean()))
        out.update({f"P(>={T})": float((t >= T).mean()) for T in self.thresholds})
        return out

    def sensitivity(self, lo: Optional[float] = None, hi: Optional[float] = None) -> pd.DataFrame:
        """Какой вид даёт наибольший математический риск потери очков:
        var_share — доля дисперсии суммы (с учётом корреляций между видами);
        ES10 — средний недобор очков вида в худших 10 % относительно его медианы;
        protect — прирост P(сумма ≥ lo), если результат вида «заморозить» на его медиане (убрать риск провала);
        upside — прирост P(сумма ≥ hi), если вид выдаст свой P90 (резерв роста).
        lo — рубеж не выше медианы суммы, hi — ближайший рубеж выше медианы."""
        t = self.total
        med = float(np.median(t))
        ths = sorted(self.thresholds)
        lo = lo or next((T for T in reversed(ths) if T <= med), ths[0])
        hi = hi or next((T for T in ths if T > med), ths[-1])
        b_lo, b_hi = (t >= lo).mean(), (t >= hi).mean()
        rows = []
        for e in EVENTS:
            x, mx = self.pts[e].to_numpy(), float(self.pts[e].median())
            q10 = np.percentile(x, 10)
            rows.append(dict(event=e, var_share=np.cov(x, t)[0, 1] / t.var(), ES10=mx - x[x <= q10].mean(),
                             P_zero=(x == 0).mean(), protect=((t - x + mx) >= lo).mean() - b_lo,
                             upside=((t - x + np.percentile(x, 90)) >= hi).mean() - b_hi))
        out = pd.DataFrame(rows).sort_values("var_share", ascending=False).round(3)
        out.attrs["lo"], out.attrs["hi"] = lo, hi
        return out


class TournamentMC:
    """Монте-Карло двух дней. Источники риска: (1) коррелированная «форма дня» (многомерное t, df=6 — тяжёлые хвосты и
    общий «плохой день», корреляции = популяционная матрица), (2) 3 попытки с заступами в LJ/SP/DT/JT (результат = лучшая
    действительная), (3) прыжки с планкой: шаг планки, 3 попытки на высоте, риск «нет высоты», (4) падение на барьере, DNF."""

    def __init__(self, corr: Optional[np.ndarray] = None, sigma: Optional[Dict[str, float]] = None, nu: float = 6.0):
        self.C = build_event_corr() if corr is None else corr
        self.sig = np.array([(sigma or DAY_SIGMA)[e] for e in EVENTS])
        self.nu = nu

    def simulate(self, mean: Dict[str, float], n: int = 10_000, seed: int = 0,
                 thresholds: Tuple[int, ...] = (6500, 7000, 7500, 8000)) -> MCResult:
        rng = np.random.default_rng(seed)
        Lc = np.linalg.cholesky(self.C * np.outer(self.sig, self.sig))
        w = rng.chisquare(self.nu, n) / self.nu
        eps = (rng.standard_normal((n, 10)) @ Lc.T) * (np.sqrt((self.nu - 2) / self.nu) / np.sqrt(w))[:, None]  # SD сохраняется
        res, pts = {}, {}
        for i, e in enumerate(EVENTS):
            m = mean[e]
            if e in ("100m", "400m", "110mH", "1500m"):
                r = m * (1 - eps[:, i] / 100.0)
                if e == "110mH":
                    r = r + (rng.random(n) < P_HURDLE_CRASH) * rng.uniform(.5, 1.5, n)
                if e == "1500m":
                    r = np.where(rng.random(n) < P_1500_DNF, np.nan, r)
            elif e in BAR:
                r = self._bar(e, m * 100.0, eps[:, i], rng, n) / 100.0
            else:
                sd = ATT_SD[e] / 100.0
                Lat = m / (1 + 0.846 * sd)                      # E[max из 3] ≈ L(1+0.846·sd) → центр так, чтобы E[best]≈m
                a = Lat * (1 + (eps[:, [i]] + rng.normal(0, ATT_SD[e], (n, 3))) / 100.0)
                best = np.where(rng.random((n, 3)) > P_FOUL[e], a, -np.inf).max(1)
                r = np.where(np.isfinite(best), best, np.nan)    # 3 заступа → нет результата → 0 очков
            r = np.where(np.isfinite(r) & (r > 0), r, np.nan)
            res[e] = r
            pts[e] = np.where(np.isfinite(r), wa_points(e, np.nan_to_num(r, nan=1.0), floor=True), 0.0)
        P = pd.DataFrame(pts)
        return MCResult(P, pd.DataFrame(res), P.sum(axis=1).to_numpy(), thresholds)

    @staticmethod
    def _bar(e: str, pred_cm: float, eps: np.ndarray, rng, n: int) -> np.ndarray:
        """Планка: на каждой высоте 3 попытки, p(попытка)=σ((C−h)/s). Медиана последней взятой высоты ≈ C+1.35s
        (P(взять)=1−(1−p)³=½ ⇒ p≈0.206), поэтому центр C = pred − 1.35·s — прогноз остаётся несмещённым."""
        b = BAR[e]
        C = pred_cm * (1 + eps / 100.0) - 1.35 * b["s"]
        h0 = np.floor((pred_cm - b["open_off"]) / b["step"]) * b["step"]
        alive = rng.random(n) > b["p_nh"]
        best = np.zeros(n)
        for k in range(60):
            h = h0 + k * b["step"]
            p = 1.0 / (1.0 + np.exp(-(C - h) / b["s"]))
            alive &= rng.random(n) < 1.0 - (1.0 - p) ** 3
            if not alive.any():
                break
            best = np.where(alive, h, best)
        return np.where(best > 0, best, np.nan)


def _selftest_tensor():
    c = PoolConstants()
    # 1. ровно половина через T½
    for k, p in enumerate(POOLS):
        s = PoolState(c).apply(StressVector(1, 1, 1)).advance(c.half_life_h[k])
        assert abs(s.P[k] - 0.5) < 1e-12, p
    # 2. аддитивность и независимость от разбиения времени
    a = PoolState(c).apply(StressVector(.9, .1, .8)).advance(10).advance(14)
    b = PoolState(c).apply(StressVector(.9, .1, .8)).advance(24)
    assert np.allclose(a.P, b.P)
    # 3. hours_until согласуется с advance
    s = PoolState(c).apply(StressVector(.92, .08, .25))
    h = s.hours_until("CNS", 0.40)
    assert abs(s.copy().advance(h).P[0] - 0.40) < 1e-9
    # 4. диапазоны из ТЗ защищены
    try:
        PoolConstants((10.0, 48.0, 84.0))
        raise SystemExit("ожидался ValueError")
    except ValueError:
        pass
    print("самопроверка пройдена")




def _selftest_library():
    for k, prof in REQUIRED.items():
        assert LIBRARY[k].levels() == prof, k
    # разница, ради которой нужна гранулярность
    assert LIBRARY["S_HJ_SHORT"].stress.cns < 0.7 <= LIBRARY["S_HJ_FULL"].stress.cns
    assert LIBRARY["S_FLY30"].stress.met < 0.35 and LIBRARY["S_GLYCO100"].stress.met >= 0.8
    assert LIBRARY["S_HILL_SL"].stress.mech >= 0.95 > LIBRARY["S_HJ_FULL"].stress.mech
    for s in LIBRARY:
        assert abs(sum(s.chan.values()) - (1.0 if s.parts and s.key != "S_MOBILITY" else sum(s.chan.values()))) < 1e-9
    st = PoolState()
    LIBRARY.apply(st, "S_EXPLOSIVE")
    assert abs(st.P[0] - LIBRARY["S_EXPLOSIVE"].stress.cns) < 1e-12
    print("самопроверка пройдена: профили совпадают с расчётом, шесть модулей ТЗ на месте")




def _selftest_gates():
    g, S = BioGates(), LIBRARY
    ev = lambda *items: [(24 * d + 17, k, 1.0) for d, k in items]
    # --- CNS Gate: после тяжёлого зала ---
    st = AthleteState(); st.record(S["S_EXPLOSIVE"], 17)
    assert not g.check(st, S["S_FLY30"], 24 + 17).allowed, "30 м с ходу на следующий день после зала должны быть запрещены"
    assert not g.check(st, S["S_PV_FULL"], 24 + 17).allowed
    assert g.check(st, S["S_THROW_TECH"], 24 + 17).allowed, "умеренный CNS шлюзом не блокируется"
    assert g.check(st, S["S_FLY30"], 48 + 17).allowed, "через 48 ч свежесть восстановлена"
    # --- кумулятивность CNS: два умеренных дня подряд + 30 м ---
    st2 = AthleteState(); st2.record(S["S_THROW_POWER"], 17); st2.record(S["S_ACCEL20"], 41)
    assert not g.check(st2, S["S_FLY30"], 65).allowed
    # --- Interference Gate ---
    st = AthleteState(); st.record(S["S_GLYCO100"], 17)
    v = g.check(st, S["S_PV_SHORT"], 24 + 17)
    assert not v.allowed and v.blocks[0].gate == "INTERFERENCE" and abs(v.free_in_h - 24) < 1e-9
    assert g.check(st, S["S_PV_SHORT"], 48 + 17).allowed
    assert g.check(st, S["S_THROW_TECH"], 24 + 17).allowed, "не-техническая сессия лактатом не блокируется"
    # --- Tendon Protection: 72 ч, а после горки 96 ч ---
    st = AthleteState(); st.record(S["S_EXPLOSIVE"], 17)
    assert not g.check(st, S["S_DROP_JUMPS"], 48 + 17).allowed and g.check(st, S["S_DROP_JUMPS"], 72 + 17).allowed
    st = AthleteState(); st.record(S["S_HILL_SL"], 17)
    t = g.check(st, S["S_EXPLOSIVE"], 72 + 17)
    assert not t.allowed and any(b.gate == "TENDON" for b in t.blocks), "после горки 72 ч мало"
    assert g.check(st, S["S_DROP_JUMPS"], 96 + 17).allowed
    # --- earliest_allowed согласован с check ---
    st = AthleteState(); st.record(S["S_EXPLOSIVE"], 17)
    for key in ("S_FLY30", "S_HJ_FULL", "S_HILL_SL"):
        h = g.earliest_allowed(st, S[key], 24 + 17)
        assert g.check(st, S[key], h + 1e-6).allowed, key
        if h > 24 + 17 + 1e-6:
            assert not g.check(st, S[key], h - 0.5).allowed, key
    # --- валидатор расписания ---
    bad = g.validate_schedule(ev((0, "S_EXPLOSIVE"), (1, "S_FLY30"), (2, "S_HILL_SL")))
    assert list(bad.allowed) == [True, False, False]
    print("самопроверка пройдена: все три шлюза и кумулятивность работают, earliest_allowed точен")




def _selftest_solver():
    model, gc = PerfModel(), GateConfig()
    ath = Athlete("test", 77.5, {"100m": 11.35, "LJ": 7.35, "SP": 12.28, "HJ": 2.06, "400m": 51.99, "110mH": 14.95,
                                 "DT": 31.84, "PV": 4.30, "JT": 47.72, "1500m": 284.53}, ("DT", "JT", "1500m"))
    st = warm_start(model)
    sol = DailySolver(model, ath, st)
    # 1) векторные шлюзы ≡ эталонный BioGates на случайных планах
    rng, names, mism = np.random.default_rng(0), sol.names, 0
    for _ in range(150):
        seq = list(rng.choice(names, 10))
        _, _, _, tr = sol.evaluate(seq, trace=True)
        ver = sol.verify(seq)
        mine = [not any(k in ("CNS", "INTERFERENCE", "TENDON") for k in t["bad"]) for t in tr]
        mism += int(mine != list(ver.allowed))
    assert mism == 0, f"векторные шлюзы разошлись с BioGates в {mism} планах"
    # 2) beam без отсечения = полный перебор (малая задача)
    sub = SessionLibrary([LIBRARY[k] for k in ("S_FLY30", "S_HILL_SL", "S_HJ_FULL", "S_EXPLOSIVE", "S_GLYCO100",
                                              "S_CROSS30", "S_THROW_TECH", "S_REST")])
    small = DailySolver(model, ath, st, library=sub, limits=PlanLimits(max_rest_per_week=4))
    H = 4
    best = max((small.evaluate(list(s))[1] for s in itertools.product(small.names, repeat=H) if small.evaluate(list(s))[0]))
    got = small.solve(H, beam=5000, top=1, refine=False)[0]["objective"]
    assert abs(best - got) < 1e-6, (best, got)
    # 3) решение допустимо и не хуже «как раньше»; принудительная сессия соблюдается
    plan = sol.solve(14, beam=150, top=1)[0]
    assert plan["feasible"] and sol.verify(plan["seq"]).allowed.all()
    assert plan["objective"] >= sol.baseline(14)[1]["objective"] - 1e-6 or not sol.baseline(14)[0]
    forced = sol.solve(7, beam=50, top=1, locked={2: "S_REST"}, refine=False)[0]["seq"]
    assert forced[2] == "S_REST"
    print("самопроверка пройдена: шлюзы идентичны эталону, beam = перебор на малой задаче, план допустим")




# ═════════════════════════════ 9b. SEASON — сезонный план по недельным моделям Ушакова/Верхошанского (шаблон + шлюзы) ═════════════════════════════
# Структура и порядок видов по дням — из журнала (WEEK_MODEL, LIT_PERIODIZATION: мезоциклы 3–4 нед., воскресенье — отдых,
# спринт с декабря–января, 2–3 облегчённых многоборья зимой). Конкретные упражнения, объёмы и темпы ниже — МОИ ориентиры
# (экспертные, не из журнала): подгоняйте по самочувствию и тестам. Оптимизатор здесь не выбирает сессии, а проверяет шаблон шлюзами.
WD_RU = ["Пн", "Вт", "Ср", "Чт", "Пт", "Сб", "Вс"]
_SLOT_KEY = {"LJ": "S_ACCEL20", "SP": "S_THROW_POWER", "HJ": "S_HJ_SHORT", "JT": "S_THROW_TECH", "DT": "S_THROW_TECH", "PV": "S_PV_SHORT",
             "110mH": "S_HURDLE_RHYTHM", "weights": "S_STRENGTH_BASE", "jump_drills": "S_DROP_JUMPS", "sprint": "S_FLY30",
             "throw*": "S_THROW_TECH"}
_RUN_KEY = {"aerobic": "S_CROSS30", "tempo": "S_TEMPO_RUN", "speed": "S_ACCEL20", "interval": "S_INT1000", "easy": "S_CROSS30"}
_RUN_DAY = {"prep": {"Пн": "aerobic", "Вт": "tempo", "Ср": "speed", "Пт": "interval", "Сб": "easy"},
            "precomp": {"Вт": "tempo", "Ср": "speed", "Чт": "speed", "Пт": "easy"}}
_FALLBACK = {"S_FLY30": "S_TEMPO_RUN", "S_ACCEL20": "S_TEMPO_RUN", "S_THROW_POWER": "S_THROW_TECH", "S_THROW_TECH": "S_MOBILITY",
             "S_PV_FULL": "S_PV_SHORT", "S_PV_SHORT": "S_MOBILITY", "S_HJ_FULL": "S_HJ_SHORT", "S_HJ_SHORT": "S_MOBILITY",
             "S_EXPLOSIVE": "S_STRENGTH_BASE", "S_STRENGTH_BASE": "S_MOBILITY", "S_INT1000": "S_TEMPO_RUN", "S_SPEEDEND300": "S_TEMPO_RUN",
             "S_HURDLE_RHYTHM": "S_MOBILITY", "S_HURDLE_PLYO": "S_MOBILITY", "S_DROP_JUMPS": "S_MOBILITY", "S_HILL_SL": "S_DROP_JUMPS",
             "S_TEMPO_RUN": "S_CROSS30", "S_GLYCO100": "S_TEMPO_RUN", "S_CROSS30": "S_MOBILITY", "S_MOBILITY": "S_REST"}
# Упражнения по слотам и фазам: A — общая/функциональная, B — специальная, C — предсоревновательная
_EX = {
    "LJ": {"A": ["Длина: 6–8 прыжков с 8–10 шагов, акцент на 3 последних шага; 4×5 многоскоков", "Длина: прыжки в шаге 5×(5–6), 4 прыжка с короткого разбега"],
           "B": ["Длина: 6–8 прыжков с 12–14 шагов, отметки, отталкивание", "Длина: 5–6 прыжков с 14–16 шагов + 3×30 м скоростные прыжки"],
           "C": ["Длина: 3–4 прыжка с полного разбега (16–18 шагов), точность отметки"]},
    "SP": {"A": ["Ядро: 20–25 бросков (5 кг/7,26 кг, стоя и со скачка) + 3×8 броски 5 кг снизу вперёд/назад", "Ядро: 15–20 бросков со скачка, 3×6 толчок из-за головы 5 кг"],
           "B": ["Ядро: 12–15 бросков 7,26 кг со скачка + 4–6 бросков 6 кг", "Ядро: 10–12 бросков с техникой в полную силу, 6 лёгких 6 кг"],
           "C": ["Ядро: 6–8 бросков в полную технику, 3–4 лёгких"]},
    "HJ": {"A": ["Высота: 12–15 прыжков с 5–7 шагов на средних высотах + 3×6 прыжков через барьеры", "Высота: 10–12 прыжков с 7–9 шагов, акцент на постановку ноги"],
           "B": ["Высота: 8–12 прыжков с полного разбега на 90–95 % личного рекорда", "Высота: 8–10 прыжков с полного разбега, серия на 3 высотах"],
           "C": ["Высота: 5–7 прыжков, тактика 3 попыток; стартовая высота ≈ на 12–15 см ниже личного рекорда"]},
    "JT": {"A": ["Копьё: 20–25 метаний (мячи 600–800 г + копьё), импульсный шаг; резина 3×10", "Копьё: 15–20 метаний с 3–5 шагов + 3×8 с резиной"],
           "B": ["Копьё: 10–14 метаний с полного разбега + 6 метаний с места на силу", "Копьё: 10–12 с полного разбега, работа с последними шагами"],
           "C": ["Копьё: 5–6 метаний с полного разбега"]},
    "DT": {"A": ["Диск: 20–25 бросков (1,75–2 кг) с поворота, работа над ногами и осью", "Диск: 15–20 бросков с полуповорота + 3×8 броски ядра 5 кг назад"],
           "B": ["Диск: 12–15 бросков с полного поворота, 2 кг и 1,75 кг", "Диск: 10–12 бросков с поворота, акцент на финальное усилие"],
           "C": ["Диск: 5–6 бросков в полную технику"]},
    "PV": {"A": ["Шест: 8–10 прыжков с 6–8 шагов, хват ниже на 20–40 см; канат/кольца 3×", "Шест: 8–10 прыжков с короткого разбега + гимнастика (упор, вис) 15 мин"],
           "B": ["Шест: 8–10 прыжков с 10–12 шагов на уровне 3,80–4,10", "Шест: 6–8 прыжков с 10–12 шагов + 3 прыжка с полного разбега"],
           "C": ["Шест: 5–7 прыжков с полного разбега, тактика на высотах 3,80–4,10"]},
    "110mH": {"A": ["Барьеры: ритм 3 шага через 5–6 барьеров ×4–5 + мобильность тазобедренных", "Барьеры: рука/нога-перемах 4×10 + 3×(5 барьеров) невысоких"],
              "B": ["Барьеры: 3×(5 барьеров) на скорости + 4 старта до 1-го барьера", "Барьеры: 4×(4–5 барьеров) на 90–95 %, старт 3×"],
              "C": ["Барьеры: 2×(3–4 барьера) на соревновательной скорости + 2 старта"]},
    "weights": {"A": ["Штанга ≈ 50 мин: присед 4×6 (75–80 %), тяга 4×5, жим 4×6, рывок 4×3", "Штанга ≈ 50 мин: присед 4×5, румынская тяга 3×8, жим 4×5, толчок 4×3"],
                "B": ["Штанга ≈ 45 мин: присед 3×3 (85–90 %), рывок 5×2, жим 4×4", "Штанга ≈ 45 мин: присед 4×3 (85 %), толчок 5×2, жим 4×4"],
                "C": ["Штанга ≈ 30 мин: присед 2×2 (90 %), рывок 3×2 — поддерживающая"]},
    "jump_drills": {"A": ["Прыжковые: 6×30 м прыжки на одной/двух ногах, 5×5 через барьеры", "Прыжковые: многоскоки 4×(6 прыжков), подскоки 3×20"],
                    "B": ["Прыжковые: горка 6×20 м одной ногой + спрыгивания 4×6", "Прыжковые: спрыгивания 4×6 + 3×(5 прыжков через барьеры)"],
                    "C": ["Прыжковые: 3×5 подскоков, 2×30 м прыжки (короткие, быстрые)"]},
    "sprint": {"A": ["Спринт: ускорения 6×40 м (90 %)"], "B": ["Спринт: 4×30 м с ходу (отдых 4–5 мин) + 3×60 м", "Спринт: 3×20 м со старта + 3×40 м с ходу"],
               "C": ["Спринт: 3×30 м с ходу, 2×60 м, 3 старта 20 м"]},
    "throw*": {"A": ["Метание на выбор (диск/копьё/ядро): 12–15 бросков"], "B": ["Метание на выбор: 10–12 бросков с полной техникой"], "C": ["Метание на выбор: 4–6 бросков, лёгкие"]}}
_RUN_EX = {
    "aerobic": {"A": ["Кросс 25–30 мин (пульс 140–150) + 6×100 м ускорения", "Кросс 30–35 мин по пересечённой местности + 4×60 м ускорения"], "B": ["Кросс 20–25 мин + 6×100 м ускорения"], "C": ["Кросс 15–20 мин + 4×60 м ускорения"]},
    "tempo": {"A": ["Темп 5×1000 м (≈3:10–3:20), отдых 2:30", "Фартлек 25 мин: 6×(1:30 быстро/1:30 трусцой)", "Темп 3×1500 м (≈5:00–5:10), отдых 4 мин", "Лесенка 400-600-800-600-400 (≈ 85 %), отдых 2–3 мин"],
              "B": ["Темп 3×1000 м (≈3:00–3:05), отдых 3 мин", "4×600 м (≈1:45–1:50), отдых 3 мин", "Фартлек 20 мин: 5×(1:00 быстро/1:30 трусцой)"], "C": ["Темп 2×600 м (≈1:45), отдых 4 мин"]},
    "speed": {"A": ["Ускорения 8×60 м (90 %), отдых 2 мин", "Эстафетные отрезки 6×80 м (92 %)"], "B": ["6×60 м со старта (95 %), отдых 3 мин", "4×80 м с ходу (95 %), отдых 4 мин"],
              "C": ["3×40 м со старта + 2×60 м (98 %), отдых 5 мин"]},
    "interval": {"A": ["Интервал 8×400 м (≈75–78 с), отдых 90 с", "Горка 8×150 м в подъём, трусцой вниз", "6×500 м (≈1:35–1:40), отдых 2:30"],
                 "B": ["6×300 м (≈41–43 с), отдых 4 мин", "4×500 м (≈1:25–1:28), отдых 5 мин", "3×(150+150 через 30 с), отдых 6 мин"], "C": ["2×300 м (≈39–41 с), отдых 8 мин"]},
    "easy": {"A": ["Восстановительный бег 20 мин + мобильность 15 мин", "Лёгкий кросс 25 мин + растяжка"], "B": ["Восстановительный бег 15–20 мин + мобильность"], "C": ["Лёгкий бег 12–15 мин + мобильность"]}}
# Фазы сезона: (название, начало, конец, шаблон, фаза упражнений, волна объёма по неделям мезоцикла)
def season_phases(hept_d1: pd.Timestamp, dec_d1: pd.Timestamp, start: pd.Timestamp):
    t = pd.Timestamp
    return [
        ("1. Общая и функциональная (2 мезоцикла по 4 нед.)", start, t("2026-12-02"), "prep", "A", [1.00, 1.10, 1.15, 0.70]),
        ("2. Специальная, спринт включён (мезоциклы по 3 нед.)", t("2026-12-03"), t("2027-01-11"), "prep", "B", [1.00, 1.10, 0.70]),
        ("3. Предсоревновательная к семиборью", t("2027-01-12"), hept_d1 - pd.Timedelta(days=7), "precomp", "C", [1.00, 0.90]),
        ("4. Подводка к семиборью", hept_d1 - pd.Timedelta(days=6), hept_d1 - pd.Timedelta(days=1), "taper", "C", [0.6]),
        ("5. Семиборье", hept_d1, hept_d1 + pd.Timedelta(days=1), "comp", "C", [1.0]),
        ("6. Восстановление после старта", hept_d1 + pd.Timedelta(days=2), hept_d1 + pd.Timedelta(days=7), "recover", "A", [0.5]),
        ("7. Техника и база (мезоциклы по 3 нед.)", hept_d1 + pd.Timedelta(days=8), t("2027-03-21"), "prep", "A", [1.05, 1.15, 0.70]),
        ("8. Весенний этап — специальная работа", t("2027-03-22"), t("2027-05-02"), "prep", "B", [1.10, 1.20, 0.75]),
        ("9. Предсоревновательная к десятиборью", t("2027-05-03"), dec_d1 - pd.Timedelta(days=7), "precomp", "C", [1.10, 1.00, 0.90]),
        ("10. Подводка к десятиборью", dec_d1 - pd.Timedelta(days=6), dec_d1 - pd.Timedelta(days=1), "taper", "C", [0.6]),
        ("11. Десятиборье", dec_d1, dec_d1 + pd.Timedelta(days=1), "comp", "C", [1.0])]


_TAPER = ["Скорость: 3×30 м с ходу (95 %), 2×60 м; ядро 6 бросков; разминка 20 мин", "Техника: шест 4–5 прыжков, барьеры 3×3, высота 4 прыжка",
          "Кросс 20 мин + мобильность 20 мин", "Короткая скорость: 2×30 м, 2×60 м со старта; метания лёгкие 4–5 бросков",
          "Мобильность 30 мин, самомассаж, сон", "Активация: трусцой 10 мин, 3×60 м ускорения, 3 старта 20 м; регистрация, питание"]
_TAPER_KEY = ["S_FLY30", "S_PV_SHORT", "S_CROSS30", "S_FLY30", "S_MOBILITY", "S_MOBILITY"]


def build_season(start: pd.Timestamp, hept_d1: pd.Timestamp, dec_d1: pd.Timestamp, solver: Optional["DailySolver"] = None) -> pd.DataFrame:
    """Календарь по дням: фаза, неделя мезоцикла и волна объёма, упражнения по порядку видов из модели Ушакова, ключ сессии для проверки
    шлюзами. solver задан — каждая сессия проверяется CNS/Interference/Tendon-шлюзами; не прошла → облегчённая замена по цепочке."""
    phases = season_phases(hept_d1, dec_d1, start)
    days = pd.date_range(start, dec_d1 - pd.Timedelta(days=1), freq="D")       # как в оптимизаторе: последний день перед D1
    rows, seq = [], []
    wi = {}       # счётчик недель внутри фазы
    for d in days:
        ph = next(p for p in phases if p[1] <= d <= p[2])
        name, p0, p1, tpl, ex, wave = ph
        wk = int((d - p0).days // 7)
        f = wave[wk % len(wave)]
        wd = WD_RU[d.weekday()]
        items, keys = [], []
        if tpl in ("prep", "precomp"):
            order = WEEK_MODEL[tpl].get(wd, [])
            ordered = list(order)
            if tpl == "prep" and name.startswith("2.") and wd == "Чт":
                ordered = ordered + ["sprint"]
            for j, slot in enumerate(ordered):
                if slot == "run":
                    rt = _RUN_DAY[tpl].get(wd, "easy")
                    menu = _RUN_EX[rt][ex]; keys.append(_RUN_KEY[rt])
                elif slot == "throw*":
                    pick = ["JT", "DT", "SP"][(wk + d.weekday()) % 3]
                    menu = _EX[pick][ex]; keys.append(_SLOT_KEY[pick])
                else:
                    menu = _EX[slot][ex]; keys.append(_SLOT_KEY[slot])
                items.append(menu[(wk + j + d.weekday()) % len(menu)])
            if tpl == "precomp" and wd == "Сб" and not ordered:
                pass
            if not items:
                items, keys = ["Отдых: сон, питание, 15 мин мобильности"], ["S_REST"]
            text = " → ".join(f"{i + 1}) {t}" for i, t in enumerate(items))
        elif tpl == "taper":
            k = (d - p0).days; text = _TAPER[k]; keys = [_TAPER_KEY[k]]
        elif tpl == "comp":
            text = "СТАРТ (день {})".format((d - p0).days + 1); keys = ["S_REST"]
        else:  # recover
            k = (d - p0).days
            text = ["Отдых", "Кросс 20 мин + мобильность", "Мобильность, бассейн/баня", "Лёгкая техника: шест/барьеры по 20 мин", "Скорость: 4×30 м (90 %) + лёгкие броски", "Кросс 25 мин + мобильность"][k % 6]
            keys = [["S_REST", "S_CROSS30", "S_MOBILITY", "S_PV_SHORT", "S_ACCEL20", "S_CROSS30"][k % 6]]
        if wd == "Вс" and tpl in ("prep", "precomp"):
            text, keys = "Отдых полный (сон, питание, прогулка)", ["S_REST"]
        key = max(keys, key=lambda k: LIBRARY[k].stress.cns) if solver is None else None
        if solver is not None:
            cand = max(keys, key=lambda k: LIBRARY[k].stress.cns)
            note = ""
            while True:
                feas, _, _, _, tr = solver._run(seq + [cand], True)
                bad = tr[-1]["bad"]
                if not bad or cand == "S_REST":
                    break
                note = f"облегчено шлюзом {','.join(bad)}: {cand}→{_FALLBACK[cand]}"
                cand = _FALLBACK[cand]
            key = cand
            if note:
                text = text + f"  ⚠ {note}"
        else:
            note = ""
        seq.append(key)
        rows.append(dict(date=d, weekday=wd, phase=name, week_in_phase=wk + 1, volume_factor=f, session_key=key, plan=text,
                         gate_note=note if solver is not None else ""))
    return pd.DataFrame(rows)


def season_to_html(df: pd.DataFrame, path: str, title: str, notes: str = "") -> str:
    h = ["<!doctype html><meta charset='utf-8'><title>" + _html.escape(title) + "</title><style>body{font:14px system-ui;margin:20px;max-width:1100px}"
         "h1{font-size:20px}h2{font-size:16px;margin:22px 0 4px;background:#e9eef5;padding:6px 8px;border-radius:4px}"
         "table{border-collapse:collapse;width:100%;margin-bottom:6px}td{border-bottom:1px solid #ddd;padding:4px 6px;vertical-align:top}"
         "td.d{white-space:nowrap;width:92px;color:#555}td.f{width:46px;color:#555}tr.rest td{color:#999}tr.comp td{background:#fff3cd;font-weight:600}"
         ".w{color:#a15c00}.note{background:#f6f6f6;padding:8px;border-radius:4px}</style><h1>" + _html.escape(title) + "</h1><div class='note'>" + _html.escape(notes) + "</div>"]
    for ph, g in df.groupby("phase", sort=False):
        h.append(f"<h2>{_html.escape(ph)} ({g.date.iloc[0]:%d.%m}–{g.date.iloc[-1]:%d.%m.%Y})</h2><table>")
        for _, r in g.iterrows():
            cls = "rest" if r.session_key == "S_REST" else ("comp" if r.plan.startswith("СТАРТ") else "")
            wk = f"нед.{r.week_in_phase} ×{r.volume_factor:.2f}" if r.weekday == "Пн" else ""
            plan = _html.escape(r.plan).replace("⚠", "<span class='w'>⚠</span>")
            gn = r.get("gate_note", "")
            if isinstance(gn, str) and gn:
                plan += f" <span class='w'>⚠ {_html.escape(gn)} (совет модели, занятие не менялось)</span>"
            h.append(f"<tr class='{cls}'><td class='d'>{r.date:%d.%m} {r.weekday}</td><td class='f'>{wk}</td><td>{plan}</td></tr>")
        h.append("</table>")
    open(path, "w", encoding="utf-8").write("".join(h))
    return path


def run_season(a):
    pd.set_option("display.width", 200)
    ath = samandyk()
    start = pd.Timestamp(a.start).normalize() if a.start else pd.Timestamp("2026-10-08")
    hept, dec = pd.Timestamp(a.hept), pd.Timestamp(a.dec)
    model = PerfModel.from_prior()
    state = warm_start(model)
    sol = DailySolver(model, ath, state)
    cmp_df = None
    if getattr(a, "template", False):
        df = build_season(start, hept, dec, sol)
    else:
        df, cmp_df = build_season_grz(start, hept, dec, sol, floor_weeks=getattr(a, 'floor', 2), emph=emphasis_from_profile(ath.current), endur_cap=5.0)
    seq = list(df.session_key)
    feas, obj, res = sol.evaluate(seq)
    pts = sum(float(wa_points(e, res[e], floor=True)) for e in EVENTS)
    cnt = df.session_key.value_counts()
    print(f"СЕЗОННЫЙ ПЛАН {start:%d.%m.%Y} → семиборье {hept:%d.%m.%Y}, десятиборье {dec:%d.%m.%Y}: {len(df)} дней")
    print(f"разных сессий-ключей: {cnt.size}; облегчено шлюзами: {int((df.gate_note != '').sum())} дн. из {len(df)}; допустим по шлюзам: {feas}")
    print("частоты ключей:", dict(cnt))
    if cmp_df is not None:
        print(grz_report(cmp_df))
    print(f"прогноз модели на {dec:%d.%m.%Y} при этом плане (без доз соревнований февраля): {pts:.0f} очк.")
    html_path = a.html or "season_plan.html"
    season_to_html(df, html_path, f"Сезонный план {ath.name}: семиборье {hept:%d.%m.%Y}, десятиборье {dec:%d.%m.%Y}",
                   notes=("Состав и объём тренировок по месяцам — из диссертации Грузенкина (1990), табл. 3: % годового объёма каждого средства по месяцам, число занятий "
                         "в месяц; годовые объёмы — средние группы мастеров спорта СССР (7859 очк.). Сочетание видов в один день — по Юшкевичу и Романову (2012): "
                         "110 м с/б+ядро+прыжок, 100 м+ядро/диск. Календарь: окт–дек = Х–XII; 1–25 янв = I–II (зимний этап); после семиборья до десятиборья = III–V, "
                         "растянутый. Акценты по слабым видам (множители 0,9–1,25 на число занятий и объём) и потолок кросса 5 км за занятие — мои правила. «(поддержка)» — мои добавки (≥1 раза в 2 недели для видов, которых нет в таблице в этом месяце). Недельная волна 1,0/1,1/0,75 — моя. "
                         "Подводка, дни старта и 4 дня контрольных испытаний — мои (идея тестов в начале и конце этапов — Грузенкин). ⚠ — совет шлюзов модели (экспертные пороги), занятия не меняются.") if not getattr(a, "template", False) else
                   "Порядок видов по дням — из журнала (Ушаков, Верхошанский); упражнения — ориентиры автора скрипта. ⚠ — занятие облегчено шлюзом.")
    df.to_csv(html_path.replace(".html", ".csv"), index=False, encoding="utf-8-sig")
    print("HTML:", html_path)
    return df


# ═════════════════════════════ 9c. ГРУЗЕНКИН — сезон из помесячной структуры нагрузок (диссертация 1990, табл. 3) ═════════════════════════════
# Таблица 3 Грузенкина: % годового объёма средства по месяцам Х…IX (эксперимент окт.1987–сент.1988, 6 мастеров спорта СССР, среднее 7859 очк.,
# прирост +211,5±118 очк.). Месяцы идут Х, XI, XII, I, II, III, IV, V, VI, VII, VIII, IX (индексы 0…11). Суммы строк проверены = 100 % (±0,3).
# Третье значение — годовой объём среднего по группе. Строки 1–26 таблицы; нулевые месяцы — средство в этом месяце не применялось.
_M = ["Х", "XI", "XII", "I", "II", "III", "IV", "V", "VI", "VII", "VIII", "IX"]
GRZ_ROWS = {
    "days":     ("дн.", 214.5, [6.0, 8.9, 10.7, 8.9, 6.8, 10.8, 11.5, 7.9, 8.3, 6.9, 7.4, 3.9]),            # табл.3 строка 1: тренировочные дни
    "sessions": ("зан.", 344.8, [9.7, 11.1, 11.6, 7.0, 5.5, 12.0, 12.3, 6.8, 8.2, 6.5, 6.1, 3.2]),           # строка 2: тренировочные занятия
    "run80_96": ("км", 11.5, [0, 0, 4.9, 22.5, 23.7, 0, 5.4, 11.5, 10.9, 10.7, 10.4, 0]),                       # 3: бег до 80 м, 96–100 %
    "run80_91": ("км", 9.6, [0, 5.7, 22.6, 24.0, 11.2, 0, 11.0, 8.1, 6.6, 6.2, 4.6, 0]),                       # 4: до 80 м, 91–95 %
    "run150_96": ("км", 8.9, [0, 17.0, 27.3, 6.3, 6.4, 0, 6.2, 9.3, 9.6, 8.5, 7.9, 1.5]),                      # 5: до 150 м, 96–100 %
    "run150_91": ("км", 9.2, [22.1, 23.2, 10.3, 0, 0, 10.1, 8.9, 6.0, 5.8, 5.3, 4.4, 3.6]),                    # 6: до 150 м, 91–95 %
    "run500_96": ("км", 10.4, [0, 0, 5.8, 11.6, 10.0, 0, 5.5, 27.5, 23.0, 8.6, 8.0, 0]),                       # 7: до 500 м, 96–100 %
    "run500_91": ("км", 15.9, [0, 25.5, 25.9, 9.9, 8.4, 0, 10.1, 5.5, 4.8, 5.2, 4.7, 0]),                      # 8: до 500 м, 91–95 %
    "run500_80": ("км", 26.2, [24.1, 0, 0, 24.7, 10.3, 9.6, 9.5, 7.3, 5.6, 5.8, 3.4, 0]),                      # 9: до 500 м, 80–90 %
    "endur":    ("км", 618.6, [18.6, 8.9, 8.3, 5.1, 4.8, 17.8, 9.7, 6.5, 6.3, 6.3, 5.2, 2.5]),                  # 10: бег для развития выносливости
    "h13":      ("шт.", 231.6, [0, 0, 6.5, 27.1, 31.6, 0, 5.6, 8.6, 8.7, 6.7, 5.2, 0]),                         # 11: барьеры, 1–3 барьера
    "h47":      ("шт.", 751.6, [0, 0, 12.4, 9.1, 9.2, 10.0, 11.7, 21.7, 10.2, 8.2, 7.5, 0]),                   # 12: 4–7 барьеров
    "h8p":      ("шт.", 257.3, [0, 0, 0, 0, 0, 0, 21.2, 25.6, 18.8, 18.3, 16.1, 0]),                           # 13: более 7 барьеров
    "lj":       ("раз", 311.8, [0, 0, 10.7, 23.9, 18.8, 0, 13.2, 11.1, 7.9, 7.7, 6.7, 0]),                     # 14: прыжки в длину
    "lj_full":  ("раз", 116.6, [0, 0, 0, 12.9, 23.2, 0, 0, 24.5, 13.6, 13.2, 12.6, 0]),                        # 15: в длину с полного разбега
    "hj":       ("раз", 369.1, [0, 0, 25.6, 14.9, 13.3, 0, 11.1, 9.4, 9.7, 9.5, 6.5, 0]),                      # 16: прыжки в высоту
    "hj_full":  ("раз", 185.0, [0, 0, 5.3, 19.8, 18.8, 0, 6.1, 13.5, 13.0, 13.5, 10.0, 0]),                    # 17: в высоту с полного разбега
    "pv":       ("раз", 313.3, [0, 0, 10.1, 23.3, 23.6, 0, 7.6, 11.3, 9.0, 9.1, 6.0, 0]),                      # 18: прыжки с шестом
    "pv_bar":   ("раз", 199.1, [0, 0, 4.3, 18.6, 22.0, 0, 5.1, 13.3, 12.9, 10.2, 13.6, 0]),                    # 19: с шестом через планку
    "sp":       ("раз", 906.8, [6.0, 13.2, 26.7, 9.9, 8.7, 6.5, 11.2, 5.4, 4.5, 3.8, 4.1, 0]),                 # 20: толчки ядра
    "sp_glide": ("раз", 666.1, [0, 15.5, 26.5, 11.9, 9.8, 5.5, 11.0, 6.0, 5.2, 4.7, 3.9, 0]),                  # 21: толчки ядра со скачка
    "dt":       ("раз", 797.0, [0, 0, 0, 0, 0, 30.5, 23.8, 21.2, 8.9, 8.3, 7.3, 0]),                           # 22: метание диска
    "dt_turn":  ("раз", 555.6, [0, 0, 0, 0, 0, 23.3, 27.4, 21.9, 10.6, 9.2, 7.6, 0]),                          # 23: диск с поворотом
    "jt":       ("раз", 264.8, [20.4, 25.3, 4.0, 3.4, 3.5, 12.3, 11.2, 6.8, 4.7, 4.7, 3.7, 0]),                # 24: метание копья
    "jt_full":  ("раз", 225.1, [0, 0, 0, 0, 0, 8.4, 30.0, 29.7, 10.7, 10.1, 11.1, 0]),                         # 25: копьё с полного разбега
    "strength": ("т", 294.0, [11.5, 20.4, 19.5, 5.0, 4.5, 9.3, 9.0, 5.6, 5.4, 4.8, 3.5, 1.4]),                 # 26: силовая подготовка (тонны); Х=11,5 — по сумме строки
}
# Группы средств → строки таблицы, по которым считается месячный профиль; K — условная базовая частота занятий в неделю при «среднем» месяце (МОЁ допущение)
GRZ_GROUPS = {
    "sprint": (["run80_96", "run80_91"], 1.0), "speed150": (["run150_96", "run150_91"], 0.6), "speedend": (["run500_96", "run500_91", "run500_80"], 0.7),
    "endurance": (["endur"], 1.0), "hurdles": (["h13", "h47", "h8p"], 1.2), "LJ": (["lj"], 0.7), "HJ": (["hj"], 0.7), "PV": (["pv"], 0.7),
    "SP": (["sp"], 1.0), "DT": (["dt"], 0.7), "JT": (["jt"], 0.7), "strength": (["strength"], 1.0)}
GRZ_LABEL = {"sprint": "Спринт", "speed150": "Бег до 150 м", "speedend": "Скоростная выносливость", "endurance": "Выносливость", "hurdles": "Барьеры",
             "LJ": "Длина", "HJ": "Высота", "PV": "Шест", "SP": "Ядро", "DT": "Диск", "JT": "Копьё", "strength": "Силовая"}
# Сочетания видов из статьи Юшкевича и Романова (2012): r множественной корреляции с суммой очков
# Табл. 2 и 3 (Юшкевич, Романов 2012): R множественной корреляции сочетания видов с суммой очков → вес «хорошего сочетания» в один день:
# пары w = 2,0·(R − 0,5)/0,141 (R=0,641 → 2,0), тройки w = 1,5·(R − 0,5)/0,245 (R=0,745 → 1,5). Это МОЯ шкала веса для расстановки по дням, но привязана к опубликованным R.
GRZ_R_PAIRS = {("hurdles", "SP"): .641, ("sprint", "SP"): .628, ("sprint", "DT"): .580, ("hurdles", "DT"): .573, ("hurdles", "JT"): .566, ("speedend", "SP"): .545}
GRZ_R_TRIPLES = {("hurdles", "SP", "LJ"): .745, ("hurdles", "SP", "HJ"): .661, ("hurdles", "SP", "PV"): .649, ("speedend", "SP", "PV"): .586, ("speedend", "SP", "HJ"): .581}
GRZ_PAIRS = {frozenset(k): round(2.0 * (R - 0.5) / 0.141, 2) for k, R in GRZ_R_PAIRS.items()}
GRZ_TRIPLES = [(frozenset(k), round(1.5 * (R - 0.5) / 0.245, 2)) for k, R in GRZ_R_TRIPLES.items()]    # (группы, вес)
_GRZ_HI = {"sprint", "speed150", "hurdles", "LJ", "HJ", "PV", "speedend"}
_GRZ_AM = ["sprint", "speed150", "hurdles", "LJ", "PV", "HJ", "SP", "DT", "JT", "speedend", "endurance", "strength"]   # порядок внутри дня: высокая ЦНС утром


_GRZ_DIM = [31, 30, 31, 31, 28, 31, 30, 31, 30, 31, 31, 30]      # длины месяцев Х…IX в эксперименте (дни)


def grz_stage(d: pd.Timestamp, hept_d1: pd.Timestamp, dec_d1: pd.Timestamp) -> int:
    """Отображение календаря атлета на месяцы эксперимента Грузенкина: Х–XII как есть; от 1 января до подводки к семиборью — I и II (зимний
    соревновательный этап, 60/40); после семиборья до десятиборья — весенний этап III–V, растянутый на весь промежуток (темп нагрузки в день сохраняется)."""
    if d < pd.Timestamp(year=hept_d1.year, month=1, day=1):
        return _mon_idx(d)
    pre_end = hept_d1 - pd.Timedelta(days=7)
    j1 = pd.Timestamp(year=hept_d1.year, month=1, day=1)
    if d <= pre_end:
        span = max(1, (pre_end - j1).days + 1)
        return 3 if (d - j1).days < 0.6 * span else 4
    s0 = hept_d1 + pd.Timedelta(days=8)
    t = (d - s0).days / max(1, (dec_d1 - s0).days)
    return 5 if t < 31 / 92 else (6 if t < 61 / 92 else 7)


def grz_row(name: str, m: int) -> float:
    unit, annual, sh = GRZ_ROWS[name]
    return annual * sh[m] / 100.0


def grz_group_share(g: str, m: int) -> float:
    """Доля группы в месяце m от годового объёма группы (взвешено по абсолютным объёмам строк)."""
    rows = GRZ_GROUPS[g][0]
    tot = sum(GRZ_ROWS[r][1] for r in rows)
    return sum(grz_row(r, m) for r in rows) / tot


def grz_selftest():
    for n, (u, a, sh) in GRZ_ROWS.items():
        s = sum(sh)
        if n == "days":          # строка 1 сканирована с суммой 98,0 — в расчётах не используется
            continue
        assert abs(s - 100.0) < 1.0, (n, s)
    return True


def _mon_idx(ts: pd.Timestamp) -> int:
    return (ts.month - 10) % 12           # окт=0 … сен=11


_GRZ_MAINT = {
    "sprint": ("Спринт (поддержка): 4×30 м с ходу на 90 %, отдых 4 мин", "S_ACCEL20"), "hurdles": ("Барьеры (поддержка): 3×(4 барьера) ритм + 10 мах-ног", "S_HURDLE_RHYTHM"),
    "LJ": ("Длина (поддержка): 6 прыжков с 8–10 шагов", "S_ACCEL20"), "HJ": ("Высота (поддержка): 8 прыжков с 5–7 шагов", "S_HJ_SHORT"),
    "PV": ("Шест (поддержка): 6 прыжков с 6–8 шагов", "S_PV_SHORT"), "SP": ("Ядро (поддержка): 10 бросков на технике", "S_THROW_TECH"),
    "DT": ("Диск (поддержка): 10 бросков с поворота на технике", "S_THROW_TECH"), "JT": ("Копьё (поддержка): 8 метаний с 3–5 шагов", "S_THROW_TECH")}


def _grz_primary(g: str, rec: Dict[str, float]) -> float:
    """Главный объём средства в единицах журнала: м (спринт, бег до 150 и 500 м), км (кросс), штуки/броски/прыжки, т (силовая)."""
    f = lambda k: rec.get(k, 0.0)
    if g == "sprint": return (f("run80_96") + f("run80_91")) * 1000
    if g == "speed150": return (f("run150_96") + f("run150_91")) * 1000
    if g == "speedend": return (f("run500_96") + f("run500_91") + f("run500_80")) * 1000
    if g == "endurance": return f("endur")
    if g == "hurdles": return f("h13") + f("h47") + f("h8p")
    return f({"LJ": "lj", "HJ": "hj", "PV": "pv", "SP": "sp", "DT": "dt", "JT": "jt", "strength": "strength"}.get(g, ""))


_MAINT_VOL = {"sprint": 120.0, "hurdles": 12.0, "LJ": 6.0, "HJ": 8.0, "PV": 6.0, "SP": 10.0, "DT": 10.0, "JT": 8.0}


def grz_session_text(g: str, m: int, amount: Dict[str, float], ex: str) -> Tuple[str, str]:
    """Текст занятия и ключ библиотеки для группы g в месяце m при объёме на занятие amount (в единицах таблицы)."""
    if g.endswith("~"):
        return _GRZ_MAINT[g[:-1]]
    f = lambda k: amount.get(k, 0.0)
    if g == "sprint":
        a, b = f("run80_96") * 1000, f("run80_91") * 1000
        L = 30 if m in (2, 3, 4) else 60
        parts = []
        if a >= 20: parts.append(f"{a:.0f} м на 96–100 % (≈{max(1, round(a / L))}×{L} м, отдых 3–5 мин)")
        if b >= 20: parts.append(f"{b:.0f} м на 91–95 % (≈{max(1, round(b / (L * 2)))}×{L * 2} м)")
        return "Спринт ≤80 м: " + " + ".join(parts or ["короткие ускорения"]), "S_ACCEL20" if m == 2 else "S_FLY30"
    if g == "speed150":
        a, b = f("run150_96") * 1000, f("run150_91") * 1000
        parts = []
        if a >= 50: parts.append(f"{a:.0f} м на 96–100 % (≈{max(1, round(a / 120))}×120 м, отдых 6–8 мин)")
        if b >= 50: parts.append(f"{b:.0f} м на 91–95 % (≈{max(1, round(b / 150))}×150 м)")
        return "Бег до 150 м: " + " + ".join(parts or ["темповые 100–150 м"]), "S_SPEEDEND300"
    if g == "speedend":
        a, b, c = f("run500_96") * 1000, f("run500_91") * 1000, f("run500_80") * 1000
        parts = [f"{v:.0f} м {lab}" for v, lab in ((a, "на 96–100 %"), (b, "на 91–95 %"), (c, "на 80–90 %")) if v >= 100]
        key = "S_SPEEDEND300" if a >= max(b, c) and a > 0 else "S_INT1000"
        return "Бег до 500 м (скоростная выносливость): " + " + ".join(parts or ["интервальный бег 200–400 м"]), key
    if g == "endurance":
        km = f('endur')
        return f"Бег для выносливости: {km:.1f} км (≈{round(km * 5.5)} мин при 5:30/км, кросс/фартлек, пульс 140–160)", "S_CROSS30"
    if g == "hurdles":
        a, b, c = f("h13"), f("h47"), f("h8p")
        parts = [f"{v:.0f} барьеров {lab}" for v, lab in ((a, "в сериях по 1–3"), (b, "в сериях по 4–7"), (c, "в сериях >7 (на скорости)")) if v >= 3]
        return "Барьеры: " + " + ".join(parts or ["ритм и мах-ноги 15 барьеров"]), "S_HURDLE_RHYTHM"
    if g == "LJ":
        n, nf = f("lj"), f("lj_full")
        return f"Длина: {n:.0f} прыжков" + (f", из них {nf:.0f} с полного разбега" if nf >= 1 else " (короткий и средний разбег)"), "S_ACCEL20"
    if g == "HJ":
        n, nf = f("hj"), f("hj_full")
        return f"Высота: {n:.0f} прыжков" + (f", из них {nf:.0f} с полного разбега" if nf >= 1 else " (короткий разбег)"), "S_HJ_FULL" if nf >= 3 else "S_HJ_SHORT"
    if g == "PV":
        n, nb = f("pv"), f("pv_bar")
        return f"Шест: {n:.0f} прыжков" + (f", из них {nb:.0f} через планку" if nb >= 1 else " (на технике, без планки)"), "S_PV_FULL" if nb >= 3 else "S_PV_SHORT"
    if g == "SP":
        n, ng = f("sp"), f("sp_glide")
        return f"Ядро: {n:.0f} толчков/бросков" + (f", из них {ng:.0f} со скачка" if ng >= 1 else ""), "S_THROW_POWER"
    if g == "DT":
        n, nt = f("dt"), f("dt_turn")
        return f"Диск: {n:.0f} бросков" + (f", из них {nt:.0f} с поворота" if nt >= 1 else ""), "S_THROW_TECH"
    if g == "JT":
        n, nf = f("jt"), f("jt_full")
        return f"Копьё: {n:.0f} метаний" + (f", из них {nf:.0f} с полного разбега" if nf >= 1 else " (с места и 3–5 шагов, мячи 600–800 г)"), "S_THROW_TECH"
    return f"Силовая: суммарный тоннаж {f('strength'):.1f} т (штанга, ядра, гири)", "S_STRENGTH_BASE"


def _grz_place(sessions: List[str], rng: np.random.Generator, prev_day_groups: List[str], ncells: int = 6, iters: int = 900) -> List[List[str]]:
    """Расставить занятия недели (список групп, возможно с повторами) по доступным дням (ncells, не более 3 групп в день): случайный поиск по штрафам и бонусам сочетаний."""
    n = len(sessions)
    best, bs = None, -1e9
    for _ in range(iters):
        days = [[] for _ in range(ncells)]
        for g in sessions:
            cand = [d for d in range(ncells) if len(days[d]) < 3 and g not in days[d]]
            if not cand:
                cand = [d for d in range(ncells) if len(days[d]) < 3]
            if not cand:
                break
            days[int(rng.choice(cand))].append(g)
        else:
            s = 0.0
            cnt = np.array([len(x) for x in days], dtype=float)
            s -= 1.6 * float(((cnt - cnt.mean()) ** 2).sum())
            for d, gs in enumerate(days):
                st = {x.rstrip('~') for x in gs}
                if len(gs) >= 3: s -= 2.0
                for pr, w in GRZ_PAIRS.items():
                    if pr <= st: s += w
                for tr_, w_ in GRZ_TRIPLES:
                    if tr_ <= st: s += w_
                if "strength" in st and st & {"sprint", "hurdles", "LJ", "speed150"}: s -= 2.5
                if "endurance" in st and st & {"sprint", "hurdles", "speed150", "speedend"}: s -= 3.0
                if "speedend" in st and st & {"sprint", "speed150", "hurdles"}: s -= 2.0
                if "endurance" in st and "strength" in st: s -= 0.5
                if len(st & _GRZ_HI) > 2: s -= 1.5 * (len(st & _GRZ_HI) - 2)
                if d > 0:
                    prv = {x.rstrip('~') for x in days[d - 1]}
                    s -= 1.0 * len(prv & st & _GRZ_HI)         # одна и та же высокая нагрузка два дня подряд
                    if "strength" in prv and st & {"sprint", "hurdles", "speed150"}: s -= 1.0
                    if "strength" in prv and "strength" in st: s -= 1.0
                    if "speedend" in prv and st & {"sprint", "speedend", "hurdles"}: s -= 1.0
                if d == 0 and prev_day_groups:
                    s -= 0.8 * len({x.rstrip('~') for x in prev_day_groups} & st & _GRZ_HI)
            if s > bs:
                best, bs = [list(x) for x in days], s
    return best if best is not None else [[] for _ in range(ncells)]


def build_season_grz(start: pd.Timestamp, hept_d1: pd.Timestamp, dec_d1: pd.Timestamp, solver: Optional["DailySolver"] = None,
                     vol_scale: float = 1.0, seed: int = 11, floor_weeks: int = 2, emph: Optional[Dict[str, float]] = None,
                     endur_cap: float = 5.0) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Сезон по Грузенкину: число занятий в неделю — из строки 2 таблицы 3 (помесячно), частота каждого средства — из его помесячного процента,
    объём на занятие = годовой объём × доля месяца / число занятий в месяце. Расстановка по дням — по сочетаниям Юшкевича–Романова.
    Подводка и старты — как в season_phases. Шлюзы только помечают (не подменяют) занятия."""
    emph = emph or {}
    rng = np.random.default_rng(seed)
    phases = season_phases(hept_d1, dec_d1, start)
    days = pd.date_range(start, dec_d1 - pd.Timedelta(days=1), freq="D")
    special = {}
    for name, p0, p1, tpl, ex, wave in phases:
        if tpl in ("taper", "comp", "recover"):
            for d in pd.date_range(p0, p1):
                special[d] = (name, tpl, int((d - p0).days))
    gs = list(GRZ_GROUPS)
    acc = {g: 0.0 for g in gs}
    last = {g: -99 for g in _GRZ_MAINT}
    wi_ = -1
    # недели: понедельники
    weeks = pd.date_range(start - pd.Timedelta(days=start.weekday()), days[-1], freq="W-MON")
    plan: Dict[pd.Timestamp, List[str]] = {}
    dm: Dict[pd.Timestamp, int] = {}
    prev = []
    for w0 in weeks:
        wdays = [w0 + pd.Timedelta(days=i) for i in range(6)]           # Пн…Сб
        usable = [d for d in wdays if d in set(days) and d not in special]
        if not usable:
            prev = []
            continue
        mid = usable[len(usable) // 2]
        m = grz_stage(mid, hept_d1, dec_d1)
        for d_ in usable:
            dm[d_] = m
        B = GRZ_ROWS["sessions"][1] * GRZ_ROWS["sessions"][2][m] / 100.0 * 7.0 / _GRZ_DIM[m]
        B = float(np.clip(B, 4.0, 10.0)) * len(usable) / 6.0
        idx = {g: grz_group_share(g, m) * 12.0 for g in gs}
        t = {g: GRZ_GROUPS[g][1] * idx[g] * emph.get(g, 1.0) for g in gs}
        tot = sum(t.values())
        sc = B / tot if tot > 0 else 0.0
        v = {g: acc[g] + t[g] * sc for g in gs}
        nB = int(round(B))
        n = {g: min(3, int(np.floor(v[g]))) for g in gs}
        while sum(n.values()) < nB:
            cand = [g for g in gs if n[g] < 3 and idx[g] > 0]
            if not cand: break
            g = max(cand, key=lambda x: v[x] - n[x]); n[g] += 1
        while sum(n.values()) > nB:
            cand = [g for g in gs if n[g] > 0]
            g = min(cand, key=lambda x: v[x] - n[x]); n[g] -= 1
        for g in gs:
            acc[g] = float(np.clip(v[g] - n[g], -1.0, 1.0))
        wi_ += 1
        for g in gs:
            if n[g] > 0 and g in last: last[g] = wi_
        sess = [g for g in gs for _ in range(n[g])]
        if floor_weeks > 0:
            stale = sorted([g for g in _GRZ_MAINT if wi_ - last[g] >= floor_weeks], key=lambda g: last[g])
            for g in stale[:2]:                      # не больше 2 поддерживающих занятий в неделю: иначе всё сваливается в один день
                if len(sess) < 11:
                    sess.append(g + "~"); last[g] = wi_
        cap_total = 3 * len(usable)
        while len(sess) > cap_total:                 # сначала выбрасываем поддерживающие, потом лишнее
            drop = [x for x in sess if x.endswith("~")] or sess
            sess.remove(drop[-1])
        placed = _grz_place(sess, rng, prev, len(usable))
        for i, lst in enumerate(placed):
            plan.setdefault(usable[i], [])
            plan[usable[i]] += [g for g in lst if g not in plan[usable[i]]]
            dm[usable[i]] = m
        prev = plan.get(usable[-1], [])
    # объёмы: по месяцам, с волной 3 недели
    wave = [1.0, 1.10, 0.75]
    perm: Dict[Tuple[str, int], List[Tuple[pd.Timestamp, float]]] = {}
    for d, lst in plan.items():
        wk = int((d - start).days // 7)
        for g in lst:
            if not g.endswith('~'):
                perm.setdefault((g, dm[d]), []).append((d, wave[wk % 3]))
    frac = {}
    for m in range(12):
        nd = sum(1 for d in days if d not in special and dm.get(d) == m)
        if nd:
            frac[m] = nd / _GRZ_DIM[m]
    amt: Dict[Tuple[pd.Timestamp, str], Dict[str, float]] = {}
    for (g, m), lst in perm.items():
        W = sum(f for _, f in lst)
        for d, f in lst:
            rec = {}
            for r in GRZ_GROUPS[g][0] + {"LJ": ["lj_full"], "HJ": ["hj_full"], "PV": ["pv_bar"], "SP": ["sp_glide"], "DT": ["dt_turn"], "JT": ["jt_full"]}.get(g, []) + \
                    (["h13", "h47", "h8p"] if g == "hurdles" else []):
                rec[r] = grz_row(r, m) * frac.get(m, 1.0) * (f / W) * vol_scale * emph.get(g, 1.0)
            if "endur" in rec:
                rec["endur"] = min(rec["endur"], endur_cap)      # потолок кросса за занятие (км): задаёт атлет, в таблице Грузенкина его нет
            amt[(d, g)] = rec
    rows, seq = [], []
    for d in days:
        wd = WD_RU[d.weekday()]
        if d in special:
            name, tpl, k = special[d]
            if tpl == "taper":
                text, key = _TAPER[k], _TAPER_KEY[k]
            elif tpl == "comp":
                text, key = f"СТАРТ (день {k + 1})", "S_REST"
            else:
                text = ["Отдых", "Кросс 20 мин + мобильность", "Мобильность, бассейн/баня", "Лёгкая техника: шест/барьеры по 20 мин",
                        "Скорость: 4×30 м (90 %) + лёгкие броски", "Кросс 25 мин + мобильность"][k % 6]
                key = ["S_REST", "S_CROSS30", "S_MOBILITY", "S_PV_SHORT", "S_ACCEL20", "S_CROSS30"][k % 6]
            keys, groups = [key], []
        elif wd == "Вс" or d not in plan or not plan[d]:
            text, key, keys, groups = "Отдых полный (сон, питание, прогулка)", "S_REST", ["S_REST"], []
        else:
            lst = sorted(plan[d], key=lambda g: _GRZ_AM.index(g.rstrip('~')))
            m = dm[d]
            items, keys = [], []
            for g in lst:
                t_, k_ = grz_session_text(g, m, amt.get((d, g), {}), "")
                items.append(t_); keys.append(k_)
            hi = [i for i, g in enumerate(lst) if g.rstrip('~') in _GRZ_HI or g.rstrip('~') in ("SP", "DT", "JT")]
            lo = [i for i in range(len(lst)) if i not in hi]
            if hi and lo:
                text = "УТРО: " + " → ".join(items[i] for i in hi) + "  ||  ВЕЧЕР: " + " → ".join(items[i] for i in lo)
            elif len(items) >= 3:
                text = "УТРО: " + items[0] + " → " + items[1] + "  ||  ВЕЧЕР: " + " → ".join(items[2:])
            else:
                text = " → ".join(items)
            groups = lst
        if d in _GRZ_TESTS and d not in special:
            text, keys, groups = _GRZ_TEST_TEXT, ["S_EXPLOSIVE"], ["test"]
        key = max(keys, key=lambda k: LIBRARY[k].stress.cns)
        ph = next(p for p in phases if p[1] <= d <= p[2])
        if d in special:
            phname, wk_in = ph[0], int((d - ph[1]).days // 7) + 1
        else:
            m_ = dm.get(d, grz_stage(d, hept_d1, dec_d1))
            phname = ("A. Осенне-зимний этап: общая подготовка, выносливость, копьё, ядро, силовая (Х–XII)" if m_ <= 2 else
                      "B. Зимний этап: скорость, барьеры, техника прыжков (I–II)" if m_ <= 4 else
                      "C. Весенний этап: специальная подготовка, метания, скоростная выносливость (III–V)")
            wk_in = int((d - start).days // 7) + 1
        import json as _j
        pv = {}
        if d not in special and d not in _GRZ_TESTS:
            for g_ in groups:
                pv[g_.rstrip("~")] = round(_MAINT_VOL.get(g_[:-1], 0.0) if g_.endswith("~") else _grz_primary(g_, amt.get((d, g_), {})), 2)
        rows.append(dict(date=d, weekday=wd, phase=phname, week_in_phase=wk_in, volume_factor=wave[int((d - start).days // 7) % 3],
                         session_key=key, plan=text, groups=",".join(groups), gate_note="", vol=_j.dumps(pv)))
        seq.append(key)
    df = pd.DataFrame(rows)
    if solver is not None:
        feas, _, _, _, tr = solver._run(seq, True)
        for i, t_ in enumerate(tr):
            if t_["bad"]:
                df.loc[i, "gate_note"] = "шлюз: " + ",".join(t_["bad"])
    # проверка: доля месячного объёма группы в плане против таблицы Грузенкина
    cmp_rows = []
    for g in gs:
        for m in range(12):
            ds = df[(df.groups.str.contains(rf"(?:^|,){g}(?:,|$)")) & (df.date.map(lambda x: dm.get(x, -1)) == m)]
            cmp_rows.append(dict(group=g, month=_M[m], sessions=len(ds), table_share=round(100 * grz_group_share(g, m), 1)))
    return df, pd.DataFrame(cmp_rows)


# Контрольные испытания (Грузенкин: тесты в начале и в конце каждого этапа): начало этапа А, конец А, начало весеннего этапа, конец весеннего (до предсоревновательных недель)
_GRZ_TESTS = {pd.Timestamp(x) for x in ("2026-10-14", "2026-12-23", "2027-02-17", "2027-05-12")}
_GRZ_TEST_TEXT = ("КОНТРОЛЬ (по Грузенкину): разминка 25 мин → бег 30 м с хода → прыжок с места и тройной с места → бросок ядра назад 7 кг → толчок ядра с места → "
                  "рывок штанги и жим лёжа (вес записать) → при желании 12 минут бега. Между тестами отдых 3–5 мин. Результаты — в приложение, вкладка «Тесты».")


def grz_report(cmp_df: pd.DataFrame) -> str:
    out = ["занятий по группам и месяцам (план) против процента таблицы Грузенкина:"]
    piv = cmp_df.pivot(index="group", columns="month", values="sessions")[_M]
    shr = cmp_df.pivot(index="group", columns="month", values="table_share")[_M]
    for g in GRZ_GROUPS:
        a, b = piv.loc[g].values.astype(float), shr.loc[g].values.astype(float)
        r = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else float("nan")
        out.append(f"  {GRZ_LABEL[g]:<24} занятий: " + " ".join(f"{int(x):2d}" for x in a) + f"   корреляция с таблицей r = {r:.2f}")
    return "\n".join(out)


# ═════════════════════════════ 9d. LIVE — пересчёт «данные → модель» и обратная задача «цель → нужные результаты» (для app.py) ═════════════════════════════
# «Потолочные» результаты по видам — МОЁ допущение (не из источников): что можно считать предельно реалистичным за сезон при очень удачной подготовке.
# Используются только в обратной задаче как масштаб «сколько ещё можно взять» в каждом виде; в приложении редактируются.
STRETCH = {"100m": 11.10, "LJ": 7.50, "SP": 13.50, "HJ": 2.09, "400m": 51.00, "110mH": 14.70, "DT": 38.00, "PV": 4.55, "JT": 54.00, "1500m": 278.00}
TYPICAL_SEQ = lambda H: [TYPICAL_WEEK[i % 7] for i in range(H)]

# Какие виды десятиборья «питают» какое средство подготовки (МОЁ отображение; для усиления акцента на отстающих видах).
GROUP_EVENTS = {"sprint": ["100m"], "speed150": ["100m", "400m"], "speedend": ["400m"], "endurance": ["1500m"], "hurdles": ["110mH"],
                "LJ": ["LJ"], "HJ": ["HJ"], "PV": ["PV"], "SP": ["SP"], "DT": ["DT"], "JT": ["JT"], "strength": ["SP", "DT", "JT", "LJ"]}


def emphasis_from_profile(current: Dict[str, float], stretch: Optional[Dict[str, float]] = None, strength: float = 1.0,
                          step: float = 0.05, lo: float = 0.9, hi: float = 1.25) -> Dict[str, float]:
    """Индивидуализация плана (МОЁ правило, не из Грузенкина): средство, питающее вид с бо́льшим запасом роста (очки до «потолка»),
    получает больше занятий и объёма; сильные виды — меньше. Множитель = 1 + 0.5·strength·(запас_группы / средний_запас − 1), в [lo, hi],
    округлён до step (чтобы план не пересобирался от каждой сотой доли). strength=0 — план Грузенкина без индивидуализации."""
    st_ = {**STRETCH, **(stretch or {})}
    cap = {e: max(float(wa_points(e, st_[e], floor=True)) - float(wa_points(e, current[e], floor=True)), 0.0) for e in EVENTS}
    gcap = {g: float(np.mean([cap[e] for e in es])) for g, es in GROUP_EVENTS.items()}
    mean = float(np.mean(list(gcap.values()))) or 1.0
    out = {}
    for g, c in gcap.items():
        w = float(np.clip(1.0 + 0.5 * strength * (c / mean - 1.0), lo, hi))
        out[g] = round(round(w / step) * step, 4)
    return out


# ───────────── часы подготовки по видам и закон убывающей отдачи (Юшкевич, Романов 2012) ─────────────
PRIMARY_EV = {"Спринт": ["100m"], "Скоростная выносливость": ["400m"], "Барьеры": ["110mH"], "Прыжок в длину": ["LJ"], "Прыжок в высоту": ["HJ"], "Шест": ["PV"],
              "Ядро": ["SP"], "Диск": ["DT"], "Копьё": ["JT"], "Выносливость": ["1500m"]}
SUPPORT_W = 0.5      # «вспомогательные» средства (сила, плиометрика, ОФП) засчитываются в виды по тегам с весом 0,5 — МОЁ допущение (статья считает «время в данном виде»)


def hours_by_event_from_log(log: Sequence[dict], since: Optional[str] = None) -> Dict[str, float]:
    """Часы подготовки по видам по журналу: профильные разделы каталога идут в свой вид с весом 1, остальное — поровну на теги вида с весом 0,5; разминка и восстановление не считаются."""
    h = {e: 0.0 for e in EVENTS}
    sub = [e for e in log if (since is None or e["date"] >= since)]
    jr = log_to_journal(sub, EXT_LIBRARY)
    for _, r in jr.iterrows():
        m = r["module"]
        mins = float(r["minutes"])
        c = CAT.get(m)
        if c is None:                                            # запись по группам (сессия библиотеки): группа → виды
            g = next((g for g, v in LOG_GROUPS.items() if v[2] == m), None)
            evs = GROUP_EVENTS.get(g, []) if g else []
            for e in evs:
                h[e] += mins / 60.0 / len(evs) * (1.0 if g != "strength" else SUPPORT_W)
            continue
        if c.cat in ("Разминка", "Восстановление"):
            continue
        if c.cat in PRIMARY_EV:
            for e in PRIMARY_EV[c.cat]:
                h[e] += mins / 60.0
        else:
            evs = [t for t in c.tags if t in h]
            for e in evs:
                h[e] += SUPPORT_W * mins / 60.0 / len(evs)
    return h


def hours_by_event_from_plan(plan: pd.DataFrame, start: Optional[pd.Timestamp] = None) -> Dict[str, float]:
    """Часы по видам по плану (столбец vol): минуты = минуты типовой позиции × объём / стандартный объём. ⚠ Минуты типовых позиций — мои оценки."""
    h = {e: 0.0 for e in EVENTS}
    for _, r in plan.iterrows():
        if start is not None and pd.Timestamp(r["date"]) < start:
            continue
        for g, x in _json_loads(r.get("vol", "{}")).items():
            c = CAT.get(PLAN_DEFAULT_ITEM.get(g, ""))
            if not c or c.log_volume(1.0) <= 0:
                continue
            mins = c.minutes * float(x) / c.log_volume(1.0)
            evs = GROUP_EVENTS.get(g, [])
            for e in evs:
                h[e] += mins / 60.0 / len(evs) * (1.0 if g != "strength" else SUPPORT_W)
    return h


def event_headroom(current: Dict[str, float], stretch: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    st_ = {**STRETCH, **(stretch or {})}
    return {e: max(float(wa_points(e, st_[e], floor=True)) - float(wa_points(e, current[e], floor=True)), 0.0) for e in EVENTS}


def _opt_share(cap: Dict[str, float], p: float, floor_share: float = 0.4) -> Dict[str, float]:
    """Оптимальные ДОЛИ часов при законе g_e(t) = cap_e·(t/T)^p (форма — из статьи, масштаб — запас до потолка): из равенства предельных приростов t_e ∝ cap_e^(1/(1−p)).
    Пол: каждый вид не меньше floor_share от равной доли (спринт и прыжки нельзя «забросить»); пол применяется итеративно."""
    ev = [e for e in EVENTS if cap[e] > 0]
    if not ev:
        return {e: 0.1 for e in EVENTS}
    raw = {e: cap[e] ** (1.0 / (1.0 - p)) for e in ev}
    fl = floor_share / len(EVENTS)
    share = {e: raw[e] / sum(raw.values()) for e in ev}
    for _ in range(20):
        low = [e for e in share if share[e] < fl]
        if not low:
            break
        fixed = {e: fl for e in low}
        rest = [e for e in share if e not in fixed]
        rem = 1.0 - fl * len(fixed)
        tot = sum(raw[e] for e in rest) or 1.0
        share = {**{e: raw[e] / tot * rem for e in rest}, **fixed}
    return {e: float(share.get(e, 0.0)) for e in EVENTS}


def hours_table(current: Dict[str, float], hours: Dict[str, float], stretch: Optional[Dict[str, float]] = None, law: Optional[Dict[str, float]] = None) -> pd.DataFrame:
    """Часы по видам против оптимума закона убывающей отдачи. Форма закона (показатель p) — из статьи; МАСШТАБ — ваши «потолки»: абсолютная кривая статьи (a·t^p, 150–490 очков на вид)
    в 4–8 раз превышает нормы Хейнлы для вашего класса (≈ 600–650 очков за 2 сезона на ВСЕ виды), поэтому в абсолютном виде к вам не применима."""
    law = law or HOURS_LAW
    cap = event_headroom(current, stretch)
    share = _opt_share(cap, law["p"])
    H = sum(hours.get(e, 0.0) for e in EVENTS) or 1.0
    rm = {e: (cap[e] * max(hours.get(e, 0.0), 1.0) ** (law["p"] - 1.0)) for e in EVENTS}
    mrm = float(np.mean(list(rm.values()))) or 1.0
    rows = []
    for e in EVENTS:
        t = float(hours.get(e, 0.0))
        ratio = (share[e] * H) / t if t > 0 else float("nan")
        rows.append(dict(event=e, Часы=round(t, 1), **{"Доля часов, %": round(100 * t / H, 1), "Оптимум закона, %": round(100 * share[e], 1), "Оптимум, ч": round(share[e] * H, 1),
                                                       "Оптимум / план": (round(ratio, 2) if ratio == ratio else None), "Запас до потолка, очки": round(cap[e]),
                                                       "Отдача +1 ч (среднее=100)": round(100 * rm[e] / mrm)}))
    return pd.DataFrame(rows)


def optimal_hours(current: Dict[str, float], total_hours: float, stretch: Optional[Dict[str, float]] = None, law: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    law = law or HOURS_LAW
    sh = _opt_share(event_headroom(current, stretch), law["p"])
    return {e: round(sh[e] * total_hours, 1) for e in EVENTS}


def emphasis_hours_law(current: Dict[str, float], hours: Dict[str, float], stretch: Optional[Dict[str, float]] = None, strength: float = 1.0, step: float = 0.05,
                       lo: float = 0.9, hi: float = 1.25, law: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """Множитель объёма по группам (МОЁ применение закона): отношение «оптимальная доля часов / доля часов в плане» усредняется по видам группы;
    множитель = 1 + 0,5·strength·(отношение − 1), в [lo, hi], округлён до step."""
    law = law or HOURS_LAW
    cap = event_headroom(current, stretch)
    sh = _opt_share(cap, law["p"])
    H = sum(hours.get(e, 0.0) for e in EVENTS) or 1.0
    x = {e: (sh[e] * H / hours[e]) if hours.get(e, 0.0) > 0 else 1.0 for e in EVENTS}
    out = {}
    for g, es in GROUP_EVENTS.items():
        m = 1.0 + 0.5 * strength * (float(np.mean([x[e] for e in es])) - 1.0)
        out[g] = float(np.clip(round(m / step) * step, lo, hi))
    return out


def emphasis_blend(e_head: Dict[str, float], e_law: Dict[str, float], w_law: float, step: float = 0.05, lo: float = 0.9, hi: float = 1.25) -> Dict[str, float]:
    """Геометрическое смешивание индивидуализации по запасу роста и по закону убывающей отдачи."""
    out = {}
    for g in set(e_head) | set(e_law):
        v = float(e_head.get(g, 1.0)) ** (1.0 - w_law) * float(e_law.get(g, 1.0)) ** w_law
        out[g] = float(np.clip(round(v / step) * step, lo, hi))
    return out


def make_model(half=None, tau1=None, K1=None, K2=None, sat_G=None) -> "PerfModel":
    """Модель с изменёнными характеристиками (None — значение холодного старта). half — T½ пулов CNS/MET/MECH в часах (допускаются любые)."""
    m = PerfModel.from_prior()
    return PerfModel(K1=m.K1 if K1 is None else np.full(R, float(K1)), K2=m.K2 if K2 is None else np.full(R, float(K2)),
                     tau1=m.tau1 if tau1 is None else np.asarray(tau1, float), sat_G=m.sat_G if sat_G is None else float(sat_G),
                     const=PoolConstants(tuple(float(x) for x in (half if half is not None else m.const.half_life_h)), strict=False))


def pts_table(current: Dict[str, float]) -> Dict[str, int]:
    return {e: int(wa_points(e, current[e], floor=True)) for e in EVENTS}


def live_forecast(current: Dict[str, float], mass: float, seq: Sequence[str], model: Optional["PerfModel"] = None,
                  already_day2: Tuple[str, ...] = ("DT", "JT", "1500m"), noise_sd: Optional[Dict[str, float]] = None,
                  sims: int = 4000, seed: int = 0, thresholds: Tuple[int, ...] = (6500, 7000, 7500, 8000), loss: Optional[float] = None,
                  state: Optional["SolverState"] = None, corr: Optional[np.ndarray] = None) -> dict:
    """Прямая задача: личные результаты + характеристики модели + план (последовательность ключей сессий) → прогноз к концу плана.
    Всё считается заново при любом изменении входов (≈0,1 с на 241 день + MC 4000 розыгрышей)."""
    model = model or PerfModel.from_prior()
    ath = Athlete("live", float(mass), dict(current), tuple(already_day2), dict(noise_sd or {e: 60.0 for e in EVENTS}))
    ref = warm_start(model)                                   # личные результаты отнесены к форме «после типовой недели» (см. DailySolver)
    sol = DailySolver(model, ath, state if state is not None else ref, ref=ref)
    seq = list(seq)
    feas, obj, res = sol.evaluate(seq)
    _, _, res_typ = sol.evaluate(TYPICAL_SEQ(len(seq)))
    mc = TournamentMC(corr=corr).simulate(res, n=sims, seed=seed, thresholds=thresholds)
    pts_cur = pts_table(current)
    base = float(sum(pts_cur.values()))
    shr, d2, pval = ProfilePrior().shrink({e: float(pts_cur[e]) for e in EVENTS}, ath.noise_sd)
    ts = mc.total_stats()
    mu, sd = LIT_HEINLA["normal_loss"]
    return dict(model=model, solver=sol, feasible=bool(feas), res=res, pts_forecast=pts_table(res), sum_forecast=float(sum(pts_table(res).values())),
                res_typical=res_typ, sum_typical=float(sum(pts_table(res_typ).values())), pts_cur=pts_cur, sum_cur=base,
                mc=mc, mc_summary=mc.summary(), mc_total=ts, sens=mc.sensitivity(), shrink=shr, d2=d2, pval=pval,
                heinla=growth_check(ts["median"], base, len(seq)), loss=(mu if loss is None else loss, sd),
                half_life=dict(zip(POOLS, model.const.half())), tau1=dict(zip(CHANNELS, model.tau1)), cns_thr=float(sol.thr),
                horizon_days=len(seq))


def marginal_points(current: Dict[str, float], pct: float = 1.0) -> pd.DataFrame:
    """Сколько очков даёт улучшение результата вида на pct % (для времён — уменьшение, для дальностей — увеличение)."""
    rows = []
    for e in EVENTS:
        x = current[e]
        y = x * (1 + SIGN[e] * pct / 100.0)
        rows.append(dict(event=e, now=x, after=y, d_pts=float(wa_points(e, y)) - float(wa_points(e, x))))
    return pd.DataFrame(rows).round(3)


def _round_safe(e: str, x: float) -> float:
    """Округление результата в сторону лучшего (чтобы очки не оказались ниже нужных): время — до 0,01 с вниз, дальность — до 1 см вверх."""
    return float(np.floor(x * 100 + 1e-9) / 100) if IS_TIME[e] else float(np.ceil(x * 100 - 1e-9) / 100)


def required_for_target(current: Dict[str, float], target: float, basis: str = "tournament", loss: float = 295.0,
                        stretch: Optional[Dict[str, float]] = None, horizon_days: Optional[int] = None) -> dict:
    """Обратная задача: нужная сумма → какие личные результаты по видам.
    basis='tournament': target — результат турнира, нужная сумма личных результатов = target + loss (норма потерь Хейнлы 295 — можно менять);
    basis='pr': target — сама сумма личных результатов.
    Распределение (МОЁ допущение): каждый вид проходит одну и ту же долю f пути от текущих очков до очков «потолочного» результата stretch;
    f = нужный прирост / суммарный запас. Если запаса не хватает — f=1 и показывается нехватка."""
    st_ = {**STRETCH, **(stretch or {})}
    cur_p = {e: float(wa_points(e, current[e], floor=True)) for e in EVENTS}
    cap = {e: max(float(wa_points(e, st_[e], floor=True)) - cur_p[e], 0.0) for e in EVENTS}
    base, cap_sum = sum(cur_p.values()), sum(cap.values())
    need = target + (loss if basis == "tournament" else 0.0)
    gain = need - base
    f = 0.0 if gain <= 0 else (min(gain / cap_sum, 1.0) if cap_sum > 0 else 1.0)
    rows = []
    for e in EVENTS:
        p_need = cur_p[e] + f * cap[e] if gain > 0 else cur_p[e]
        r_need = current[e] if gain <= 0 or cap[e] == 0 else _round_safe(e, float(wa_result(e, p_need)))
        rows.append(dict(event=e, current=current[e], pts_now=cur_p[e], stretch=st_[e], stretch_pts=cur_p[e] + cap[e],
                         required=r_need, pts_required=float(wa_points(e, r_need, floor=True)), d_pts=float(wa_points(e, r_need, floor=True)) - cur_p[e]))
    df = pd.DataFrame(rows)
    out = dict(table=df, need_sum=need, base=base, gain=gain, capacity=cap_sum, fraction=f, shortfall=max(gain - cap_sum, 0.0),
               reachable_sum=float(df.pts_required.sum()), loss=loss if basis == "tournament" else 0.0)
    if horizon_days:
        out["heinla"] = growth_check(target if basis == "tournament" else need - 295.0, base, horizon_days)
    return out


# ═════════════════════════════ 9d-2. ПРОГНОЗ И ТРЕНИРОВОЧНЫЕ РЕЗУЛЬТАТЫ ═════════════════════════════
def forecast_seq(plan_df: pd.DataFrame, last_logged: Optional[pd.Timestamp], today: pd.Timestamp, with_dates: bool = False):
    """Последовательность сессий для прогноза: дни после последней записи до сегодняшнего (не включая) — ОТДЫХ (нет записи = не тренировался; так же
    считает журнал), с сегодняшнего дня — план. Раньше проигрывание плана начиналось на следующий день после последней записи, и двухмесячная
    пауза в записях прогноз почти не меняла. Без журнала (last_logged=None) — просто план с сегодняшнего дня."""
    today = pd.Timestamp(today).normalize()
    pl = plan_df.assign(_d=pd.to_datetime(plan_df["date"]).dt.normalize()).sort_values("_d")
    if last_logged is None:
        start, gap = today, 0
    else:
        nxt = pd.Timestamp(last_logged).normalize() + pd.Timedelta(days=1)
        start = max(today, nxt)
        gap = (start - nxt).days
    tail = pl[pl["_d"] >= start]
    keys = ["S_REST"] * gap + list(tail["session_key"])
    if not with_dates:
        return keys
    dates = [start - pd.Timedelta(days=gap - i) for i in range(gap)] + list(tail["_d"])
    return keys, dates


def forecast_sum(current: Dict[str, float], mass: float, seq: Sequence[str], model: Optional["PerfModel"] = None, state: Optional["SolverState"] = None,
                 noise_sd: Optional[Dict[str, float]] = None, ref: Optional["SolverState"] = None) -> float:
    """Сумма очков прогноза на последний день seq (без Монте-Карло, ≈0,1 с). Так же, как live_forecast: личные результаты отнесены к типовой форме."""
    model = model or PerfModel.from_prior()
    ath = Athlete("live", float(mass), dict(current), ("DT", "JT", "1500m"), dict(noise_sd or {e: 60.0 for e in EVENTS}))
    ref = ref or warm_start(model)
    sol = DailySolver(model, ath, state if state is not None else ref, ref=ref)
    _, _, res = sol.evaluate(list(seq))
    return float(sum(pts_table(res).values()))


def pause_scenario(plan_df: pd.DataFrame, today: pd.Timestamp, start: pd.Timestamp, days: int, targets: Dict[str, pd.Timestamp], current: Dict[str, float],
                   mass: float, model: Optional["PerfModel"] = None, state: Optional["SolverState"] = None, last_logged: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    """«Что если пауза»: те же дни плана, но с `start` на `days` дней — отдых. Для каждой даты-цели: прогноз формы в этот день по плану и с паузой.
    Очки считаются по шкале десятиборья на указанный день (для даты семиборья это «форма на эту дату», а не прогноз семиборья)."""
    keys, dates = forecast_seq(plan_df, last_logged, today, with_dates=True)
    st0, en0 = pd.Timestamp(start).normalize(), pd.Timestamp(start).normalize() + pd.Timedelta(days=int(days))
    paused = ["S_REST" if st0 <= d < en0 else k for k, d in zip(keys, dates)]
    model = model or PerfModel.from_prior()
    ref = warm_start(model)
    rows = []
    for name, d in targets.items():
        d = pd.Timestamp(d).normalize()
        if d not in dates:
            continue
        i = dates.index(d)
        a = forecast_sum(current, mass, keys[:i + 1], model, state, ref=ref)
        b = forecast_sum(current, mass, paused[:i + 1], model, state, ref=ref)
        rows.append(dict(target=name, date=str(d.date()), plan=round(a, 1), pause=round(b, 1), delta=round(b - a, 1)))
    return pd.DataFrame(rows, columns=["target", "date", "plan", "pause", "delta"])


# Упражнения каталога, которые и есть сам вид (полное движение, соревновательный снаряд) → вид десятиборья, дистанция (м) для времён
EVENT_EQUIV = {"sp_full": ("SP", None), "sp_comp": ("SP", None), "dt_full": ("DT", None), "dt_comp": ("DT", None), "jt_full": ("JT", None), "jt_comp": ("JT", None),
               "lj_full": ("LJ", None), "lj_comp": ("LJ", None), "hj_full": ("HJ", None), "hj_bar": ("HJ", None), "hj_prog": ("HJ", None),
               "pv_full": ("PV", None), "pv_bar": ("PV", None), "ts_100": ("100m", 100.0), "h_110": ("110mH", 110.0), "se_400": ("400m", 400.0), "en_1500": ("1500m", 1500.0)}
STD_COMP_IMPL = {"SP": 7.26, "DT": 2.0, "JT": 0.8}          # соревновательные снаряды десятиборья, кг


def pr_candidates(log: Sequence[dict], current: Dict[str, float]) -> pd.DataFrame:
    """Результаты тренировок, которые ЛУЧШЕ личного результата вида — но только в упражнениях, являющихся самим видом (полный бросок/прыжок/забег
    с соревновательным снарядом). Метание с места, тесты, тяжёлые/лёгкие снаряды и силовые сюда не попадают: связь с результатом вида у них иная.
    Это КАНДИДАТЫ: тренировочный результат не рекорд (ветер, замер, правила), решение — за вами."""
    h = result_history(log)
    best: Dict[str, dict] = {}
    for lst in h.values():
        for p in lst:
            q = EVENT_EQUIV.get(p["id"])
            if not q:
                continue
            e, dist = q
            c = CAT[p["id"]]
            if e in STD_COMP_IMPL and p.get("impl") and abs(float(p["impl"]) - STD_COMP_IMPL[e]) > 1e-6:
                continue
            if dist:
                d_m = float(p.get("size") or 0) * (1000.0 if c.size_unit == "км" else 1.0)
                if not d_m or abs(d_m - dist) > 0.01 * dist:
                    continue
            v, cur = float(p["best"]), float(current[e])
            if (v < cur) if IS_TIME[e] else (v > cur):
                if e not in best or ((v < best[e]["value"]) if IS_TIME[e] else (v > best[e]["value"])):
                    best[e] = dict(event=e, exercise=c.name, date=p["date"], value=v, current=cur,
                                   d_pts=float(wa_points(e, v, floor=True)) - float(wa_points(e, cur, floor=True)))
    rows = [r for r in best.values() if r["d_pts"] >= 1]
    return pd.DataFrame(rows, columns=["event", "exercise", "date", "value", "current", "d_pts"]).sort_values("d_pts", ascending=False).reset_index(drop=True)


# «упрощённое → полное»: сколько составляет бросок/прыжок с места от полного именно у вас (из ваших пар в журнале; чужих коэффициентов я не подставляю)
PAIR_DEFS = {"DT": (("dt_stand", "ts_dt_stand", "dt_stand_full"), ("dt_full", "dt_comp")),
             "SP": (("sp_stand", "sp_fwd_stand", "ts_sp_fwd"), ("sp_full", "sp_comp")),
             "LJ": (("lj_stand", "ts_sj"), ("lj_full", "lj_comp")),
             "JT": (("jt_stand", "jt_3step", "jt_short"), ("jt_full", "jt_comp"))}


def pair_ratio(log: Sequence[dict], current: Optional[Dict[str, float]] = None, window_days: int = 3) -> pd.DataFrame:
    """По каждому упрощённому упражнению: пары «оно ↔ полный бросок/прыжок того же вида в пределах window_days (по умолчанию 3 дня)», отношение (простое/полное),
    медиана, и если пар ≥ 2 — оценка «полного» по последнему простому результату. Это ОЦЕНКА по вашим данным, не коэффициент из литературы."""
    h = result_history(log)
    rows = []
    for e, (simple, full) in PAIR_DEFS.items():
        F_ = [(pd.Timestamp(p["date"]), p["best"]) for lst in h.values() for p in lst if p["id"] in full
              and not (e in STD_COMP_IMPL and p.get("impl") and abs(float(p["impl"]) - STD_COMP_IMPL[e]) > 1e-6)]
        for sid in simple:
            S_ = sorted([(pd.Timestamp(p["date"]), p["best"]) for lst in h.values() for p in lst if p["id"] == sid and
                         not (e in STD_COMP_IMPL and p.get("impl") and abs(float(p["impl"]) - STD_COMP_IMPL[e]) > 1e-6)])
            if not S_:
                continue
            rat = []
            for d_s, v_s in S_:
                near = sorted((abs((d_f - d_s).days), v_f) for d_f, v_f in F_ if abs((d_f - d_s).days) <= window_days)
                if near:
                    rat.append(v_s / near[0][1])
            med = float(np.median(rat)) if rat else None
            last_d, last_v = S_[-1]
            est = last_v / med if (med and len(rat) >= 2) else None
            rows.append(dict(event=e, exercise=CAT[sid].name, pairs=len(rat), ratio=med, last_date=str(last_d.date()), last_value=last_v, estimate=est,
                             pr=(float(current[e]) if current else None)))
    return pd.DataFrame(rows, columns=["event", "exercise", "pairs", "ratio", "last_date", "last_value", "estimate", "pr"])


# ═════════════════════════════ 9h. БЫСТРАЯ ЗАПИСЬ ТЕКСТОМ: «длина 6.85, ядро 14.20, присед 180х3» → позиции журнала ═════════════════════════════
# Разбор — обычный код по таблице синонимов, не нейросеть: один и тот же текст всегда даёт один и тот же результат, и его можно проверить глазами в предпросмотре.
# Правила (МОИ): «ядро/диск/копьё/длина/высота/шест» без уточнения = полные попытки (…_full); со словом «соревнование/старт» = соревновательная серия; «ядро назад/снизу» = тест
# Нариманова (7,26 кг назад через голову); «… с места» = метание/прыжок с места; жим/присед/рывок — вес × повторы (× серии), одно число = один подход на 1 повторение.
_NUM = r"\d+(?::\d+)?(?:[.,]\d+)?"
_X = r"[хx×*]"
_NB = r"(?<![\d.,:])"
QUICK_ALIASES: List[Tuple[str, str]] = [
    (r"(?:бросок\s+)?ядр\w*\s+(?:назад|снизу)(?:\s+(?:через\s+голову|назад))?", "ts_sp_back"),
    (r"ядр\w*\s+(?:вперед\s+)?с\s+места|толч\w*\s+с\s+места", "sp_fwd_stand"),
    (r"диск\w*\s+с\s+места", "dt_stand"),
    (r"тройн\w*(?:\s+прыжок)?(?:\s+с\s+места)?", "ts_triple"),
    (r"прыж\w*\s+с\s+места|длин\w*\s+с\s+места", "ts_sj"),
    (_NB + r"30\s*м?\s*с\s+ход\w*", "ts_fly30"),
    (r"ядр\w*\s+(?:соревнов\w*|старт\w*)", "sp_comp"), (r"ядр\w*", "sp_full"),
    (r"диск\w*\s+(?:соревнов\w*|старт\w*)", "dt_comp"), (r"диск\w*", "dt_full"),
    (r"копь\w*\s+(?:соревнов\w*|старт\w*)", "jt_comp"), (r"копь\w*", "jt_full"),
    (r"(?:прыж\w*\s+в\s+)?длин\w*\s+(?:соревнов\w*|старт\w*)", "lj_comp"), (r"(?:прыж\w*\s+в\s+)?длин\w*", "lj_full"),
    (r"высот\w*", "hj_full"), (r"шест\w*", "pv_full"),
    (_NB + r"110\s*м?\s*(?:с\s*/\s*б|мб|б)\b|барьер\w*\s*110", "h_110"),
    (_NB + r"1500(?!\d)\s*м?", "en_1500"), (_NB + r"1000(?!\d)\s*м?", "en_test1000"), (_NB + r"400(?!\d)\s*м?", "se_400"), (_NB + r"100(?!\d)\s*м?", "ts_100"),
    (r"жим\w*", "st_bench"), (r"присед\w*|присел\w*", "st_squat_h"), (r"рыв\w*", "st_snatch"), (r"подрыв\w*|взяти\w*", "st_clean"), (r"станов\w*|тяг\w*", "st_dead"),
]
QUICK_COMPILED = [(re.compile(p), i) for p, i in QUICK_ALIASES]


def parse_quick(text: str) -> dict:
    """Текст → {'items': [позиции журнала], 'lines': [читаемые строки предпросмотра], 'warnings': [...], 'unparsed': [...], 'date_word': 'сегодня'|'вчера'|None}.
    Позиция: id, sets, reps, size, [impl], res. Значения вне физически возможных пропускаются с предупреждением."""
    raw = str(text or "").lower().replace("ё", "е")
    date_word = None
    for w in ("позавчера", "вчера", "сегодня"):
        if w in raw:
            date_word = date_word or w
            raw = raw.replace(w, " ")
    segs = [x.strip(" .:-—") for x in re.split(r"[;\n]|,\s+|,(?=\D)|\s+и\s+", raw) if x.strip(" .:-—")]
    items: List[dict] = []
    lines: List[str] = []
    warns: List[str] = []
    unparsed: List[str] = []
    for seg in segs:
        best = None
        for rx, iid in QUICK_COMPILED:
            m = rx.search(seg)
            if m and (best is None or m.start() < best[0].start()):
                best = (m, iid)
        if best is None:
            unparsed.append(seg)
            continue
        m, iid = best
        c = CAT[iid]
        rest = (seg[:m.start()] + " " + seg[m.end():]).strip()
        impl = IMPL_STD.get(iid)
        mk = re.search(r"(" + _NUM + r")\s*кг", rest) if c.cat in ("Ядро", "Диск", "Копьё", "Контроль") and not c.id.startswith("st_") else None
        if mk:
            impl = float(mk.group(1).replace(",", "."))
            rest = rest[:mk.start()] + " " + rest[mk.end():]
        sp = res_spec(c)
        if sp:
            vals = parse_results(" ".join(re.findall(_NUM, rest)))
            if not vals:
                warns.append(f"«{seg}»: не нашёл числа — упражнение «{c.name}» пропущено.")
                continue
            it = dict(id=iid, sets=1, reps=max(1, len(vals)), size=c.size, res=[])
            if impl:
                it["impl"] = impl
            ok_v = [v for v in vals if _sane(c, it, sp[0], sp[1], v)]
            bad = [v for v in vals if v not in ok_v]
            if bad:
                warns.append(f"«{seg}»: значение {', '.join(f'{v:g}' for v in bad)} {sp[1]} вне допустимого диапазона для «{c.name}»"
                             + (" (время на длинные дистанции пишите как 4:40)" if sp[0] == "time" and (c.size or 0) >= 400 else "") + " — пропущено.")
            if not ok_v:
                continue
            it["res"], it["reps"] = ok_v, len(ok_v)
            items.append(it)
            lines.append(f"{c.name}: " + " ".join(fmt_res(sp[0], v) for v in ok_v) + f" {sp[1]}" + (f" (снаряд {impl:g} кг)" if impl and impl != IMPL_STD.get(iid) else ""))
        else:
            mm = re.search(r"(" + _NUM + r")\s*(?:кг)?\s*" + _X + r"\s*(\d+)(?:\s*" + _X + r"\s*(\d+))?", rest)
            if mm:
                w_, reps_, sets_ = float(mm.group(1).replace(",", ".")), int(mm.group(2)), int(mm.group(3) or 1)
            else:
                nums = re.findall(_NUM, rest)
                if not nums:
                    warns.append(f"«{seg}»: не нашёл вес — «{c.name}» пропущено.")
                    continue
                w_, reps_, sets_ = float(nums[0].replace(",", ".")), 1, 1
            if not (5.0 <= w_ <= 400.0) or reps_ > 60 or sets_ > 30:
                warns.append(f"«{seg}»: вес {w_:g} кг или повторы вне допустимого диапазона — пропущено.")
                continue
            items.append(dict(id=iid, sets=sets_, reps=reps_, size=w_))
            lines.append(f"{c.name}: {w_:g} кг × {reps_}" + (f" × {sets_} сер." if sets_ > 1 else " (1 подход)"))
    return dict(items=items, lines=lines, warnings=warns, unparsed=unparsed, date_word=date_word)


# ═════════════════════════════ 9g. ПОТЕНЦИАЛ: тесты СФП из журнала → слабое звено → акценты; рост к старту ═════════════════════════════
# Что здесь из источников, а что моё.
#  • ИЗ ИСТОЧНИКОВ: уровни тестов по сумме очков — таблица Нариманова (NAR); нормы роста за сезон — Хейнла (LIT_HEINLA, heinla_cap);
#    динамика тестов за подготовительный период — Грузенкин (GRZ_TEST_DYN, средняя группа мастеров спорта СССР, 7859 очк.).
#  • МОЁ: профиль «7500» (REF_PROFILE); перевод «уровня» в очки вида через этот профиль; пороги ±500 («сила»/«техника»); множители акцентов;
#    линейная зависимость роста от доли времени, когда вы тренировались; распределение роста по видам пропорционально запасу; оценка 1ПМ по формуле Эпли
#    (общепринятая формула силовой подготовки, не из источников проекта). Всё это ориентиры для решений, а не измерения и не предсказания.
REF_PROFILE = {"100m": 11.32, "LJ": 7.13, "SP": 13.70, "HJ": 1.96, "400m": 50.90, "110mH": 15.20, "DT": 41.80, "PV": 4.40, "JT": 56.80, "1500m": 283.0}

# Тест Нариманова → (позиции каталога, подпись, кому «принадлежит»: группа плана и виды)
SFP_TESTS = {
    "shot_bwd": dict(ids=("ts_sp_back", "sp_back"), label="Бросок ядра назад (снизу через голову), м", group="strength", events=("SP", "DT", "JT"), impl=7.26),
    "triple_sj": dict(ids=("ts_triple", "lj_triple_st"), label="Тройной с места, м", group="LJ", events=("LJ",)),
    "vmax": dict(ids=("ts_fly30",), label="Vmax, м/с (30 м с хода: 30 / время)", group="sprint", events=("100m", "110mH")),
    "t1000": dict(ids=("en_test1000",), label="Бег 1000 м, с", group="endurance", events=("1500m",)),
    "bench": dict(ids=("st_bench",), label="Жим лёжа, кг (оценка 1ПМ по Эпли)", group="strength", events=("SP", "JT")),
}
# Динамика за подготовительный период (Грузенкин, средняя группа МС СССР): начало → конец (для времени ниже = лучше)
THROW_TESTS = {
    "stand_put": dict(ids=("sp_fwd_stand", "sp_stand"), label="Толчок ядра с места, м", event="SP", impl=7.26),
    "stand_dt": dict(ids=("ts_dt_stand", "dt_stand_full", "dt_stand"), label="Бросок диска с места, м", event="DT", impl=2.0),
}
EXTRA_TESTS = {"snatch": dict(ids=("st_snatch",), label="Рывок штанги, кг")}       # тест без таблицы Нариманова: только групповые средние Грузенкина
GRZ_TEST_DYN = {"shot_bwd": (15.18, 15.99), "triple_sj": (9.07, 9.45), "bench": (106.8, 119.6), "fly30": (3.16, 3.06), "snatch": (80.3, 87.2)}
WEAK_GAP = 500.0       # МОЁ: на сколько очков уровень теста должен отличаться от уровня результата, чтобы считать разницу значимой


def epley_1rm(weight: float, reps: float) -> float:
    return float(weight) if reps <= 1 else float(weight) * (1.0 + float(reps) / 30.0)


def tests_from_journal(log: Sequence[dict], today=None, window_days: int = 120) -> Dict[str, dict]:
    """Последние лучшие значения тестов Нариманова из журнала за `window_days` дней: {тест: {value, date, source, n}}.
    Ядро — только стандартным весом 7,26 кг (иначе сравнение с таблицей некорректно); Vmax — из 30 м с хода как 30/время; жим — оценка 1ПМ по Эпли."""
    out: Dict[str, dict] = {}
    t0 = pd.Timestamp(today).normalize() if today is not None else None
    for e in sorted(log, key=lambda x: x["date"]):
        d = pd.Timestamp(e["date"])
        if t0 is not None and (d > t0 or (t0 - d).days > window_days):
            continue
        for it in (e.get("items") or []):
            iid = it.get("id")
            for tk, sp in {**SFP_TESTS, **THROW_TESTS, **EXTRA_TESTS}.items():
                if iid not in sp["ids"]:
                    continue
                vals = [float(v) for v in (it.get("res") or []) if v is not None]
                if tk == "snatch":                     # рывок: берём вес на ≤ 3 повторениях как оценку максимума (формула Эпли для рывка ненадёжна)
                    if not it.get("size") or float(it["size"]) <= 0 or float(it.get("reps") or 1) > 3 or not (30.0 <= float(it["size"]) <= 200.0):
                        continue
                    v, better = float(it["size"]), (lambda a, b: a > b)
                elif tk in THROW_TESTS:
                    w = it.get("impl")
                    if w is not None and abs(float(w) - sp["impl"]) > 0.25:
                        continue
                    lim = (6.0, 22.0) if tk == "stand_put" else (15.0, 65.0)
                    vals = [x for x in vals if lim[0] <= x <= lim[1]]
                    if not vals:
                        continue
                    v, better = max(vals), (lambda a, b: a > b)
                elif tk == "bench":
                    if not it.get("size") or float(it["size"]) <= 0:
                        continue
                    v = epley_1rm(float(it["size"]), float(it.get("reps") or 1))
                    better = lambda a, b: a > b
                elif tk == "shot_bwd":
                    w = it.get("impl")
                    if w is not None and abs(float(w) - sp["impl"]) > 0.25:
                        continue
                    vals = [x for x in vals if 5.0 <= x <= 22.0]
                    if not vals:
                        continue
                    v, better = max(vals), (lambda a, b: a > b)
                elif tk == "triple_sj":
                    vals = [x for x in vals if 5.0 <= x <= 11.5]
                    if not vals:
                        continue
                    v, better = max(vals), (lambda a, b: a > b)
                elif tk == "vmax":
                    if float(it.get("size") or 30) != 30.0:
                        continue
                    vals = [x for x in vals if 2.4 <= x <= 5.0]
                    if not vals:
                        continue
                    v, better = 30.0 / min(vals), (lambda a, b: a > b)
                else:                                  # t1000
                    vals = [x for x in vals if 120.0 <= x <= 400.0]
                    if not vals:
                        continue
                    v, better = min(vals), (lambda a, b: a < b)
                cur = out.get(tk)
                if cur is None or better(v, cur["value"]) or (not better(cur["value"], v) and e["date"] > cur["date"]):
                    out[tk] = dict(value=float(v), date=e["date"], source="журнал", n=(cur["n"] + 1 if cur else 1))
                elif cur is not None:
                    cur["n"] += 1
    return out


def event_level(event: str, result: float) -> float:
    """«Уровень» вида в очках суммы: 7500 · очки / очки профиля 7500 (МОЁ определение; у профиля 7500 ровно 7500)."""
    ref = float(wa_points(event, REF_PROFILE[event]))
    return 7500.0 * float(wa_points(event, result)) / max(ref, 1e-9)


def tests_assessment(tests: Dict[str, float], current: Dict[str, float]) -> pd.DataFrame:
    """По каждому введённому тесту: уровень по таблице Нариманова, уровень связанных видов (среднее), разница и вывод.
    Вывод: тест ниже результата на ≥ 500 → упираетесь в физическую базу (сила/СФП); выше на ≥ 500 → база есть, запас в технике/реализации."""
    rows = []
    for tk, sp in SFP_TESTS.items():
        if tk not in tests or tests[tk] is None:
            continue
        v = float(tests[tk])
        L_t = nar_level(tk, v)
        L_e = float(np.mean([event_level(e, current[e]) for e in sp["events"]]))
        gap = L_t - L_e
        verdict = ("сила/СФП отстаёт от результата" if gap <= -WEAK_GAP else ("запас в технике и реализации" if gap >= WEAK_GAP else "соответствует результату"))
        rows.append(dict(test=tk, label=sp["label"], value=v, level_test=L_t, level_events=L_e, gap=gap, verdict=verdict,
                         events=",".join(sp["events"]), group=sp["group"]))
    return pd.DataFrame(rows, columns=["test", "label", "value", "level_test", "level_events", "gap", "verdict", "events", "group"])


def tests_emphasis(assess: pd.DataFrame, step: float = 0.05, lo: float = 0.9, hi: float = 1.25, fgap: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """Множители акцентов плана от слабых звеньев (МОЁ правило): тест ниже результата на ≥500 → +5 % группе, питающей тест (+5 % ещё при ≥1000);
    тест выше результата на ≥500 → +5 % группам видов (техника/реализация). Пусто, если тестов нет."""
    m: Dict[str, float] = {}
    fgap = fgap or {}
    if (assess is None or len(assess) == 0) and not fgap:
        return m
    for r in (assess.itertuples() if assess is not None else []):
        if r.gap <= -WEAK_GAP:
            k = 2 if r.gap <= -2 * WEAK_GAP else 1
            m[r.group] = m.get(r.group, 1.0) * (1.0 + step * k)
        elif r.gap >= WEAK_GAP:
            for e in r.events.split(","):
                g = e if e in ("SP", "DT", "JT", "LJ", "HJ", "PV") else None
                if g and e not in fgap:                      # для видов с функцией по Грузенкину берём её
                    m[g] = m.get(g, 1.0) * (1.0 + step)
    for e, g in fgap.items():                                # тест выше результата на ≥ 50 очков → техника этого вида; ниже на ≥ 50 → сила
        if g >= WEAK_GAP / 10.0:
            m[e] = m.get(e, 1.0) * (1.0 + step)
        elif g <= -WEAK_GAP / 10.0:
            m["strength"] = m.get("strength", 1.0) * (1.0 + step)
    return {g: round(float(np.clip(v, lo, hi)), 4) for g, v in m.items()}


def apply_emphasis(base: Dict[str, float], mult: Dict[str, float], lo: float = 0.9, hi: float = 1.25, step: float = 0.05) -> Dict[str, float]:
    out = dict(base)
    for g, m in (mult or {}).items():
        if g in out:
            out[g] = round(round(float(np.clip(out[g] * m, lo, hi)) / step) * step, 4)
    return out


def growth_forecast(current: Dict[str, float], horizon_days: int, pause_days: int = 0, plan_share: float = 1.0, loss: float = 295.0,
                    stretch: Optional[Dict[str, float]] = None, assess: Optional[pd.DataFrame] = None, fgap: Optional[Dict[str, float]] = None) -> dict:
    """Ожидаемый рост личных результатов к старту и его распределение по видам.
    ВЕЛИЧИНА — норма Хейнлы для вашего класса, пересчитанная на срок (heinla_cap), умноженная на долю времени, когда вы тренировались
    ((срок − пауза)/срок · доля плана) — линейность МОЯ. РАСПРЕДЕЛЕНИЕ — пропорционально запасу до «потолка» (STRETCH, мои допущения); у видов, где тест
    показывает запас в технике, вес ×1,15, где сила отстаёт — ×0,85 (МОЁ). Рост вида не превышает его запас."""
    st_ = {**STRETCH, **(stretch or {})}
    pts0 = {e: float(wa_points(e, current[e])) for e in EVENTS}
    reserve = {e: max(float(wa_points(e, st_[e])) - pts0[e], 0.0) for e in EVENTS}
    sum_pr = float(sum(np.floor(list(wa_points_vec(current).values()))))
    lo_c, hi_c = heinla_cap(sum_pr, max(horizon_days, 0) / 365.0)
    share = float(np.clip((max(horizon_days - pause_days, 0) / max(horizon_days, 1)) * plan_share, 0.0, 1.0))
    w = {e: reserve[e] for e in EVENTS}
    fgap = fgap or {}
    if assess is not None and len(assess):
        for r in assess.itertuples():
            f = 1.15 if r.gap >= WEAK_GAP else (0.85 if r.gap <= -WEAK_GAP else 1.0)
            for e in r.events.split(","):
                if e not in fgap:                                 # для видов с функцией по Грузенкину берём её, а не уровень по Нариманову
                    w[e] *= f
    for e, g in fgap.items():
        w[e] *= 1.15 if g >= WEAK_GAP / 10.0 else (0.85 if g <= -WEAK_GAP / 10.0 else 1.0)       # порог 50 очков на вид — тот же 500 «уровня», деленный на 10 видов (МОЁ)
    out = {}
    for tag, total in (("lo", lo_c * share), ("hi", hi_c * share)):
        gain = {e: 0.0 for e in EVENTS}
        left = total
        free = {e for e in EVENTS if reserve[e] > 0}
        for _ in range(6):
            sw = sum(w[e] for e in free)
            if left <= 1e-9 or sw <= 0:
                break
            for e in list(free):
                add = left * w[e] / sw
                room = reserve[e] - gain[e]
                gain[e] += min(add, room)
            left = total - sum(gain.values())
            free = {e for e in free if reserve[e] - gain[e] > 1e-9}
        out[tag] = gain
    rows = []
    for e in EVENTS:
        rows.append(dict(event=e, current=float(current[e]), pts_now=int(np.floor(pts0[e])), reserve=round(reserve[e]), gain_lo=round(out["lo"][e], 1), gain_hi=round(out["hi"][e], 1),
                         result_lo=float(wa_result(e, pts0[e] + out["lo"][e])), result_hi=float(wa_result(e, pts0[e] + out["hi"][e])), stretch=float(st_[e])))
    tab = pd.DataFrame(rows)
    return dict(table=tab, budget=(lo_c, hi_c), share=share, sum_pr=sum_pr, sum_lo=sum_pr + lo_c * share, sum_hi=sum_pr + hi_c * share,
                tour_lo=sum_pr + lo_c * share - loss, tour_hi=sum_pr + hi_c * share - loss, loss=loss)


def wa_points_vec(current: Dict[str, float]) -> Dict[str, float]:
    return {e: float(wa_points(e, current[e])) for e in EVENTS}


# ───────────── Грузенкин (автореферат, 1990): функция «тест → результат вида» ─────────────
# Табл. 1 (динамика показателей, мастера спорта СССР, эксперимент окт.1987 – сент.1988). Колонки: лучший до эксперимента, исходный (начало подготовительного периода),
# итоговый (конец подготовительного периода), лучший за период эксперимента. Единицы — как в таблице (см, кг, с).
GRZ_T1 = {"snatch_kg": (82.6, 80.3, 87.2, 87.6), "bench_kg": (113.7, 106.8, 119.6, 120.8), "bwd_cm": (1570.3, 1518.3, 1599.2, 1604.0),
          "stand_put_cm": (1321.5, 1282.4, 1345.2, 1358.0), "stand_dt_cm": (3783.0, 3685.0, 3833.7, 3933.0), "triple_cm": (930.7, 907.1, 944.7, 959.7),
          "stand_lj_cm": (307.6, 294.0, 306.6, 312.0), "fly30_s": (3.09, 3.16, 3.06, 3.02)}
# Табл. 4 (личные рекорды ДО эксперимента → лучшие результаты 1988 г.), 6 спортсменов. Суммы очков сверены с табл. 5: среднее 7647,5 → 7859,0, прирост +211,5.
GRZ_T4 = {"SP": [(11.37, 12.54), (13.79, 14.31), (15.96, 16.38), (14.83, 15.82), (13.98, 14.56), (13.98, 14.34)],
          "DT": [(39.18, 41.74), (41.70, 44.22), (49.98, 49.86), (42.48, 47.12), (40.52, 41.74), (41.42, 42.40)],
          "JT": [(51.20, 54.12), (58.86, 59.18), (56.50, 57.12), (55.74, 57.00), (57.52, 58.12), (55.52, 57.02)],
          "LJ_cm": [(727, 713), (724, 726), (697, 702), (718, 710), (666, 681), (716, 708)],
          "sum": [(7637, 7989), (7852, 7922), (7735, 7916), (7756, 7878), (7400, 7756), (7505, 7693)]}
# пары «вид ↔ тест»: (вид, ключ в GRZ_T1, перевод теста в метры). «Лучший против лучшего»: рекорд до эксперимента ↔ лучший тест до; лучший 1988 ↔ лучший тест за эксперимент.
GRZ_COUPLE = {"SP_bwd": ("SP", "bwd_cm", 0.01, "shot_bwd"), "SP_stand": ("SP", "stand_put_cm", 0.01, "stand_put"), "DT_stand": ("DT", "stand_dt_cm", 0.01, "stand_dt")}
GRZ_BAND = 0.04     # МОЁ: минимальная полуширина коридора вокруг среднего отношения (по двум точкам и шести спортсменам выборочный разброс не оценить)


def grz_ratios() -> pd.DataFrame:
    """Отношения «результат вида / тест» по группе Грузенкина: до эксперимента и после (лучший против лучшего), среднее, коридор.
    Это групповые средние шести–восьми мастеров (7647 → 7859), а не индивидуальные зависимости: годится как ориентир «что в тестах должно соответствовать результату»."""
    rows = []
    for k, (ev, tk, sc, test) in GRZ_COUPLE.items():
        a = np.array(GRZ_T4[ev], float)
        r1 = float(a[:, 0].mean()) / (GRZ_T1[tk][0] * sc)
        r2 = float(a[:, 1].mean()) / (GRZ_T1[tk][3] * sc)
        m = (r1 + r2) / 2.0
        lo, hi = min(r1, r2, m * (1 - GRZ_BAND)), max(r1, r2, m * (1 + GRZ_BAND))
        rows.append(dict(pair=k, event=ev, test=test, ratio_before=r1, ratio_after=r2, ratio=m, lo=lo, hi=hi))
    return pd.DataFrame(rows)


def expected_from_test(pair: str, value: float) -> Tuple[float, float, float]:
    """Ожидаемый результат вида (м) при значении теста: (низ, середина, верх) по группе Грузенкина."""
    r = grz_ratios().set_index("pair").loc[pair]
    return float(r.lo * value), float(r.ratio * value), float(r.hi * value)


def tests_to_results(tests: Dict[str, float], current: Dict[str, float]) -> pd.DataFrame:
    """Для каждого введённого метательного теста: ожидаемый результат вида, ваш результат и разница в метрах и очках.
    Разница > 0: результат ниже, чем «поддерживает» тест (резерв в технике/реализации); < 0: результат выше теста (упираетесь в силу)."""
    R = grz_ratios().set_index("pair")
    rows = []
    for k, (ev, tk, sc, test) in GRZ_COUPLE.items():
        if test not in tests or tests[test] is None:
            continue
        lo, mid, hi = expected_from_test(k, float(tests[test]))
        cur = float(current[ev])
        rows.append(dict(pair=k, event=ev, test=test, value=float(tests[test]), exp_lo=lo, exp_mid=mid, exp_hi=hi, current=cur, d_m=mid - cur,
                         d_pts=float(wa_points(ev, mid)) - float(wa_points(ev, cur)), ratio=float(R.loc[k, "ratio"])))
    return pd.DataFrame(rows, columns=["pair", "event", "test", "value", "exp_lo", "exp_mid", "exp_hi", "current", "d_m", "d_pts", "ratio"])


def function_gaps(tr: pd.DataFrame) -> Dict[str, float]:
    """Разница в очках «ожидание по тесту − ваш результат» по видам (среднее, если тестов несколько)."""
    if tr is None or len(tr) == 0:
        return {}
    return {e: float(g.d_pts.mean()) for e, g in tr.groupby("event")}



def test_targets(tests: Dict[str, float], today, prep_start, prep_end, dates: Sequence) -> pd.DataFrame:
    """Ориентиры тестов на контрольные дни: линейный рост от сегодня до конца подготовительного периода на ту долю, на какую выросла средняя группа Грузенкина
    (ядро назад +5,3 %, тройной +4,2 %, жим +12 %, Vmax +3,3 %). Группа сильнее вас — для вас это ориентир, не норма (МОЁ допущение о линейности и переносе)."""
    pct = {"shot_bwd": GRZ_TEST_DYN["shot_bwd"][1] / GRZ_TEST_DYN["shot_bwd"][0] - 1, "triple_sj": GRZ_TEST_DYN["triple_sj"][1] / GRZ_TEST_DYN["triple_sj"][0] - 1,
           "bench": GRZ_TEST_DYN["bench"][1] / GRZ_TEST_DYN["bench"][0] - 1, "vmax": GRZ_TEST_DYN["fly30"][0] / GRZ_TEST_DYN["fly30"][1] - 1}
    t0, t1 = pd.Timestamp(prep_start).normalize(), pd.Timestamp(prep_end).normalize()
    rows = []
    for tk, p in pct.items():
        if tk not in tests or tests[tk] is None:
            continue
        for d in dates:
            d = pd.Timestamp(d).normalize()
            f = float(np.clip((d - t0).days / max((t1 - t0).days, 1), 0.0, 1.0))
            rows.append(dict(test=tk, label=SFP_TESTS[tk]["label"], date=str(d.date()), target=float(tests[tk]) * (1.0 + p * f), now=float(tests[tk])))
    return pd.DataFrame(rows, columns=["test", "label", "date", "target", "now"])


# ═════════════════════════════ 9k. СИНТЕЗ КОРРЕЛЯЦИЙ «ТЕСТ × ВИД» ИЗ ЛИТЕРАТУРЫ (поиск в интернете, 12.10.2026) ═════════════════════════════
# Прямых работ «силовой/прыжковый тест ↔ результат вида у десятиборцев» найти не удалось. Что нашлось и как использовано:
#  • СВЯЗИ МЕЖДУ ВИДАМИ: arXiv 2602.17043 (2026), табл. A7 — 8668 десятиборий 1007 спортсменов 2001–2022 (≥6400 очков). Это самая большая выборка; проверено двумя прочтениями.
#  • ТЕСТЫ У СПЕЦИАЛИСТОВ (корреляции с результатом ИХ вида): Judge & Bellar 2012 (ядро, n=53; жим .767, присед .771, взятие на грудь .868, частные r с поправкой на пол);
#    Maeda и др. 2024 (молот, женщины, n=82; рывок .79, присед .69, жим .56, тройной с места .61, прыжок с места .40, бросок ядра назад 4 кг .61, бег 30 м −.10 по времени; значения из таблицы 2,
#    в тексте статьи для приседа .59 — расхождение); Abd Rahim и др. 2020 (длина, n=30 студентов; присед .906, прыжок с места .892, прыжок вверх .557, бег 30 м −.598);
#    Reis и др. 2006 (длина, n=45 новичков); Zhou & Kang (длина, n=33, скорость разбега .84); Loturco 2015 (100 м, n=14, SJ −.82, CMJ −.85, прыжок с места −.81), Mackala 2015 (n=11), Almuzaini 2008 (n=38);
#    Healy 2022 (100 м, n=82, Vmax r ≥ .90 по модулю); Lin и др. 2023 (мета-анализ: прыжок с места ↔ скорость 30–100 м −.73).
#  • СВЯЗИ ТЕСТОВ МЕЖДУ СОБОЙ: Guthrie 2021 (жим ↔ присед .56, n=663; прыжок вверх ↔ с места .64, n=160); Graham-Smith & Brice (элитные прыгуны, R² CMJ ↔ прыжок с места .76).
#  • Единственные данные о десятиборцах: Романов 2018 (100 сильнейших; масса тела ↔ ядро .42, диск .36, сумма .29) — показываются отдельно.
#  • НЕ ПРОЧИТАНО: PLOS ONE 0295083 (сервер отказал по лимиту запросов), PMC и PubMed (капча) — из них ничего не взято.
# ЧТО МОЁ: (1) относительная пригодность выборки (rel) — специалисты с широким диапазоном результата завышают корреляцию для десятиборцев, у которых диапазон уже; (2) стягивание каждой клетки
# к слабой априорной r = 0,30 с весом «15 наблюдений»; (3) клетки без данных: «по аналогии» (ядро назад, толчок и диск с места ← бросок назад у молота .61; 1000 м ← VO₂max .85, R² = .72, Bliss 2021)
# и «через вид» (r теста с его «домашним» видом × связь вида с данным видом); (4) ρ между тестами одного вида = 0,55 (Guthrie .56/.64).
LIT3_ROWS = {   # табл. A7 arXiv 2602.17043: корреляции РЕЗУЛЬТАТОВ (время: меньше = лучше), порядок EVENTS; строка → значения против видов справа
    "100m": [-.54, -.20, -.20, .66, .52, -.18, -.25, -.13, .10], "LJ": [.33, .43, -.45, -.47, .29, .34, .24, -.14], "SP": [.31, -.15, -.36, .73, .38, .51, -.01],
    "HJ": [-.21, -.34, .28, .29, .22, -.10], "400m": [.46, -.13, -.25, -.11, .46], "110mH": [-.32, -.37, -.24, .12], "DT": [.40, .48, -.02], "PV": [.31, -.20], "JT": [-.09]}
LIT3_N_ATHLETES, LIT3_N_PERF = 1007, 8668


def lit3_corr_results() -> np.ndarray:
    C = np.eye(10)
    for ev, vals in LIT3_ROWS.items():
        i = EVENTS.index(ev)
        for k, r in enumerate(vals):
            C[i, i + 1 + k] = C[i + 1 + k, i] = r
    return C


def lit3_corr_points() -> np.ndarray:
    """Корреляции ОЧКОВ между видами (у времён знак меняется) — арXiv 2026, 1007 десятиборцев. Это связи МЕЖДУ спортсменами (срез), а не разброс одного спортсмена в турнире."""
    s = _sgn(EVENTS)
    return lit3_corr_results() * np.outer(s, s)


# (тест, вид, r в ориентации «выше результат теста → лучше результат вида», n, пригодность rel (МОЁ), вид записи: d — прямая, a — по аналогии, источник)
LIT_TEST_R = [
    ("snatch", "SP", .868, 53, .45, "d", "Judge & Bellar 2012, взятие на грудь как замена рывка"),
    ("bench", "SP", .767, 53, .50, "d", "Judge & Bellar 2012"), ("squat", "SP", .771, 53, .50, "d", "Judge & Bellar 2012"),
    ("snatch", "DT", .79, 82, .40, "d", "Maeda 2024, молот (жен.) → диск"), ("bench", "DT", .56, 82, .40, "d", "Maeda 2024"), ("squat", "DT", .69, 82, .40, "d", "Maeda 2024"),
    ("triple_sj", "DT", .61, 82, .40, "d", "Maeda 2024"), ("slj", "DT", .40, 82, .40, "d", "Maeda 2024"), ("fly30", "DT", .10, 82, .40, "d", "Maeda 2024 (30 м по времени −.10)"),
    ("shot_bwd", "DT", .61, 82, .40, "d", "Maeda 2024, бросок ядра назад 4 кг"),
    ("squat", "LJ", .906, 30, .35, "d", "Abd Rahim 2020, студенты (одна малая выборка — сомнительно высоко)"), ("slj", "LJ", .892, 30, .35, "d", "Abd Rahim 2020"),
    ("cmj", "LJ", .557, 30, .35, "d", "Abd Rahim 2020"), ("fly30", "LJ", .598, 30, .35, "d", "Abd Rahim 2020 (30 м по времени −.598)"),
    ("fly30", "LJ", .54, 45, .25, "d", "Reis 2006, новички (20 м с хода)"), ("slj", "LJ", .45, 45, .25, "d", "Reis 2006"), ("triple_sj", "LJ", .62, 45, .25, "d", "Reis 2006"),
    ("cmj", "LJ", .37, 45, .25, "d", "Reis 2006"), ("fly30", "LJ", .84, 33, .50, "d", "Zhou & Kang, пик скорости разбега"),
    ("slj", "100m", .81, 14, .50, "d", "Loturco 2015, горизонтальный прыжок"), ("slj", "100m", .82, 11, .50, "d", "Mackala 2015"), ("slj", "100m", .42, 38, .30, "d", "Almuzaini 2008"),
    ("cmj", "100m", .85, 14, .50, "d", "Loturco 2015"), ("cmj", "100m", .80, 15, .30, "a", "по аналогии: Loturco 2015, паралимпийские спринтеры с нарушением зрения, прыжок SJ (100 м: r −.80)"), ("fly30", "100m", .90, 82, .70, "d", "Healy 2022, Vmax"),
    ("shot_bwd", "SP", .61, 82, .30, "a", "по аналогии: бросок назад у молота (Maeda 2024)"), ("stand_put", "SP", .61, 82, .30, "a", "по аналогии: бросок назад у молота"),
    ("stand_dt", "DT", .61, 82, .30, "a", "по аналогии: бросок назад у молота"), ("t1000", "1500m", .85, 15, .30, "a", "по аналогии: VO₂max ↔ 1500 м, R²=.72 (Bliss 2021)")]
TEST_PRIOR = (0.30, 15.0)      # МОЁ: слабая априорная корреляция и её вес (число «наблюдений») — стягивает клетки с малым числом данных
TEST_RHO = 0.55                # корреляция между тестами одного вида (Guthrie 2021: .56 и .64)
R_MIN = 0.35                   # МОЁ: тест считается «связанным» с видом, если r ≥ 0,35
TEST_LABEL = {"snatch": "Рывок", "bench": "Жим лёжа", "squat": "Присед", "triple_sj": "Тройной с места", "slj": "Прыжок с места", "cmj": "Прыжок вверх (CMJ)",
              "shot_bwd": "Ядро назад", "stand_put": "Толчок ядра с места", "stand_dt": "Диск с места", "fly30": "Скорость (30 м с хода)", "t1000": "Бег 1000 м"}
ROMANOV_MASS = {"100m": .02, "LJ": .03, "SP": .42, "HJ": -.04, "400m": -.01, "110mH": -.11, "DT": .36, "PV": -.06, "JT": .04, "1500m": -.12, "sum": .29}   # n=100 лучших десятиборцев мира; знак как в статье


def test_event_pooled() -> pd.DataFrame:
    """Клетки «тест × вид», где есть данные: Фишер-усреднение с весом (n−3)·rel и стягиванием к слабой априорной r (TEST_PRIOR)."""
    rows = []
    d = pd.DataFrame(LIT_TEST_R, columns=["test", "event", "r", "n", "rel", "kind", "src"])
    for (t, e), g in d.groupby(["test", "event"], sort=False):
        w = np.maximum(g.n.values - 3.0, 1.0) * g.rel.values
        z = (w * np.arctanh(np.clip(g.r.values, -.99, .99))).sum()
        z0, w0 = float(np.arctanh(TEST_PRIOR[0])), TEST_PRIOR[1]
        rows.append(dict(test=t, event=e, r_raw=float(np.tanh(z / w.sum())), r=float(np.tanh((z + z0 * w0) / (w.sum() + w0))), w=float(w.sum()),
                         kind=("прямая" if (g.kind == "d").any() else "по аналогии"), n_src=len(g), src="; ".join(g.src)))
    return pd.DataFrame(rows)


def test_event_matrix() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Синтезированная матрица r «тест × вид» (в ориентации очков) и таблица происхождения клеток: прямая / по аналогии / через вид.
    Пустая клетка = r теста с его лучшим «домашним» видом × связь этого вида с данным (arXiv 2026), берётся лучший путь."""
    P = test_event_pooled()
    Cp = lit3_corr_points()
    tests = list(dict.fromkeys(P.test))
    M = pd.DataFrame(np.nan, index=tests, columns=list(EVENTS))
    K = pd.DataFrame("", index=tests, columns=list(EVENTS))
    for r in P.itertuples():
        M.loc[r.test, r.event], K.loc[r.test, r.event] = r.r, r.kind
    for t in tests:
        have = [e for e in EVENTS if not np.isnan(M.loc[t, e])]
        for e in EVENTS:
            if np.isnan(M.loc[t, e]):
                h = max(have, key=lambda x: abs(M.loc[t, x] * Cp[EVENTS.index(x), EVENTS.index(e)]))
                M.loc[t, e], K.loc[t, e] = M.loc[t, h] * Cp[EVENTS.index(h), EVENTS.index(e)], "через вид"
    return M, K


def test_event_sources() -> pd.DataFrame:
    """Все исходные корреляции «тест × вид» построчно (для показа в приложении): что, где, на какой выборке, насколько я их ослабил по пригодности."""
    d = pd.DataFrame(LIT_TEST_R, columns=["test", "event", "r", "n", "rel", "kind", "src"])
    d["test_label"] = d.test.map(TEST_LABEL)
    d["origin"] = d.kind.map({"d": "прямая", "a": "по аналогии"})
    return d


def test_sum_corr(M: Optional[pd.DataFrame] = None) -> pd.Series:
    """r теста с суммой очков при равном разбросе очков видов: Σ r(тест, вид) / √ΣΣ R(вид, вид)  (МОЁ допущение о равном разбросе)."""
    M = test_event_matrix()[0] if M is None else M
    return M.sum(axis=1) / float(np.sqrt(lit3_corr_points().sum()))


def multi_R2(r: Sequence[float], rho: float = TEST_RHO) -> float:
    """Множественный R² вида по набору тестов с одинаковой корреляцией ρ между тестами."""
    r = np.asarray(r, float)
    if r.size == 0:
        return 0.0
    R = np.full((r.size, r.size), rho)
    np.fill_diagonal(R, 1.0)
    return float(np.clip(r @ np.linalg.solve(R, r), 0.0, 0.95))


# ═════════════════════════════ 9j. «ЧТО ЕСЛИ»: профиль силы и прыжков → результаты видов и сумма очков ═════════════════════════════
# Откуда что (это же показано в приложении).
#  • ИЗ ИСТОЧНИКОВ: (а) Нариманов — какое значение теста соответствует какой сумме очков (таблица по уровням 4000…9000, срез по многим спортсменам: bench, triple_sj, shot_bwd, t1000);
#    (б) Грузенкин — групповые средние тестов в двух точках (до эксперимента 7647,5 очк. и лучшие за период 7859,0), в том числе рывок и бег 30 м с хода, которых у Нариманова нет;
#    его же отношения «результат вида / тест» (ядро назад, толчок с места, диск с места); (в) связи между видами и тестов с видами — синтез из литературы (раздел 9k).
#  • МОЁ: (1) профиль «7500» (REF_PROFILE) как способ перевести «уровень» в результат вида; (2) линия Грузенкина как «значение теста → уровень» — это динамика ОДНОЙ группы за сезон,
#    а не срез по спортсменам, поэтому вне диапазона группы (для рывка 82,6…87,6 кг) это экстраполяция; (3) стягивание оценки к уровню 7500 с коэффициентом R² (доля дисперсии вида, объяснимая тестами) —
#    таблицы Нариманова/Грузенкина читаются «в обратную сторону», что завышает отклонение (проверено на группе Грузенкина: см. profile_validation); (4) уровень видов без теста = средний уровень
#    протестированных + поправка по связям между видами при равном разбросе очков; (5) ширина коридора.
#  • ЧЕГО НЕТ: индивидуальной регрессии «рывок → результат». У Грузенкина тесты приведены только средними по группе (6–8 человек), у Рудских силовых тестов нет, у Нариманова нет рывка и приседа.
GRZ_SUM = (float(np.mean([a for a, _ in GRZ_T4["sum"]])), float(np.mean([b for _, b in GRZ_T4["sum"]])))        # 7647,5 → 7859,0 (табл. 4/5 Грузенкина)
_REF_PTS = {e: float(wa_points(e, REF_PROFILE[e])) for e in EVENTS}
WHATIF_L_POP = 7500.0       # центр, к которому стягиваются оценки: середина таблицы Нариманова и профиль «7500» (МОЁ; по двум точкам группы Грузенкина эмпирический коэффициент 0,42–0,46)
PROFILE_TESTS = {
    "snatch": dict(label="Рывок штанги, кг", short="рывок", grz="snatch_kg", scale=1.0, nar=None, step=5.0, rng=(30.0, 160.0)),
    "bench": dict(label="Жим лёжа (1ПМ), кг", short="жим", grz="bench_kg", scale=1.0, nar="bench", step=5.0, rng=(30.0, 220.0)),
    "triple_sj": dict(label="Тройной с места, м", short="тройной", grz="triple_cm", scale=.01, nar="triple_sj", step=0.10, rng=(5.0, 11.5)),
    "shot_bwd": dict(label="Ядро назад 7,26 кг, м", short="ядро назад", grz="bwd_cm", scale=.01, nar="shot_bwd", pair={"SP": "SP_bwd"}, step=0.5, rng=(5.0, 22.0)),
    "stand_put": dict(label="Толчок ядра с места, м", short="толчок с места", grz="stand_put_cm", scale=.01, nar=None, pair={"SP": "SP_stand"}, step=0.5, rng=(5.0, 20.0)),
    "stand_dt": dict(label="Бросок диска с места 2 кг, м", short="диск с места", grz="stand_dt_cm", scale=.01, nar=None, pair={"DT": "DT_stand"}, step=2.0, rng=(15.0, 65.0)),
    "fly30": dict(label="Бег 30 м с хода, с", short="30 м с хода", grz="fly30_s", scale=1.0, nar=None, step=-0.05, rng=(2.4, 4.5)),
    "t1000": dict(label="Бег 1000 м, с", short="1000 м", grz=None, scale=1.0, nar="t1000", step=-5.0, rng=(120.0, 400.0)),
}
LEVEL_CLIP = (5000.0, 9500.0)


def _nar_group_errors() -> List[float]:
    """Ошибки «уровень по Нариманову − сумма очков группы Грузенкина» (жим, ядро назад, тройной; до и после): независимая проверка таблицы Нариманова."""
    errs = []
    for idx, S in ((0, GRZ_SUM[0]), (3, GRZ_SUM[1])):
        for nk, gk, sc in (("bench", "bench_kg", 1.0), ("shot_bwd", "bwd_cm", .01), ("triple_sj", "triple_cm", .01)):
            errs.append(nar_level(nk, GRZ_T1[gk][idx] * sc) - S)
    return errs


def _pooled_event_sd() -> float:
    """Разброс очков вида между спортсменами внутри группы мастеров: среднеквадратичное по ядру, диску, копью, длине (шесть человек, до и после; табл. 4 Грузенкина)."""
    v = []
    for ev in ("SP", "DT", "JT", "LJ_cm"):
        for k in (0, 1):
            e_ = "LJ" if ev == "LJ_cm" else ev
            p = [float(wa_points(e_, a[k] / (100.0 if ev == "LJ_cm" else 1.0))) for a in GRZ_T4[ev]]
            v.append(float(np.var(p, ddof=1)))
    return float(np.sqrt(np.mean(v)))


WHATIF_SIG_L = float(np.sqrt(np.mean(np.square(_nar_group_errors()))))      # ≈ 450 очков суммы: насколько один тест по Нариманову «промахивается» мимо уровня группы
WHATIF_SIG_E = _pooled_event_sd()                                            # ≈ 64 очка вида


def profile_levels(tk: str, v: float, sources: Sequence[str] = ("nar", "grz")) -> Dict[str, Tuple[float, bool]]:
    """Значение теста → «уровень» (сумма очков) по каждому источнику: {источник: (уровень, вне диапазона источника)}.
    nar — линейная интерполяция таблицы Нариманова; grz — прямая через две точки группы Грузенкина (7647,5 при «лучшем до», 7859,0 при «лучшем за период»)."""
    sp = PROFILE_TESTS[tk]
    out: Dict[str, Tuple[float, bool]] = {}
    if sp["nar"] and "nar" in sources:
        arr = NAR[sp["nar"]][0]
        out["nar"] = (nar_level(sp["nar"], v), not (min(arr) <= v <= max(arr)))
    if sp["grz"] and "grz" in sources:
        row = np.array(GRZ_T1[sp["grz"]], float) * sp["scale"]
        a, b = float(row[0]), float(row[3])
        L = GRZ_SUM[0] + (v - a) / (b - a) * (GRZ_SUM[1] - GRZ_SUM[0])
        out["grz"] = (float(np.clip(L, *LEVEL_CLIP)), not (float(row.min()) <= v <= float(row.max())))
    return out


def _pts_at(e: str, level: float) -> float:
    return max(_REF_PTS[e] * float(level) / 7500.0, 30.0)


def profile_whatif(tests: Dict[str, float], current: Optional[Dict[str, float]] = None, loss: float = 295.0, sources: Sequence[str] = ("nar", "grz"), shrink: bool = True) -> dict:
    """Ожидаемые результаты видов и сумма очков по введённому профилю тестов («что если»). Пустые/нулевые тесты пропускаются.
    1) Каждый тест → уровень (profile_levels). Вид «связан» с тестом, если синтезированная r(тест, вид) ≥ R_MIN; вес теста для вида = r².
       Для ядра назад / толчка с места / диска с места свой вид считается ещё и прямой функцией Грузенкина «результат = отношение × тест».
    2) Надёжность вида K = множественный R² по связанным тестам; уровень вида = 7500 + K·(взвешенный уровень − 7500) (shrink=False — без стягивания).
    3) Результат вида = очки профиля 7500 × уровень / 7500 (МОЁ).
    4) Виды без теста: средний уровень по тестам + поправка по связям между видами (arXiv 2026) от отклонений протестированных видов; разброс — WHATIF_SIG_E."""
    notes: List[str] = []
    clean: Dict[str, float] = {}
    for k, v in (tests or {}).items():
        if k not in PROFILE_TESTS or v is None or not np.isfinite(float(v)) or float(v) <= 0:
            continue
        lo_r, hi_r = PROFILE_TESTS[k]["rng"]
        if not (lo_r <= float(v) <= hi_r):
            notes.append(f"{PROFILE_TESTS[k]['label']}: значение {float(v):g} вне допустимого диапазона {lo_r:g}…{hi_r:g} — не учтено.")
            continue
        clean[k] = float(v)
    M, _K = test_event_matrix()
    Cp = lit3_corr_points()
    R = grz_ratios().set_index("pair")
    est: Dict[str, List[tuple]] = {e: [] for e in EVENTS}        # (уровень, низ, верх, r, подпись)
    trows = []
    for tk, v in clean.items():
        sp = PROFILE_TESTS[tk]
        lv = profile_levels(tk, v, sources)
        use_pair = bool(sp.get("pair")) and "grz" in sources
        if not lv and not use_pair:
            notes.append(f"{sp['label']}: нет источника при выбранных настройках — не учтено.")
            continue
        if lv:
            Lt = float(np.mean([x[0] for x in lv.values()]))
            lo = min(min(x[0] for x in lv.values()), Lt - WHATIF_SIG_L)
            hi = max(max(x[0] for x in lv.values()), Lt + WHATIF_SIG_L)
            out_r = [("Нариманов" if s == "nar" else "Грузенкин") for s, x in lv.items() if x[1]]
        else:
            Lt = lo = hi = float("nan")
            out_r = []
        trows.append(dict(test=tk, label=sp["label"], value=v, nar=(lv["nar"][0] if "nar" in lv else np.nan), grz=(lv["grz"][0] if "grz" in lv else np.nan),
                          level=Lt, outside=", ".join(out_r)))
        for e in EVENTS:
            r_te = float(M.loc[tk, e]) if tk in M.index else 0.0
            pair = sp.get("pair", {}).get(e) if use_pair else None
            if pair:
                r = R.loc[pair]
                est[e].append((event_level(e, r.ratio * v), event_level(e, r.lo * v), event_level(e, r.hi * v), max(r_te, R_MIN), sp["short"] + " (по отношению Грузенкина)"))
            elif lv and r_te >= R_MIN:
                est[e].append((Lt, lo, hi, r_te, sp["short"]))
    tested = [e for e in EVENTS if est[e]]
    empty = dict(tests=pd.DataFrame(trows, columns=["test", "label", "value", "nar", "grz", "level", "outside"]), events=pd.DataFrame(), level=float("nan"), sum_pts=0.0, sum_lo=0.0, sum_hi=0.0,
                 cur_sum_pr=None, cur_tour=None, d_vs_tour=None, notes=notes, tested=[])
    if not tested:
        return empty
    L_e, lo_e, hi_e, K_e, r_best = {}, {}, {}, {}, {}
    for e in tested:
        w = np.array([x[3] ** 2 for x in est[e]])
        raw = float(np.dot(w, [x[0] for x in est[e]]) / w.sum())
        rlo = float(np.dot(w, [x[1] for x in est[e]]) / w.sum())
        rhi = float(np.dot(w, [x[2] for x in est[e]]) / w.sum())
        K = multi_R2([x[3] for x in est[e]]) if shrink else 1.0
        K_e[e], r_best[e] = K, max(x[3] for x in est[e])
        L_e[e] = WHATIF_L_POP + K * (raw - WHATIF_L_POP)
        lo_e[e] = WHATIF_L_POP + K * (rlo - WHATIF_L_POP)
        hi_e[e] = WHATIF_L_POP + K * (rhi - WHATIF_L_POP)
    L0 = sum(L_e[e] * K_e[e] for e in tested) / max(sum(K_e[e] for e in tested), 1e-9)
    pts, lo_p, hi_p, basis = {}, {}, {}, {}
    for e in tested:
        pts[e] = _pts_at(e, L_e[e])
        spread = max((hi_e[e] - lo_e[e]) / 2.0, 0.0)
        h = float(np.sqrt(WHATIF_SIG_E ** 2 * (1.0 - K_e[e]) + (_REF_PTS[e] * spread / 7500.0) ** 2)) if shrink else _REF_PTS[e] * spread / 7500.0
        lo_p[e], hi_p[e] = max(pts[e] - h, 30.0), pts[e] + h
        basis[e] = ", ".join(dict.fromkeys(x[4] for x in est[e]))
    untested = [e for e in EVENTS if e not in tested]
    if untested:
        ix, iu = [EVENTS.index(e) for e in tested], [EVENTS.index(e) for e in untested]
        d = np.array([pts[e] - _pts_at(e, L0) for e in tested])
        Ctt = Cp[np.ix_(ix, ix)] + 0.05 * np.eye(len(ix))
        Cut = Cp[np.ix_(iu, ix)]
        corr = Cut @ np.linalg.solve(Ctt, d)
        R2 = np.einsum("ij,ji->i", Cut, np.linalg.solve(Ctt, Cut.T))
        for k, e in enumerate(untested):
            pts[e] = max(_pts_at(e, L0) + float(corr[k]), 30.0)
            h = float(np.sqrt(WHATIF_SIG_E ** 2 * (1.0 - min(float(R2[k]), 0.9)) + (_REF_PTS[e] * WHATIF_SIG_L / 7500.0) ** 2))
            lo_p[e], hi_p[e] = max(pts[e] - h, 30.0), pts[e] + h
            basis[e] = "по остальным тестам + связь с видами, где тест есть" if len(tested) > 1 else "по уровню единственного вида с тестом"
    rows = []
    for e in EVENTS:
        a, b = float(wa_result(e, lo_p[e])), float(wa_result(e, hi_p[e]))
        cur = float(current[e]) if current is not None and e in current else None
        cp = float(wa_points(e, cur)) if cur is not None else None
        rows.append(dict(event=e, tested=e in tested, pts=pts[e], pts_lo=lo_p[e], pts_hi=hi_p[e], result=float(wa_result(e, pts[e])), res_lo=min(a, b), res_hi=max(a, b),
                         current=cur, cur_pts=cp, d_pts=(pts[e] - cp if cp is not None else None), basis=basis[e], K=K_e.get(e), r_best=r_best.get(e)))
    ev = pd.DataFrame(rows)
    sum_pts = float(np.sum(np.floor(ev.pts)))
    dl_t = float(sum(pts[e] - lo_p[e] for e in tested)); dh_t = float(sum(hi_p[e] - pts[e] for e in tested))
    dl_u = float(np.sqrt(sum((pts[e] - lo_p[e]) ** 2 for e in untested))); dh_u = float(np.sqrt(sum((hi_p[e] - pts[e]) ** 2 for e in untested)))
    cur_pr = float(np.sum(np.floor([float(wa_points(e, current[e])) for e in EVENTS]))) if current is not None else None
    cur_tour = (cur_pr - loss) if cur_pr is not None else None
    return dict(tests=pd.DataFrame(trows, columns=["test", "label", "value", "nar", "grz", "level", "outside"]), events=ev, level=float(L0), sum_pts=sum_pts,
                sum_lo=sum_pts - float(np.sqrt(dl_t ** 2 + dl_u ** 2)), sum_hi=sum_pts + float(np.sqrt(dh_t ** 2 + dh_u ** 2)), cur_sum_pr=cur_pr, cur_tour=cur_tour,
                d_vs_tour=(sum_pts - cur_tour if cur_tour is not None else None), notes=notes, tested=tested)


def profile_sensitivity(tests: Dict[str, float], current: Optional[Dict[str, float]] = None, sources: Sequence[str] = ("nar", "grz")) -> pd.DataFrame:
    """Что даёт «шаг» в каждом введённом тесте (рывок +5 кг, жим +5 кг, тройной +0,10 м, ядро назад +0,5 м, толчок +0,5 м, диск +2 м, 30 м −0,05 с, 1000 м −5 с) при остальных без изменений."""
    base = profile_whatif(tests, current, sources=sources)
    rows = []
    for tk, v in (tests or {}).items():
        if tk not in PROFILE_TESTS or v is None or float(v) <= 0 or not len(base["events"]) or tk not in set(base["tests"].test):
            continue
        sp = PROFILE_TESTS[tk]
        t2 = dict(tests); t2[tk] = float(v) + sp["step"]
        r2 = profile_whatif(t2, current, sources=sources)
        evb, ev2 = base["events"].set_index("event"), r2["events"].set_index("event")
        dd = (ev2.pts - evb.pts)
        top = dd.abs().sort_values(ascending=False).index[:3]
        rows.append(dict(test=tk, label=sp["label"], step=sp["step"], d_sum=r2["sum_pts"] - base["sum_pts"], d_level=r2["level"] - base["level"],
                         main=", ".join(f"{e} {dd[e]:+.0f}" for e in top if abs(dd[e]) >= 0.5)))
    return pd.DataFrame(rows, columns=["test", "label", "step", "d_sum", "d_level", "main"])


def corr_top_pairs(k: int = 8) -> pd.DataFrame:
    """Самые сильные связи очков между видами (arXiv 2026, 1007 десятиборцев; сверка с прежним синтезом — corr_events_compare)."""
    C = lit3_corr_points()
    rows = [dict(a=EVENTS[i], b=EVENTS[j], r=float(C[i, j]), r_old=float(synth_corr()[i, j])) for i in range(10) for j in range(i + 1, 10)]
    return pd.DataFrame(rows).sort_values("r", key=lambda s: -s.abs()).head(k).reset_index(drop=True)


def profile_validation() -> dict:
    """Проверка на группе Грузенкина (там известны и тесты, и результаты). Независима только таблица Нариманова: линия Грузенкина по построению проходит через сумму группы.
    Тесты группы (жим, ядро назад, тройной) → сумма ТОЛЬКО по Нариманову: без стягивания и со стягиванием; сравнение с фактической суммой группы и очками ядра/длины/диска/копья.
    Эмпирический коэффициент стягивания k = (факт − 7500)/(оценка по тестам − 7500) по двум точкам и такой же коэффициент по литературной матрице (K_lit)."""
    rows, kemp = [], []
    M, _ = test_event_matrix()
    for lab, idx, S in (("до эксперимента", 0, GRZ_SUM[0]), ("лучшие за период", 3, GRZ_SUM[1])):
        t = {"bench": GRZ_T1["bench_kg"][idx], "shot_bwd": GRZ_T1["bwd_cm"][idx] / 100.0, "triple_sj": GRZ_T1["triple_cm"][idx] / 100.0}
        r0 = profile_whatif(t, sources=("nar",), shrink=False)
        r1 = profile_whatif(t, sources=("nar",), shrink=True)
        ev = r1["events"].set_index("event")
        k_ = 0 if idx == 0 else 1
        act = {e: float(np.mean([float(wa_points(e, a[k_])) for a in GRZ_T4[e]])) for e in ("SP", "DT", "JT")}
        act["LJ"] = float(np.mean([float(wa_points("LJ", a[k_] / 100.0)) for a in GRZ_T4["LJ_cm"]]))
        lev = float(np.mean([nar_level(nk, GRZ_T1[gk][idx] * sc) for nk, gk, sc in (("bench", "bench_kg", 1.0), ("shot_bwd", "bwd_cm", .01), ("triple_sj", "triple_cm", .01))]))
        kemp.append((S - WHATIF_L_POP) / (lev - WHATIF_L_POP))
        rows.append(dict(period=lab, sum_fact=S, sum_raw=r0["sum_pts"], err_raw=r0["sum_pts"] - S, sum_model=r1["sum_pts"], err=r1["sum_pts"] - S,
                         **{f"{e}_fact": act[e] for e in act}, **{f"{e}_model": float(ev.loc[e, "pts"]) for e in act}))
    rs = test_sum_corr(M)
    K_lit = multi_R2([float(rs[t]) for t in ("bench", "shot_bwd", "triple_sj")])
    return dict(table=pd.DataFrame(rows), sig_level=WHATIF_SIG_L, sig_event=WHATIF_SIG_E, nar_errors=_nar_group_errors(), k_emp=float(np.mean(kemp)), k_emp_pts=kemp, k_lit=K_lit)


def whatif_selftest():
    cur = samandyk().current
    # константы получены из данных, а не заданы
    assert abs(GRZ_SUM[0] - 7647.5) < 1e-9 and abs(GRZ_SUM[1] - 7859.0) < 1e-9
    assert 300 < WHATIF_SIG_L < 650 and 50 < WHATIF_SIG_E < 80, (WHATIF_SIG_L, WHATIF_SIG_E)
    # литературная матрица связей между видами (arXiv 2026): симметрична, положительно определена, ключевые числа переписаны верно
    C3 = lit3_corr_results()
    assert np.allclose(C3, C3.T) and np.linalg.eigvalsh(lit3_corr_points()).min() > 0
    assert C3[EVENTS.index("SP"), EVENTS.index("DT")] == .73 and C3[EVENTS.index("100m"), EVENTS.index("LJ")] == -.54 and C3[EVENTS.index("400m"), EVENTS.index("1500m")] == .46
    P3 = lit3_corr_points()
    assert P3[EVENTS.index("100m"), EVENTS.index("LJ")] == .54 and P3[EVENTS.index("100m"), EVENTS.index("400m")] == .66, "знак меняется только у пар «время–поле»"
    # матрица «тест × вид»: клетки с данными, аналогиями и расчётом; значения в разумных границах; стягивание уменьшает |r|
    Mx, Kx = test_event_matrix()
    assert (Mx.abs() <= 1).all().all() and Mx.shape[1] == 10 and {"snatch", "bench", "squat", "triple_sj", "slj", "cmj", "shot_bwd", "fly30", "t1000"} <= set(Mx.index)
    PP = test_event_pooled().set_index(["test", "event"])
    lo_r, hi_r = np.minimum(PP.r_raw, TEST_PRIOR[0]), np.maximum(PP.r_raw, TEST_PRIOR[0])
    assert ((PP.r >= lo_r - 1e-12) & (PP.r <= hi_r + 1e-12)).all(), "стягивание сдвигает клетку к априорной r, но не перескакивает её"
    assert (PP.r[PP.r_raw > TEST_PRIOR[0]] < PP.r_raw[PP.r_raw > TEST_PRIOR[0]]).all(), "сильная клетка после стягивания слабее"
    assert Kx.loc["snatch", "SP"] == "прямая" and Kx.loc["shot_bwd", "SP"] == "по аналогии" and Kx.loc["bench", "HJ"] == "через вид"
    assert Mx.loc["fly30", "100m"] > Mx.loc["fly30", "DT"] and Mx.loc["bench", "SP"] > Mx.loc["bench", "100m"], "скорость связана со спринтом, жим — с ядром"
    rs = test_sum_corr(Mx)
    assert (rs > 0).all() and (rs < 1).all()
    src_ = test_event_sources()
    assert len(src_) == len(LIT_TEST_R) and src_.origin.notna().all() and src_.test_label.notna().all() and (src_.n >= 10).all() and src_.r.abs().le(1).all()
    assert abs(multi_R2([.6]) - .36) < 1e-12 and multi_R2([.6, .6]) < 2 * .36 and multi_R2([.6, .6]) > .36 and multi_R2([]) == 0.0
    # линия Грузенкина проходит через оба якоря и продолжается за ними
    a, b = GRZ_T1["snatch_kg"][0], GRZ_T1["snatch_kg"][3]
    assert abs(profile_levels("snatch", a)["grz"][0] - GRZ_SUM[0]) < 1e-9 and abs(profile_levels("snatch", b)["grz"][0] - GRZ_SUM[1]) < 1e-9
    L90, out90 = profile_levels("snatch", 90.0)["grz"]
    assert out90 and L90 > GRZ_SUM[1], "90 кг выше лучших у группы: экстраполяция помечена"
    assert not profile_levels("snatch", 85.0)["grz"][1]
    lt = profile_levels("fly30", 3.0)["grz"][0]
    assert lt > profile_levels("fly30", 3.1)["grz"][0], "время меньше — уровень выше"
    assert profile_levels("t1000", 160.0)["nar"][0] > profile_levels("t1000", 190.0)["nar"][0]
    # нулевые/пустые тесты
    assert profile_whatif({}, cur)["events"].empty and profile_whatif({"snatch": 0, "bench": None}, cur)["events"].empty
    bad = profile_whatif({"bench": 900.0}, cur)
    assert bad["events"].empty and bad["notes"], "значение вне диапазона не принимается"
    # пример пользователя: рывок 90, жим 110, тройной 9,40
    r = profile_whatif({"snatch": 90.0, "bench": 110.0, "triple_sj": 9.40}, cur)
    ev = r["events"].set_index("event")
    assert {"SP", "LJ"} <= set(r["tested"]) and len(ev) == 10
    assert 7400 < r["sum_pts"] < 8000 and r["sum_lo"] < r["sum_pts"] < r["sum_hi"], r["sum_pts"]
    assert (ev.pts_lo <= ev.pts).all() and (ev.pts <= ev.pts_hi).all() and (ev.res_lo <= ev.result + 1e-9).all() and (ev.result <= ev.res_hi + 1e-9).all()
    assert ev.loc["LJ", "tested"] and not ev.loc["HJ", "tested"]
    # стягивание: без него оценка дальше от 7500, чем со стягиванием; надёжность K в (0, 1)
    r_raw = profile_whatif({"snatch": 90.0, "bench": 110.0, "triple_sj": 9.40}, cur, shrink=False)
    assert abs(r["sum_pts"] - 7500) < abs(r_raw["sum_pts"] - 7500) and (ev.K.dropna() > 0).all() and (ev.K.dropna() < 1).all()
    # монотонность: больше рывок/жим/тройной — больше сумма; меньше время — больше сумма
    base = profile_whatif({"snatch": 85.0, "bench": 105.0, "triple_sj": 9.2, "fly30": 3.1, "t1000": 175.0}, cur)["sum_pts"]
    for k, dv in (("snatch", 5.0), ("bench", 5.0), ("triple_sj", 0.1), ("fly30", -0.05), ("t1000", -5.0)):
        t = {"snatch": 85.0, "bench": 105.0, "triple_sj": 9.2, "fly30": 3.1, "t1000": 175.0}; t[k] += dv
        assert profile_whatif(t, cur)["sum_pts"] > base, k
    # прямая функция Грузенкина: диск с места 33 → результат диска между «по 7500» и «по отношению Грузенкина» (стягивание)
    r2 = profile_whatif({"stand_dt": 33.0}, cur)
    lo_, mid_, hi_ = expected_from_test("DT_stand", 33.0)
    ev2 = r2["events"].set_index("event")
    d7500 = float(wa_result("DT", _REF_PTS["DT"]))
    assert "DT" in r2["tested"] and min(mid_, d7500) - 0.05 <= ev2.loc["DT", "result"] <= max(mid_, d7500) + 0.05, (ev2.loc["DT", "result"], mid_, d7500)
    # корреляции работают: ядро с места высоко/низко при одинаковом тройном — диск (r=0,73 с ядром) без теста сдвигается в ту же сторону
    ra = profile_whatif({"stand_put": 15.5, "triple_sj": 8.6}, cur); rb = profile_whatif({"stand_put": 11.0, "triple_sj": 8.6}, cur)
    ea, eb = ra["events"].set_index("event"), rb["events"].set_index("event")
    assert ea.loc["JT", "pts"] > eb.loc["JT", "pts"] and ea.loc["SP", "pts"] > eb.loc["SP", "pts"]
    # чувствительность
    s = profile_sensitivity({"snatch": 85.0, "bench": 105.0, "triple_sj": 9.2}, cur)
    assert len(s) == 3 and (s.d_sum > 0).all(), s
    # проверка по группе Грузенкина (только Нариманов): стягивание уменьшает ошибку по сумме; коэффициенты стягивания из данных и из литературы одного порядка
    pv = profile_validation(); v = pv["table"]
    assert (v.err.abs() < v.err_raw.abs()).all(), (v.err.tolist(), v.err_raw.tolist())
    assert (v.err.abs() < 250).all(), v.err.tolist()
    assert 0.2 < pv["k_emp"] < 0.7 and 0.1 < pv["k_lit"] < 0.7 and abs(pv["k_emp"] - pv["k_lit"]) < 0.3, (pv["k_emp"], pv["k_lit"])
    assert len(corr_top_pairs(5)) == 5 and corr_top_pairs(1).iloc[0].r > 0.7
    # рывок из журнала: ≤ 3 повторений, вес
    lg = [dict(date="2026-10-10", items=[dict(id="st_snatch", sets=3, reps=2, size=80.0), dict(id="st_snatch", sets=1, reps=8, size=100.0)])]
    tj = tests_from_journal(lg, "2026-10-12")
    assert tj["snatch"]["value"] == 80.0, "подходы на 8 повторений не считаются оценкой максимума"


def quick_selftest():
    r = parse_quick("Сегодня длина 6.85, ядро 14.20")
    assert [(i["id"], i["res"]) for i in r["items"]] == [("lj_full", [6.85]), ("sp_full", [14.2])] and r["date_word"] == "сегодня" and not r["warnings"]
    r = parse_quick("ядро снизу 14,5, диск с места 33, ядро с места 13")
    assert [(i["id"], i["res"]) for i in r["items"]] == [("ts_sp_back", [14.5]), ("dt_stand", [33.0]), ("sp_fwd_stand", [13.0])], "десятичная запятая и слова «снизу/с места»"
    r = parse_quick("присед 180х3, жим 100, рывок 80х2х3")
    assert [(i["id"], i["sets"], i["reps"], i["size"]) for i in r["items"]] == [("st_squat_h", 1, 3, 180.0), ("st_bench", 1, 1, 100.0), ("st_snatch", 3, 2, 80.0)]
    r = parse_quick("100 м 11.2, 400 м 51.8, 110 с/б 15.1, 1500 м 4:40.5, 1000 м 2:50, 30 м с хода 3.05 3.08")
    assert [i["id"] for i in r["items"]] == ["ts_100", "se_400", "h_110", "en_1500", "en_test1000", "ts_fly30"] and r["items"][3]["res"] == [280.5] and r["items"][5]["res"] == [3.05, 3.08]
    r = parse_quick("длина 68.5, бла 5, 1500 м 4.40")
    assert not r["items"] and len(r["warnings"]) == 2 and r["unparsed"] == ["бла 5"], "ошибки ввода не попадают в журнал"
    r = parse_quick("ядро 6 кг 14.5 14.8")
    assert r["items"][0]["impl"] == 6.0 and r["items"][0]["res"] == [14.5, 14.8] and r["items"][0]["reps"] == 2
    assert parse_quick("")["items"] == [] and parse_quick("присед")["items"] == []
    # результат разбора читается журналом: тесты и кандидаты в личные
    lg = [dict(date="2026-10-10", items=parse_quick("ядро снизу 14.5, диск с места 33, жим 100х3")["items"])]
    tj = tests_from_journal(lg, "2026-10-12")
    assert tj["shot_bwd"]["value"] == 14.5 and tj["stand_dt"]["value"] == 33.0 and abs(tj["bench"]["value"] - 110.0) < 1e-9


def potential_selftest():
    cur = samandyk().current
    # функция «тест → результат» по Грузенкину
    gr = grz_ratios().set_index("pair")
    assert abs(np.mean(GRZ_T4["sum"], axis=0)[0] - 7647.5) < 1e-9 and abs(np.mean(GRZ_T4["sum"], axis=0)[1] - 7859.0) < 1e-9, "суммы табл. 4 = табл. 5"
    assert 0.88 < gr.loc["SP_bwd", "ratio"] < 0.93 and 1.05 < gr.loc["SP_stand", "ratio"] < 1.10 and 1.10 < gr.loc["DT_stand", "ratio"] < 1.17
    assert (gr.lo < gr.ratio).all() and (gr.hi > gr.ratio).all() and (gr.lo <= gr[["ratio_before", "ratio_after"]].min(axis=1) + 1e-12).all()
    lo_, mid_, hi_ = expected_from_test("DT_stand", 33.0)
    assert 35.0 < lo_ < mid_ < hi_ < 39.5, (lo_, mid_, hi_)
    tr = tests_to_results({"shot_bwd": 14.5, "stand_dt": 33.0}, cur)
    assert set(tr.pair) == {"SP_bwd", "DT_stand"} and (tr.d_m > 0).all() and (tr.d_pts > 0).all(), "тесты выше результатов → резерв"
    assert tests_to_results({}, cur).empty and function_gaps(tests_to_results({}, cur)) == {}
    fg = function_gaps(tr)
    assert fg["DT"] > 25 and fg["SP"] > 0
    g_f = growth_forecast(cur, 237, fgap={"DT": 200.0}); g_n = growth_forecast(cur, 237, fgap={"DT": -200.0})
    assert g_f["table"].set_index("event").loc["DT", "gain_hi"] >= g_n["table"].set_index("event").loc["DT", "gain_hi"]
    lj_ = [dict(date="2026-10-05", items=[dict(id="dt_stand", res=[33.0], impl=2.0), dict(id="sp_fwd_stand", res=[13.0], impl=7.26), dict(id="dt_stand", res=[31.0], impl=1.75)])]
    tj_ = tests_from_journal(lj_, "2026-10-10")
    assert tj_["stand_dt"]["value"] == 33.0 and tj_["stand_put"]["value"] == 13.0, "диск другого веса не подменяет стандартный"
    g1 = growth_forecast(cur, 237)
    lo, hi = heinla_cap(g1["sum_pr"], 237 / 365.0)
    assert abs(g1["table"].gain_hi.sum() - hi) < 1.0 and abs(g1["table"].gain_lo.sum() - lo) < 1.0, "сумма роста = норма Хейнлы на срок"
    assert (g1["table"].gain_hi <= g1["table"].reserve + 1e-6).all()
    g2 = growth_forecast(cur, 237, pause_days=61)
    assert abs(g2["table"].gain_hi.sum() / g1["table"].gain_hi.sum() - (237 - 61) / 237) < 0.02, "пауза уменьшает рост пропорционально"
    g3 = growth_forecast(cur, 237, plan_share=0.5)
    assert abs(g3["table"].gain_hi.sum() / g1["table"].gain_hi.sum() - 0.5) < 0.02
    assert growth_forecast(cur, 237, pause_days=300)["table"].gain_hi.sum() == 0
    # диск (самый большой запас) получает бо́льшую долю, чем длина (малый запас)
    t = g1["table"].set_index("event")
    assert t.loc["DT", "gain_hi"] > t.loc["LJ", "gain_hi"]
    # тесты из журнала
    lg = [dict(date="2026-10-05", items=[dict(id="ts_sp_back", sets=3, reps=3, size=None, res=[14.2, 14.5, 14.4]), dict(id="ts_triple", res=[9.1, 9.2]),
                                        dict(id="ts_fly30", size=30, res=[3.05, 3.10]), dict(id="en_test1000", size=1000, res=[170.0]), dict(id="st_bench", sets=4, reps=6, size=90)]),
          dict(date="2026-10-08", items=[dict(id="ts_sp_back", res=[12.0], impl=5.0)])]
    tj = tests_from_journal(lg, "2026-10-10")
    assert abs(tj["shot_bwd"]["value"] - 14.5) < 1e-9, "ядро другого веса не должно подменять стандартное"
    assert abs(tj["bench"]["value"] - 108.0) < 1e-9 and abs(tj["vmax"]["value"] - 30 / 3.05) < 1e-9 and tj["t1000"]["value"] == 170.0 and tj["triple_sj"]["value"] == 9.2
    assert tests_from_journal(lg, "2027-05-01") == {}, "тесты старше 120 дней не считаются текущими"
    A = tests_assessment({k: v["value"] for k, v in tj.items()}, cur)
    r = A.set_index("test")
    assert abs(r.loc["shot_bwd", "level_test"] - nar_level("shot_bwd", 14.5)) < 1e-9 and len(A) == 5
    assert tests_emphasis(pd.DataFrame(columns=A.columns)) == {} and apply_emphasis({"SP": 1.0}, {}) == {"SP": 1.0}
    weak = tests_assessment({"shot_bwd": 13.0}, cur)
    assert weak.iloc[0].gap < -WEAK_GAP and tests_emphasis(weak).get("strength", 1.0) > 1.0
    strong = tests_assessment({"shot_bwd": 16.5}, cur)
    assert strong.iloc[0].gap > WEAK_GAP and tests_emphasis(strong).get("SP", 1.0) > 1.0
    assert growth_forecast(cur, 237, assess=strong)["table"].set_index("event").loc["SP", "gain_hi"] >= growth_forecast(cur, 237, assess=weak)["table"].set_index("event").loc["SP", "gain_hi"]
    tt = test_targets({"shot_bwd": 14.5}, "2026-10-12", "2026-10-12", "2027-02-01", ["2026-12-23", "2027-02-17"])
    assert tt.target.iloc[0] > 14.5 and abs(tt.target.iloc[1] - 14.5 * GRZ_TEST_DYN["shot_bwd"][1] / GRZ_TEST_DYN["shot_bwd"][0]) < 1e-9


def live_selftest():
    cur = samandyk().current
    seq = list(pd.DataFrame({"s": TYPICAL_SEQ(120)}).s)
    a = live_forecast(cur, 77.5, seq, sims=500)
    cur2 = dict(cur); cur2["SP"] = 13.20
    b = live_forecast(cur2, 77.5, seq, sims=500)
    assert b["sum_forecast"] > a["sum_forecast"] + 20, "рост ядра обязан поднять прогноз"
    c = live_forecast(cur, 77.5, seq, model=make_model(half=(24, 24, 72)), sims=500)
    d = live_forecast(cur, 77.5, seq, model=make_model(half=(48, 72, 96)), sims=500)
    assert abs(c["sum_forecast"] - d["sum_forecast"]) > 0.5, "характеристики модели обязаны менять прогноз"
    r = required_for_target(cur, 7500, "tournament", 295.0)
    assert abs(r["need_sum"] - 7795) < 1e-9 and r["reachable_sum"] >= min(r["need_sum"], r["base"] + r["capacity"]) - 11
    r0 = required_for_target(cur, 100, "pr")
    assert r0["fraction"] == 0 and (r0["table"].required == r0["table"].current).all()
    # журнал: синтетические 24 дня → анализ и добор
    import json as _j
    rng = np.random.default_rng(0)
    df, _ = build_season_grz(pd.Timestamp("2026-10-08"), pd.Timestamp("2027-02-01"), pd.Timestamp("2027-06-06"), None)
    assert df.vol.map(lambda x: isinstance(_j.loads(x), dict)).all() and df.groups.str.count("endurance").sum() > 0
    log = []
    for _, r in df.iloc[:24].iterrows():
        pv = _j.loads(r.vol)
        grp = {g: round(v * rng.uniform(0.4, 1.0), 1) for g, v in pv.items()}
        log.append(dict(date=str(r.date.date()), rest=not grp, groups=grp, minutes=75, rpe=6, sleep_h=7.5, sore=3, rhr=50, hrv=80, cmj=44))
    an = analyze_log(log, df, pd.Timestamp("2026-11-01"), PerfModel.from_prior())
    assert an["acwr"] is not None and "pools" in an and len(an["compliance"]) > 0 and an["advice"]
    de = deficit_emph(log, df, pd.Timestamp("2026-11-01"))
    assert de and max(de.values()) <= 1.15 + 1e-9 and max(de.values()) > 1.0
    assert analyze_log(log, df, pd.Timestamp("2026-10-01"), PerfModel.from_prior()).get("n_days") == 0
    assert parse_results("3,82 3.85; 1:02.4") == [3.82, 3.85, 62.4]
    h_ = [dict(date="2026-10-01", items=[dict(id="sp_fly30", sets=1, reps=4, size=30, res=[3.30, 3.32])])]
    r_ = analyze_day([dict(id="sp_fly30", sets=1, reps=4, size=30, res=[3.40, 3.42, 3.45, 3.60])], None, PerfModel.from_prior(), history=h_, date="2026-10-02")
    assert len(r_["results"]) == 1 and 0.75 <= r_["results"]["Множитель дозы"].iloc[0] < 1.0 and any("ухудшился" in n for n in r_["result_notes"])
    assert not item_result_stats(CAT["lj_stand"], dict(id="lj_stand", res=[9.9]), {})["n"]       # 9,9 м с места — ошибка ввода
    lg_ = [dict(date=str((pd.Timestamp("2026-09-10") + pd.Timedelta(days=5 * k)).date()), items=[dict(id="sp_fly30", sets=1, reps=4, size=30, res=[3.40 - .02 * k]),
           dict(id="ts_sp_back", sets=3, reps=3, size=None, res=[14.5 - .1 * k])]) for k in range(6)]
    tr_ = result_trends(lg_, pd.Timestamp("2026-10-07"))
    ef_ = event_form(tr_)
    st_ = dict(zip(ef_["event"], ef_["Статус"]))
    assert st_["100m"] == "растёт" and st_["SP"] == "падает" and form_emph(ef_).get("SP") == 1.10
    ra_ = result_adjust(dict(samandyk().current), ef_)
    assert ra_["100m"] < samandyk().current["100m"] and ra_["SP"] < samandyk().current["SP"]
    it_ = [dict(id="sp_fly30", sets=1, reps=4, size=30), dict(id="dt_full", sets=3, reps=5, size=None, impl=2.5)]
    assert any("Нет разминки" in w for w in analyze_day(it_, None, PerfModel.from_prior())["warnings"])
    ok_ = analyze_day(suggest_warmup(it_) + it_ + suggest_cooldown(it_), None, PerfModel.from_prior(), readiness=dict(sleep=4, sore=7))
    assert ok_["warmup_min"] >= 15 and not any("Нет разминки" in w for w in ok_["warnings"]) and ok_["readiness_factor"] < 1.0
    assert impl_factor(CAT["dt_full"], dict(impl=2.5)) > 1.0 > impl_factor(CAT["dt_full"], dict(impl=1.5))
    cr_ = corr_report()
    assert (abs(cr_["Табл. 1 (2012)"] - cr_["Опубликовано"]) <= 0.01).sum() >= 11, cr_      # Табл. 2/3 воспроизводятся из Табл. 1 (кроме одной подписи)
    Cs_ = synth_corr()
    assert np.allclose(Cs_, Cs_.T) and np.linalg.eigvalsh(Cs_).min() > 0 and np.all(np.abs(Cs_) <= 1 + 1e-9)
    hf_ = HOURS_LAW
    assert 0.40 < hf_["p"] < 0.55 and abs(hours_gain(50.0) - 250) < 15 and abs(hours_gain(100.0) - hours_gain(50.0) - 100) < 10
    cur_ = samandyk().current
    H_ = {e: 30.0 for e in EVENTS}
    em_ = emphasis_hours_law(cur_, H_)
    assert all(0.9 <= v <= 1.25 for v in em_.values()) and abs(sum(optimal_hours(cur_, 300.0).values()) - 300.0) < 2.0
    # прогноз после паузы в записях, точка отсчёта, кандидаты в рекорды, пары «с места → полный»
    last_ = pd.Timestamp("2026-10-20")
    ks_, ds_ = forecast_seq(df, last_, pd.Timestamp("2026-10-25"), with_dates=True)
    assert ks_[:4] == ["S_REST"] * 4 and ds_[0] == pd.Timestamp("2026-10-21") and ds_[4] == pd.Timestamp("2026-10-25") and len(ks_) == 4 + int((df.date >= pd.Timestamp("2026-10-25")).sum())
    assert forecast_seq(df, None, pd.Timestamp("2026-10-25"))[0] == df[df.date >= pd.Timestamp("2026-10-25")].session_key.iloc[0]
    day_ = lambda d, its: dict(date=d, rest=False, items=its, groups={}, minutes=70, rpe=6.5)
    mdl_ = PerfModel.from_prior()
    st_a = analyze_log(log, df, pd.Timestamp("2026-11-01"), mdl_)["state"]
    st_b = analyze_log(log + [day_("2026-11-01", [dict(id="st_squat_h", sets=3, reps=3, size=200)])], df, pd.Timestamp("2026-11-02"), mdl_)["state"]
    sq_ = list(df[df.date >= pd.Timestamp("2026-11-02")].session_key)
    fa_, fb_ = forecast_sum(cur_, 77.5, sq_, mdl_, st_a), forecast_sum(cur_, 77.5, sq_, mdl_, st_b)
    assert abs(fa_ - fb_) < 8, f"одна тяжёлая тренировка за полгода до старта не должна двигать прогноз: {fa_:.0f} vs {fb_:.0f}"
    pz_ = pause_scenario(df, pd.Timestamp("2026-11-02"), pd.Timestamp("2026-11-03"), 60, {"сем": pd.Timestamp("2027-02-01")}, cur_, 77.5, mdl_, st_a, pd.Timestamp("2026-11-01"))
    pz2_ = pause_scenario(df, pd.Timestamp("2026-11-02"), pd.Timestamp("2026-12-03"), 60, {"сем": pd.Timestamp("2027-02-01")}, cur_, 77.5, mdl_, st_a, pd.Timestamp("2026-11-01"))
    assert len(pz_) == 1 and pz_.delta.iloc[0] <= 0 and pz2_.delta.iloc[0] < pz_.delta.iloc[0], "пауза ближе к старту обязана стоить больше ранней"
    assert len(pr_candidates([day_("2026-11-01", [dict(id="sp_full", sets=3, reps=5, impl=7.26, res=[13.0, 12.9])])], cur_)) == 1
    assert len(pr_candidates([day_("2026-11-01", [dict(id="sp_full", sets=3, reps=5, impl=6.0, res=[14.5])])], cur_)) == 0          # не соревновательный снаряд
    assert len(pr_candidates([day_("2026-11-01", [dict(id="dt_stand", sets=3, reps=5, impl=2.0, res=[33.0])])], cur_)) == 0         # с места ≠ полный вид
    assert len(pr_candidates([day_("2026-11-01", [dict(id="sp_full", sets=3, reps=5, impl=7.26, res=[11.5])])], cur_)) == 0
    pl_ = [day_(f"2026-11-{d:02d}", [dict(id="dt_stand", sets=3, reps=5, impl=2.0, res=[a_]), dict(id="dt_full", sets=3, reps=5, impl=2.0, res=[b_])]) for d, a_, b_ in ((1, 26.0, 30.0), (8, 27.0, 31.0), (15, 27.6, 32.0))]
    pr_ = pair_ratio(pl_ + [day_("2026-11-20", [dict(id="dt_stand", sets=3, reps=5, impl=2.0, res=[33.0])])], cur_)
    assert int(pr_.pairs.iloc[0]) == 3 and 0.85 < pr_.ratio.iloc[0] < 0.88 and pr_.estimate.iloc[0] > 33
    assert pair_ratio([day_("2026-11-20", [dict(id="dt_stand", sets=3, reps=5, impl=2.0, res=[33.0])])], cur_).estimate.iloc[0] is None
    print("самопроверка live: данные → прогноз, модель → прогноз, цель → результаты, журнал → анализ и добор — пройдена")


# ═════════════════════════════ 9e. ЖУРНАЛ — ввод фактических тренировок и анализ ═════════════════════════════
# Запись дня (dict, хранится в state["log"]): date (ISO), rest (bool), groups {группа: объём в единицах журнала}, minutes, rpe (1–10),
# чек-ин утром: sleep_h, sore (1–10), rhr (пульс покоя), hrv (RMSSD, мс), cmj (прыжок, см), mass, notes.
LOG_GROUPS = {
    "sprint": ("Спринт ≤80 м", "м", "S_FLY30"), "speed150": ("Бег до 150 м", "м", "S_SPEEDEND300"), "speedend": ("Бег до 500 м", "м", "S_SPEEDEND300"),
    "endurance": ("Кросс / выносливость", "км", "S_CROSS30"), "hurdles": ("Барьеры", "барьеров", "S_HURDLE_RHYTHM"), "LJ": ("Длина", "прыжков", "S_ACCEL20"),
    "HJ": ("Высота", "прыжков", "S_HJ_SHORT"), "PV": ("Шест", "прыжков", "S_PV_SHORT"), "SP": ("Ядро", "бросков", "S_THROW_POWER"),
    "DT": ("Диск", "бросков", "S_THROW_TECH"), "JT": ("Копьё", "метаний", "S_THROW_TECH"), "strength": ("Силовая", "т", "S_STRENGTH_BASE"),
    "explosive": ("Взрывная сила (присед + рывок)", "т", "S_EXPLOSIVE"), "plyo": ("Прыжковая / плиометрика", "контактов", "S_DROP_JUMPS"),
    "mobility": ("Мобильность / восстановление", "мин", "S_MOBILITY")}

CNS_EXPLAIN = """**Как в модели считается «утомление ЦНС» — и чем оно НЕ является.**

1. У каждого упражнения я задал вектор нагрузки [ЦНС, метаболизм, механика] от 0 до 1. Пример: спринт 30 м с ходу — ЦНС 0,92, метаболизм 0,08, механика 0,25.
   Эти числа — **мои экспертные оценки**, не измерения.
2. После занятия «пул» ЦНС увеличивается на вектор × объём (относительно стандартного занятия), потом экспоненциально убывает: через T½ остаётся половина.
   T½ ЦНС по умолчанию 36 ч (диапазон 24–48 ч из исходного ТЗ). У вас он **не измерен**.
3. Шлюз ЦНС: если пул выше порога 0,44 (это 0,70 · 0,5^(24/36) — «сутки спустя после эталонной дозы 0,70»), то занятия с ЦНС ≥ 0,8 (максимальная скорость, шест/высота с полного разбега, тяжёлая сила)
   помечаются как нежелательные. Порог и 0,8 — тоже мои допущения.

**Что это значит:** это условный индекс «накопленной нервной нагрузки», а не измеренная величина. Единицы условные. «Усталость ЦНС» в литературе вообще нечёткий термин
(центральная против периферической). Измерить можно следствия: высоту прыжка (CMJ), время 30 м, HRV, пульс покоя, субъективные оценки.

**Что делает приложение:** вы вводите чек-ин (сон, боль, пульс покоя, HRV, CMJ). Когда накопится ≥ 21 день с объективными показателями, приложение подберёт T½ ЦНС,
при котором индекс лучше всего совпадает с вашим реальным самочувствием/показателями (вкладка «Факт и анализ»). До этого значение — заглушка."""


def log_by_date(log: Sequence[dict]) -> Dict[str, dict]:
    return {e["date"]: e for e in log}


def log_to_journal(log: Sequence[dict], library: Optional[SessionLibrary] = None) -> pd.DataFrame:
    """Журнал → таблица для state_from_journal: date, module, minutes, rpe. Если в записи есть items (выбор из каталога), берутся они: модуль = id упражнения,
    минуты = доза × стандартные минуты. Иначе — группы: минуты делятся между средствами дня пропорционально стандартной длительности сессий."""
    rows = []
    for e in log:
        if e.get("rest"):
            continue
        its = e.get("items") or []
        if its and "CAT" in globals():
            hist = result_history(log, before=e["date"])
            for it in its:
                c = CAT.get(it.get("id"))
                if not c:
                    continue
                phi = c.phi(it.get("sets"), it.get("reps"), it.get("size"))
                stt = item_result_stats(c, it, hist)
                m_ = (stt["m"] if stt else 1.0) * impl_factor(c, it)      # интенсивность по результатам (доля от личного лучшего) × вес снаряда
                rows.append(dict(date=e["date"], module=c.id, minutes=c.minutes * phi, rpe=(e.get("rpe_ovr") or c.rpe) * m_))
            continue
        gs = [g for g, v in (e.get("groups") or {}).items() if g in LOG_GROUPS and (v or 0) > 0]
        if not gs:
            continue
        base = np.array([LIBRARY[LOG_GROUPS[g][2]].minutes for g in gs], float)
        tot = float(e.get("minutes") or base.sum())
        for g, b in zip(gs, base):
            rows.append(dict(date=e["date"], module=LOG_GROUPS[g][2], minutes=tot * b / base.sum(), rpe=e.get("rpe") or np.nan))
    return pd.DataFrame(rows, columns=["date", "module", "minutes", "rpe"])


def log_daily_au(log: Sequence[dict], end: pd.Timestamp) -> pd.DataFrame:
    """Нагрузка по календарным дням (sRPE = минуты × RPE); дни без записи считаются отдыхом, но помечены logged=False."""
    if not log:
        return pd.DataFrame(columns=["date", "AU", "logged"])
    d0 = pd.Timestamp(min(e["date"] for e in log))
    days = pd.date_range(d0, max(pd.Timestamp(end), pd.Timestamp(max(e["date"] for e in log))), freq="D")
    au = {e["date"]: 0.0 for e in log}
    jr = log_to_journal(log, EXT_LIBRARY)
    if len(jr):
        _, mods, a = _session_rows(jr, EXT_LIBRARY)
        for d_, v in zip(jr["date"], a):
            au[d_] = au.get(d_, 0.0) + float(v)
    return pd.DataFrame({"date": days, "AU": [au.get(str(d.date()), 0.0) for d in days], "logged": [str(d.date()) in au for d in days]})


def _z(series: pd.Series, window: int = 28) -> Optional[float]:
    x = series.dropna().astype(float)
    if len(x) < 6:
        return None
    base = x.iloc[:-1].tail(window)
    sd = base.std()
    return None if not sd or np.isnan(sd) else float((x.iloc[-1] - base.mean()) / sd)


def checkin_table(log: Sequence[dict]) -> pd.DataFrame:
    rows = [dict(date=pd.Timestamp(e["date"]), sleep_h=e.get("sleep_h"), sore=e.get("sore"), rhr=e.get("rhr"), hrv=e.get("hrv"), cmj=e.get("cmj"), mass=e.get("mass"))
            for e in log]
    df = pd.DataFrame(rows).sort_values("date") if rows else pd.DataFrame(columns=["date", "sleep_h", "sore", "rhr", "hrv", "cmj", "mass"])
    for c in ("sleep_h", "sore", "rhr", "hrv", "cmj", "mass"):
        df[c] = pd.to_numeric(df[c], errors="coerce").where(lambda x: x > 0)
    return df.reset_index(drop=True)


READY_SIGN = {"cmj": -1.0, "hrv": -1.0, "rhr": +1.0, "sleep_h": -1.0, "sore": +1.0}      # знак: куда «хуже»


def readiness_signals(log: Sequence[dict]) -> pd.DataFrame:
    """Последний замер против вашей же базы (предыдущие ≤28 записей): z-оценка и сигнал. Порог |z|≥1 — мой условный ориентир."""
    ck = checkin_table(log)
    names = {"cmj": "Прыжок CMJ, см", "hrv": "HRV (RMSSD), мс", "rhr": "Пульс покоя, уд/мин", "sleep_h": "Сон, ч", "sore": "Боль, 1–10"}
    rows = []
    for c, lab in names.items():
        x = ck[c].dropna()
        z = _z(ck[c])
        if z is None:
            rows.append(dict(metric=lab, last=(float(x.iloc[-1]) if len(x) else None), base=None, z=None, signal="мало данных (нужно ≥ 6 замеров)"))
            continue
        base = float(x.iloc[:-1].tail(28).mean())
        bad = READY_SIGN[c] * z
        rows.append(dict(metric=lab, last=float(x.iloc[-1]), base=round(base, 2), z=round(z, 2),
                         signal="хуже обычного" if bad >= 1 else ("лучше обычного" if bad <= -1 else "в норме")))
    return pd.DataFrame(rows)


def fatigue_score(log: Sequence[dict]) -> pd.Series:
    """Сводный «индекс самочувствия» дня (больше = хуже): среднее z-оценок доступных показателей со знаком (CMJ↓, HRV↓, пульс↑, сон↓, боль↑). Индекс — мой."""
    ck = checkin_table(log).set_index("date")
    zs = []
    for c, sg in READY_SIGN.items():
        x = ck[c]
        mu, sd = x.expanding(min_periods=6).mean().shift(1), x.expanding(min_periods=6).std().shift(1)
        zs.append(sg * (x - mu) / sd.replace(0, np.nan))
    return pd.concat(zs, axis=1).mean(axis=1, skipna=True)


def fit_cns_halflife(log: Sequence[dict], grid: Sequence[float] = tuple(range(12, 85, 4)), min_days: int = 21) -> Optional[pd.DataFrame]:
    """Подбор T½ ЦНС: при каком T½ индекс пула ЦНС (утром, до тренировки) лучше всего коррелирует с индексом самочувствия. Нужны ≥ min_days дней
    с объективными показателями (CMJ, HRV или пульс покоя). Корреляция — не причинность; при |r| < 0,3 вывод ненадёжен."""
    fs = fatigue_score(log).dropna()
    ck = checkin_table(log).set_index("date")
    obj = ck[["cmj", "hrv", "rhr"]].notna().any(axis=1)
    fs = fs[fs.index.isin(obj[obj].index)]
    if len(fs) < min_days:
        return None
    jr = log_to_journal(log)
    if jr.empty:
        return None
    s_, mods, au = _session_rows(jr, EXT_LIBRARY)
    start = min(s_["date"].min(), fs.index.min())
    out = []
    for h in grid:
        ast = AthleteState(PoolConstants((float(h), 48.0, 84.0), strict=False), GateConfig())
        rec = []
        by = {}
        for k, d in enumerate(s_["date"]):
            by.setdefault(d, []).append(k)
        for d in pd.date_range(start, fs.index.max(), freq="D"):
            hour_morning = 24.0 * (d - start).days + 7.0
            if d in fs.index:
                rec.append((d, ast.pools_at(hour_morning).P[0]))
            for k in by.get(d, []):
                m = mods[k]
                ast.record(m, 24.0 * (d - start).days + 17.0, au[k] / m.AU if m.AU > 0 else 0.0)
        pool = pd.Series(dict(rec))
        a = pd.concat([pool, fs], axis=1, join="inner").dropna()
        r = float(a.iloc[:, 0].corr(a.iloc[:, 1])) if len(a) >= min_days and a.iloc[:, 0].std() > 0 else np.nan
        out.append(dict(half_life_h=float(h), r=r, n=len(a)))
    return pd.DataFrame(out)


def analyze_log(log: Sequence[dict], plan_df: Optional[pd.DataFrame], today: pd.Timestamp, model: "PerfModel") -> dict:
    """Анализ фактически сделанного: нагрузка (sRPE), ACWR, монотонность/напряжение, выполнение плана по группам, состояние пулов по журналу, чек-ин."""
    today = pd.Timestamp(today).normalize()
    log = [e for e in log if pd.Timestamp(e["date"]) <= today]       # записи «из будущего» в анализ не берём
    out: dict = dict(n_days=len(log))
    if not log:
        return out
    day = log_daily_au(log, today)
    day = day[day.date <= today]
    out["daily"] = day
    a = day.AU.to_numpy(float)
    ewma = lambda x, n: pd.Series(x).ewm(alpha=2.0 / (n + 1), adjust=False).mean().to_numpy()
    acute, chronic = ewma(a, 7), ewma(a, 28)
    out["acute"], out["chronic"] = float(acute[-1]), float(chronic[-1])
    out["acwr"] = float(acute[-1] / chronic[-1]) if chronic[-1] > 0 and len(a) >= 14 else None
    w = a[-7:]
    out["week_au"] = float(w.sum())
    out["monotony"] = float(w.mean() / w.std(ddof=1)) if len(w) >= 7 and w.std(ddof=1) > 0 else None
    out["strain"] = float(w.sum() * out["monotony"]) if out["monotony"] else None
    wk = day.assign(week=day.date.dt.to_period("W").dt.start_time).groupby("week").agg(AU=("AU", "sum"), trained=("AU", lambda x: int((x > 0).sum())))
    out["weekly"] = wk.reset_index()
    # выполнение плана (только по дням, где есть запись)
    if plan_df is not None and len(plan_df):
        by = log_by_date(log)
        rows = {}
        for _, r in plan_df.iterrows():
            iso = str(pd.Timestamp(r["date"]).date())
            if iso not in by or pd.Timestamp(r["date"]) > today:
                continue
            pv = _json_loads(r.get("vol", "{}"))
            done = by[iso].get("groups") or {}
            for g in set(pv) | {g for g in done if g in LOG_GROUPS}:
                rec = rows.setdefault(g, dict(planned_sessions=0, done_sessions=0, planned_vol=0.0, done_vol=0.0))
                if g in pv:
                    rec["planned_sessions"] += 1
                    rec["planned_vol"] += float(pv[g])
                if (done.get(g) or 0) > 0 and not by[iso].get("rest"):
                    rec["done_sessions"] += 1
                    rec["done_vol"] += float(done[g])
        comp = pd.DataFrame([dict(group=g, label=LOG_GROUPS[g][0], unit=LOG_GROUPS[g][1], **v) for g, v in rows.items() if g in LOG_GROUPS])
        if len(comp):
            comp["pct_vol"] = np.where(comp.planned_vol > 0, 100.0 * comp.done_vol / comp.planned_vol.replace(0, np.nan), np.nan).round(0)
        out["compliance"] = comp
    # состояние по журналу
    jr = log_to_journal(log)
    if len(jr) and (pd.Timestamp(jr.date.max()) - pd.Timestamp(jr.date.min())).days >= 6:
        st_ = state_from_journal(jr, model, library=EXT_LIBRARY)
        last = pd.Timestamp(jr.date.max())
        hour = st_.t0 + 24.0 * max(0, (today - last).days - 1)
        gates = BioGates(GateConfig(), model.const)
        pools = st_.ast.pools_at(hour).P
        out["state"] = st_
        out["last_logged"] = last
        out["pools"] = dict(zip(POOLS, [float(x) for x in pools]))
        out["cns_thr"] = float(gates.cns_thr)
        out["gates"] = gates.check_all(st_.ast, hour, EXT_LIBRARY)
    out["readiness"] = readiness_signals(log)
    out["cns_fit"] = fit_cns_halflife(log)
    out["advice"] = log_advice(out)
    return out


def _json_loads(x) -> dict:
    import json as _j
    try:
        return _j.loads(x) if isinstance(x, str) and x else {}
    except Exception:  # noqa: BLE001
        return {}


def deficit_emph(log: Sequence[dict], plan_df: pd.DataFrame, today: pd.Timestamp, window: int = 14, gain: float = 0.5, cap: float = 0.15) -> Dict[str, float]:
    """Добор недовыполненного (МОЁ правило): за последние `window` дней по дням с записью считаем долю невыполненного объёма средства;
    множитель на ближайшие недели = 1 + min(cap, gain · доля). Если записанных дней < 5 — ничего не меняем (отсутствие записи ≠ пропуск)."""
    today = pd.Timestamp(today).normalize()
    by = log_by_date(log)
    lo = today - pd.Timedelta(days=window)
    rows = [r for _, r in plan_df.iterrows() if lo <= pd.Timestamp(r["date"]) < today and str(pd.Timestamp(r["date"]).date()) in by]
    if len(rows) < 5:
        return {}
    plan_v, done_v = {}, {}
    for r in rows:
        e = by[str(pd.Timestamp(r["date"]).date())]
        for g, v in _json_loads(r.get("vol", "{}")).items():
            plan_v[g] = plan_v.get(g, 0.0) + float(v)
            done_v[g] = done_v.get(g, 0.0) + (0.0 if e.get("rest") else float((e.get("groups") or {}).get(g, 0) or 0))
    return {g: round(1.0 + min(cap, gain * max(0.0, 1.0 - done_v.get(g, 0.0) / v)), 3) for g, v in plan_v.items() if v > 0}


def log_advice(an: dict) -> List[str]:
    """Автоматические замечания по правилам (пороги — ориентиры из литературы по нагрузке или мои; не диагноз)."""
    adv = []
    ac = an.get("acwr")
    if ac is not None:
        if ac >= 1.5:
            adv.append(f"Нагрузка выросла резко: острая/хроническая = {ac:.2f} (≥ 1,5). Ориентир из литературы по ACWR (оспариваемый): рост травм. Снизьте 2–3 дня.")
        elif ac > 1.3:
            adv.append(f"Острая/хроническая нагрузка {ac:.2f} выше «привычной зоны» 0,8–1,3 — не добавляйте объём на этой неделе.")
        elif ac < 0.8:
            adv.append(f"Острая/хроническая нагрузка {ac:.2f} ниже 0,8: недогруз относительно последнего месяца (или вы мало записывали).")
    if an.get("monotony") and an["monotony"] > 2.0:
        adv.append(f"Монотонность недели {an['monotony']:.1f} (> 2 по Фостеру): дни слишком похожи по нагрузке — добавьте контраст лёгких и тяжёлых дней.")
    comp = an.get("compliance")
    if comp is not None and len(comp):
        for _, r in comp.iterrows():
            if r.planned_sessions >= 3 and r.done_sessions / r.planned_sessions < 0.6:
                adv.append(f"{r.label}: сделано {int(r.done_sessions)} из {int(r.planned_sessions)} запланированных занятий — приложение добавит добор в следующие недели (не более +15 %).")
    rd = an.get("readiness")
    if rd is not None and len(rd):
        bad = rd[rd.signal == "хуже обычного"]
        if len(bad) >= 2:
            adv.append("Два и более показателя самочувствия хуже вашей обычной нормы (" + ", ".join(bad.metric) + "): облегчите ближайший тяжёлый день.")
    pools = an.get("pools")
    if pools and an.get("cns_thr") and pools["CNS"] > an["cns_thr"]:
        adv.append(f"Индекс ЦНС по журналу {pools['CNS']:.2f} выше порога {an['cns_thr']:.2f}: максимальные скорости и шест/высота с полного разбега лучше сдвинуть.")
    return adv or ["Замечаний по правилам нет."]


# ═════════════════════════════ 9f. КАТАЛОГ — упражнения и отрезки на выбор, анализ собранной тренировки ═════════════════════════════
# Каждая запись: стандартная доза (подходы × повторы × размер: дистанция м / вес кг), длительность и тяжесть стандартной дозы, вектор нагрузки
# [ЦНС, метаболизм, механика] на стандартную дозу и вклад в каналы фитнеса. ВСЕ числа — МОИ экспертные оценки (как и у 25 базовых упражнений), не измерения и не из источников;
# перечень упражнений — типовой для многоборья. Дозу вы меняете (подходы, повторы, дистанция, вес) — нагрузка масштабируется пропорционально.
@dataclass(frozen=True)
class CatItem:
    id: str
    name: str
    cat: str
    kind: str                      # ОФП / СФП / Техника / Скорость / Контроль / Восстановление
    sets: int
    reps: int
    size: Optional[float]          # дистанция (м) или вес (кг); None — не применимо
    size_unit: str
    scale_size: bool               # доза пропорциональна размеру (дистанция, вес) — иначе только подходы×повторы
    minutes: float                 # длительность стандартной дозы, мин (с отдыхом)
    rpe: float
    stress: Tuple[float, float, float]
    chan: Tuple[Tuple[str, float], ...]
    group: Optional[str]           # группа журнала (LOG_GROUPS)
    tags: Tuple[str, ...]          # виды десятиборья, которым помогает
    phases: str                    # A: окт–дек, B: янв–фев, C: март–июнь
    note: str = ""
    coord: bool = False

    def std_text(self) -> str:
        t = f"{self.sets}×{self.reps}"
        if self.size is not None:
            t += f" по {self.size:g} {self.size_unit}"
        return t

    def total(self, sets=None, reps=None, size=None) -> float:
        sets, reps = (self.sets if sets is None else sets), (self.reps if reps is None else reps)
        t = float(sets) * float(reps)
        if self.scale_size and self.size:
            t *= float(self.size if size is None else size)
        return t

    def phi(self, sets=None, reps=None, size=None) -> float:
        return self.total(sets, reps, size) / self.total()

    def log_volume(self, phi: float) -> float:
        """Объём в единицах журнала для группы self.group при дозе phi (φ=1 — стандартная доза)."""
        if not self.group:
            return 0.0
        unit = LOG_GROUPS[self.group][1]
        std = self.sets * self.reps * (self.size if (self.size and unit in ("м", "км", "т")) else 1.0)
        if unit == "т":
            std = std / 1000.0
        elif unit == "км":
            std = std / 1000.0 if self.size_unit == "м" else std
        elif unit == "мин":
            std = self.minutes
        return std * phi


def _cc(spec: str) -> Tuple[Tuple[str, float], ...]:
    return tuple((k, float(v)) for k, v in (x.split(":") for x in spec.split(",") if x)) if spec else ()


def _i(id, name, cat, kind, sets, reps, size, su, scale, mins, rpe, cns, met, mech, chan, group, tags, ph="ABC", note="", coord=False) -> CatItem:
    return CatItem(id, name, cat, kind, sets, reps, size, su, scale, mins, rpe, (cns, met, mech), _cc(chan), group, tuple(tags.split(",")) if tags else (), ph, note, coord)


_SPR = "100m,LJ,110mH"
CATALOG: List[CatItem] = [
    # ── разминка и подготовка ──
    _i("wu_jog", "Разминочный бег", "Разминка", "ОФП", 1, 1, 12, "мин", True, 12, 3, .03, .15, .08, "END:1", None, "", "ABC", "10–15 мин, лёгко"),
    _i("wu_abc", "Беговая азбука (A/B-скип, приставной, захлёст)", "Разминка", "ОФП", 4, 2, 20, "м", True, 12, 3, .12, .05, .12, "SPR:.6,JMP:.4", None, _SPR),
    _i("wu_dyn", "Динамическая мобильность (мах-ноги, выпады с поворотом)", "Разминка", "ОФП", 1, 1, 10, "мин", True, 10, 2, .03, .03, .06, "", None, ""),
    _i("wu_act", "Активация: стопа, ягодицы, корпус (мини-бэнд)", "Разминка", "ОФП", 2, 12, None, "", False, 8, 3, .05, .05, .08, "", None, ""),
    _i("wu_strides", "Ускорения 80 % перед основной частью", "Разминка", "Скорость", 4, 1, 60, "м", True, 8, 4, .35, .05, .12, "SPR:1", None, _SPR),
    # ── спринт и ускорение ──
    _i("sp_acc10", "Старты с места 10 м", "Спринт", "Скорость", 6, 1, 10, "м", True, 22, 6, .75, .08, .18, "SPR:1", "sprint", "100m,LJ,110mH"),
    _i("sp_acc20", "Старты 20 м с низкого старта", "Спринт", "Скорость", 6, 1, 20, "м", True, 30, 7, .85, .12, .30, "SPR:1", "sprint", "100m,LJ,110mH,400m"),
    _i("sp_acc30", "Старты 30 м", "Спринт", "Скорость", 5, 1, 30, "м", True, 32, 7, .88, .15, .32, "SPR:1", "sprint", "100m,110mH"),
    _i("sp_sled", "Ускорения с санками (10 % массы)", "Спринт", "СФП", 5, 1, 20, "м", True, 28, 7, .75, .20, .55, "SPR:.8,THR:.2", "sprint", "100m,110mH,LJ", "ABC", "Сила старта; не ставить перед тонкой техникой"),
    _i("sp_fly20", "Бег с ходу 20 м (разгон 25–30 м)", "Спринт", "Скорость", 4, 1, 20, "м", True, 28, 7, .90, .06, .22, "SPR:1", "sprint", _SPR),
    _i("sp_fly30", "Бег с ходу 30 м (макс.)", "Спринт", "Скорость", 4, 1, 30, "м", True, 55, 7, .92, .08, .25, "SPR:1", "sprint", "100m,110mH,LJ", "ABC", "База из библиотеки: 4×30 м"),
    _i("sp_fly40", "Бег с ходу 40 м", "Спринт", "Скорость", 3, 1, 40, "м", True, 35, 8, .93, .10, .30, "SPR:1", "sprint", "100m,400m", "BC"),
    _i("sp_60", "60 м максимально со старта", "Спринт", "Скорость", 3, 1, 60, "м", True, 40, 8, .90, .22, .32, "SPR:1", "sprint", "100m", "BC"),
    _i("sp_80", "80 м максимально", "Спринт", "Скорость", 2, 1, 80, "м", True, 35, 8, .88, .30, .35, "SPR:.9,SE:.1", "sprint", "100m,400m", "BC"),
    _i("sp_tempo100", "Темповый бег 100 м (75–80 %)", "Спринт", "ОФП", 6, 1, 100, "м", True, 35, 5, .25, .35, .22, "SPR:.5,SE:.3,END:.2", "speed150", "100m,400m,1500m", "AB"),
    _i("sp_ladder", "«Лесенка» 30-60-90-60-30 м", "Спринт", "Скорость", 2, 5, 30, "м", False, 40, 7, .82, .30, .30, "SPR:.7,SE:.3", "sprint", "100m,400m", "BC"),
    _i("sp_resist", "Бег с резинкой/парашютом", "Спринт", "СФП", 5, 1, 30, "м", True, 28, 7, .75, .20, .45, "SPR:.8,THR:.2", "sprint", "100m,110mH", "AB"),
    _i("sp_down", "Бег под уклон (сверхскорость) 2–3°", "Спринт", "Скорость", 4, 1, 40, "м", True, 30, 7, .90, .08, .50, "SPR:1", "sprint", "100m", "BC", "Высокая механика: не в день тяжёлой силы"),
    _i("sp_reaction", "Старты на реакцию/сигнал", "Спринт", "Скорость", 6, 1, 10, "м", True, 22, 5, .55, .05, .12, "SPR:1", "sprint", "100m"),
    # ── бег 100–500 м ──
    _i("se_tempo200", "Темповый бег 200 м (80 %)", "Скоростная выносливость", "ОФП", 4, 1, 200, "м", True, 35, 6, .25, .55, .28, "SE:.6,SPR:.2,END:.2", "speed150", "400m,1500m", "AB"),
    _i("se_150", "150 м на 96–100 %, отдых 6–8 мин", "Скоростная выносливость", "Скорость", 3, 1, 150, "м", True, 40, 8, .65, .55, .38, "SE:.6,SPR:.4", "speed150", "100m,400m", "BC"),
    _i("se_200", "200 м на 90–95 %, отдых 6–8 мин", "Скоростная выносливость", "СФП", 3, 1, 200, "м", True, 40, 8, .55, .70, .38, "SE:.8,SPR:.2", "speedend", "400m", "BC"),
    _i("se_250", "250 м на 90–95 %", "Скоростная выносливость", "СФП", 3, 1, 250, "м", True, 42, 8, .58, .80, .40, "SE:.85,END:.15", "speedend", "400m", "BC"),
    _i("se_300", "300 м повторно (библиотека: 3–4×300)", "Скоростная выносливость", "СФП", 3, 1, 300, "м", True, 65, 8, .60, .93, .45, "SE:.85,END:.15", "speedend", "400m,1500m", "BC"),
    _i("se_350", "350 м на 90 %", "Скоростная выносливость", "СФП", 2, 1, 350, "м", True, 40, 9, .60, .95, .45, "SE:.85,END:.15", "speedend", "400m", "C"),
    _i("se_500", "500 м в 400-м темпе (контроль скоростной выносливости)", "Скоростная выносливость", "Контроль", 1, 1, 500, "м", True, 18, 9, .55, .95, .42, "SE:.7,END:.3", "speedend", "400m,1500m", "C"),
    _i("se_sub100", "100 м субмакс 85–90 % с короткими паузами", "Скоростная выносливость", "СФП", 4, 1, 100, "м", True, 70, 8, .55, .92, .50, "SE:.75,SPR:.25", "speed150", "400m,100m", "BC"),
    _i("se_hill", "Бег в горку 60 м", "Скоростная выносливость", "СФП", 6, 1, 60, "м", True, 35, 8, .60, .55, .70, "SPR:.5,SE:.3,THR:.2", "speed150", "400m,100m,LJ", "AB"),
    # ── барьеры ──
    _i("h_drills", "Барьерные упражнения: проходка толчковой/маховой ногой", "Барьеры", "Техника", 4, 5, None, "", False, 25, 4, .35, .05, .20, "SPR:.5,JMP:.5", "hurdles", "110mH", "ABC", "", True),
    _i("h_low", "Низкие барьеры, скоростной ритм", "Барьеры", "Скорость", 4, 5, None, "", False, 30, 6, .65, .10, .22, "SPR:.7,JMP:.3", "hurdles", "110mH,100m", "AB", "", True),
    _i("h_start", "Старт с барьерами (3 барьера)", "Барьеры", "Техника", 5, 3, None, "", False, 30, 7, .78, .15, .28, "SPR:.7,JMP:.3", "hurdles", "110mH", "BC", "", True),
    _i("h_rhythm", "Барьеры: ритм 3–5 барьеров (библиотека)", "Барьеры", "Техника", 4, 4, None, "", False, 70, 6, .72, .15, .30, "SPR:.7,JMP:.3", "hurdles", "110mH", "ABC", "", True),
    _i("h_full", "5 барьеров на скорость", "Барьеры", "Скорость", 3, 5, None, "", False, 35, 8, .85, .20, .35, "SPR:.7,JMP:.3", "hurdles", "110mH", "BC", "", True),
    _i("h_110", "Полная дистанция 110 м с/б", "Барьеры", "Контроль", 2, 10, None, "", False, 25, 9, .88, .45, .40, "SPR:.7,JMP:.3,SE:.1", "hurdles", "110mH", "C", "", True),
    _i("h_plyo", "Барьерная плиометрика (подскоки через барьеры)", "Барьеры", "СФП", 4, 6, None, "", False, 50, 7, .72, .15, .88, "JMP:.6,SPR:.4", "plyo", "110mH,LJ,HJ", "AB"),
    # ── прыжок в длину ──
    _i("lj_run", "Отработка разбега (отметки, без прыжка)", "Прыжок в длину", "Техника", 1, 6, None, "", False, 20, 4, .55, .05, .12, "SPR:.6,JMP:.4", None, "LJ", "ABC", "", True),
    _i("lj_short", "Прыжки с 6–8 шагов", "Прыжок в длину", "Техника", 1, 6, None, "", False, 25, 5, .62, .06, .25, "JMP:.8,SPR:.2", "LJ", "LJ", "ABC", "", True),
    _i("lj_mid", "Прыжки с 10–12 шагов", "Прыжок в длину", "Техника", 1, 6, None, "", False, 30, 6, .78, .08, .30, "JMP:.8,SPR:.2", "LJ", "LJ", "BC", "", True),
    _i("lj_full", "Прыжки с полного разбега", "Прыжок в длину", "Техника", 1, 6, None, "", False, 40, 7, .90, .10, .40, "JMP:.7,SPR:.3", "LJ", "LJ", "BC", "", True),
    _i("lj_stand", "Прыжок в длину с места", "Прыжок в длину", "СФП", 4, 3, None, "", False, 18, 6, .65, .05, .45, "JMP:.9,THR:.1", "LJ", "LJ", "ABC"),
    _i("lj_triple_st", "Тройной прыжок с места", "Прыжок в длину", "СФП", 4, 2, None, "", False, 18, 7, .72, .05, .65, "JMP:.9,SPR:.1", "plyo", "LJ,100m", "ABC", "Тест Нариманова"),
    _i("lj_five", "Пятерной прыжок / «прыжки с ноги на ногу»", "Прыжок в длину", "СФП", 4, 1, None, "", False, 20, 7, .70, .08, .75, "JMP:.8,SPR:.2", "plyo", "LJ,100m", "AB"),
    # ── прыжок в высоту ──
    _i("hj_low", "Высота: прыжки с 3–5 шагов, низкая планка", "Прыжок в высоту", "Техника", 1, 10, None, "", False, 35, 4, .50, .05, .20, "JMP:1", "HJ", "HJ", "ABC", "", True),
    _i("hj_short", "Высота: короткий разбег (библиотека)", "Прыжок в высоту", "Техника", 1, 10, None, "", False, 75, 5, .62, .10, .28, "JMP:1", "HJ", "HJ", "ABC", "", True),
    _i("hj_curve", "Высота: бег по дуге/отработка разбега", "Прыжок в высоту", "Техника", 1, 8, None, "", False, 20, 4, .55, .05, .15, "SPR:.5,JMP:.5", None, "HJ", "ABC", "", True),
    _i("hj_full", "Высота: полный разбег (библиотека)", "Прыжок в высоту", "Техника", 1, 8, None, "", False, 90, 6, .88, .12, .28, "JMP:.8,SPR:.2", "HJ", "HJ", "BC", "", True),
    _i("hj_bar", "Высота: серия на планке, соревновательная схема", "Прыжок в высоту", "Контроль", 1, 10, None, "", False, 60, 8, .90, .14, .32, "JMP:.8,SPR:.2", "HJ", "HJ", "C", "", True),
    # ── шест ──
    _i("pv_gym", "Шест: гимнастика (раскачивания на брусьях/кольцах, канат)", "Шест", "СФП", 4, 6, None, "", False, 30, 5, .30, .20, .30, "THR:.5,JMP:.5", None, "PV", "AB"),
    _i("pv_run", "Шест: разбег с шестом, постановка в упор", "Шест", "Техника", 1, 8, None, "", False, 35, 5, .60, .06, .18, "SPR:.5,JMP:.5", "PV", "PV", "ABC", "", True),
    _i("pv_short", "Шест: короткий разбег, технические звенья (библиотека)", "Шест", "Техника", 1, 8, None, "", False, 75, 5, .65, .12, .30, "JMP:.7,SPR:.3", "PV", "PV", "ABC", "", True),
    _i("pv_mid", "Шест: средний разбег (6–8 шагов)", "Шест", "Техника", 1, 8, None, "", False, 60, 6, .75, .12, .32, "JMP:.7,SPR:.3", "PV", "PV", "BC", "", True),
    _i("pv_full", "Шест: полный разбег (библиотека)", "Шест", "Техника", 1, 6, None, "", False, 90, 6, .90, .15, .30, "JMP:.7,SPR:.3", "PV", "PV", "BC", "", True),
    _i("pv_bar", "Шест: соревновательная серия через планку", "Шест", "Контроль", 1, 8, None, "", False, 90, 8, .92, .18, .35, "JMP:.7,SPR:.3", "PV", "PV", "C", "", True),
    # ── ядро ──
    _i("sp_stand", "Ядро: толчки с места", "Ядро", "Техника", 3, 4, None, "", False, 25, 5, .55, .08, .30, "THR:1", "SP", "SP", "ABC"),
    _i("sp_glide", "Ядро: толчки со скачка", "Ядро", "Техника", 3, 4, None, "", False, 30, 6, .72, .10, .38, "THR:1", "SP", "SP", "ABC"),
    _i("sp_full", "Ядро: полные броски (соревновательное ядро)", "Ядро", "Скорость", 3, 5, None, "", False, 35, 7, .85, .12, .50, "THR:1", "SP", "SP", "BC"),
    _i("sp_heavy", "Ядро тяжёлое (8–9 кг), толчки", "Ядро", "СФП", 3, 5, None, "", False, 30, 7, .70, .10, .55, "THR:1", "SP", "SP", "AB"),
    _i("sp_light", "Ядро лёгкое (5–6 кг), скорость", "Ядро", "СФП", 3, 6, None, "", False, 30, 6, .78, .10, .40, "THR:1", "SP", "SP", "BC"),
    _i("sp_back", "Бросок ядра назад через голову", "Ядро", "СФП", 4, 3, None, "", False, 18, 6, .80, .06, .40, "THR:.8,JMP:.2", "SP", "SP,DT,JT", "ABC", "Тест Нариманова"),
    _i("sp_chest", "Бросок ядра от груди двумя руками", "Ядро", "СФП", 4, 4, None, "", False, 18, 5, .72, .06, .30, "THR:.9,JMP:.1", "SP", "SP", "ABC"),
    # ── диск ──
    _i("dt_drill", "Диск: поворот без снаряда, имитация", "Диск", "Техника", 3, 6, None, "", False, 20, 3, .30, .04, .12, "THR:1", None, "DT", "ABC"),
    _i("dt_stand", "Диск: броски с места/полуповорота", "Диск", "Техника", 3, 5, None, "", False, 25, 5, .52, .06, .25, "THR:1", "DT", "DT", "ABC"),
    _i("dt_turn", "Диск: броски с поворота на 70–80 %", "Диск", "Техника", 3, 5, None, "", False, 30, 6, .65, .08, .30, "THR:1", "DT", "DT", "ABC"),
    _i("dt_full", "Диск: полные броски (2 кг)", "Диск", "Скорость", 3, 5, None, "", False, 35, 7, .82, .10, .45, "THR:1", "DT", "DT", "BC"),
    _i("dt_heavy", "Диск тяжёлый (2,5–3 кг)", "Диск", "СФП", 3, 5, None, "", False, 30, 7, .72, .10, .50, "THR:1", "DT", "DT", "AB"),
    _i("dt_light", "Диск лёгкий (1,5 кг), скорость", "Диск", "СФП", 3, 6, None, "", False, 30, 6, .78, .10, .38, "THR:1", "DT", "DT", "BC"),
    # ── копьё ──
    _i("jt_ball", "Копьё: метание мяча/дротика (600–800 г) с места и 3–5 шагов", "Копьё", "Техника", 3, 4, None, "", False, 22, 5, .58, .06, .25, "THR:1", "JT", "JT", "ABC"),
    _i("jt_cross", "Копьё: скрестный шаг, финальное усилие", "Копьё", "Техника", 3, 5, None, "", False, 25, 5, .62, .06, .28, "THR:1", "JT", "JT", "ABC"),
    _i("jt_short", "Копьё: с 5–7 шагов, 800 г", "Копьё", "Техника", 3, 4, None, "", False, 28, 6, .72, .08, .35, "THR:1", "JT", "JT", "BC"),
    _i("jt_full", "Копьё: с полного разбега", "Копьё", "Скорость", 3, 4, None, "", False, 35, 7, .85, .10, .50, "THR:1", "JT", "JT", "BC"),
    _i("jt_heavy", "Копьё/мяч тяжёлый (1 кг), через голову", "Копьё", "СФП", 3, 6, None, "", False, 25, 6, .70, .08, .45, "THR:1", "JT", "JT", "AB"),
    _i("jt_gym", "Плечевой пояс: резина/блок, ротация (профилактика локтя и плеча)", "Копьё", "ОФП", 3, 12, None, "", False, 15, 3, .12, .10, .15, "THR:.4", None, "JT", "ABC"),
    # ── силовые ──
    _i("st_squat_h", "Приседания тяжёлые (3×3 на 90 %)", "Силовая", "СФП", 3, 3, 130, "кг", True, 30, 8, .88, .18, .94, "THR:.5,JMP:.3,SPR:.2", "explosive", "SP,DT,JT,LJ,100m", "ABC", "База из библиотеки"),
    _i("st_squat_v", "Приседания объёмные (4×6 на 75 %)", "Силовая", "ОФП", 4, 6, 105, "кг", True, 35, 7, .40, .35, .75, "THR:.5,JMP:.25,SPR:.25", "strength", "SP,DT,JT", "A"),
    _i("st_front", "Фронтальные приседания", "Силовая", "СФП", 4, 5, 90, "кг", True, 30, 7, .55, .25, .70, "THR:.4,JMP:.3,SPR:.3", "strength", "SP,LJ,HJ", "AB"),
    _i("st_dead", "Становая тяга", "Силовая", "ОФП", 4, 4, 140, "кг", True, 32, 8, .70, .20, .85, "THR:.6,SPR:.2,JMP:.2", "strength", "SP,DT,JT", "AB"),
    _i("st_rdl", "Румынская тяга", "Силовая", "ОФП", 3, 8, 90, "кг", True, 22, 6, .35, .20, .65, "SPR:.4,THR:.4,JMP:.2", "strength", "100m,110mH", "AB", "Задняя поверхность бедра, профилактика"),
    _i("st_snatch", "Рывок 3×2–3 (библиотека)", "Силовая", "СФП", 3, 3, 70, "кг", True, 30, 8, .92, .10, .62, "THR:.4,JMP:.4,SPR:.2", "explosive", "SP,DT,JT,LJ,HJ,PV", "ABC"),
    _i("st_clean", "Взятие на грудь / подрыв", "Силовая", "СФП", 5, 3, 90, "кг", True, 30, 8, .85, .12, .60, "THR:.4,JMP:.4,SPR:.2", "explosive", "SP,DT,JT,LJ", "ABC"),
    _i("st_pushpress", "Швунг жимовой / жим стоя", "Силовая", "ОФП", 4, 5, 60, "кг", True, 25, 6, .55, .15, .35, "THR:.8,JMP:.2", "strength", "SP,JT,PV", "AB"),
    _i("st_bench", "Жим лёжа", "Силовая", "ОФП", 4, 6, 90, "кг", True, 28, 7, .45, .22, .40, "THR:1", "strength", "SP,DT,JT", "AB", "Тест Нариманова"),
    _i("st_pull", "Подтягивания", "Силовая", "ОФП", 4, 8, None, "", False, 20, 6, .35, .20, .30, "THR:.6,JMP:.4", "strength", "PV,JT,HJ", "ABC", "Нужны для шеста"),
    _i("st_row", "Тяга штанги в наклоне / гантели", "Силовая", "ОФП", 4, 8, 60, "кг", True, 22, 6, .30, .20, .30, "THR:.7,JMP:.3", "strength", "JT,SP,DT", "AB"),
    _i("st_lunge", "Выпады с гантелями/штангой", "Силовая", "ОФП", 3, 10, 40, "кг", True, 25, 7, .30, .35, .65, "JMP:.4,SPR:.3,THR:.3", "strength", "110mH,LJ,HJ", "AB"),
    _i("st_stepup", "Зашагивания на тумбу с отягощением", "Силовая", "СФП", 3, 8, 30, "кг", True, 22, 6, .45, .25, .60, "JMP:.6,SPR:.4", "strength", "HJ,LJ,100m", "AB"),
    _i("st_hipthrust", "Ягодичный мост со штангой", "Силовая", "ОФП", 4, 8, 100, "кг", True, 20, 6, .30, .18, .45, "SPR:.5,THR:.3,JMP:.2", "strength", "100m,110mH", "AB"),
    _i("st_nordic", "Скандинавские сгибания (профилактика бицепса бедра)", "Силовая", "ОФП", 3, 6, None, "", False, 15, 6, .25, .12, .60, "SPR:.5", None, "100m,110mH", "ABC"),
    _i("st_calf", "Подъёмы на носки (икры, ахилл)", "Силовая", "ОФП", 4, 12, 40, "кг", True, 15, 5, .15, .15, .40, "JMP:.5,SPR:.5", None, "110mH,LJ,HJ", "ABC"),
    _i("st_core", "Корпус: планка, V-ups, ротации (блок 10 мин)", "Силовая", "ОФП", 3, 12, None, "", False, 12, 4, .15, .10, .20, "THR:.3,JMP:.2", None, "", "ABC"),
    _i("st_hyper", "Гиперэкстензия / спина", "Силовая", "ОФП", 3, 12, None, "", False, 12, 4, .15, .10, .20, "THR:.3,SPR:.2", None, "", "ABC"),
    # ── плиометрика и СФП-прыжки/броски ──
    _i("pl_cmj", "Прыжки вверх с приседом (CMJ) × серии", "Плиометрика и СФП", "СФП", 4, 5, None, "", False, 15, 6, .62, .05, .50, "JMP:.9,SPR:.1", "plyo", "HJ,LJ,PV", "ABC"),
    _i("pl_drop", "Спрыгивания 40–60 см (библиотека)", "Плиометрика и СФП", "СФП", 4, 6, None, "", False, 45, 7, .62, .10, .92, "JMP:.85,SPR:.15", "plyo", "HJ,LJ,110mH", "AB", "Высокая механика"),
    _i("pl_box", "Запрыгивания на тумбу", "Плиометрика и СФП", "СФП", 4, 5, None, "", False, 18, 6, .55, .06, .45, "JMP:1", "plyo", "HJ,LJ", "ABC"),
    _i("pl_bound", "Прыжки «шаг-прыжок» (баундинг) 30 м", "Плиометрика и СФП", "СФП", 4, 1, 30, "м", True, 22, 7, .70, .15, .80, "JMP:.7,SPR:.3", "plyo", "LJ,100m,110mH", "AB"),
    _i("pl_hop", "Прыжки на одной ноге (10 м)", "Плиометрика и СФП", "СФП", 4, 2, 10, "м", True, 20, 7, .65, .12, .85, "JMP:.8,SPR:.2", "plyo", "LJ,HJ,110mH", "AB"),
    _i("pl_hill", "Прыжки на одной ноге в горку (библиотека)", "Плиометрика и СФП", "СФП", 5, 1, 30, "м", True, 50, 7, .55, .20, .98, "JMP:.8,SPR:.2", "plyo", "LJ,110mH,PV", "A", "Экстремальная механика: интервал ≥ 96 ч"),
    _i("pl_rope", "Скакалка / низкие подскоки", "Плиометрика и СФП", "ОФП", 4, 30, None, "", False, 12, 4, .30, .12, .35, "JMP:.7,SPR:.3", "plyo", "110mH,HJ,LJ", "ABC"),
    _i("pl_stairs", "Прыжки по ступеням / на двух ногах", "Плиометрика и СФП", "СФП", 4, 8, None, "", False, 18, 6, .55, .12, .70, "JMP:1", "plyo", "LJ,HJ", "AB"),
    _i("pl_kb", "Махи гирей / швунги", "Плиометрика и СФП", "ОФП", 4, 12, 24, "кг", True, 15, 6, .45, .30, .35, "THR:.5,SPR:.5", "strength", "SP,DT,100m", "AB"),
    _i("pl_med_over", "Набивной 3–5 кг: вперёд-вверх", "Плиометрика и СФП", "СФП", 4, 6, None, "", False, 15, 6, .75, .10, .35, "THR:.9,JMP:.1", "SP", "SP,JT", "ABC"),
    _i("pl_med_rot", "Набивной 3–5 кг: ротационные броски в стену", "Плиометрика и СФП", "СФП", 4, 6, None, "", False, 15, 6, .72, .10, .32, "THR:1", "DT", "DT,SP", "ABC"),
    _i("pl_med_back", "Набивной назад через голову", "Плиометрика и СФП", "СФП", 4, 6, None, "", False, 15, 6, .75, .08, .35, "THR:.9,JMP:.1", "SP", "SP,DT,JT", "ABC"),
    _i("pl_medball", "Взрывные броски ядра/мяча (библиотека)", "Плиометрика и СФП", "СФП", 4, 6, None, "", False, 40, 7, .78, .15, .45, "THR:.9,JMP:.1", "SP", "SP,DT,JT", "ABC"),
    # ── выносливость ──
    _i("en_cross", "Кросс (кроссовая часть)", "Выносливость", "ОФП", 1, 1, 5, "км", True, 28, 5, .05, .45, .15, "END:1", "endurance", "1500m", "ABC", "5 км ≈ 28 мин при 5:30/км"),
    _i("en_easy", "Лёгкий бег восстановительный", "Выносливость", "Восстановление", 1, 1, 3, "км", True, 18, 3, .03, .25, .10, "END:1", "endurance", "1500m", "ABC"),
    _i("en_fartlek", "Фартлек 20 мин", "Выносливость", "ОФП", 1, 1, 4, "км", True, 20, 6, .12, .60, .18, "END:.8,SE:.2", "endurance", "1500m,400m", "AB"),
    _i("en_tempo", "Темповый бег (библиотека)", "Выносливость", "ОФП", 1, 1, 4, "км", True, 35, 4, .08, .40, .18, "END:.8,SPR:.2", "endurance", "1500m", "ABC"),
    _i("en_int400", "Интервалы 6×400 м (темп 1500 м)", "Выносливость", "СФП", 6, 1, 400, "м", True, 40, 8, .20, .80, .28, "END:.6,SE:.4", "speedend", "1500m,400m", "BC"),
    _i("en_int600", "Интервалы 4×600 м", "Выносливость", "СФП", 4, 1, 600, "м", True, 38, 8, .18, .82, .25, "END:.7,SE:.3", "speedend", "1500m", "BC"),
    _i("en_int800", "Интервалы 3×800 м", "Выносливость", "СФП", 3, 1, 800, "м", True, 36, 8, .15, .85, .22, "END:.8,SE:.2", "speedend", "1500m", "BC"),
    _i("en_int1000", "Интервалы 4–5×1000 м (библиотека)", "Выносливость", "СФП", 4, 1, 1000, "м", True, 60, 7, .20, .85, .20, "END:.8,SE:.2", "speedend", "1500m", "ABC"),
    _i("en_test1000", "Контрольный бег 1000 м", "Выносливость", "Контроль", 1, 1, 1000, "м", True, 15, 9, .25, .90, .25, "END:.7,SE:.3", "speedend", "1500m", "ABC", "Тест Нариманова"),
    _i("en_bike", "Велотренажёр/эллипс 30 мин (без удара)", "Выносливость", "Восстановление", 1, 1, 30, "мин", True, 30, 4, .03, .35, .04, "END:.9", None, "1500m", "ABC"),
    _i("en_pool", "Плавание/аквабег 20 мин", "Выносливость", "Восстановление", 1, 1, 20, "мин", True, 20, 3, .03, .30, .03, "END:.8", None, "1500m", "ABC"),
    # ── ОФП ──
    _i("of_circuit", "Круговая тренировка (8 станций × 3 круга)", "ОФП", "ОФП", 3, 8, None, "", False, 40, 6, .20, .55, .35, "END:.4,THR:.3,JMP:.3", None, "1500m,SP", "A"),
    _i("of_gym", "Гимнастика: стойки, мосты, перекаты, кувырки", "ОФП", "ОФП", 1, 1, 20, "мин", True, 20, 3, .15, .08, .12, "JMP:.5,THR:.5", None, "PV,HJ", "AB"),
    _i("of_games", "Подвижные игры (баскетбол/футбол/эстафеты)", "ОФП", "ОФП", 1, 1, 30, "мин", True, 30, 5, .25, .45, .25, "SPR:.4,END:.4,JMP:.2", None, "100m,400m,1500m", "AB"),
    _i("of_swim", "Плавание (общая подготовка)", "ОФП", "ОФП", 1, 1, 30, "мин", True, 30, 4, .05, .35, .05, "END:.8", None, "1500m", "ABC"),
    _i("of_core_circ", "Корпус и стабилизация, круг 15 мин", "ОФП", "ОФП", 3, 5, None, "", False, 15, 4, .12, .10, .18, "THR:.3,JMP:.2", None, "", "ABC"),
    # ── восстановление ──
    _i("rc_mob", "Мобильность и профилактика", "Восстановление", "Восстановление", 1, 1, 30, "мин", True, 30, 3, .03, .03, .05, "", "mobility", "", "ABC"),
    _i("rc_foam", "Ролл/самомассаж", "Восстановление", "Восстановление", 1, 1, 15, "мин", True, 15, 2, .01, .02, .02, "", "mobility", "", "ABC"),
    _i("rc_stretch", "Растяжка статическая", "Восстановление", "Восстановление", 1, 1, 20, "мин", True, 20, 2, .01, .02, .03, "", "mobility", "", "ABC"),
    _i("rc_contrast", "Контрастный душ / баня / бассейн", "Восстановление", "Восстановление", 1, 1, 20, "мин", True, 20, 2, .01, .02, .02, "", None, "", "ABC"),
    _i("rc_cool", "Заминка: лёгкий бег + растяжка", "Восстановление", "Восстановление", 1, 1, 10, "мин", True, 10, 2, .02, .08, .05, "END:.5", None, "", "ABC"),
    # ── контроль ──
    _i("ts_fly30", "Тест: бег 30 м с хода", "Контроль", "Контроль", 3, 1, 30, "м", True, 20, 8, .92, .06, .25, "SPR:1", "sprint", "100m", "ABC", "Тест Грузенкина", True),
    _i("ts_cmj", "Тест: прыжок вверх (CMJ)", "Контроль", "Контроль", 3, 3, None, "", False, 10, 5, .55, .04, .30, "JMP:1", None, "HJ,LJ", "ABC"),
    _i("ts_60", "Тест: 60 м со старта", "Контроль", "Контроль", 2, 1, 60, "м", True, 20, 8, .90, .20, .30, "SPR:1", "sprint", "100m", "BC"),
    _i("ts_100", "Контрольный бег 100 м", "Контроль", "Контроль", 1, 1, 100, "м", True, 12, 9, .90, .35, .35, "SPR:.8,SE:.2", "sprint", "100m", "C"),
    _i("ts_shot", "Тест: ядро назад / с места", "Контроль", "Контроль", 3, 3, None, "", False, 15, 6, .78, .05, .35, "THR:1", "SP", "SP", "ABC", "Тест Нариманова"),
]
CATALOG += [
    # ── разминка и подготовка (ещё) ──
    _i("wu_skips", "Прыжковая разминка: подскоки, прыжки на месте, «пружинки»", "Разминка", "ОФП", 3, 10, None, "", False, 8, 3, .10, .05, .15, "JMP:1", None, "LJ,HJ,110mH"),
    _i("wu_minihurd", "Мини-барьеры: ритмовая проходка (6–8 шт.)", "Разминка", "ОФП", 4, 1, 8, "шт", True, 10, 3, .15, .04, .12, "SPR:.5,JMP:.5", None, "110mH,100m"),
    _i("wu_throw", "Метательная разминка: резина, набивной 2 кг, вращения", "Разминка", "ОФП", 2, 10, None, "", False, 10, 3, .10, .05, .10, "THR:1", None, "SP,DT,JT"),
    _i("wu_hurdle_mob", "Барьерная мобильность (проход боком, мах, «колесо»)", "Разминка", "ОФП", 3, 6, None, "", False, 8, 2, .05, .03, .08, "SPR:.3,JMP:.2", None, "110mH"),
    _i("wu_pole", "Шестовая разминка: бег с шестом 20 м, висы", "Разминка", "ОФП", 3, 1, 20, "м", True, 10, 3, .15, .05, .10, "SPR:.4,JMP:.4,THR:.2", None, "PV"),
    # ── спринт (ещё) ──
    _i("sp_block30", "Старт с колодок 30 м", "Спринт", "Скорость", 5, 1, 30, "м", True, 32, 8, .90, .15, .32, "SPR:1", "sprint", "100m,110mH", "BC"),
    _i("sp_45", "45 м со старта (с колодок или высокого)", "Спринт", "Скорость", 4, 1, 45, "м", True, 35, 8, .90, .18, .32, "SPR:1", "sprint", "100m,400m", "BC"),
    _i("sp_50", "50 м максимально", "Спринт", "Скорость", 3, 1, 50, "м", True, 32, 8, .90, .20, .33, "SPR:1", "sprint", "100m", "BC"),
    _i("sp_70", "70 м максимально", "Спринт", "Скорость", 3, 1, 70, "м", True, 38, 8, .88, .26, .34, "SPR:.95,SE:.05", "sprint", "100m,400m", "BC"),
    _i("sp_120", "120 м на 95 %", "Спринт", "Скорость", 3, 1, 120, "м", True, 40, 8, .80, .40, .36, "SPR:.8,SE:.2", "speed150", "100m,400m", "BC"),
    _i("sp_wicket", "Бег через мини-барьеры (частота, шаги)", "Спринт", "Техника", 5, 1, 30, "м", True, 25, 6, .62, .08, .20, "SPR:1", "sprint", "100m,110mH", "ABC", "", True),
    _i("sp_tow", "Сверхскорость: буксировка/эластичный трос", "Спринт", "Скорость", 4, 1, 30, "м", True, 30, 7, .88, .08, .55, "SPR:1", "sprint", "100m", "BC", "Высокая механика — не после тяжёлой силы"),
    _i("sp_highknee", "Бег с высоким подниманием бедра на скорость", "Спринт", "Техника", 5, 1, 20, "м", True, 20, 5, .45, .08, .15, "SPR:1", "sprint", "100m,110mH", "ABC"),
    _i("sp_curve", "Бег по виражу 60 м (вход в поворот)", "Спринт", "Скорость", 4, 1, 60, "м", True, 35, 7, .75, .28, .40, "SPR:.7,SE:.3", "sprint", "400m,100m", "BC"),
    _i("sp_flat200", "200 м максимально", "Спринт", "Скорость", 2, 1, 200, "м", True, 35, 9, .65, .75, .42, "SE:.6,SPR:.4", "speedend", "400m,100m", "C"),
    _i("sp_inertia", "Разгон–инерция–разгон 30+30+30 м", "Спринт", "Скорость", 3, 1, 90, "м", True, 35, 7, .80, .22, .30, "SPR:.8,SE:.2", "sprint", "100m,400m", "BC"),
    _i("sp_stairs", "Бег по лестнице/ступеням вверх (силовая скорость)", "Спринт", "СФП", 5, 1, 12, "м", True, 22, 7, .60, .20, .60, "SPR:.6,JMP:.4", "sprint", "100m,LJ", "AB"),
    # ── скоростная выносливость (ещё) ──
    _i("se_120", "120 м на 95 %, отдых 5–6 мин", "Скоростная выносливость", "Скорость", 3, 1, 120, "м", True, 35, 8, .70, .45, .36, "SE:.5,SPR:.5", "speed150", "100m,400m", "BC"),
    _i("se_2x200", "Парные 200 м: 2×(200 + 200) с отдыхом 3 мин внутри пары", "Скоростная выносливость", "СФП", 2, 2, 200, "м", True, 55, 9, .55, .90, .42, "SE:.85,END:.15", "speedend", "400m", "C"),
    _i("se_broken400", "Ломаный 400: 4×100 с паузой 30–40 с", "Скоростная выносливость", "СФП", 1, 4, 100, "м", True, 20, 9, .55, .92, .40, "SE:.85,END:.15", "speedend", "400m", "C"),
    _i("se_400", "400 м контрольный/на скорость", "Скоростная выносливость", "Контроль", 1, 1, 400, "м", True, 15, 10, .55, 1.0, .42, "SE:.8,END:.2", "speedend", "400m,1500m", "C"),
    _i("se_600", "600 м в 90 % от лучшего темпа", "Скоростная выносливость", "СФП", 2, 1, 600, "м", True, 35, 8, .35, .90, .30, "END:.5,SE:.5", "speedend", "400m,1500m", "BC"),
    _i("se_pyramid", "Пирамида 100-150-200-150-100 м", "Скоростная выносливость", "СФП", 1, 5, 140, "м", True, 40, 8, .55, .80, .38, "SE:.7,SPR:.3", "speedend", "400m,100m", "BC"),
    _i("se_300x2", "2×300 м на 92–95 %, отдых 8–10 мин", "Скоростная выносливость", "СФП", 2, 1, 300, "м", True, 35, 9, .58, .95, .45, "SE:.85,END:.15", "speedend", "400m", "C"),
    _i("se_short_rec", "10×100 м (пауза 60–90 с) в темпе 400 м", "Скоростная выносливость", "ОФП", 10, 1, 100, "м", True, 35, 7, .35, .75, .30, "SE:.6,END:.4", "speed150", "400m,1500m", "AB"),
    _i("se_hillrun", "Бег в горку 150 м на 85–90 %", "Скоростная выносливость", "СФП", 4, 1, 150, "м", True, 35, 8, .40, .75, .65, "SE:.5,END:.3,SPR:.2", "speedend", "400m,1500m", "AB"),
    # ── барьеры (ещё) ──
    _i("h_1h", "Один барьер: проход с разбега 3–5 шагов (атака)", "Барьеры", "Техника", 4, 6, None, "", False, 25, 5, .55, .05, .20, "SPR:.5,JMP:.5", "hurdles", "110mH", "ABC", "", True),
    _i("h_trail", "Проходка маховой ногой через 5 барьеров", "Барьеры", "Техника", 4, 4, None, "", False, 22, 4, .30, .04, .18, "SPR:.4,JMP:.6", None, "110mH", "ABC", "", True),
    _i("h_lead", "Проходка толчковой ногой через 5 барьеров", "Барьеры", "Техника", 4, 4, None, "", False, 22, 4, .30, .04, .18, "SPR:.4,JMP:.6", None, "110mH", "ABC", "", True),
    _i("h_narrow", "Сокращённые расстояния между барьерами (ритм)", "Барьеры", "Техника", 4, 4, None, "", False, 28, 6, .65, .10, .25, "SPR:.7,JMP:.3", "hurdles", "110mH", "AB", "", True),
    _i("h_high", "Барьеры повышенные/расстояния увеличены (запас)", "Барьеры", "Скорость", 3, 4, None, "", False, 28, 7, .72, .12, .30, "SPR:.7,JMP:.3", "hurdles", "110mH", "BC", "", True),
    _i("h_3step", "Старт и три барьера в соревновательном ритме (3 шага)", "Барьеры", "Скорость", 5, 1, 3, "шт", True, 30, 8, .85, .12, .30, "SPR:.7,JMP:.3", "hurdles", "110mH", "BC", "", True),
    _i("h_8", "8 барьеров на скорость (контроль выносливости на барьерах)", "Барьеры", "Контроль", 2, 1, 8, "шт", True, 25, 9, .85, .35, .40, "SPR:.7,JMP:.3,SE:.1", "hurdles", "110mH", "C", "", True),
    _i("h_sled_hop", "Подскоки через барьеры (45–60 см) — частота", "Барьеры", "СФП", 4, 8, None, "", False, 22, 6, .55, .10, .60, "JMP:.7,SPR:.3", "plyo", "110mH,LJ", "AB"),
    # ── прыжок в длину (ещё) ──
    _i("lj_3step", "Отталкивание с 3 шагов (стойка, вынос, посадка)", "Прыжок в длину", "Техника", 1, 8, None, "", False, 22, 4, .45, .04, .15, "JMP:1", "LJ", "LJ", "ABC", "", True),
    _i("lj_5step", "Прыжки с 5 шагов", "Прыжок в длину", "Техника", 1, 6, None, "", False, 22, 5, .55, .05, .20, "JMP:.9,SPR:.1", "LJ", "LJ", "ABC", "", True),
    _i("lj_hang", "Прыжок «в шаге/виса» с 8 шагов — работа в воздухе", "Прыжок в длину", "Техника", 1, 6, None, "", False, 25, 5, .62, .06, .24, "JMP:.9,SPR:.1", "LJ", "LJ", "ABC", "", True),
    _i("lj_comp", "Прыжки в длину: соревновательная серия (6 попыток)", "Прыжок в длину", "Контроль", 1, 6, None, "", False, 45, 8, .92, .12, .45, "JMP:.7,SPR:.3", "LJ", "LJ", "C", "", True),
    _i("lj_zone", "Прыжок с попаданием на брусок (контроль разбега)", "Прыжок в длину", "Техника", 1, 8, None, "", False, 25, 6, .70, .08, .25, "JMP:.7,SPR:.3", "LJ", "LJ", "BC", "", True),
    _i("lj_sl", "Прыжок в длину с места на одной ноге", "Прыжок в длину", "СФП", 3, 3, None, "", False, 15, 6, .55, .05, .60, "JMP:1", "plyo", "LJ", "ABC"),
    _i("lj_ten", "Десятерной прыжок с места (с ноги на ногу)", "Прыжок в длину", "СФП", 3, 1, None, "", False, 15, 7, .65, .12, .75, "JMP:.8,SPR:.2", "plyo", "LJ,100m", "AB"),
    _i("lj_triple_run", "Тройной с короткого разбега 6 шагов", "Прыжок в длину", "СФП", 3, 2, None, "", False, 18, 7, .75, .08, .75, "JMP:.8,SPR:.2", "plyo", "LJ,100m", "AB"),
    # ── прыжок в высоту (ещё) ──
    _i("hj_stand", "Высота: толчок с 1–3 шагов + постановка стопы", "Прыжок в высоту", "Техника", 1, 10, None, "", False, 22, 4, .40, .04, .15, "JMP:1", "HJ", "HJ", "ABC", "", True),
    _i("hj_back", "Высота: «мост» спиной через планку на низкой высоте", "Прыжок в высоту", "Техника", 1, 8, None, "", False, 20, 4, .40, .04, .15, "JMP:1", "HJ", "HJ", "ABC", "", True),
    _i("hj_med", "Высота: 7–9 шагов, 85–90 % от лучшего", "Прыжок в высоту", "Техника", 1, 8, None, "", False, 45, 6, .75, .10, .28, "JMP:.9,SPR:.1", "HJ", "HJ", "BC", "", True),
    _i("hj_prog", "Высота: набор высоты по шагам (соревновательная схема)", "Прыжок в высоту", "Контроль", 1, 12, None, "", False, 70, 9, .92, .14, .36, "JMP:.8,SPR:.2", "HJ", "HJ", "C", "", True),
    _i("hj_depth", "Высота: прыжок с «глубины» + толчок", "Прыжок в высоту", "СФП", 4, 5, None, "", False, 25, 7, .70, .08, .70, "JMP:1", "plyo", "HJ", "AB"),
    # ── шест (ещё) ──
    _i("pv_swing", "Шест: раскачивания на брусьях/перекладине", "Шест", "СФП", 4, 6, None, "", False, 25, 5, .25, .15, .20, "THR:.5,JMP:.5", None, "PV", "ABC"),
    _i("pv_inv", "Шест: вис на кольцах, подъём ног, проход в стойку", "Шест", "СФП", 4, 5, None, "", False, 25, 6, .30, .18, .25, "THR:.6,JMP:.4", None, "PV", "ABC"),
    _i("pv_plant", "Шест: постановка в ящик 5–6 шагов", "Шест", "Техника", 1, 10, None, "", False, 28, 5, .55, .06, .20, "JMP:.6,SPR:.4", "PV", "PV", "ABC", "", True),
    _i("pv_rope", "Лазание по канату без ног / подъём 5 м", "Шест", "СФП", 3, 2, 5, "м", True, 20, 6, .30, .20, .30, "THR:.7,JMP:.3", None, "PV", "AB"),
    _i("pv_hangup", "Вис на перекладине: подъём прямых ног к перекладине", "Шест", "СФП", 4, 8, None, "", False, 15, 5, .20, .12, .20, "THR:.6,JMP:.4", None, "PV", "ABC"),
    _i("pv_high", "Шест: увеличенный хват/мягкий шест — работа на «взлёте»", "Шест", "Техника", 1, 6, None, "", False, 45, 6, .75, .10, .32, "JMP:.7,SPR:.3", "PV", "PV", "BC", "", True),
    # ── ядро (ещё) ──
    _i("sp_fwd_stand", "Ядро вперёд с места (контрольные)", "Ядро", "Контроль", 3, 3, None, "", False, 15, 6, .75, .05, .35, "THR:1", "SP", "SP", "ABC", "Тест"),
    _i("sp_shift", "Ядро: «скачок» без добавления ноги — работа ног", "Ядро", "Техника", 3, 5, None, "", False, 25, 5, .60, .08, .30, "THR:1", "SP", "SP", "ABC"),
    _i("sp_over", "Ядро через голову вперёд (7,26 кг и 6 кг)", "Ядро", "СФП", 4, 4, None, "", False, 18, 6, .78, .06, .40, "THR:.9,JMP:.1", "SP", "SP,DT,JT", "ABC", "Тест Нариманова"),
    _i("sp_comp", "Ядро: соревновательная серия (6 бросков)", "Ядро", "Контроль", 1, 6, None, "", False, 30, 8, .90, .12, .50, "THR:1", "SP", "SP", "C"),
    _i("sp_hip", "Ядро: бросок с поворота корпуса на низкой стойке", "Ядро", "Техника", 3, 5, None, "", False, 22, 5, .55, .06, .25, "THR:1", "SP", "SP", "ABC"),
    # ── диск (ещё) ──
    _i("dt_wall", "Диск: имитация с резиной, стена/блок", "Диск", "СФП", 3, 8, None, "", False, 18, 4, .35, .05, .15, "THR:1", None, "DT", "ABC"),
    _i("dt_step", "Диск: бросок с шага + вращение стопы", "Диск", "Техника", 3, 6, None, "", False, 22, 5, .55, .06, .25, "THR:1", "DT", "DT", "ABC"),
    _i("dt_comp", "Диск: соревновательная серия (6 бросков)", "Диск", "Контроль", 1, 6, None, "", False, 30, 8, .90, .12, .50, "THR:1", "DT", "DT", "C"),
    _i("dt_med", "Набивной мяч через голову в поворот (3 кг)", "Диск", "СФП", 4, 6, None, "", False, 15, 6, .72, .08, .30, "THR:1", "DT", "DT,SP", "ABC"),
    _i("dt_stand_full", "Диск: бросок с места (контроль силы без разворота)", "Диск", "Контроль", 3, 3, None, "", False, 18, 6, .65, .06, .28, "THR:1", "DT", "DT", "ABC"),
    # ── копьё (ещё) ──
    _i("jt_stand", "Копьё: метание с места (стойка, финальное усилие)", "Копьё", "Контроль", 3, 5, None, "", False, 22, 6, .70, .06, .30, "THR:1", "JT", "JT", "ABC", "Тест"),
    _i("jt_3step", "Копьё: 3 шага до метания", "Копьё", "Техника", 3, 4, None, "", False, 25, 5, .60, .06, .28, "THR:1", "JT", "JT", "ABC"),
    _i("jt_run", "Копьё: разбег без метания + отведение", "Копьё", "Техника", 3, 4, None, "", False, 20, 4, .45, .06, .15, "THR:.6,SPR:.4", "JT", "JT", "ABC", "", True),
    _i("jt_comp", "Копьё: соревновательная серия (6 метаний)", "Копьё", "Контроль", 1, 6, None, "", False, 35, 8, .90, .12, .55, "THR:1", "JT", "JT", "C"),
    _i("jt_med", "Набивной 2 кг одной рукой через плечо (имитация)", "Копьё", "СФП", 4, 6, None, "", False, 15, 5, .60, .08, .30, "THR:1", "JT", "JT", "ABC"),
    _i("jt_pullover", "Набивной через голову двумя руками стоя/с шага", "Копьё", "СФП", 4, 6, None, "", False, 15, 5, .65, .08, .30, "THR:1", "JT", "JT,SP", "ABC"),
    # ── силовые (ещё) ──
    _i("st_hangclean", "Подрыв из виса", "Силовая", "СФП", 5, 3, 70, "кг", True, 28, 8, .85, .12, .55, "THR:.4,JMP:.4,SPR:.2", "explosive", "SP,DT,JT,LJ,HJ", "ABC"),
    _i("st_jerk", "Толчок со стоек / швунг толчковый", "Силовая", "СФП", 4, 3, 65, "кг", True, 25, 8, .88, .12, .55, "THR:.5,JMP:.3,SPR:.2", "explosive", "SP,JT,PV", "ABC"),
    _i("st_jumpsq", "Приседания с прыжком (30 % от максимума)", "Силовая", "СФП", 4, 5, 40, "кг", True, 20, 7, .80, .10, .60, "JMP:.8,SPR:.2", "explosive", "LJ,HJ,100m", "ABC"),
    _i("st_bulg", "Болгарские сплит-приседания", "Силовая", "ОФП", 3, 8, 35, "кг", True, 24, 7, .30, .30, .65, "JMP:.4,SPR:.3,THR:.3", "strength", "110mH,LJ,HJ", "AB"),
    _i("st_incline", "Жим на наклонной / гантели", "Силовая", "ОФП", 4, 8, 30, "кг", True, 24, 6, .35, .20, .30, "THR:1", "strength", "SP,JT", "AB"),
    _i("st_dips", "Отжимания на брусьях (с отягощением)", "Силовая", "ОФП", 4, 8, 10, "кг", True, 18, 6, .30, .18, .30, "THR:1", "strength", "SP,PV", "AB"),
    _i("st_ohp", "Жим штанги стоя", "Силовая", "ОФП", 4, 6, 50, "кг", True, 22, 6, .45, .18, .30, "THR:1", "strength", "SP,JT,PV", "AB"),
    _i("st_legpress", "Жим ногами", "Силовая", "ОФП", 4, 10, 180, "кг", True, 22, 6, .15, .30, .45, "JMP:.4,SPR:.3,THR:.3", "strength", "100m,LJ", "A"),
    _i("st_goodm", "Наклоны со штангой (good morning)", "Силовая", "ОФП", 3, 8, 60, "кг", True, 18, 6, .25, .15, .55, "SPR:.5,THR:.3,JMP:.2", "strength", "100m,110mH", "AB"),
    _i("st_trap", "Тяга с трэп-грифом (динамическая)", "Силовая", "СФП", 4, 3, 120, "кг", True, 25, 8, .72, .15, .75, "THR:.5,SPR:.2,JMP:.3", "explosive", "SP,DT,JT,LJ", "AB"),
    _i("st_farmer", "Прогулка фермера / удержание", "Силовая", "ОФП", 3, 1, 30, "м", True, 14, 5, .10, .25, .25, "THR:.5,END:.5", None, "", "AB"),
    _i("st_sledpush", "Толкание тяжёлых санок 20 м", "Силовая", "СФП", 5, 1, 20, "м", True, 22, 7, .55, .30, .65, "SPR:.6,THR:.4", "strength", "100m,SP", "AB"),
    _i("st_ghr", "Глют-хэм-рейз / ножной сгиб", "Силовая", "ОФП", 3, 6, None, "", False, 15, 6, .20, .10, .50, "SPR:.5", None, "100m,110mH", "ABC"),
    _i("st_pullw", "Подтягивания с отягощением", "Силовая", "ОФП", 4, 5, 10, "кг", True, 20, 7, .38, .20, .35, "THR:.6,JMP:.4", "strength", "PV,JT", "AB"),
    _i("st_abwheel", "Колесо для пресса / ролик", "Силовая", "ОФП", 3, 10, None, "", False, 10, 5, .15, .10, .18, "THR:.3", None, "", "ABC"),
    # ── плиометрика (ещё) ──
    _i("pl_depth", "Глубинные прыжки (спрыг + выпрыгивание)", "Плиометрика и СФП", "СФП", 4, 5, None, "", False, 22, 8, .75, .10, .90, "JMP:1", "plyo", "HJ,LJ", "AB", "Высокая механика"),
    _i("pl_hurdle_hop", "Прыжки через барьеры 60–76 см подряд", "Плиометрика и СФП", "СФП", 4, 6, None, "", False, 22, 7, .65, .12, .75, "JMP:.9,SPR:.1", "plyo", "HJ,LJ,110mH", "AB"),
    _i("pl_split", "Выпрыгивания из выпада со сменой ног", "Плиометрика и СФП", "СФП", 3, 10, None, "", False, 15, 6, .50, .15, .55, "JMP:.8,SPR:.2", "plyo", "110mH,LJ,HJ", "AB"),
    _i("pl_lateral", "Боковые прыжки и скачки через ленту", "Плиометрика и СФП", "ОФП", 3, 12, None, "", False, 12, 5, .35, .12, .30, "JMP:.7,SPR:.3", "plyo", "HJ,110mH", "ABC"),
    _i("pl_scoop", "Набивной 3–5 кг снизу-вперёд (дальность)", "Плиометрика и СФП", "СФП", 4, 5, None, "", False, 15, 6, .75, .08, .35, "THR:.8,JMP:.2", "SP", "SP,JT,DT", "ABC", "Тест Нариманова"),
    _i("pl_medjump", "Прыжок с набивным: вверх и вперёд", "Плиометрика и СФП", "СФП", 4, 5, None, "", False, 15, 6, .70, .10, .50, "JMP:.6,THR:.4", "plyo", "LJ,SP,HJ", "ABC"),
    _i("pl_reach", "Прыжок вверх с отметкой (тест/тренировка)", "Плиометрика и СФП", "Контроль", 3, 5, None, "", False, 12, 6, .62, .04, .45, "JMP:1", None, "HJ,LJ,PV", "ABC", "Результат — высота прыжка"),
    _i("pl_ankle", "Подскоки на жёстких голеностопах, низкий амплитудный", "Плиометрика и СФП", "СФП", 4, 15, None, "", False, 12, 5, .40, .10, .40, "JMP:.7,SPR:.3", "plyo", "100m,110mH", "ABC"),
    _i("pl_sprint_bound", "Баундинг 40–50 м (дальность шага/время)", "Плиометрика и СФП", "СФП", 3, 1, 40, "м", True, 22, 7, .70, .20, .80, "JMP:.6,SPR:.4", "plyo", "LJ,100m", "AB"),
    _i("pl_standing5", "Пятерной с места (дальность)", "Плиометрика и СФП", "СФП", 3, 1, None, "", False, 15, 7, .65, .10, .80, "JMP:.9,SPR:.1", "plyo", "LJ,100m", "AB"),
    # ── выносливость (ещё) ──
    _i("en_strides", "Ускорения 8×100 м после лёгкого бега", "Выносливость", "ОФП", 8, 1, 100, "м", True, 20, 5, .25, .20, .20, "SPR:.5,END:.5", "endurance", "100m,1500m", "ABC"),
    _i("en_300rep", "10×300 м (по 55–60 с, пауза 2 мин)", "Выносливость", "СФП", 10, 1, 300, "м", True, 45, 8, .25, .85, .30, "END:.7,SE:.3", "speedend", "1500m,400m", "BC"),
    _i("en_3x1000", "Темповые 3×1000 м (темп 3 км)", "Выносливость", "СФП", 3, 1, 1000, "м", True, 40, 7, .15, .75, .20, "END:.9,SE:.1", "speedend", "1500m", "AB"),
    _i("en_ladder", "Лесенка 200-400-600-400-200", "Выносливость", "СФП", 1, 5, 360, "м", True, 35, 8, .22, .85, .25, "END:.6,SE:.4", "speedend", "1500m,400m", "BC"),
    _i("en_500pace", "3×500 м в темпе 1500 м", "Выносливость", "СФП", 3, 1, 500, "м", True, 35, 8, .22, .90, .25, "END:.7,SE:.3", "speedend", "1500m", "BC"),
    _i("en_1200", "Тренировочный бег 1200 м в 90 % от темпа 1500", "Выносливость", "СФП", 2, 1, 1200, "м", True, 30, 8, .15, .85, .22, "END:.8,SE:.2", "speedend", "1500m", "BC"),
    _i("en_1500", "Контрольный бег 1500 м", "Выносливость", "Контроль", 1, 1, 1500, "м", True, 10, 10, .22, 1.0, .28, "END:.7,SE:.3", "speedend", "1500m", "C"),
    _i("en_hillrep", "Повторный бег в горку 200 м (3–5 повторений)", "Выносливость", "СФП", 4, 1, 200, "м", True, 28, 8, .30, .80, .60, "END:.4,SE:.4,SPR:.2", "speedend", "1500m,400m", "AB"),
    _i("en_tempo_short", "Темповый бег 2 км (умеренно)", "Выносливость", "ОФП", 1, 1, 2, "км", True, 14, 5, .06, .45, .16, "END:1", "endurance", "1500m", "ABC"),
    _i("en_run_walk", "Бег и ходьба 3 км (реабилитационный)", "Выносливость", "Восстановление", 1, 1, 3, "км", True, 25, 2, .02, .15, .06, "END:1", None, "1500m", "ABC"),
    _i("en_row", "Гребной эргометр 20 мин", "Выносливость", "Восстановление", 1, 1, 20, "мин", True, 20, 4, .04, .35, .06, "END:.8,THR:.2", None, "1500m", "ABC"),
    # ── ОФП (ещё) ──
    _i("of_ladder", "Координационная лестница (10 вариантов)", "ОФП", "ОФП", 3, 6, None, "", False, 15, 4, .20, .06, .15, "SPR:.5,JMP:.5", None, "110mH,100m", "ABC"),
    _i("of_sand", "Бег и прыжки по песку", "ОФП", "ОФП", 1, 1, 25, "мин", True, 25, 5, .08, .45, .30, "END:.4,SPR:.3,JMP:.3", None, "1500m,LJ", "A"),
    _i("of_acro", "Акробатика/батут: сальто, перевороты", "ОФП", "ОФП", 1, 1, 20, "мин", True, 20, 3, .15, .05, .10, "JMP:.7,THR:.3", None, "PV,HJ", "AB"),
    _i("of_ball_drills", "Баскетбол/ручной мяч: передачи, броски, рывки", "ОФП", "ОФП", 1, 1, 25, "мин", True, 25, 4, .20, .40, .20, "SPR:.4,THR:.3,END:.3", None, "100m,400m,JT", "AB"),
    _i("of_stairs", "Подъёмы по лестнице бегом 6–8 этажей", "ОФП", "ОФП", 6, 1, None, "", False, 25, 6, .18, .50, .50, "END:.5,SPR:.3,JMP:.2", None, "1500m,100m", "A"),
    _i("of_stab", "Стабилизация и баланс: одна нога, фитбол", "ОФП", "ОФП", 3, 8, None, "", False, 14, 3, .08, .06, .12, "JMP:.3,THR:.2", None, "", "ABC"),
    _i("of_rope_climb", "Лазание по канату/шведской стенке", "ОФП", "ОФП", 4, 2, None, "", False, 15, 5, .20, .15, .22, "THR:.6,JMP:.4", None, "PV,JT", "AB"),
    # ── профилактика и мобильность (ещё) ──
    _i("pre_ham", "Профилактика бицепса бедра: скольжение, нордик-лайт, мост на одной", "Восстановление", "Восстановление", 3, 8, None, "", False, 12, 4, .05, .06, .20, "SPR:.3", "mobility", "100m,110mH", "ABC"),
    _i("pre_ankle", "Профилактика голеностопа и ахилла (изометрия, подъёмы)", "Восстановление", "Восстановление", 3, 12, None, "", False, 10, 3, .03, .04, .15, "JMP:.2", "mobility", "110mH,LJ,HJ", "ABC"),
    _i("pre_shoulder", "Профилактика плеча/локтя: ротаторы, лопатки", "Восстановление", "Восстановление", 3, 15, None, "", False, 10, 3, .03, .04, .10, "THR:.2", "mobility", "JT,PV", "ABC"),
    _i("pre_hip", "Мобильность таза и бедра, 10 упражнений", "Восстановление", "Восстановление", 1, 1, 15, "мин", True, 15, 2, .02, .02, .03, "", "mobility", "110mH,LJ", "ABC"),
    _i("rc_sauna", "Баня/сауна 2–3 захода", "Восстановление", "Восстановление", 1, 1, 30, "мин", True, 30, 1, .0, .02, .0, "", None, "", "ABC"),
    _i("rc_massage", "Массаж (спортивный)", "Восстановление", "Восстановление", 1, 1, 45, "мин", True, 45, 1, .0, .0, .0, "", None, "", "ABC"),
    _i("rc_walk", "Прогулка 30–40 мин", "Восстановление", "Восстановление", 1, 1, 35, "мин", True, 35, 1, .0, .06, .02, "", None, "", "ABC"),
    # ── контроль и тесты (ещё) ──
    _i("ts_sj", "Тест: прыжок в длину с места", "Контроль", "Контроль", 3, 3, None, "", False, 12, 6, .62, .04, .45, "JMP:1", None, "LJ", "ABC", "Нормативы Нариманова"),
    _i("ts_triple", "Тест: тройной прыжок с места", "Контроль", "Контроль", 3, 2, None, "", False, 12, 7, .70, .05, .65, "JMP:1", "plyo", "LJ,100m", "ABC", "Тест Нариманова"),
    _i("ts_5hop", "Тест: пятерной прыжок с места", "Контроль", "Контроль", 2, 2, None, "", False, 12, 7, .65, .08, .75, "JMP:1", "plyo", "LJ", "ABC"),
    _i("ts_10hop", "Тест: десятерной прыжок с места", "Контроль", "Контроль", 2, 1, None, "", False, 10, 7, .65, .12, .78, "JMP:1", "plyo", "LJ", "ABC"),
    _i("ts_sp_back", "Тест: ядро 7,26 кг назад через голову", "Контроль", "Контроль", 3, 3, None, "", False, 12, 6, .78, .05, .35, "THR:1", "SP", "SP,DT,JT", "ABC", "Тест Нариманова"),
    _i("ts_sp_fwd", "Тест: ядро 7,26 кг вперёд двумя руками", "Контроль", "Контроль", 3, 3, None, "", False, 12, 6, .75, .05, .35, "THR:1", "SP", "SP", "ABC"),
    _i("ts_dt_stand", "Тест: диск с места", "Контроль", "Контроль", 3, 3, None, "", False, 15, 6, .65, .06, .30, "THR:1", "DT", "DT", "ABC"),
    _i("ts_jt_ball", "Тест: метание мяча 150 г / копья с места", "Контроль", "Контроль", 3, 3, None, "", False, 15, 6, .70, .06, .30, "THR:1", "JT", "JT", "ABC"),
    _i("ts_30start", "Тест: 30 м со старта", "Контроль", "Контроль", 3, 1, 30, "м", True, 14, 8, .90, .10, .28, "SPR:1", "sprint", "100m", "ABC"),
    _i("ts_150", "Тест: 150 м", "Контроль", "Контроль", 2, 1, 150, "м", True, 18, 9, .70, .60, .40, "SE:.6,SPR:.4", "speed150", "100m,400m", "BC"),
    _i("ts_300", "Тест: 300 м", "Контроль", "Контроль", 1, 1, 300, "м", True, 12, 10, .60, .95, .45, "SE:.85,END:.15", "speedend", "400m", "C"),
    _i("ts_1500", "Тест: 1500 м", "Контроль", "Контроль", 1, 1, 1500, "м", True, 10, 10, .22, 1.0, .28, "END:.7,SE:.3", "speedend", "1500m", "C"),
    _i("ts_hj_standing", "Тест: высота с 3 шагов (максимум)", "Контроль", "Контроль", 1, 8, None, "", False, 25, 8, .70, .08, .28, "JMP:1", "HJ", "HJ", "BC", "", True),
    _i("ts_pv_short", "Тест: шест с 6 шагов (максимальная высота)", "Контроль", "Контроль", 1, 8, None, "", False, 35, 8, .80, .12, .30, "JMP:.7,SPR:.3", "PV", "PV", "BC", "", True),
]
CAT = {c.id: c for c in CATALOG}
assert len(CAT) == len(CATALOG), "дублирующиеся id каталога"
for _c_ in CATALOG:
    assert _c_.group is None or _c_.group in LOG_GROUPS, (_c_.id, _c_.group)


# ───────────── результаты на тренировке: секунды, метры, высота ─────────────
RES_OVERRIDE: Dict[str, Optional[Tuple[str, str]]] = {
    "lj_run": None, "hj_curve": None, "pv_gym": None, "pv_run": None, "pv_swing": None, "pv_inv": None, "pv_rope": None, "pv_hangup": None,
    "dt_drill": None, "dt_wall": None, "jt_gym": None, "jt_run": None, "pl_box": None, "pl_rope": None, "pl_stairs": None, "pl_hop": ("time", "с"),
    "pl_cmj": ("height", "см"), "ts_cmj": ("height", "см"), "pl_reach": ("height", "см"), "pl_med_over": ("dist", "м"), "pl_med_rot": ("dist", "м"),
    "pl_med_back": ("dist", "м"), "pl_medball": ("dist", "м"), "pl_scoop": ("dist", "м"), "pl_medjump": ("dist", "м"), "dt_med": ("dist", "м"),
    "jt_med": ("dist", "м"), "jt_pullover": ("dist", "м"), "jt_ball": ("dist", "м"), "sp_chest": ("dist", "м"), "sp_back": ("dist", "м"),
    "lj_stand": ("dist", "м"), "lj_triple_st": ("dist", "м"), "lj_five": ("dist", "м"), "lj_sl": ("dist", "м"), "lj_ten": ("dist", "м"),
    "lj_triple_run": ("dist", "м"), "pl_standing5": ("dist", "м"), "ts_sj": ("dist", "м"), "ts_triple": ("dist", "м"), "ts_5hop": ("dist", "м"),
    "ts_10hop": ("dist", "м"), "ts_shot": ("dist", "м"), "ts_sp_back": ("dist", "м"), "ts_sp_fwd": ("dist", "м"), "ts_dt_stand": ("dist", "м"),
    "ts_jt_ball": ("dist", "м"), "pl_jump_none": None, "h_drills": None, "h_1h": None, "h_trail": None, "h_lead": None, "h_narrow": None, "h_high": None,
    "h_low": None, "h_rhythm": None, "h_plyo": None, "h_sled_hop": None, "wu_strides": ("time", "с"), "st_farmer": None, "st_sledpush": ("time", "с"),
    "sp_wicket": ("time", "с"), "sp_stairs": ("time", "с"), "en_run_walk": None, "en_easy": None, "en_bike": None, "en_pool": None, "en_row": None,
    "en_cross": ("time", "с"), "en_fartlek": ("time", "с"), "en_tempo": ("time", "с"), "en_tempo_short": ("time", "с"),
}
RES_LABEL = {"time": "время", "dist": "дальность", "height": "высота"}
RES_HINT = {"time": "секунды на повтор через пробел: 3.82 3.85; можно 1:02.4",
            "dist": "метры по попыткам через пробел: 6.45 6.52", "height": "высота по попыткам через пробел"}
# предел потери результата внутри занятия (best → last), % — ПРАКТИЧЕСКОЕ правило тренеров, пороги мои
DROP_LIMIT = {"Спринт": 3.0, "Барьеры": 3.0, "Контроль": 4.0, "Скоростная выносливость": 6.0, "Выносливость": 6.0}
DROP_DEFAULT = 8.0


def res_spec(c: "CatItem") -> Optional[Tuple[str, str]]:
    """(тип результата, единица) для позиции каталога или None, если результат не измеряется."""
    if c.id in RES_OVERRIDE:
        return RES_OVERRIDE[c.id]
    if c.cat in ("Спринт", "Скоростная выносливость") and c.size_unit == "м":
        return ("time", "с")
    if c.cat == "Выносливость" and c.size_unit in ("м", "км"):
        return ("time", "с")
    if c.cat == "Барьеры" and c.kind in ("Скорость", "Контроль"):
        return ("time", "с")
    if c.cat == "Контроль" and c.size_unit == "м":
        return ("time", "с")
    if c.cat in ("Ядро", "Диск", "Копьё", "Прыжок в длину"):
        return ("dist", "м")
    if c.cat in ("Прыжок в высоту", "Шест"):
        return ("height", "м")
    return None


def parse_results(text: str) -> List[float]:
    """«3.82 3,85; 1:02.4» → [3.82, 3.85, 62.4]. Пробел/;/новая строка — разделитель; запятая — десятичная; m:ss — минуты и секунды."""
    out = []
    for tok in re.split(r"[;\s]+", str(text or "").strip()):
        tok = tok.strip().replace(",", ".")
        if not tok:
            continue
        try:
            if ":" in tok:
                parts = [float(x) for x in tok.split(":")]
                v = 0.0
                for x in parts:
                    v = v * 60.0 + x
                out.append(v)
            else:
                out.append(float(tok))
        except ValueError:
            continue
    return out


def fmt_res(kind: str, v: float) -> str:
    if kind == "time" and v >= 60:
        return f"{int(v // 60)}:{v % 60:04.1f}"
    return f"{v:.2f}"


def res_key(c: "CatItem", it: dict) -> str:
    """Ключ для сравнения результатов: время зависит от дистанции (30 м ≠ 40 м), остальное — от упражнения."""
    sp = res_spec(c)
    k = c.id
    if sp and sp[0] == "time" and c.size:
        k = f"{c.id}@{float(it.get('size') or c.size):g}"
    if IMPL_STD.get(c.id) and it.get("impl"):
        k += f"#{float(it['impl']):g}кг"
    return k


DIST_MAX = {"lj_stand": 4.2, "ts_sj": 4.2, "lj_sl": 3.6, "lj_triple_st": 11.5, "ts_triple": 11.5, "lj_five": 17.0, "pl_standing5": 17.0, "ts_5hop": 17.0,
            "lj_ten": 32.0, "ts_10hop": 32.0, "lj_triple_run": 14.0}
CAT_DIST_MAX = {"Прыжок в длину": 9.5, "Ядро": 30.0, "Диск": 80.0, "Копьё": 110.0, "Плиометрика и СФП": 30.0}


def _sane(c: "CatItem", it: dict, kind: str, unit: str, v: float) -> bool:
    if kind == "time":
        if c.size and c.size_unit in ("м", "км"):
            d = float(it.get("size") or c.size) * (1000.0 if c.size_unit == "км" else 1.0)
            sp = d / v if v > 0 else 0.0
            return 1.5 <= sp <= (12.5 if d <= 400 else 9.0)       # скорость м/с: рекорд мира в спринте ≈ 12,4
        return 0.3 <= v <= 3600
    if kind == "dist":
        return 0.2 <= v <= DIST_MAX.get(c.id, CAT_DIST_MAX.get(c.cat, 110.0))
    if unit == "см":
        return 5 <= v <= 150
    return 0.3 <= v <= 6.5


def _better(kind: str, a: float, b: float) -> bool:
    return a < b if kind == "time" else a > b


def result_history(log: Sequence[dict], before: Optional[str] = None) -> Dict[str, List[dict]]:
    """Журнал → ключ упражнения → [{date, best, mean, n, vals}] в порядке дат (только записи строго до `before`, если задано)."""
    h: Dict[str, List[dict]] = {}
    for e in sorted(log, key=lambda x: x["date"]):
        if before is not None and e["date"] >= before:
            continue
        for it in (e.get("items") or []):
            c = CAT.get(it.get("id"))
            sp = res_spec(c) if c else None
            vals = [float(v) for v in (it.get("res") or [])]
            if not sp or not vals:
                continue
            vals = [v for v in vals if _sane(c, it, sp[0], sp[1], v)]
            if not vals:
                continue
            best = min(vals) if sp[0] == "time" else max(vals)
            h.setdefault(res_key(c, it), []).append(dict(date=e["date"], best=best, mean=float(np.mean(vals)), n=len(vals), vals=vals, id=c.id, kind=sp[0], unit=sp[1],
                                                        size=float(it.get("size") or c.size or 0), impl=it.get("impl")))
    return h


def item_result_stats(c: "CatItem", it: dict, hist: Optional[Dict[str, List[dict]]] = None) -> Optional[dict]:
    """Статистика результатов по позиции: лучший/средний/последний, потеря внутри занятия, сравнение с личным лучшим и прошлым разом,
    интенсивность (доля от личного лучшего) и множитель дозы m. Пороги и формула m — мои допущения."""
    sp = res_spec(c)
    raw = [float(v) for v in (it.get("res") or [])]
    if not sp or not raw:
        return None
    kind, unit = sp
    vals = [v for v in raw if _sane(c, it, kind, unit, v)]
    bad = len(raw) - len(vals)
    if not vals:
        return dict(kind=kind, unit=unit, n=0, bad=bad, flags=["Значения вне физически возможного диапазона — проверьте ввод (единицы: " + unit + ")."], m=1.0)
    best = min(vals) if kind == "time" else max(vals)
    worst = max(vals) if kind == "time" else min(vals)
    mean = float(np.mean(vals))
    last = vals[-1]
    loss = (last - best) / best * 100.0 if kind == "time" else (best - last) / best * 100.0
    flags: List[str] = []
    if bad:
        flags.append(f"{bad} знач. вне допустимого диапазона проигнорировано.")
    key = res_key(c, it)
    prev = (hist or {}).get(key, [])
    pb = None
    if prev:
        pb = min(p["best"] for p in prev) if kind == "time" else max(p["best"] for p in prev)
    lastsess = prev[-1] if prev else None
    out = dict(kind=kind, unit=unit, n=len(vals), bad=bad, best=best, mean=mean, worst=worst, last=last, loss=loss, pb=pb, vals=vals,
               prev_best=(lastsess["best"] if lastsess else None), prev_date=(lastsess["date"] if lastsess else None), m=1.0, ratio=None)
    lim = DROP_LIMIT.get(c.cat, DROP_DEFAULT)
    if len(vals) >= 3 and loss > lim:
        flags.append(f"«{c.name}»: к последней попытке результат ухудшился на {loss:.1f} % от лучшего (порог {lim:g} %): дальше нервная работа теряет смысл "
                     f"(практическое правило, порог мой).")
    if kind == "time" and c.size and c.size_unit in ("м", "км"):
        d = float(it.get("size") or c.size) * (1000.0 if c.size_unit == "км" else 1.0)
        out["speed"] = d / best
    if pb is not None:
        ratio = (pb / mean) if kind == "time" else (mean / pb)
        out["ratio"] = ratio
        out["vs_pb"] = (pb - best) / pb * 100.0 if kind == "time" else (best - pb) / pb * 100.0     # + = лучше личного лучшего
        if _better(kind, best, pb):
            flags.append(f"«{c.name}»: новый лучший результат за все записи ({fmt_res(kind, best)} {unit} против {fmt_res(kind, pb)}).")
        elif c.kind in ("Скорость", "Контроль") and ratio < 0.90:
            flags.append(f"«{c.name}»: в среднем {100 * ratio:.0f} % от вашего лучшего — это не максимальная интенсивность; стимул для скорости слабый.")
        # множитель дозы: 1 при 97 % от лучшего; ±2 за 1 «единицу» отклонения, в пределах 0.75–1.15 (мой допуск; не для восстановительных)
        if c.kind != "Восстановление":
            out["m"] = float(np.clip(1.0 + 2.0 * (ratio - 0.97), 0.75, 1.15))
    else:
        flags.append(f"«{c.name}»: раньше этого упражнения с результатами не было — записал как стартовую точку.")
    out["flags"] = flags
    return out


def pb_table(log: Sequence[dict]) -> pd.DataFrame:
    """Таблица личных лучших по упражнениям (по журналу): лучший результат, дата, число занятий, последний результат."""
    h = result_history(log)
    rows = []
    for k, lst in h.items():
        kind, unit = lst[-1]["kind"], lst[-1]["unit"]
        bst = min(lst, key=lambda p: p["best"]) if kind == "time" else max(lst, key=lambda p: p["best"])
        c = CAT[lst[-1]["id"]]
        nm = c.name + (f" ({lst[-1]['size']:g} {c.size_unit})" if kind == "time" and c.size else "") + (f", снаряд {lst[-1]['impl']:g} кг" if lst[-1].get("impl") else "")
        rows.append(dict(key=k, Упражнение=nm, Тип=RES_LABEL[kind], Единица=unit, Лучший=round(bst["best"], 2), Дата=bst["date"], Занятий=len(lst),
                         Последний=round(lst[-1]["best"], 2), Был=lst[-1]["date"]))
    return pd.DataFrame(rows)



# ───────────── форма по результатам тренировок: тренды → виды → план/прогноз ─────────────
FORM_MIN_SESS, FORM_WINDOW, FORM_FLAT = 3, 56, 0.8     # мин. занятий, окно (дн.), порог «значимого» изменения за 4 недели (%) — мои допущения
FORM_KINDS = ("Скорость", "Контроль", "СФП", "Техника")


def result_trends(log: Sequence[dict], today: Optional[pd.Timestamp] = None, window: int = FORM_WINDOW) -> pd.DataFrame:
    """Тренд лучшего результата по каждому упражнению за последние `window` дней: наклон Тейла–Сена, в % за 4 недели (плюс = лучше: для времени меньше).
    Нужно ≥ 3 разных дат с результатами в окне и разброс ≥ 10 дней. Порог «рост/падение» — мой (0,8 % за 4 недели и не меньше половины шума)."""
    today = pd.Timestamp(today if today is not None else pd.Timestamp.today()).normalize()
    lo = today - pd.Timedelta(days=window)
    rows = []
    for k, lst in result_history(log).items():
        pts = [(pd.Timestamp(p["date"]), p["best"]) for p in lst if lo <= pd.Timestamp(p["date"]) <= today]
        kind, unit = lst[-1]["kind"], lst[-1]["unit"]
        c = CAT[lst[-1]["id"]]
        if len({d for d, _ in pts}) < FORM_MIN_SESS or (pts[-1][0] - pts[0][0]).days < 10:
            continue
        x = np.array([(d - pts[0][0]).days for d, _ in pts], float)
        y = np.array([v for _, v in pts], float)
        sign = -1.0 if kind == "time" else 1.0
        yp = sign * (y / y.mean() - 1.0) * 100.0                       # % относительно среднего, плюс = лучше
        sl = [(yp[j] - yp[i]) / (x[j] - x[i]) for i in range(len(x)) for j in range(i + 1, len(x)) if x[j] > x[i]]
        if not sl:
            continue
        b = float(np.median(sl))
        a = float(np.median(yp - b * x))
        noise = float(1.4826 * np.median(np.abs(yp - (a + b * x)))) if len(x) >= 4 else float(np.std(yp))
        slope4 = b * 28.0
        st_ = "растёт" if slope4 >= FORM_FLAT and slope4 >= 0.5 * noise else ("падает" if slope4 <= -FORM_FLAT and -slope4 >= 0.5 * noise else "без изменений")
        nm = c.name + (f" ({lst[-1]['size']:g} {c.size_unit})" if kind == "time" and c.size else "") + (f", снаряд {lst[-1]['impl']:g} кг" if lst[-1].get("impl") else "")
        rows.append(dict(key=k, id=c.id, Упражнение=nm, Занятий=len(pts), С=str(pts[0][0].date()), По=str(pts[-1][0].date()), Единица=unit,
                         **{"Тренд, % за 4 нед.": round(slope4, 2), "Шум, %": round(noise, 2)}, Статус=st_, tags=c.tags, kind_cat=c.kind))
    return pd.DataFrame(rows)


def event_form(tr: pd.DataFrame) -> pd.DataFrame:
    """Сводка по видам: среднее направление результатов упражнений, помогающих виду (разминка и ОФП-упражнения не считаются)."""
    rows = []
    for e in EVENTS:
        sub = tr[[e in t and kc in FORM_KINDS for t, kc in zip(tr["tags"], tr["kind_cat"])]] if len(tr) else tr
        if not len(sub):
            rows.append(dict(event=e, Вид=RU_EV.get(e, e), Упражнений=0, **{"Тренд, % за 4 нед.": None}, Статус="нет данных", Упражнения=""))
            continue
        m = float(sub["Тренд, % за 4 нед."].mean())
        st_ = "растёт" if m >= FORM_FLAT else ("падает" if m <= -FORM_FLAT else "без изменений")
        rows.append(dict(event=e, Вид=RU_EV.get(e, e), Упражнений=len(sub), **{"Тренд, % за 4 нед.": round(m, 2)}, Статус=st_, Упражнения="; ".join(sub["Упражнение"])))
    return pd.DataFrame(rows)


def form_emph(ef: pd.DataFrame) -> Dict[str, float]:
    """МОЁ правило: средство, питающее вид, у которого результаты тренировок падают (×1,10) или стоят (×1,05), получает больше объёма; растущие — без изменений."""
    st = dict(zip(ef["event"], ef["Статус"]))
    out = {}
    for g, es in GROUP_EVENTS.items():
        ss = [st.get(e, "нет данных") for e in es if st.get(e, "нет данных") != "нет данных"]
        if not ss:
            continue
        out[g] = 1.10 if ss.count("падает") > len(ss) / 2 else (1.05 if "падает" in ss or ss.count("без изменений") > len(ss) / 2 else 1.0)
    return {g: v for g, v in out.items() if v != 1.0}


def result_adjust(res: Dict[str, float], ef: pd.DataFrame, k: float = 0.3, cap: float = 2.0) -> Dict[str, float]:
    """Необязательная поправка результата вида по тренду тренировочных результатов: сдвиг = k · тренд (за 4 недели), в пределах ±cap %.
    k=0,3 — МОЁ допущение (перенос тренировочного прироста на соревновательный результат неполный); достоверно это не измерено."""
    out = dict(res)
    for _, r in ef.iterrows():
        t = r["Тренд, % за 4 нед."]
        if t is None or pd.isna(t) or r["event"] not in out:
            continue
        pct = float(np.clip(k * t, -cap, cap))
        out[r["event"]] = out[r["event"]] * (1 - pct / 100.0 if IS_TIME[r["event"]] else 1 + pct / 100.0)
    return out


# ───────────── части тренировки, вес снаряда, готовность (сон/боль), подбор разминки ─────────────
PARTS = ("Разминка", "Основная часть", "Заминка")
IMPL_STD: Dict[str, float] = {}                                  # вес снаряда по умолчанию, кг (по id)
for _c_ in CATALOG:
    if _c_.cat == "Ядро":
        IMPL_STD[_c_.id] = 8.5 if _c_.id == "sp_heavy" else (5.5 if _c_.id == "sp_light" else 7.26)
    elif _c_.cat == "Диск":
        IMPL_STD[_c_.id] = {"dt_heavy": 2.5, "dt_light": 1.5, "dt_med": 3.0, "dt_wall": 2.0}.get(_c_.id, 2.0)
    elif _c_.cat == "Копьё":
        IMPL_STD[_c_.id] = {"jt_heavy": 1.0, "jt_ball": 0.7, "jt_med": 2.0, "jt_pullover": 3.0, "jt_gym": 1.0}.get(_c_.id, 0.8)
IMPL_STD.update({"pl_med_over": 4.0, "pl_med_rot": 4.0, "pl_med_back": 4.0, "pl_medball": 4.0, "pl_scoop": 4.0, "pl_medjump": 3.0, "sp_back": 7.26, "ts_shot": 7.26,
                 "ts_sp_back": 7.26, "ts_sp_fwd": 7.26, "ts_dt_stand": 2.0, "ts_jt_ball": 0.8, "sp_chest": 7.26})


def part_of(c: "CatItem") -> int:
    return 0 if c.cat == "Разминка" else (2 if c.cat == "Восстановление" else 1)


def impl_factor(c: "CatItem", it: dict) -> float:
    """Множитель дозы от веса снаряда относительно стандартного: √(вес/стандарт) в пределах 0.6–1.5 (моё допущение: тяжелее — выше механика и нагрузка)."""
    std = IMPL_STD.get(c.id)
    w = it.get("impl")
    if not std or not w:
        return 1.0
    return float(np.sqrt(np.clip(float(w) / std, 0.6, 1.5)))


def sort_parts(items: List[dict]) -> List[dict]:
    """Разминка → основная часть → заминка (порядок внутри части сохраняется)."""
    return sorted(items, key=lambda it: part_of(CAT[it["id"]]) if it.get("id") in CAT else 1)


def readiness_factor(sleep: Optional[float], sore: Optional[float]) -> float:
    """Допуск нагрузки дня по самочувствию (МОЁ правило): 1,0 при сне ≥ 7/10 и боли ≤ 3/10; −5 % за каждый балл сна ниже 7 и −4 % за балл боли выше 3; не ниже 0,6."""
    f = 1.0
    if sleep:
        f -= 0.05 * max(0.0, 7.0 - float(sleep))
    if sore:
        f -= 0.04 * max(0.0, float(sore) - 3.0)
    return float(np.clip(f, 0.6, 1.0))


def _dflt(cid: str, size_factor: float = 1.0, sets: Optional[int] = None) -> dict:
    c = CAT[cid]
    return dict(id=cid, sets=(sets or c.sets), reps=c.reps, size=(c.size * size_factor if c.size else None))


def suggest_warmup(items: Sequence[dict]) -> List[dict]:
    """Разминка под состав основной части (МОЁ правило): общая часть 12 мин бега + мобильность + активация, затем специальная под то, что стоит в плане."""
    main = [CAT[i["id"]] for i in items if i.get("id") in CAT and part_of(CAT[i["id"]]) == 1]
    if not main:
        return []
    out = [_dflt("wu_jog", 0.67), _dflt("wu_dyn", 0.8)]
    cats = {c.cat for c in main}
    hi = [c for c in main if c.stress[0] >= 0.7]
    if cats & {"Спринт", "Скоростная выносливость", "Барьеры", "Прыжок в длину"} or any(c.id.startswith(("lj_", "pl_")) for c in main):
        out += [_dflt("wu_abc", sets=2), _dflt("wu_strides", sets=3)]
    if "Барьеры" in cats:
        out += [_dflt("wu_hurdle_mob"), _dflt("wu_minihurd")]
    if cats & {"Прыжок в длину", "Прыжок в высоту", "Шест", "Плиометрика и СФП"}:
        out.append(_dflt("wu_skips"))
    if "Шест" in cats:
        out.append(_dflt("wu_pole"))
    if cats & {"Ядро", "Диск", "Копьё"} or any(c.id.startswith("pl_med") for c in main):
        out.append(_dflt("wu_throw"))
    if not hi:
        out = [x for x in out if x["id"] not in ("wu_strides",)]
    seen, res = set(), []
    for x in out:
        if x["id"] not in seen:
            seen.add(x["id"])
            res.append(x)
    return res


def suggest_cooldown(items: Sequence[dict]) -> List[dict]:
    mins = sum(CAT[i["id"]].minutes for i in items if i.get("id") in CAT)
    return [_dflt("rc_cool")] + ([_dflt("rc_stretch")] if mins > 60 else [])


def _register_catalog() -> SessionLibrary:
    """Каждая запись каталога становится упражнением и сессией стандартной дозы; расширенная библиотека = базовая + каталог."""
    sess = []
    for c in CATALOG:
        EXERCISES[c.id] = Exercise(c.id, c.name, StressVector(*c.stress), c.chan)
        lv = StressVector(*c.stress).levels()
        sess.append(Session(c.id, c.name, Category.RECOVERY if c.kind == "Восстановление" else Category.TECHNIQUE, ((c.id, 1.0),), c.minutes, c.rpe, lv, c.coord, c.note))
    return SessionLibrary(list(_SESSIONS) + sess)


EXT_LIBRARY = _register_catalog()
LEVEL_RU = {"LOW": "низк.", "MID": "средн.", "HIGH": "высок.", "EXTREME": "экстр."}

# соответствие группы плана и записи каталога по умолчанию («заполнить по плану»)
PLAN_DEFAULT_ITEM = {"sprint": "sp_fly30", "speed150": "se_150", "speedend": "se_300", "endurance": "en_cross", "hurdles": "h_rhythm", "LJ": "lj_short",
                     "HJ": "hj_short", "PV": "pv_short", "SP": "sp_glide", "DT": "dt_turn", "JT": "jt_ball", "strength": "st_squat_v"}


def items_from_plan(plan_vol: Dict[str, float]) -> List[dict]:
    """План дня (группа → объём в единицах журнала) → список записей каталога с дозой, дающей этот объём."""
    out = []
    for g, v in plan_vol.items():
        c = CAT.get(PLAN_DEFAULT_ITEM.get(g, ""))
        if not c or v <= 0:
            continue
        per = c.log_volume(1.0)
        phi = v / per if per > 0 else 1.0
        sets, reps, size = c.sets, c.reps, c.size
        if c.group == "endurance" and c.size:                      # длина кросса/темпа — через дистанцию
            size = max(1.0, round(c.size * phi, 1))
        else:                                                      # остальное — через число серий, затем повторов (размер не трогаем)
            smax = max(3 * c.sets, 3)
            sets = int(min(smax, max(1, round(c.sets * phi))))
            reps = int(min(c.reps + 3, max(1, round(c.reps * phi * c.sets / sets))))
        out.append(dict(id=c.id, sets=sets, reps=reps, size=size))
    return out


def item_dose(it: dict) -> Tuple[CatItem, float]:
    c = CAT[it["id"]]
    return c, c.phi(it.get("sets"), it.get("reps"), it.get("size"))


def analyze_day(items: Sequence[dict], state: Optional["SolverState"], model: "PerfModel", week_channels: Optional[np.ndarray] = None,
                week_au_ref: Optional[float] = None, rpe_override: Optional[float] = None, hour: Optional[float] = None, stage: str = "A",
                weak: Optional[Dict[str, float]] = None, gate_cfg: GateConfig = GateConfig(), history: Optional[Sequence[dict]] = None,
                date: Optional[str] = None, form: Optional[Dict[str, str]] = None, readiness: Optional[dict] = None, mass: Optional[float] = None) -> dict:
    """Анализ собранной тренировки: время, нагрузка (AU), дозы пулов, баланс каналов, проверки шлюзов, конфликты и подсказки (все пороги — мои)."""
    rows, tot_min, tot_au = [], 0.0, 0.0
    dose = np.zeros(3)
    chan = np.zeros(R)
    items = sort_parts([dict(i) for i in items])
    hist = result_history(history or [], before=date) if history is not None else {}
    res_rows, res_flags, load_rows, load_flags = [], [], [], []
    lifts = {}
    for e_ in sorted([e for e in (history or []) if date is None or e["date"] < date], key=lambda x: x["date"]):
        for j_ in (e_.get("items") or []):
            lifts[j_.get("id")] = dict(date=e_["date"], **{k: j_.get(k) for k in ("sets", "reps", "size")})
    maxw = {}
    for e_ in (history or []):
        for j_ in (e_.get("items") or []):
            if j_.get("size") and CAT.get(j_.get("id")) and CAT[j_["id"]].size_unit == "кг":
                maxw[j_["id"]] = max(maxw.get(j_["id"], 0.0), float(j_["size"]))
    for it in items:
        c, phi = item_dose(it)
        minutes = c.minutes * phi
        stt = item_result_stats(c, it, hist)
        m_ = (stt["m"] if stt else 1.0) * impl_factor(c, it)
        if c.size_unit == "кг" and c.scale_size and it.get("size"):
            w_ = float(it["size"])
            tn = float(it.get("sets", c.sets)) * float(it.get("reps", c.reps)) * w_
            lr = {"Упражнение": c.name, "Вес, кг": w_, "Серий×повт.": f"{it.get('sets', c.sets)}×{it.get('reps', c.reps)}", "Тоннаж, кг": round(tn),
                  "× массы тела": (round(w_ / mass, 2) if mass else None), "% от макс. в журнале": (round(100 * w_ / maxw[c.id]) if maxw.get(c.id) else None)}
            pl = lifts.get(c.id)
            if pl and pl.get("size"):
                tn0 = float(pl.get("sets") or c.sets) * float(pl.get("reps") or c.reps) * float(pl["size"])
                lr["Вес к прошлому, %"] = round(100 * (w_ / float(pl["size"]) - 1), 1)
                lr["Тоннаж к прошлому, %"] = round(100 * (tn / tn0 - 1), 1)
                if w_ > 1.10 * float(pl["size"]):
                    load_flags.append(f"«{c.name}»: вес +{100 * (w_ / float(pl['size']) - 1):.0f} % к прошлому разу ({pl['size']:g} → {w_:g} кг): резкий скачок (практическое правило: до ~10 % за шаг).")
                if tn > 1.25 * tn0:
                    load_flags.append(f"«{c.name}»: тоннаж +{100 * (tn / tn0 - 1):.0f} % к прошлому разу — выше +25 %.")
            load_rows.append(lr)
        if it.get("impl") and IMPL_STD.get(c.id):
            load_rows.append({"Упражнение": f"{c.name} — снаряд", "Вес, кг": float(it["impl"]), "Серий×повт.": f"{it.get('sets', c.sets)}×{it.get('reps', c.reps)}",
                              "Тоннаж, кг": None, "× массы тела": None, "% от макс. в журнале": None, "Вес к прошлому, %": round(100 * (float(it["impl"]) / IMPL_STD[c.id] - 1), 1)})
        au = minutes * (rpe_override or c.rpe) * m_
        v = np.array(c.stress) * phi * m_
        if stt:
            res_flags += stt.get("flags", [])
            if stt.get("n"):
                res_rows.append({"Упражнение": c.name, "Попыток": stt["n"], "Лучший": fmt_res(stt["kind"], stt["best"]), "Средний": fmt_res(stt["kind"], stt["mean"]),
                                 "Последний": fmt_res(stt["kind"], stt["last"]), "Ед.": stt["unit"], "Потеря к последней, %": round(stt["loss"], 1),
                                 "Скорость, м/с": (round(stt["speed"], 2) if stt.get("speed") else None),
                                 "Личный лучший": (fmt_res(stt["kind"], stt["pb"]) if stt["pb"] is not None else "—"),
                                 "К лучшему, %": (round(stt["vs_pb"], 1) if stt.get("vs_pb") is not None else None),
                                 "Интенсивность": (f"{100 * stt['ratio']:.0f} %" if stt.get("ratio") else "—"), "Множитель дозы": round(m_, 2)})
        dose += v
        chan += (au / AU_REF) * np.array([dict(c.chan).get(k, 0.0) for k in CHANNELS]) / CHAN_SCALE
        tot_min += minutes
        tot_au += au
        rows.append(dict(id=c.id, name=c.name, cat=c.cat, kind=c.kind, dose=f"{it.get('sets', c.sets)}×{it.get('reps', c.reps)}" + (f" по {it.get('size', c.size):g} {c.size_unit}" if c.size else ""),
                         phi=round(phi, 2), minutes=round(minutes, 1), AU=round(au), cns=round(v[0], 2), met=round(v[1], 2), mech=round(v[2], 2), stress=c.stress, coord=c.coord, tags=c.tags))
    out: dict = dict(items=pd.DataFrame(rows), minutes=tot_min, AU=tot_au, dose=dose, chan=chan, results=pd.DataFrame(res_rows), result_notes=res_flags,
                     loads=pd.DataFrame(load_rows), load_notes=load_flags, ordered=items)
    warn: List[str] = []
    if not rows:
        out["warnings"] = ["Тренировка пуста."]
        out["suggest"] = []
        return out
    ids = [r["id"] for r in rows]
    # готовность (сон/боль) снижает порог ЦНС для шлюза и допуск нагрузки (мои правила)
    rf = readiness_factor((readiness or {}).get("sleep"), (readiness or {}).get("sore")) if readiness else 1.0
    out["readiness_factor"] = rf
    if rf < 1.0:
        base_thr = BioGates(gate_cfg, model.const).cns_thr
        gate_cfg = dataclasses.replace(gate_cfg, cns_thr_override=float(base_thr * rf))
    # шлюзы от состояния
    gates = BioGates(gate_cfg, model.const)
    if state is not None:
        h = hour if hour is not None else state.t0
        verdicts = []
        for r in rows:
            v = gates.check(state.ast, EXT_LIBRARY[r["id"]], h)
            verdicts.append("" if v.allowed else v.text())
        out["items"]["gate"] = verdicts
        pools_now = state.ast.pools_at(h).P
    else:
        out["items"]["gate"] = [""] * len(rows)
        pools_now = np.zeros(3)
    out["pools_now"] = pools_now
    out["pools_after"] = pools_now * model.const.decay_factor(0.0) + dose
    hi_cns = [r for r in rows if r["stress"][0] >= 0.8]
    hi_mech = [r for r in rows if r["stress"][2] >= 0.8]
    kinds = {r["kind"] for r in rows}
    cats = {r["cat"] for r in rows}
    tagset = set()
    for r in rows:
        tagset |= set(r["tags"])
    out["events_touched"] = sorted(tagset - {""})
    if len(hi_cns) > 3:
        warn.append(f"{len(hi_cns)} позиций с высокой нагрузкой на ЦНС (≥ 0,8): для одной тренировки много, обычно 2–3 «нервных» блока (моё правило).")
    if len(hi_mech) >= 2:
        warn.append("Две и более позиции с высокой механикой (≥ 0,8): удар по сухожилиям и связкам; оставьте одну.")
    if any(r["cat"] == "Силовая" and r["stress"][2] >= 0.8 for r in rows) and any(r["cat"] in ("Спринт", "Барьеры") and r["stress"][0] >= 0.8 for r in rows):
        warn.append("Тяжёлая силовая и максимальная скорость в один день: скорость лучше утром, силовую после, либо разнести по дням.")
    if any(r["cat"] == "Выносливость" and r["kind"] != "Восстановление" and r["stress"][1] >= 0.8 for r in rows) and any(r["stress"][0] >= 0.8 for r in rows):
        warn.append("Тяжёлые интервалы выносливости вместе с «нервными» блоками: метаболическое утомление испортит скорость и технику. Разнесите.")
    if any(r["coord"] for r in rows) and any(r["stress"][1] >= 0.8 for r in rows):
        warn.append("Тонкая техника (шест/высота/барьеры) и лактатная работа в один день: технику ставьте первой или в другой день.")
    mainr = [r for r in rows if r["cat"] not in ("Разминка", "Восстановление")]
    order_bad = [i for i in range(1, len(mainr)) if mainr[i]["stress"][0] > mainr[i - 1]["stress"][0] + 0.3]
    wu = [r for r in rows if r["cat"] == "Разминка"]
    wu_min = float(sum(r["minutes"] for r in wu))
    wu_ids = {r["id"] for r in wu}
    out["warmup_min"] = wu_min
    need_wu = [r for r in mainr if r["stress"][0] >= 0.5 or r["stress"][2] >= 0.6]
    if need_wu and not wu:
        warn.append("Нет разминки, а в основной части есть скоростные/прыжковые/силовые блоки: добавьте её (кнопка «Подобрать разминку»).")
    elif need_wu and any(r["stress"][0] >= 0.7 for r in mainr) and wu_min < 15:
        warn.append(f"Разминка {wu_min:.0f} мин: перед максимальной скоростью, прыжками и метаниями нужно не меньше 15 мин (моё правило).")
    if wu:
        if any(r["cat"] in ("Спринт", "Барьеры", "Скоростная выносливость") and r["stress"][0] >= 0.7 for r in mainr) and not (wu_ids & {"wu_strides", "wu_abc", "wu_minihurd"}):
            warn.append("В разминке нет ускорений/беговой азбуки перед максимальным бегом.")
        if any(r["cat"] in ("Ядро", "Диск", "Копьё") for r in mainr) and "wu_throw" not in wu_ids:
            warn.append("Перед метаниями нужна метательная разминка (резина, набивной, вращения).")
        if any(r["cat"] in ("Прыжок в высоту", "Шест", "Прыжок в длину", "Плиометрика и СФП") and r["stress"][0] >= 0.6 for r in mainr) and not (wu_ids & {"wu_skips", "wu_dyn", "wu_pole"}):
            warn.append("Перед прыжками нет прыжковой разминки/динамической мобильности.")
        if any(r["cat"] == "Барьеры" for r in mainr) and not (wu_ids & {"wu_hurdle_mob", "wu_minihurd"}):
            warn.append("Перед барьерами добавьте барьерную мобильность или мини-барьеры.")
    if rf < 1.0:
        hi_ = [r["name"] for r in mainr if r["stress"][0] >= 0.8]
        warn.append(f"Самочувствие (сон {readiness.get('sleep')}/10, боль {readiness.get('sore')}/10): допуск нагрузки {100 * rf:.0f} %, порог ЦНС для шлюза снижен ×{rf:.2f}"
                    + (f". Облегчите или уберите: {', '.join(hi_)}." if hi_ else ".") + " Коэффициенты мои.")
    if week_au_ref and tot_au > 0.28 * rf * week_au_ref and rf < 1.0:
        warn.append(f"При вашем самочувствии тяжёлым днём считается нагрузка выше {0.28 * rf * week_au_ref:.0f} AU — сейчас {tot_au:.0f}.")
    warn += load_flags
    if order_bad:
        warn.append("Порядок: блоки с самой высокой нагрузкой на ЦНС ставьте раньше (скорость и техника → метания → сила → выносливость → заминка).")
    if tot_min > 150:
        warn.append(f"Длительность {tot_min:.0f} мин: больше 2,5 часов — вторая тренировка дня лучше, чем одна длинная.")
    if week_au_ref and tot_au > 0.28 * week_au_ref:
        warn.append(f"Нагрузка дня {tot_au:.0f} AU — {100 * tot_au / week_au_ref:.0f} % типовой недели (≥ 28 % — тяжёлый день).")
    if not any(r["kind"] == "Восстановление" for r in rows) and tot_min > 90:
        warn.append("Нет заминки/мобильности: добавьте 10 минут.")
    blocked = out["items"][out["items"]["gate"] != ""]
    for _, r in blocked.iterrows():
        warn.append(f"Шлюз по «{r['name']}»: {r['gate']}.")
    out["warnings"] = warn or ["Конфликтов по правилам нет."]
    # подсказки: что добавить
    out["suggest"] = suggest_items(ids, state, model, week_channels, stage, weak, gate_cfg, hour, dose, form=form)
    return out


def suggest_items(chosen: Sequence[str], state, model, week_channels, stage: str, weak: Optional[Dict[str, float]], gate_cfg, hour, dose, top: int = 8,
                  form: Optional[Dict[str, str]] = None) -> List[dict]:
    """Что добавить: бонус за виды с большим запасом роста (weak: вид → очки запаса), за каналы, недобравшие за неделю; штрафы за шлюзы, чужую фазу, повтор категории."""
    gates = BioGates(gate_cfg, model.const)
    cats_chosen = [CAT[i].cat for i in chosen]
    wk = np.zeros(R) if week_channels is None else np.asarray(week_channels, float)
    need = np.full(R, 5.0) - wk                      # недельная цель по каналу ≈ 5 условных единиц (моё), минус сделанное
    wsum = sum((weak or {}).values()) or 1.0
    out = []
    for c in CATALOG:
        if c.id in chosen or c.kind in ("Восстановление",) or c.cat in ("Разминка",):
            continue
        if stage not in c.phases:
            continue
        if state is not None:
            v = gates.check(state.ast, EXT_LIBRARY[c.id], hour if hour is not None else state.t0)
            if not v.allowed:
                continue
        sc = 0.0
        why = []
        ev = [e for e in c.tags if e in (weak or {})]
        if ev:
            b = sum(weak[e] for e in ev) / wsum * 6.0
            sc += b
            why.append("помогает слабым видам: " + ", ".join(RU_EV.get(e, e) for e in ev))
        fv = [e for e in c.tags if (form or {}).get(e) in ("падает", "без изменений") and c.kind in FORM_KINDS]
        if fv:
            b2 = 2.0 if any(form[e] == "падает" for e in fv) else 1.0
            sc += b2
            why.append("результаты по виду " + ", ".join(RU_EV.get(e, e) for e in fv) + (" падают" if b2 == 2.0 else " стоят на месте"))
        cl = np.array([dict(c.chan).get(k, 0.0) for k in CHANNELS])
        d = float((np.maximum(need, 0) * cl).sum() / max(need.max(), 1e-6)) if need.max() > 0 else 0.0
        if d > 0.3:
            sc += d
            why.append("канал недобран за неделю")
        if cats_chosen.count(c.cat) >= 2:
            sc -= 1.5
        if c.stress[0] >= 0.8 and sum(1 for i in chosen if CAT[i].stress[0] >= 0.8) >= 2:
            sc -= 3
        if sc > 0.2:
            out.append(dict(id=c.id, name=c.name, score=round(sc, 2), why="; ".join(why) or "дополняет набор"))
    return sorted(out, key=lambda x: -x["score"])[:top]


RU_EV = {"100m": "100 м", "LJ": "длина", "SP": "ядро", "HJ": "высота", "400m": "400 м", "110mH": "110 м с/б", "DT": "диск", "PV": "шест", "JT": "копьё", "1500m": "1500 м"}


def week_channel_load(log: Sequence[dict], today: pd.Timestamp) -> np.ndarray:
    """Нагрузка по каналам за текущую неделю (с понедельника) по журналу — для подсказок «что недобрано»."""
    today = pd.Timestamp(today).normalize()
    mon = today - pd.Timedelta(days=today.weekday())
    sub = [e for e in log if mon <= pd.Timestamp(e["date"]) <= today]
    jr = log_to_journal(sub, EXT_LIBRARY)
    if jr.empty:
        return np.zeros(R)
    _, mods, au = _session_rows(jr, EXT_LIBRARY)
    tot = np.zeros(R)
    for m, a in zip(mods, au):
        tot += (a / AU_REF) * np.array([m.chan.get(c, 0.0) for c in CHANNELS]) / CHAN_SCALE
    return tot


# ═════════════════════════════ 10. DEMO — профиль атлета, синтетика, самопроверка, CLI ═════════════════════════════
def samandyk() -> Athlete:
    """Самандык Д.В. (2004 г.р.). Источники: протокол Кубка РБ (июнь 2026) + значения, присланные вручную.
    ⚠ 100м/400м/ядро/высота/диск/копьё/1500м — из июньского протокола (давние); LJ 7.35, 110мH 14.95, шест 4.30 — свежие.
    1500 м (4:44.5) — июньский результат 2-го дня; 1000 м 2:47 в модель не вводился (нет пересчёта) — замените при желании."""
    cur = {"100m": 11.22, "LJ": 7.39, "SP": 12.67, "HJ": 2.06, "400m": 51.66, "110mH": 14.95, "DT": 31.84,
           "PV": 4.30, "JT": 50.00, "1500m": 284.53}       # обновлено по вашим сообщениям: сумма 7266
    noise = {e: 70.0 for e in EVENTS}                       # старые данные — неопределённость выше
    noise.update({"100m": 35.0, "LJ": 35.0, "SP": 35.0, "400m": 35.0, "110mH": 35.0, "PV": 35.0, "JT": 35.0})   # свежие замеры
    return Athlete("Самандык Д.В.", 77.5, cur, already_day2=("DT", "JT", "1500m"), noise_sd=noise)


def make_synthetic(days: int = 420, seed: int = 3, noise: float = 1.0):
    """Журнал из библиотеки сессий + «истинные» параметры + наблюдения (старты и тесты) — проверка калибровки."""
    rng = np.random.default_rng(seed)
    cfg = ModelConfig()
    truth = Params(K1=np.array([5., 4.5, 5., 4., 6.]), K2=np.array([2.5, 2.5, 2.2, 3.0, 3.2]),
                   tau1=np.array([37., 34., 41., 40., 47.]), half_h=np.array([30., 60., 90.]))
    names = [n for n in LIBRARY.keys() if n != "S_REST"]
    dates = pd.date_range("2025-01-06", periods=days, freq="D")
    rows = []
    for i, d in enumerate(dates):
        if i % 7 == 6 or rng.random() < .08:
            continue
        m = rng.choice(names)
        w = 1 + .5 * np.sin(2 * np.pi * i / 90)
        rows.append((d, m, LIBRARY[m].minutes * w * rng.lognormal(0, .15), min(10, LIBRARY[m].rpe * rng.lognormal(0, .08))))
    jr = pd.DataFrame(rows, columns=["date", "module", "minutes", "rpe"])
    daily = build_daily(jr)
    g, pools, _ = forward(daily["U"], daily["P"], truth, cfg)
    y = form(g, pools, truth, cfg)
    specs = {**default_specs(), **test_specs()}
    refs = {"100m": 11.3, "LJ": 7.3, "SP": 13.0, "HJ": 2.04, "400m": 50.5, "110mH": 15.0, "DT": 38.0, "PV": 4.4, "JT": 52.0,
            "1500m": 270.0, "T30f": 3.5, "TSLJ": 3.1, "TSHF": 13.0, "T300": 38.0, "T1000": 165.0}
    ob = []
    for nm, sp in specs.items():
        n_o = 16 if nm in EVENTS else 45
        di = np.sort(rng.choice(np.arange(50, len(y)), n_o, replace=False))
        L_ = np.array([sp.loadings.get(c, 0.0) for c in CHANNELS])
        sg = 1.0 if sp.higher_is_better else -1.0
        for i in di:
            ob.append((daily["dates"][i], nm, refs[nm] * (1 + sg * (y[i] @ L_ + rng.normal(0, noise * sp.sigma)) / 100)))
    return jr, daily, pd.DataFrame(ob, columns=["date", "obs", "value"]), specs, truth, cfg


def _selftest_pipeline():
    """Калибровка на синтетике → персональная модель → план → объяснение (сквозная проверка)."""
    jr, daily, obs, specs, truth, cfg = make_synthetic(days=300)
    cal = Calibrator(daily, obs, specs, KineticPrior(), cfg)
    fit = cal.fit(n_starts=2, max_nfev=60)
    assert fit.rmse < 3.5, fit.rmse
    lo, hi = Layout(cfg).bounds()
    assert np.all(fit.theta >= lo - 1e-9) and np.all(fit.theta <= hi + 1e-9)
    model = fit.params.to_model()
    ath = Athlete("synthetic", 77.5, {"100m": 11.3, "LJ": 7.3, "SP": 13.0, "HJ": 2.04, "400m": 50.5, "110mH": 15.0,
                                      "DT": 38.0, "PV": 4.4, "JT": 52.0, "1500m": 270.0})
    st = state_from_journal(jr, model)
    sol = DailySolver(model, ath, st)
    seq = sol.solve(10, beam=60, top=1)[0]["seq"]
    assert sol.verify(seq).allowed.all()
    ex = PlanExplainer(sol)
    df = ex.explain(seq, pd.Timestamp("2026-10-05"))
    assert len(df) == 10 and df.why_ok.str.len().min() > 0
    first = seq[1] if seq[1] != "S_REST" else seq[0]
    assert "стоит на" in ex.explain_session(seq, pd.Timestamp("2026-10-05"), first)
    # объяснение: заблокированные сессии в тексте совпадают с эталонным шлюзом
    v0 = sol.gates.check_all(sol.s0.ast, sol.s0.t0)
    if (~v0.allowed).any():
        assert "Gate" in df.blocked_today.iloc[0]
    print("самопроверка пайплайна пройдена: калибровка → состояние по журналу → план → объяснение")


def selftest():
    _selftest_tensor()
    _selftest_library()
    _selftest_gates()
    _selftest_solver()
    _selftest_pipeline()
    lit_selftest()
    grz_selftest()
    live_selftest()
    potential_selftest()
    quick_selftest()
    whatif_selftest()


def run_library():
    pd.set_option("display.width", 160)
    pd.set_option("display.max_colwidth", 48)
    print(LIBRARY.table().to_string(index=False))
    print("\nПрофили проверены: расчётный вектор стресса каждой сессии соответствует заявленному LOW/MID/HIGH/EXTREME.")
    print("\nОстаток утомления после дозы, % (T½ CNS/MET/MECH = 36/48/84 ч):\n", recovery_table())


def run_plan(a):
    pd.set_option("display.width", 160)
    pd.set_option("display.max_colwidth", 70)
    ath0 = samandyk()
    # ── Слой 1: усадка профиля к популяционной ковариации ──
    pts0 = {e: float(wa_points(e, ath0.current[e], floor=True)) for e in EVENTS}
    tab, d2, pval = ProfilePrior().shrink(pts0, ath0.noise_sd)
    print(f"ПРОФИЛЬ {ath0.name}: сумма по текущим результатам = {sum(pts0.values()):.0f} очк.")
    print(tab.to_string(index=False))
    print(f"совместимость с популяцией: d²={d2:.1f}, p={pval:.3f}\n")
    # Свежие замеры (noise_sd ≤ 40) оставляем как измерено — усадка применяется к «старым» видам.
    prof = {e: (ath0.current[e] if ath0.noise_sd.get(e, 60) <= 40 else float(wa_result(e, p)))
            for e, p in zip(EVENTS, tab.pts_post)}
    ath = Athlete(ath0.name, ath0.mass, prof, ath0.already_day2, ath0.noise_sd)
    # ── Модель и состояние: калибровка по журналу или холодный старт ──
    start = pd.Timestamp(a.start).normalize() if a.start else None
    if a.journal:
        jr = pd.read_csv(a.journal)
        if a.results:
            daily = build_daily(jr)
            cal = Calibrator(daily, pd.read_csv(a.results), {**default_specs(), **test_specs()}, KineticPrior(), ModelConfig())
            fit = cal.fit(n_starts=4)
            model = fit.params.to_model()
            print(f"КАЛИБРОВКА: наблюдений {fit.n_obs}, RMSE {fit.rmse:.2f} %\n{fit.params.classical()}\n{fit.params.pools_table()}\n")
        else:
            model = PerfModel.from_prior()
            print("журнал есть, результатов нет → параметры = априоры, состояние — по журналу\n")
        state = state_from_journal(jr, model)
        start = start or (pd.to_datetime(jr["date"]).max().normalize() + pd.Timedelta(days=1))
    else:
        model = PerfModel.from_prior()
        state = warm_start(model)
        start = start or (pd.Timestamp.today().normalize() + pd.Timedelta(days=1))
        print("журнала нет → холодный старт: параметры = априоры, состояние — «типовая неделя»\n")
    print("T½ пулов (ч):", {p: round(float(h), 1) for p, h in zip(POOLS, model.const.half())})
    sol = DailySolver(model, ath, state)
    H = a.horizon
    meet = start + pd.Timedelta(days=H)
    print(f"порог CNS Gate: {sol.thr:.2f} | план: {start:%d.%m.%Y} ({WD[start.weekday()]}) → старт (D1) {meet:%d.%m.%Y}, {H} дн.\n")
    # ── Солвер ──
    t = time.time()
    plans = sol.solve(H, beam=a.beam)
    week = TYPICAL_WEEK[start.weekday():] + TYPICAL_WEEK[:start.weekday()]
    f0, b0 = sol.baseline(H, week)
    best = plans[0]
    print(f"поиск: {time.time() - t:.1f} с | «как раньше»: {b0['points']:.0f} очк. (допустим по шлюзам: {f0}) | "
          f"оптимальный план: {best['points']:.0f} очк., штраф баланса {best['penalty']:.0f}, допустим: {best['feasible']}\n")
    # ── Объяснение и визуализация ──
    ex = PlanExplainer(sol)
    df = ex.explain(best["seq"], start)
    if a.why_day:
        print(render_text(df[df.day == a.why_day], full=True))
    elif a.why:
        print(ex.explain_session(best["seq"], start, a.why))
    else:
        print(render_text(df, full=a.full))
    if a.why or a.why_day:
        return
    print("\nСЕГОДНЯ (день 1) — оптимизатор дня при текущем векторе восстановления:")
    rec = sol.recommend_today(H)
    print(rec[["session", "allowed", "free_in_h", "plan_points", "delta_vs_best"]].head(8).to_string(index=False))
    # ── Монте-Карло ──
    res = sol.evaluate(best["seq"])[2]
    mc = TournamentMC().simulate(res, n=a.sims, thresholds=(6500, 7000, 7500, 8000))
    summ = mc.summary()
    print(f"\nMONTE CARLO, N={a.sims}")
    print(summ.to_string(index=False))
    ts = mc.total_stats()
    print("ИТОГО:", {k: (round(v, 3) if k.startswith("P(") else round(v)) for k, v in ts.items()})
    gc = growth_check(ts["median"], float(sum(pts0.values())), H)
    print(f"\nСВЕРКА С ЛИТЕРАТУРОЙ (Хейнла): медиана {gc['forecast']:.0f} при сумме текущих {gc['baseline']:.0f} (+{gc['gain']:.0f}); "
          f"за {H} дн. нормы роста ≈ {gc['cap_for_horizon'][0]:.0f}–{gc['cap_for_horizon'][1]:.0f} (при равномерном росте); "
          f"при нормальной потере 295±90 прогноз означает суммы личных рекордов ≈ {gc['implied_PR_sum']:.0f} (+{gc['implied_PR_growth']:.0f}). "
          f"Вывод: прогноз — оптимистичный сценарий.")
    sens = mc.sensitivity()
    print(f"\nЧУВСТВИТЕЛЬНОСТЬ (protect — рубеж {sens.attrs['lo']}, upside — рубеж {sens.attrs['hi']}):\n", sens.to_string(index=False))
    if a.html:
        p = plan_to_html(df, summ, f"План {ath.name}: {start:%d.%m}–{meet:%d.%m}", a.html,
                         notes="Холодный старт: τ1, K и T½ — априоры, не персональная калибровка; векторы стресса — экспертные."
                         if not a.results else "Параметры откалиброваны по вашим данным (MAP).")
        print("\nHTML-отчёт:", p)


def main():
    ap = argparse.ArgumentParser(description="Планирование и прогнозирование в десятиборье")
    ap.add_argument("--mode", choices=["plan", "selftest", "library", "benchmarks", "season"], default="plan")
    ap.add_argument("--start", default=None, help="дата первого дня плана YYYY-MM-DD (по умолчанию — завтра или день после журнала)")
    ap.add_argument("--horizon", type=int, default=21, help="дней до старта (2–4 недели)")
    ap.add_argument("--beam", type=int, default=300)
    ap.add_argument("--sims", type=int, default=10_000)
    ap.add_argument("--html", default=None, help="путь для HTML-отчёта")
    ap.add_argument("--journal", default=None, help="CSV журнала сессий: date,module,minutes,rpe")
    ap.add_argument("--results", default=None, help="CSV результатов/тестов: date,obs,value")
    ap.add_argument("--why", default=None, help="ключ сессии: почему она стоит там, а не раньше")
    ap.add_argument("--why-day", type=int, default=None, help="номер дня плана для подробного объяснения")
    ap.add_argument("--full", action="store_true", help="печатать полное объяснение каждого дня")
    ap.add_argument("--hept", default="2027-02-01", help="первый день семиборья (режим season)")
    ap.add_argument("--dec", default="2027-06-06", help="первый день десятиборья (режим season)")
    ap.add_argument("--floor", type=int, default=2, help="season: поддерживающее занятие технического вида, если его не было N недель (0 — выкл.; по Ушакову ≥1/нед.)")
    ap.add_argument("--template", action="store_true", help="season: старый шаблон Ушакова/Верхошанского вместо структуры Грузенкина")
    a = ap.parse_args()
    {"plan": lambda: run_plan(a), "selftest": selftest, "library": run_library, "benchmarks": lambda: run_benchmarks(a), "season": lambda: run_season(a)}[a.mode]()




if __name__ == "__main__" and any(a.startswith("--mode") for a in sys.argv[1:]):
    main()
    sys.exit(0)


# ═════════════════════════════ 11. STREAMLIT-ИНТЕРФЕЙС ═════════════════════════════
# Decathlon Suite — живое приложение для Самандыка Д.В.
# Запуск: streamlit run app.py     (рядом лежат decathlon_suite.py и season_plan.csv)
#
# Как это работает:
#   • меняете результаты / вес / даты / характеристики модели в боковой панели — всё пересчитывается сразу:
#     прогноз модели (decathlon_suite.py), Монте-Карло турнира, чувствительность, усадка профиля, нормы Хейнлы;
#   • вкладка «Цель → результаты» решает обратную задачу: задаёте нужную сумму — получаете, какие результаты нужны по видам;
#   • план берётся из season_plan.csv; если вы меняете даты или масштаб объёма — план строится заново (≈15 с, кэшируется).
#
# Что из источников, а что моё:
#   • очки 1985 г. и манежные формулы — проверены по протоколу Универсиады (5117);
#   • нормы Хейнлы (прирост за 2 сезона, потери 295±90), таблица Нариманова (КД-2), помесячная структура нагрузок и динамика тестов — Грузенкин (1990);
#     сочетания видов в день — Юшкевич/Романов (2012);
#   • МОИ допущения: «потолочные» результаты по видам (обратная задача), профили ≈7500/≈7850, пороги самочувствия, шлюзы модели,
#     стартовые параметры модели (середины диапазонов, K — заглушки, пока нет вашего журнала), недельная волна, поддерживающие занятия, тест-дни.
import os
import sys
import types

import streamlit as st

HERE = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
ds = types.SimpleNamespace(**globals())   # доступ к функциям модели как ds.xxx (всё в одном файле)

st.set_page_config(page_title="Decathlon Suite", page_icon="🏃", layout="wide")
st.title("🏃 Decathlon Suite: данные ⇄ модель")
st.caption("Меняете данные или характеристики модели — прогноз пересчитывается; задаёте цель — получаете нужные результаты. "
           "Условные ориентиры и мои допущения помечены.")

EV = list(ds.EVENTS)
RU = {"100m": "100 м", "LJ": "Длина", "SP": "Ядро", "HJ": "Высота", "400m": "400 м", "110mH": "110 м с/б", "DT": "Диск", "PV": "Шест",
      "JT": "Копьё", "1500m": "1500 м"}
IND = {"60m": (58.015, 11.5, 1.81), "60mH": (20.5173, 15.5, 1.92), "1000m": (0.08713, 305.5, 1.85)}      # манеж; проверено на протоколе 5117


def calc_pts(e, v):
    return int(ds.wa_points(e, float(v), floor=True))


def calc_ind(e, v):
    if e in IND:
        a, b, c = IND[e]
        return int(np.floor(a * max(b - float(v), 0.0) ** c))
    return calc_pts(e, v)


GL = {"sprint": "Спринт", "speed150": "Бег до 150 м", "speedend": "Скор. выносливость", "endurance": "Выносливость", "hurdles": "Барьеры", "LJ": "Длина",
      "HJ": "Высота", "PV": "Шест", "SP": "Ядро", "DT": "Диск", "JT": "Копьё", "strength": "Силовая", "test": "Контроль"}


def fmt_t(sec):
    m, s = divmod(float(sec), 60)
    return f"{int(m)}:{s:05.2f}"


D = ds.samandyk()            # значения по умолчанию = ваши последние результаты (сумма 7266)
DEF_START, DEF_HEPT, DEF_DEC = pd.Timestamp("2026-10-08"), pd.Timestamp("2027-02-01"), pd.Timestamp("2027-06-06")

# ───────────────────────── сохранение данных ─────────────────────────
# Что сохраняется: все значения в боковой панели (результаты, вес, модель, даты, масштаб) + история срезов результатов.
# Куда: (1) облако Supabase, если заданы секреты SUPABASE_URL / SUPABASE_KEY (переживает перезапуск сервера);
#       (2) локальный файл athlete_state.json рядом с программой (переживает перезапуск на вашем компьютере/сервере с диском);
#       (3) ручная резервная копия: кнопка «Скачать копию» / «Загрузить копию» (работает везде).
# На бесплатном Streamlit Cloud файлы стираются при перезапуске, поэтому там нужен вариант (1) — см. DEPLOY.md.
import datetime as _dt
import hashlib
import json
import urllib.request

STATE_FILE = os.environ.get("DECA_STATE", os.path.join(HERE, "athlete_state.json"))
STATE_ID = "main"


def _secret(name):
    try:
        v = st.secrets[name]
        return str(v) if v else None
    except Exception:  # noqa: BLE001
        return os.environ.get(name)


def _sb_req(method, path, body=None, extra=None):
    url, key = _secret("SUPABASE_URL"), _secret("SUPABASE_KEY")
    if not (url and key):
        return None
    hdr = {"apikey": key, "Authorization": f"Bearer {key}", "Content-Type": "application/json"}
    hdr.update(extra or {})
    rq = urllib.request.Request(url.rstrip("/") + "/rest/v1/" + path, data=None if body is None else json.dumps(body).encode("utf-8"),
                                headers=hdr, method=method)
    with urllib.request.urlopen(rq, timeout=8) as r:
        raw = r.read().decode("utf-8")
    return json.loads(raw) if raw else []


def load_state():
    """Облако → файл → пусто. Возвращает (state, источник)."""
    try:
        rows = _sb_req("GET", f"app_state?id=eq.{STATE_ID}&select=data")
        if rows:
            return rows[0]["data"], "облако"
    except Exception:  # noqa: BLE001
        pass
    try:
        with open(STATE_FILE, encoding="utf-8") as f:
            return json.load(f), "файл"
    except Exception:  # noqa: BLE001
        return {}, "нет"


try:
    from zoneinfo import ZoneInfo
    _TZ = ZoneInfo("Europe/Minsk")
except Exception:  # noqa: BLE001
    _TZ = _dt.timezone(_dt.timedelta(hours=3))


def now_local():
    """Время по Минску (сервер Streamlit Cloud живёт по UTC — без этого ночью «сегодня» было бы вчерашним днём)."""
    return _dt.datetime.now(_TZ)


def today_local():
    return pd.Timestamp(now_local().date())


def save_state(state, verify=False):
    """Пишет везде, где можно. Возвращает список мест, куда записалось. verify=True: после записи в облако читает метку времени обратно
    и сверяет — так «записано» значит «действительно лежит в облаке»."""
    ok = []
    st.session_state.pop("_save_err", None)
    try:
        if _secret("SUPABASE_URL") and _secret("SUPABASE_KEY"):
            _sb_req("POST", "app_state?on_conflict=id", [{"id": STATE_ID, "data": state}],
                    {"Prefer": "resolution=merge-duplicates,return=minimal"})
            good = True
            if verify and state.get("saved_at"):
                try:
                    back = _sb_req("GET", f"app_state?id=eq.{STATE_ID}&select=s:data->>saved_at")
                    if not back or back[0].get("s") != state.get("saved_at"):
                        good = False
                        st.session_state["_save_err"] = "облако: запись не подтвердилась при чтении обратно"
                except Exception:  # noqa: BLE001
                    pass          # проверка не удалась технически — запись сама прошла
            if good:
                ok.append("облако")
    except Exception as exc:  # noqa: BLE001
        st.session_state["_save_err"] = f"облако: {exc}"
    try:
        with open(STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=1)
        ok.append("файл")
    except Exception:  # noqa: BLE001
        pass
    return ok


# пароль (необязательно): задайте APP_PASSWORD в секретах, и без него ничего не откроется
_pw = _secret("APP_PASSWORD")
if _pw and st.session_state.get("_auth") != _pw:
    _in = st.text_input("Пароль", type="password")
    if _in == _pw:
        st.session_state["_auth"] = _pw
        st.rerun()
    st.stop()

if "_state" not in st.session_state:
    _s, _src = load_state()
    st.session_state["_state"], st.session_state["_src"] = _s, _src
STATE = st.session_state["_state"]                  # то, что сохранено (виджеты + история); меняется при сохранении
INIT = dict(STATE.get("widgets", {}))                 # начальные значения виджетов в этой сессии
if "_init" not in st.session_state:
    st.session_state["_init"] = INIT
INIT = st.session_state["_init"]
CUR_W = {}                                            # значения виджетов в этом прогоне


def _mk(ns, kind):
    def inner(label, *a, **k):
        persist = k.pop("_save", True)
        key = k.pop("key", None) or "w:" + label
        sv = INIT.get(key) if persist else None
        if sv is not None:
            try:
                if kind == "slider":
                    a = list(a)
                    a[2] = type(a[2])(sv)
                elif kind == "date_input":
                    k["value"] = _dt.date.fromisoformat(sv)
                elif kind == "number_input":
                    k["value"] = type(k["value"])(sv)
                elif kind in ("select_slider", "checkbox"):
                    k["value"] = sv
            except Exception:  # noqa: BLE001
                pass
        out = getattr(ns.t, kind)(label, *a, key=key, **k)
        if persist:
            CUR_W[key] = out.isoformat() if isinstance(out, _dt.date) else (bool(out) if isinstance(out, (bool, np.bool_)) else (float(out) if isinstance(out, (float, np.floating)) else (int(out) if isinstance(out, (int, np.integer)) else out)))
        return out
    return inner


class _Ns:
    """Виджеты с сохранением значений. Куда рисовать — задаётся полем t (контейнер/вкладка); по умолчанию — текущее место в странице."""
    def __init__(self, target):
        self.t = target
        for kind in ("number_input", "slider", "checkbox", "select_slider", "date_input"):
            setattr(self, kind, _mk(self, kind))


P = _Ns(st)               # настройки: перед каждым блоком ставится P.t = нужная вкладка раздела «Данные и модель»
X = _Ns(st)               # виджеты внутри expander / вкладок


def persist_now():
    """Сохраняет, если значения виджетов изменились с прошлого сохранения."""
    if CUR_W and json.dumps(CUR_W, sort_keys=True) != json.dumps(STATE.get("widgets", {}), sort_keys=True):
        new = dict(STATE)
        new["widgets"] = dict(CUR_W)
        new["saved_at"] = now_local().isoformat(timespec="seconds")
        where = save_state(new)
        st.session_state["_state"] = new
        st.session_state["_where"] = where
        STATE.clear()
        STATE.update(new)


def commit(new, what):
    """Записывает изменённое состояние и оставляет сообщение об итоге, которое переживёт перезапуск страницы (раньше после «Записать» не было видно ничего)."""
    new = dict(new)
    w_ = dict(STATE.get("widgets", {}))
    w_.update(CUR_W)              # виджеты, созданные ниже по странице, в этом прогоне ещё не известны — не теряем их прежние значения
    new["widgets"] = w_
    new["saved_at"] = now_local().isoformat(timespec="seconds")
    where = save_state(new, verify=True)
    st.session_state["_where"] = where
    st.session_state["_state"] = new
    STATE.clear()
    STATE.update(new)
    t_ = now_local().strftime("%H:%M")
    cloud_cfg = bool(_secret("SUPABASE_URL") and _secret("SUPABASE_KEY"))
    err_ = st.session_state.get("_save_err", "")
    if "облако" in where:
        flash = ("success", f"✅ {what} · {t_} · лежит в облаке (проверено чтением обратно)")
    elif where and cloud_cfg:
        flash = ("warning", f"⚠️ {what} · {t_} · записано только на сервере: {err_ or 'облако не ответило'}. После перезапуска приложения запись может пропасть — "
                            "скачайте копию (⚙️ Данные и модель → 💾 Копия данных).")
    elif where:
        flash = ("warning", f"⚠️ {what} · {t_} · записано на сервере, но облако не подключено: после перезапуска приложения запись пропадёт. "
                            "Скачайте копию (⚙️ Данные и модель → 💾 Копия данных) или подключите облако (DEPLOY.md).")
    else:
        flash = ("error", f"❌ {what}: не удалось сохранить нигде. Скачайте копию (⚙️ Данные и модель → 💾 Копия данных) и сообщите об ошибке.")
    st.session_state["_flash"] = flash
    st.rerun()


# ───────────────────────── каркас страницы ─────────────────────────
# Пять разделов вместо тринадцати вкладок; ежедневное — «Сегодня», остальное — по смыслу. Старые номера вкладок (tabs[N]) сохранены как словарь.
TOP = st.container()                                    # выбор дня и статус дня (заполняется ниже, когда известен план)
SEC = st.tabs(["🏠 Сегодня", "📓 Дневник", "🗓 План", "🎯 Прогноз", "⚙️ Данные и модель"])
with SEC[1]:
    DIA = st.tabs(["Запись дня", "Нагрузка и форма", "История результатов"])
with SEC[2]:
    PLN = st.tabs(["Весь сезон", "Связи и часы"])
with SEC[3]:
    FC_TOP = st.container()                             # сводные показатели над вкладками прогноза
    FC = st.tabs(["Прогноз модели", "Цель → результаты", "Нормы Хейнлы", "Семиборье", "Профили", "Тесты СФП"])
with SEC[4]:
    SET = st.tabs(["👤 Профиль", "📅 Календарь и объём", "🧠 Модель", "🔧 Правила плана", "ℹ️ Как это работает", "💾 Копия данных"])
tabs = {0: SEC[0], 1: DIA[0], 2: DIA[1], 11: DIA[2], 3: PLN[0], 12: PLN[1], 4: FC[0], 5: FC[1], 6: FC[2], 7: FC[3], 8: FC[4], 9: FC[5], 10: SET[4]}
APP_BUILD = "12.10.2026 13:56 (Минск)"                                # подставляется сборщиком (build_single.py)
SIDE_STATUS = st.sidebar.container()                   # статус сохранения (рисуется в самом конце, когда известен итог прогона)

# ───────────────────────── боковая панель: данные ─────────────────────────
# ── 💾 Копия данных
with SET[5]:
    st.subheader("💾 Копия данных")
    st.caption("Всё (записи дневника, настройки, срезы результатов) лежит одним файлом. Копия — страховка на случай, если облако недоступно или вы меняете место запуска.")
    st.download_button("⬇️ Скачать копию (JSON)", json.dumps(STATE, ensure_ascii=False, indent=1).encode("utf-8"), "athlete_state.json", "application/json")
    _up = st.file_uploader("Загрузить копию (заменит текущие данные)", type="json", key="restore_up")
    if _up is not None:
        _raw = _up.getvalue()
        _h = hashlib.md5(_raw).hexdigest()
        if st.session_state.get("_restored") != _h:
            try:
                _new = json.loads(_raw.decode("utf-8"))
                assert isinstance(_new, dict)
                save_state(_new)
                for _k in [k for k in st.session_state if str(k).startswith("w:") or str(k).startswith("st_")]:
                    del st.session_state[_k]
                st.session_state["_state"], st.session_state["_init"], st.session_state["_restored"] = _new, dict(_new.get("widgets", {})), _h
                st.rerun()
            except Exception as exc:  # noqa: BLE001
                st.error(f"Не удалось прочитать копию: {exc}")
_src = st.session_state.get("_src")

# ── 👤 Профиль
sb = SET[0]
P.t = sb
sb.subheader("👤 Самандык Д.В.")
sb.caption("Эти значения сохраняются автоматически при каждом изменении. Результаты — ваши личные рекорды (от них считаются очки, прогноз и план).")
ath_height = P.number_input("Рост (м)", value=1.85, step=0.01, help="Рост я не знаю — введите свой (нужен только для КД-2).")
ath_mass = P.number_input("Вес (кг)", value=77.5, step=0.5)

sb.subheader("Десятиборье: личные результаты")
res = {"100m": P.number_input("100 м (с)", value=D.current["100m"], step=0.01, format="%.2f"),
       "LJ": P.number_input("Длина (м)", value=D.current["LJ"], step=0.01, format="%.2f"),
       "SP": P.number_input("Ядро (м)", value=D.current["SP"], step=0.01, format="%.2f"),
       "HJ": P.number_input("Высота (м)", value=D.current["HJ"], step=0.01, format="%.2f"),
       "400m": P.number_input("400 м (с)", value=D.current["400m"], step=0.05, format="%.2f"),
       "110mH": P.number_input("110 м с/б (с)", value=D.current["110mH"], step=0.01, format="%.2f"),
       "DT": P.number_input("Диск (м)", value=D.current["DT"], step=0.05, format="%.2f"),
       "PV": P.number_input("Шест (м)", value=D.current["PV"], step=0.05, format="%.2f"),
       "JT": P.number_input("Копьё (м)", value=D.current["JT"], step=0.05, format="%.2f"),
       "1500m": P.number_input("1500 м (с)", value=D.current["1500m"], step=0.5, format="%.2f", help="4:44.53 = 284,53 с")}

with sb.expander("Давность замеров (влияет на усадку профиля)"):
    fresh = {e: X.checkbox(f"{RU[e]}: свежий (≤2 мес.)", value=D.noise_sd[e] <= 35) for e in EV}
noise = {e: (35.0 if fresh[e] else 70.0) for e in EV}      # 35/70 очков — моя условная шкала неопределённости

sb.subheader("Семиборье (манеж): личные результаты")
hep = {"60m": P.number_input("60 м (с)", value=7.12, step=0.01),
       "LJ": P.number_input("Длина, манеж (м)", value=7.13, step=0.01),
       "SP": P.number_input("Ядро, манеж (м)", value=12.30, step=0.01),
       "HJ": P.number_input("Высота, манеж (м)", value=2.03, step=0.01),
       "60mH": P.number_input("60 м с/б (с)", value=8.50, step=0.01),
       "PV": P.number_input("Шест, манеж (м)", value=4.20, step=0.05),
       "1000m": P.number_input("1000 м (с)", value=167.0, step=1.0, help="2:47 = 167 с")}

sb.subheader("Тесты СФП")
has_tests = P.checkbox("Тесты замерены лично", value=False, help="Отметьте, только если это ваши реальные замеры. Замеры, внесённые в журнал («Тест: …» в «Контроль», «Жим лёжа»), подхватываются сами и имеют приоритет.")
v_max = P.number_input("Vmax (м/с)", value=10.40, step=0.05)
t_triple = P.number_input("Тройной с места (м)", value=9.40, step=0.05)
t_shot = P.number_input("Бросок ядра назад (м)", value=14.50, step=0.1)
t_bench = P.number_input("Жим лёжа (кг)", value=110.0, step=2.5)
t_1000 = P.number_input("Бег 1000 м (с)", value=167.0, step=1.0)

# ───────────────────────── боковая панель: характеристики модели ─────────────────────────
sb = SET[2]
P.t = sb
sb.subheader("🧠 Характеристики модели")
sb.caption("Стартовые значения — середины диапазонов из ТЗ. Пока нет вашего журнала, это не измерено, а заложено: двигайте и смотрите, что меняется.")
with sb.expander("Полураспад утомления (часы)", expanded=False):
    h_cns = X.slider("ЦНС (диапазон ТЗ 24–48)", 12.0, 72.0, 36.0, 1.0)
    h_met = X.slider("Метаболический (24–72)", 12.0, 96.0, 48.0, 1.0)
    h_mech = X.slider("Механический (72–96)", 48.0, 144.0, 84.0, 1.0)
with sb.expander("Фитнес и отклик", expanded=False):
    tau_names = {"SPR": "Спринт", "JMP": "Прыжки", "THR": "Метания", "SE": "Скор. выносл.", "END": "Выносливость"}
    tau_def = {"SPR": 36, "JMP": 36, "THR": 40, "SE": 41, "END": 46}
    taus = [X.slider(f"τ фитнеса: {tau_names[c]} (сут.)", 20, 70, tau_def[c]) for c in ds.CHANNELS]
    k1 = X.slider("K1 — отдача фитнеса", 0.5, 8.0, 3.0, 0.1)
    k2 = X.slider("K2 — цена усталости", 0.2, 6.0, 2.0, 0.1)
    sat = X.slider("Насыщение нагрузки", 0.5, 5.0, 2.0, 0.1)
sims = P.select_slider("Розыгрышей Монте-Карло", options=[1000, 2000, 4000, 10000, 20000], value=4000)
loss_n = P.slider("Потери «сумма личных − турнир» (Хейнла: 295±90)", 150, 450, 295, 5)

# ───────────────────────── боковая панель: даты и план ─────────────────────────
sb = SET[1]
P.t = sb
sb.subheader("📅 Календарь и объём")
d_start = pd.Timestamp(P.date_input("Начало плана", value=DEF_START.date()))
d_hept = pd.Timestamp(P.date_input("Семиборье (старт)", value=DEF_HEPT.date()))
d_dec = pd.Timestamp(P.date_input("Десятиборье (старт)", value=DEF_DEC.date()))
vol_scale = P.slider("Масштаб объёма к Грузенкину", 0.6, 1.3, 1.0, 0.05,
                      help="1.0 = годовые объёмы средней группы мастеров спорта СССР. Для вас могут быть тяжелее или легче.")
sb = SET[3]
P.t = sb
sb.subheader("🔧 Правила адаптации плана")
sb.caption("Как план подстраивается под ваши результаты и записи. Все правила — мои допущения (в подсказках «?» указано, что из литературы, а что моё).")
indiv = P.slider("Индивидуализация плана по моим результатам", 0.0, 1.5, 1.0, 0.1,
                 help="0 = чистый план Грузенкина для средней группы. Больше = больше занятий и объёма для средств, питающих виды с наибольшим запасом роста "
                      "(очки до «потолка»), и меньше для сильных видов. Правило моё, не из источника.")
use_deficit = P.checkbox("Добирать недовыполненное по журналу", value=True,
                         help="Если за последние 2 недели по записанным дням средство выполнено хуже плана, ближайшие недели получают до +15 % к нему (правило моё).")
use_form = P.checkbox("Учитывать тренд результатов тренировок в плане", value=True,
                      help="Если за последние 8 недель результаты упражнений, питающих вид, падают (×1,10) или стоят на месте (×1,05), средство получает больше объёма. Нужно ≥ 3 занятий с результатами. Правило моё.")
use_adj = P.checkbox("Поправка прогноза по тренду результатов", value=False,
                     help="Сдвигает результат вида в прогнозе на k × тренд тренировочных результатов за 4 недели (не более ±2 %). Коэффициент k — моё допущение, по данным не проверен.")
adj_k = P.slider("k переноса тренировочного тренда на результат вида", 0.0, 0.6, 0.3, 0.05, help="0 = не переносить; 1 = тренировочный прирост полностью переходит на результат вида (нереалистично).")
CORR_OPTS = {"Синтез: 2012 + 1952–60 + ТЗ (по умолчанию)": "synth", "Только Табл. 1 (2012, сильнейшие)": "new", "Только 1952–60 (Олимпиады)": "old", "Только ТЗ/эксперт": "prior"}
corr_label = P.select_slider("Связи между видами в Монте-Карло", options=list(CORR_OPTS), value=list(CORR_OPTS)[0],
                             help="Корреляции результатов между видами в турнирном розыгрыше. Синтез = обратно-дисперсионное усреднение в z-пространстве Фишера (веса — на вкладке «Связи и часы»).")
w_hours = P.slider("Вес закона убывающей отдачи в индивидуализации", 0.0, 1.0, 0.5, 0.1,
                   help="Смешивает «запас роста» с законом убывающей отдачи (Юшкевич, Романов 2012): форма из статьи (показатель ≈0,47), масштаб — ваши «потолки». 0 = только запас роста.")
use_tests_emph = P.checkbox("Слабое звено по тестам СФП влияет на акценты плана", value=True,
                            help="Если тест ниже результата соответствующих видов на ≥ 500 очков по таблице Нариманова — группе, питающей тест, +5 % (при ≥ 1000 — +10 %); "
                                 "если выше на ≥ 500 — +5 % техническим группам этих видов. Работает только при наличии тестов (из журнала или профиля). Правило и пороги мои, не из источника.")
sb = SET[1]
P.t = sb
endur_cap = P.slider("Потолок кросса за занятие (км)", 2.0, 12.0, 5.0, 0.5,
                     help="В таблице Грузенкина кросс доходит до ~10 км за занятие (618 км в год, 1987–88). Здесь вы задаёте потолок сами.")


# тесты СФП: из журнала (приоритет), иначе из профиля, если отмечено «замерены лично»
_TJ = ds.tests_from_journal(list(STATE.get("log", [])), pd.Timestamp(os.environ.get("DECA_TODAY") or today_local()))
TEST_VALS = {k: v["value"] for k, v in _TJ.items()}
TEST_SRC = {k: f"журнал, {v['date'][8:10]}.{v['date'][5:7]}" for k, v in _TJ.items()}
if has_tests:
    for _k, _v in (("vmax", v_max), ("triple_sj", t_triple), ("shot_bwd", t_shot), ("bench", t_bench), ("t1000", t_1000)):
        if _k not in TEST_VALS:
            TEST_VALS[_k] = float(_v)
            TEST_SRC[_k] = "профиль (вручную)"
TEST_ASSESS = ds.tests_assessment(TEST_VALS, res)
TEST_RES = ds.tests_to_results(TEST_VALS, res)             # функция «тест → результат вида» по Грузенкину (ядро назад / толчок с места / диск с места)
FGAP = ds.function_gaps(TEST_RES)
T_MULT = ds.tests_emphasis(TEST_ASSESS, fgap=FGAP) if use_tests_emph else {}
STRETCH_UI = {e: float(st.session_state.get(f"st_{e}", INIT.get(f"st_{e}", ds.STRETCH[e]))) for e in EV}      # «потолки» из вкладки «Цель → результаты»
EMPH = ds.emphasis_from_profile(res, STRETCH_UI, strength=float(indiv))
EMPH_HEAD = dict(EMPH)
EMPH_LAWD, HB = {}, None
CORR_M = ds.build_event_corr(source=CORR_OPTS[corr_label])


@st.cache_data(show_spinner="Строю план (≈5 с)…")
def gen_plan(start, hept, dec, scale, emph_items, cap_km):
    out, _ = ds.build_season_grz(pd.Timestamp(start), pd.Timestamp(hept), pd.Timestamp(dec), None, vol_scale=scale,
                                 emph=dict(emph_items), endur_cap=cap_km)
    return out


def load_csv():
    for pth in (os.path.join(HERE, "season_plan.csv"), "season_plan.csv"):
        if os.path.exists(pth):
            return pd.read_csv(pth)
    return None


plan_note = ""
df = None
if not (d_start + pd.Timedelta(days=30) < d_hept < d_dec - pd.Timedelta(days=30)):
    plan_note = "Даты должны идти так: начало → (≥30 дн.) → семиборье → (≥30 дн.) → десятиборье. Показан запасной план из файла."
else:
    try:
        if w_hours > 0:
            _bp = gen_plan(str(d_start.date()), str(d_hept.date()), str(d_dec.date()), float(vol_scale), tuple(), float(endur_cap))     # нейтральный план → часы по видам
            HB = ds.hours_by_event_from_plan(_bp)
            EMPH_LAWD = ds.emphasis_hours_law(res, HB, STRETCH_UI, strength=float(indiv))
            EMPH = ds.emphasis_blend(EMPH_HEAD, EMPH_LAWD, float(w_hours))
        EMPH = ds.apply_emphasis(EMPH, T_MULT)
        df = gen_plan(str(d_start.date()), str(d_hept.date()), str(d_dec.date()), float(vol_scale), tuple(sorted(EMPH.items())), float(endur_cap))
    except Exception as exc:  # noqa: BLE001
        plan_note = f"Не удалось построить план для этих параметров ({exc}). Показан запасной план из файла."
if df is None:
    df = load_csv()
if df is None:
    up = SET[1].file_uploader("Загрузите season_plan.csv", type="csv")
    if up is not None:
        df = pd.read_csv(up)


def _norm(d):
    d = d.copy()
    d["date"] = pd.to_datetime(d["date"]).dt.normalize()
    for c in ("gate_note", "groups"):
        if c in d:
            d[c] = d[c].fillna("").astype(str)
    if "vol" not in d:
        d["vol"] = "{}"
    d["vol"] = d["vol"].fillna("{}").astype(str)
    return d


real_today = pd.Timestamp(os.environ.get("DECA_TODAY") or today_local()).normalize()      # «сегодня» по Минску
LOG = list(STATE.get("log", []))
LOGD = ds.log_by_date(LOG)
if df is not None:
    df = _norm(df)
df_base = df                                   # план без добора: по нему считается выполнение
DEF = {}
TR = ds.result_trends(LOG, real_today) if LOG else pd.DataFrame()
EF = ds.event_form(TR) if len(TR) else pd.DataFrame()
FORMX = ds.form_emph(EF) if (use_form and len(EF)) else {}
if df is not None and LOG and use_deficit:
    DEF = ds.deficit_emph(LOG, df_base, real_today)
if df is not None and (DEF or FORMX):
    em2 = {g: round(min(1.4, round(EMPH.get(g, 1.0) * DEF.get(g, 1.0) * FORMX.get(g, 1.0) / 0.05) * 0.05), 4) for g in set(EMPH) | set(DEF) | set(FORMX)}
    try:
        df_adj = _norm(gen_plan(str(d_start.date()), str(d_hept.date()), str(d_dec.date()), float(vol_scale), tuple(sorted(em2.items())), float(endur_cap)))
        df = pd.concat([df_base[df_base["date"] < real_today], df_adj[df_adj["date"] >= real_today]], ignore_index=True)
    except Exception:  # noqa: BLE001
        DEF, FORMX = {}, {}
p0, p1 = (df["date"].min(), df["date"].max()) if df is not None else (DEF_START, DEF_DEC - pd.Timedelta(days=1))
today = min(max(real_today, p0), p1)
WD_RU = ["пн", "вт", "ср", "чт", "пт", "сб", "вс"]


def _goto(d):
    st.session_state["w:Дата"] = min(max(pd.Timestamp(d).normalize(), p0), p1).date()


def _shift(n):
    _goto(pd.Timestamp(st.session_state.get("w:Дата") or today.date()) + pd.Timedelta(days=n))


def day_status(d):
    """(значок, текст) о том, что известно про день d: записан ли он."""
    e = LOGD.get(str(d.date()))
    if e:
        mins = float(e.get("minutes") or 0)
        its = e.get("items") or []
        if e.get("rest") and not its:
            return "😴", "записан как отдых"
        au = float(e.get("rpe_ovr") or e.get("rpe") or 0) * mins
        return "✅", (f"записано: {len(its)} упр., {mins:.0f} мин, нагрузка {au:.0f}" if its else f"записано в журнале: {mins:.0f} мин, нагрузка {au:.0f}")
    if d > real_today:
        return "📅", "день ещё не наступил"
    if d == real_today:
        return "⏳", "сегодня ещё не записано"
    return "➖", "записи нет (для модели это день отдыха)"


PR_LABEL = {"100m": "100 м (с)", "LJ": "Длина (м)", "SP": "Ядро (м)", "HJ": "Высота (м)", "400m": "400 м (с)", "110mH": "110 м с/б (с)", "DT": "Диск (м)",
            "PV": "Шест (м)", "JT": "Копьё (м)", "1500m": "1500 м (с)"}          # подписи полей профиля (ключ виджета = "w:" + подпись)


def _accept_pr(e, v):
    st.session_state["w:" + PR_LABEL[e]] = float(v)           # выполняется до перерисовки: поле профиля получит новое значение, прогноз и очки пересчитаются


def render_pr(cands, prefix):
    """Кандидаты в личные результаты: тренировочный результат полного вида лучше записанного в профиле. Решение — кнопкой, автоматически ничего не меняется."""
    for _, r_ in cands.iterrows():
        c1_, c2_ = st.columns([5, 2])
        unit_ = "с" if r_["event"] in ("100m", "400m", "110mH", "1500m") else "м"
        c1_.markdown(f"🏅 **{RU[r_['event']]}**: {r_['value']:g} {unit_} ({r_['exercise']}, {r_['date']}) лучше результата в профиле {r_['current']:g} {unit_} → **+{r_['d_pts']:.0f} очк.**")
        c2_.button("Принять как личный", on_click=_accept_pr, args=(r_["event"], r_["value"]), key=f"{prefix}_pr_{r_['event']}", use_container_width=True)


with TOP:
    _fl = st.session_state.pop("_flash", None)
    if _fl:
        getattr(st, _fl[0])(_fl[1])
        if _fl[0] == "success" and hasattr(st, "toast"):
            st.toast(_fl[1])
    if plan_note:
        st.warning(plan_note)
    nv1, nv2, nv3, nv4 = st.columns([1.2, 3, 1.2, 2.4])
    nv1.button("◀ день", on_click=_shift, args=(-1,), key="nav_prev", use_container_width=True)
    P.t = nv2
    cur_date = pd.Timestamp(P.date_input("Дата", value=today.date(), min_value=p0.date(), max_value=p1.date(), _save=False, label_visibility="collapsed"))
    nv3.button("день ▶", on_click=_shift, args=(1,), key="nav_next", use_container_width=True)
    nv4.button("К сегодняшнему дню", on_click=_goto, args=(today,), key="nav_today", use_container_width=True)

# ───────────────────────── расчёты (всё живое) ─────────────────────────
pts = {e: calc_pts(e, v) for e, v in res.items()}
sum_cur = sum(pts.values())
hep_pts = {e: calc_ind(e, v) for e, v in hep.items()}
sum_hep = sum(hep_pts.values())
horizon_end = d_dec
horizon = max(0, (horizon_end - cur_date).days)

model = ds.make_model(half=(h_cns, h_met, h_mech), tau1=taus, K1=k1, K2=k2, sat_G=sat)
model0 = ds.make_model(half=(36.0, 48.0, 84.0), tau1=[tau_def[c] for c in ds.CHANNELS], K1=3.0, K2=2.0, sat_G=2.0)   # стартовая модель (эталон «что изменилось»)

AN = ds.analyze_log(LOG, df_base, real_today, model) if LOG and df_base is not None else {"n_days": len(LOG)}
if AN.get("n_days", 0) == 0:
    AN = {"n_days": 0}
ST_REAL = AN.get("state")                       # состояние фитнеса/пулов по фактическому журналу (если записей достаточно)
LAST = AN.get("last_logged")

# пометки шлюзов считаются заново с вашими параметрами модели (и, если есть журнал, от реального состояния)
# последовательность для прогноза: пропущенные (не записанные) дни после последней записи — отдых, с сегодняшнего дня — план
if df is not None:
    seq, seq_dates = ds.forecast_seq(df, LAST if (ST_REAL is not None and LAST is not None) else None, real_today, with_dates=True)
    if len(seq) < 7:
        seq, seq_dates = list(df.session_key)[-7:], list(df["date"])[-7:]
else:
    seq, seq_dates = ds.TYPICAL_SEQ(max(horizon, 7)), []
GAP_DAYS = max(0, (real_today - LAST).days - 1) if (ST_REAL is not None and LAST is not None) else 0

with TOP:                         # строка статуса дня и предупреждение о пропущенных днях (нужны журнал и состояние, поэтому ниже)
    _rw = df[df["date"] == cur_date] if df is not None else None
    _ph = str(_rw.iloc[0]["phase"]) if (_rw is not None and len(_rw)) else "вне плана"
    _ic, _tx = day_status(cur_date)
    _line = f"{_ic} **{cur_date:%d.%m.%Y}, {WD_RU[cur_date.weekday()]}** · {_ph} · {_tx}"
    if _ic == "✅":
        st.success(_line)
    elif _ic == "⏳":
        st.warning(_line)
    else:
        st.info(_line)
    if GAP_DAYS >= 3:
        st.warning(f"Последняя запись в журнале — {LAST:%d.%m} ({GAP_DAYS} дн. назад). Дни без записей модель считает отдыхом, и прогноз уже учитывает паузу. Если вы тренировались — внесите дни.")

# пометки шлюзов по дням плана (с реального состояния, если есть журнал; иначе с типовой недели по всему плану)
if df is not None:
    ath_ = ds.Athlete("live", float(ath_mass), dict(res), ("DT", "JT", "1500m"), dict(noise))
    notes = [""] * len(df)
    if ST_REAL is not None and LAST is not None:
        keys_, dates_, st_g = seq, seq_dates, ST_REAL
    else:
        keys_, dates_, st_g = list(df.session_key), list(df["date"]), ds.warm_start(model)
    if keys_ and len(dates_) == len(keys_):
        _tr = ds.DailySolver(model, ath_, st_g)._run(list(keys_), True)[4]
        pos_ = {pd.Timestamp(d_).normalize(): i_ for i_, d_ in enumerate(df["date"])}
        for d_, t_ in zip(dates_, _tr):
            i_ = pos_.get(pd.Timestamp(d_).normalize())
            if i_ is not None and t_["bad"]:
                notes[i_] = "шлюз: " + ",".join(t_["bad"])
    df["gate_note"] = notes
RES_F = ds.result_adjust(res, EF, float(adj_k)) if (use_adj and len(EF)) else res
F = ds.live_forecast(RES_F, ath_mass, seq, model=model, noise_sd=noise, sims=sims, seed=0, loss=loss_n, state=ST_REAL, corr=CORR_M)
F0 = ds.live_forecast(dict(D.current), 77.5, seq, model=model0, noise_sd=dict(D.noise_sd), sims=sims, seed=0, state=ST_REAL, corr=CORR_M)   # эталон: стартовые данные и модель

NAR_LV = list(ds.NAR_LEVELS)
NARUI = {"Vmax (м/с)": ds.NAR["vmax"][0], "Тройной с места (м)": ds.NAR["triple_sj"][0], "Бросок ядра назад (м)": ds.NAR["shot_bwd"][0],
       "Жим лёжа (кг)": ds.NAR["bench"][0], "Бег 1000 м (с)": ds.NAR["t1000"][0]}
KD2_TAB = ds.NAR["KD2"][0]


def interp_level(tab, val):
    x, y = np.array(tab, float), np.array(NAR_LV, float)
    if x[-1] < x[0]:
        x, y = x[::-1], y[::-1]
    return float(np.interp(val, x, y))


kd2v = (ds.kd2(TEST_VALS["vmax"], ath_height, ath_mass, TEST_VALS["triple_sj"], TEST_VALS["shot_bwd"], TEST_VALS["bench"], TEST_VALS["t1000"])
        if all(k in TEST_VALS for k in ("vmax", "triple_sj", "shot_bwd", "bench", "t1000")) else None)

mt = F["mc_total"]
with FC_TOP:
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Сумма личных результатов (10 видов)", f"{sum_cur}", f"{sum_cur - 7500:+d} к уровню 7500")
    m2.metric("Прогноз модели к старту (по плану)", f"{F['sum_forecast']:.0f}", f"{F['sum_forecast'] - F0['sum_forecast']:+.0f} к эталону")
    m3.metric("Монте-Карло: медиана турнира", f"{mt['median']:.0f}", f"{mt['median'] - F0['mc_total']['median']:+.0f} к эталону")
    m4.metric(f"Консервативно (прогноз − {loss_n})", f"{F['sum_forecast'] - loss_n:.0f}", f"P(≥7000) = {mt['P(>=7000)']:.0%}")
    m5.metric("Семиборье, сумма личных (7 видов)", f"{sum_hep}", f"Универсиада 2026: 5117 ({sum_hep - 5117:+d})")
    if ST_REAL is not None and LAST is not None:
        _basis = (f"Старт прогноза — состояние по вашему журналу (последняя запись {LAST:%d.%m.%Y}). "
                  + (f"**{GAP_DAYS} дн. без записей после неё считаются отдыхом** (так же считает нагрузка недели): если вы тренировались, внесите эти дни. " if GAP_DAYS else "")
                  + f"Дальше — план с {real_today:%d.%m.%Y} ({len(seq) - GAP_DAYS} дн.).")
    else:
        _basis = (f"Журнала с ≥ 7 днями тренировок пока нет, поэтому старт прогноза — «типовая неделя», а записи его не меняют. Прогноз считается по плану с {real_today:%d.%m.%Y} "
                  f"({len(seq)} дн.).")
    st.caption("«Эталон» = стартовые данные (сумма 7266) и стартовая модель без правок. " + _basis + " Личные результаты отнесены к форме «после типовой недели»: "
               "усталость от вчерашней тренировки не делает рекорд «выше», а прогноз — «лучше».")

# ───────────────────────── сегодня ─────────────────────────
def _sig(items):
    """Подпись состава тренировки для сравнения «на экране» и «в дневнике» (без служебных полей, с округлением)."""
    out = []
    for x in items:
        d = {k: v for k, v in x.items() if k not in ("uid", "res_txt", "res")}
        d = {k: (round(float(v), 3) if isinstance(v, (int, float)) and not isinstance(v, bool) else v) for k, v in d.items()}
        d["res"] = [round(float(v), 3) for v in (x.get("res") or [])]
        out.append(json.dumps(d, sort_keys=True, ensure_ascii=False))
    return sorted(out)


with tabs[0]:
    if df is None:
        st.error("Файл season_plan.csv не найден. Положите его рядом с программой или загрузите в «⚙️ Данные и модель → 📅 Календарь и объём».")
    else:
        row = df[df["date"] == cur_date]
        r = row.iloc[0] if len(row) else df.iloc[0]
        if not LOG:
            with st.expander("👋 Как пользоваться (три шага)", expanded=True):
                st.markdown("1. Вверху выберите день (по умолчанию сегодня).\n"
                            "2. Ниже соберите тренировку из каталога (или нажмите «Заполнить по плану дня») и впишите результаты: секунды, метры, высоту.\n"
                            "3. Нажмите **«Записать тренировку»**. Вверху страницы появится зелёная строка «✅ записано», а под кнопкой — где именно лежат данные.\n\n"
                            "Если в боковой панели написано «Облако не подключено», записи пропадут при перезапуске сервера: подключите облако (DEPLOY.md) или скачивайте копию.")

        # ── 1. что по плану
        st.markdown("### 1 · Что по плану")
        st.caption(f"Этап: {r['phase']} · волна объёма ×{r['volume_factor']:.2f} (недельная волна — моя: 1,0 / 1,1 / 0,75) · ключ сессии `{r['session_key']}`")
        text = str(r["plan"])
        if "||" in text:
            am, pm = text.split("||", 1)
            st.markdown(f"**Утро:** {am.replace('УТРО:', '').strip()}")
            st.markdown(f"**Вечер:** {pm.replace('ВЕЧЕР:', '').strip()}")
        else:
            st.markdown(f"**План:** {text}")

        sc1, sc2 = st.columns(2)
        with sc1:
            sleep = X.slider("Сон / восстановление (1–10)", 1, 10, 8)
        with sc2:
            sore = X.slider("Боль в мышцах/связках (1–10)", 1, 10, 3)

        no_load = str(r["session_key"]) == "S_REST" or text.startswith(("СТАРТ", "Отдых"))
        if no_load:
            st.success("День без нагрузки или старт.")
        elif sleep <= 5 or sore >= 7:
            st.error(f"Самочувствие ниже условного порога (сон {sleep}/10, боль {sore}/10).")
            LIGHT = {"sprint": "техника бега + 3×20 м в 80 %", "hurdles": "ритм 3×(3 барьера) без скорости", "LJ": "4 прыжка с короткого разбега",
                     "HJ": "6 прыжков с короткого разбега", "PV": "техника на коротком разбеге, без планки", "SP": "8 бросков на технике",
                     "DT": "8 бросков на технике", "JT": "5 метаний с места", "speedend": "вместо бега до 500 м — кросс 15 мин", "speed150": "вместо бега до 150 м — кросс 15 мин",
                     "endurance": "кросс 20 мин, пульс 130–140", "strength": "мобильность и корпус 25 мин вместо силовой", "test": "перенести контроль на 1–2 дня"}
            gl = [g.rstrip("~") for g in str(r.get("groups", "")).split(",") if g]
            st.warning("Облегчённый вариант на сегодня (вместо плана): " + (" → ".join(f"{GL.get(g, g)}: {LIGHT[g]}" for g in gl if g in LIGHT) or "кросс 20 мин + мобильность")
                       + ". Пороги условные, не из литературы.")
        elif str(r.get("gate_note", "")):
            st.warning(f"Совет шлюзов модели: `{r['gate_note']}` (экспертные пороги, занятие не менялось). Следите за пульсом и техникой.")

        with st.expander("Неделя вокруг выбранного дня", expanded=False):
            wk = df[(df["date"] >= cur_date - pd.Timedelta(days=2)) & (df["date"] <= cur_date + pd.Timedelta(days=4))].copy()
            wk["Дата"] = [("▶ " if d_ == cur_date else "") + d_.strftime("%d.%m") for d_ in wk["date"]]
            wk["Статус"] = [" ".join(day_status(d_)) for d_ in wk["date"]]
            st.dataframe(wk[["Дата", "weekday", "Статус", "plan", "gate_note"]].rename(columns={"weekday": "День", "plan": "План", "gate_note": "Шлюз"}),
                         use_container_width=True, hide_index=True)

        # ── 2. конструктор
        st.markdown("---")
        st.markdown("### 2 · Собрать тренировку")
        st.caption(f"В каталоге {len(ds.CATALOG)} позиций (разминка, ОФП/СФП, бег по отрезкам, барьеры, прыжки, метания, сила, плиометрика, выносливость, ядро, восстановление). "
                   "Вы выбираете и задаёте дозу — система считает нагрузку, смотрит шлюзы и конфликты и предлагает, что добавить. "
                   "**Все стрессы упражнений, пороги и правила — мои экспертные допущения, а не измерения и не нормы из литературы.**")
        iso_c = str(cur_date.date())
        ck = f"cons_{iso_c}"
        if ck not in st.session_state:
            ex_c = LOGD.get(iso_c) or {}
            st.session_state[ck] = [dict(x) for x in (ex_c.get("items") or [])]
        CL = st.session_state[ck]

        def _uid():
            st.session_state["_uid"] = st.session_state.get("_uid", 0) + 1
            return st.session_state["_uid"]

        for it_ in CL:
            it_.setdefault("uid", _uid())

        def _add(cid, lst=CL):
            c_ = ds.CAT[cid]
            d_ = dict(id=cid, sets=c_.sets, reps=c_.reps, size=c_.size, uid=_uid())
            if ds.IMPL_STD.get(cid):
                d_["impl"] = ds.IMPL_STD[cid]
            lst.append(d_)

        def _rm(uid, lst=CL):
            lst[:] = [x for x in lst if x["uid"] != uid]

        def _fill_plan(lst=CL, pv=None):
            lst[:] = [dict(x, uid=_uid(), **({"impl": ds.IMPL_STD[x["id"]]} if ds.IMPL_STD.get(x["id"]) else {})) for x in ds.items_from_plan(pv or {})]

        def _set_part(part, new_items, lst=CL):
            lst[:] = [x for x in lst if ds.part_of(ds.CAT[x["id"]]) != part]
            add_ = [dict(x, uid=_uid()) for x in new_items]
            lst[:] = (add_ + lst) if part == 0 else (lst + add_)

        def _auto_wu(lst=CL):
            _set_part(0, ds.suggest_warmup([x for x in lst]))

        def _auto_cd(lst=CL):
            _set_part(2, ds.suggest_cooldown([x for x in lst]))

        def _clear(lst=CL):
            lst.clear()

        HIST_C = ds.result_history(LOG, before=iso_c)
        planned_c = ds._json_loads(r["vol"]) if ("vol" in r.index) else {}
        fb1, fb2, fb3, fb4 = st.columns(4)
        fb1.button("📋 Заполнить по плану дня", on_click=_fill_plan, kwargs=dict(pv=planned_c), key=f"cf_{iso_c}", use_container_width=True,
                   help="Подставляет по одной типовой позиции на каждое средство плана; дальше правьте как хотите.")
        fb2.button("🧹 Очистить", on_click=_clear, key=f"cc_{iso_c}", use_container_width=True)
        fb3.button("🔥 Подобрать разминку", on_click=_auto_wu, key=f"cw_{iso_c}", use_container_width=True,
                   help="Разминка под основную часть: общая + специальная (ускорения, прыжковая, метательная, барьерная). Правило моё.")
        fb4.button("🧊 Подобрать заминку", on_click=_auto_cd, key=f"cd_{iso_c}", use_container_width=True)

        with st.expander("⚡ Быстрая запись текстом", expanded=True):
            qt_key = f"qt_{iso_c}"
            qtext = st.text_input("Что было (через запятую)", key=qt_key, placeholder="длина 6.85, ядро 14.20, диск с места 33, присед 180х3, 100 м 11.2")
            QP = ds.parse_quick(qtext) if qtext.strip() else None
            if QP:
                for l_ in QP["lines"]:
                    st.markdown("• " + l_)
                for w_ in QP["warnings"]:
                    st.warning(w_)
                if QP["unparsed"]:
                    st.caption("Не понял: " + "; ".join(QP["unparsed"]) + ". Добавьте это из каталога ниже.")
                if QP["date_word"] in ("вчера", "позавчера"):
                    st.caption(f"Слово «{QP['date_word']}» не меняет день: запись пойдёт в выбранный сверху ({cur_date:%d.%m}). Переключите день стрелками, если нужно другое.")

            def _add_quick(items, key=qt_key, lst=CL):
                for x_ in items:
                    lst.append(dict(x_, uid=_uid(), res_txt=" ".join(f"{v:g}" for v in x_.get("res", []))))
                st.session_state[key] = ""
            st.button("➕ Добавить в тренировку", on_click=_add_quick, args=((QP["items"] if QP else []),), disabled=not (QP and QP["items"]), key=f"qadd_{iso_c}")
            st.caption("Слова: длина, высота, шест, ядро, диск, копьё (без уточнения — полные попытки; «соревнование» — серия), «ядро снизу/назад» (тест 7,26 кг), «диск с места», «ядро с места», «тройной», "
                       "«30 м с хода», 100 м, 110 с/б, 400 м, 1000 м, 1500 м (время 4:40), жим/присед/рывок/становая — вес×повторы×серии («присед 180х3», одно число — один подход). "
                       "Разбор по таблице слов, не нейросетью: что распознано, видно выше. После добавления проверьте список ниже и нажмите «Записать тренировку».")

        with st.expander("➕ Добавить упражнения из каталога", expanded=True):
            f1, f2, f3, f4 = st.columns([2, 2, 2, 3])
            cats_all = sorted({c.cat for c in ds.CATALOG})
            kinds_all = sorted({c.kind for c in ds.CATALOG})
            fcat = f1.multiselect("Раздел", cats_all, key="cat_f")
            fkind = f2.multiselect("Тип", kinds_all, key="cat_k")
            fev = f3.multiselect("Для вида", list(ds.RU_EV), format_func=lambda e: ds.RU_EV[e], key="cat_e")
            fq = f4.text_input("Поиск", key="cat_q")
            only_ph = st.checkbox("Только подходящие этапу подготовки", value=True, key="cat_ph")
            stage_c = str(r["phase"])[:1] if str(r["phase"])[:1] in "ABC" else "C"
            cand = [c for c in ds.CATALOG if (not fcat or c.cat in fcat) and (not fkind or c.kind in fkind) and (not fev or set(fev) & set(c.tags))
                    and (not fq or fq.lower() in (c.name + " " + c.note + " " + c.cat).lower()) and (not only_ph or stage_c in c.phases)]
            pick = st.multiselect(f"Найдено: {len(cand)}. Выберите позиции и нажмите «Добавить»", [c.id for c in cand], key=f"cat_pick_{len(cand)}",
                                  format_func=lambda i: f"{ds.CAT[i].cat} · {ds.CAT[i].name} — {ds.CAT[i].std_text()}")

            def _add_many(ids):
                for i_ in ids:
                    _add(i_)
            st.button("➕ Добавить выбранное", on_click=_add_many, args=(list(pick),), key="cat_add")
            if st.checkbox("Показать найденное таблицей", key="cat_tbl"):
                st.dataframe(pd.DataFrame([dict(Раздел=c.cat, Название=c.name, Тип=c.kind, Доза=c.std_text(), Мин=c.minutes, RPE=c.rpe,
                                                ЦНС=c.stress[0], Метаб=c.stress[1], Механика=c.stress[2], Виды=", ".join(c.tags), Этапы=c.phases, Заметка=c.note) for c in cand]),
                             use_container_width=True, hide_index=True)

        st.markdown("#### Моя тренировка")
        if not CL:
            st.info("Пока пусто: добавьте позиции из каталога или нажмите «Заполнить по плану дня».")
        CL.sort(key=lambda x: ds.part_of(ds.CAT[x["id"]]))
        prev_part = -1
        for k_, it_ in enumerate(CL):
            c_ = ds.CAT[it_["id"]]
            pt_ = ds.part_of(c_)
            if pt_ != prev_part:
                st.markdown(f"##### {ds.PARTS[pt_]}")
                prev_part = pt_
            a0, a1, a2, a3, a4 = st.columns([4, 1.2, 1.2, 1.6, 0.8])
            a0.markdown(f"**{k_ + 1}. {c_.name}**  \n<small>{c_.cat} · стандарт {c_.std_text()}</small>", unsafe_allow_html=True)
            it_["sets"] = a1.number_input("серии", min_value=1, max_value=30, value=int(it_["sets"]), key=f"cs_{it_['uid']}")
            it_["reps"] = a2.number_input("повт.", min_value=1, max_value=60, value=int(it_["reps"]), key=f"cr_{it_['uid']}")
            if c_.size:
                it_["size"] = a3.number_input(c_.size_unit, min_value=0.1, value=float(it_["size"] or c_.size), step=(0.5 if c_.size < 20 else 5.0), key=f"cz_{it_['uid']}")
            a4.button("✖", on_click=_rm, args=(it_["uid"],), key=f"cx_{it_['uid']}")
            if ds.IMPL_STD.get(c_.id):
                it_["impl"] = st.number_input("Вес снаряда, кг", min_value=0.1, max_value=20.0, value=float(it_.get("impl") or ds.IMPL_STD[c_.id]), step=0.25, key=f"cw_{it_['uid']}",
                                              help=f"Стандарт для этого упражнения {ds.IMPL_STD[c_.id]:g} кг. Другой вес меняет нагрузку (√ отношения), а результаты считаются отдельно для каждого веса.")
            if c_.size_unit == "кг" and it_.get("size"):
                st.caption(f"Тоннаж {float(it_['sets']) * float(it_['reps']) * float(it_['size']) / 1000:.2f} т · {float(it_['size']) / ath_mass:.2f} × массы тела ({ath_mass:g} кг)")
            sp_ = ds.res_spec(c_)
            if sp_:
                kind_, unit_ = sp_
                hk_ = HIST_C.get(ds.res_key(c_, it_), [])
                if kind_ == "time":
                    hint_ = f"Время каждого повтора, {unit_} (через пробел; можно 1:02.4)"
                elif kind_ == "dist":
                    hint_ = f"Дальность по попыткам, {unit_} (через пробел)"
                else:
                    hint_ = f"Высота по попыткам, {unit_} (через пробел)"
                if "res_txt" not in it_:
                    it_["res_txt"] = " ".join(f"{v:g}" for v in (it_.get("res") or []))
                it_["res_txt"] = st.text_input(hint_, value=it_["res_txt"], key=f"cres_{it_['uid']}", placeholder="например: 3.82 3.85 3.80")
                it_["res"] = ds.parse_results(it_["res_txt"])
                if hk_:
                    pbv_ = min(p_["best"] for p_ in hk_) if kind_ == "time" else max(p_["best"] for p_ in hk_)
                    st.caption(f"Ваш лучший: {ds.fmt_res(kind_, pbv_)} {unit_} · прошлый раз ({hk_[-1]['date']}): {ds.fmt_res(kind_, hk_[-1]['best'])} {unit_}")
                else:
                    st.caption("Результатов этого упражнения в журнале ещё нет.")

        # ── 3. анализ (считается по составу, но показывается после блока записи)
        A = None
        rpe_ov = 0.0
        if CL:
            tr = ds.week_channel_load(LOG, cur_date) if LOG else None
            hour_c = (ST_REAL.t0 + 24.0 * max(0, (cur_date - LAST).days - 1)) if (ST_REAL is not None and LAST is not None) else None
            weak_c = {e: max(float(ds.wa_points(e, STRETCH_UI[e], floor=True)) - float(pts[e]), 0.0) for e in ds.EVENTS}
            rpe_ov = st.number_input("RPE дня (0 = по упражнениям)", min_value=0.0, max_value=10.0, value=0.0, step=0.5, key=f"crpe_{iso_c}")
            A = ds.analyze_day([{k: v for k, v in x.items() if k not in ("uid", "res_txt")} for x in CL], ST_REAL, model, week_channels=tr, week_au_ref=float(sum(ds.LIBRARY[k].AU for k in ds.TYPICAL_WEEK)),
                               rpe_override=(rpe_ov or None), hour=hour_c, stage=stage_c, weak=weak_c, history=LOG, date=iso_c,
                               readiness=dict(sleep=sleep, sore=sore), mass=float(ath_mass),
                               form=(dict(zip(EF['event'], EF['Статус'])) if len(EF) else None))

        # ── запись
        st.markdown("---")
        st.markdown("### 3 · Записать")
        ex_s = LOGD.get(iso_c)
        saved_items = bool(ex_s and ex_s.get("items"))
        can_rec = cur_date <= real_today

        def _record_training():
            gv = {}
            for x in CL:
                cc_, phi_ = ds.item_dose(x)
                if cc_.group:
                    gv[cc_.group] = round(gv.get(cc_.group, 0.0) + cc_.log_volume(phi_), 3)
            ex_c = dict(LOGD.get(iso_c) or {})
            ex_c.update({"sleep_score": int(sleep)})
            if not ex_c.get("sore"):
                ex_c["sore"] = int(sore)
            ex_c.update({"date": iso_c, "rest": False, "items": [{k: v for k, v in x.items() if k not in ("uid", "res_txt")} for x in CL], "groups": gv,
                         "minutes": int(round(A["minutes"])), "rpe": round(A["AU"] / max(A["minutes"], 1), 2)})
            if rpe_ov:
                ex_c["rpe_ovr"] = float(rpe_ov)
            else:
                ex_c.pop("rpe_ovr", None)
            newlog = sorted([e for e in LOG if e["date"] != iso_c] + [ex_c], key=lambda e: e["date"])
            commit(dict(STATE, log=newlog), f"Тренировка {cur_date:%d.%m} записана: {len(CL)} упр., {A['minutes']:.0f} мин, нагрузка {A['AU']:.0f}")

        def _record_rest():
            ex_c = {k: v for k, v in (LOGD.get(iso_c) or {}).items() if k not in ("items", "groups", "minutes", "rpe", "rpe_ovr")}
            ex_c.update({"date": iso_c, "rest": True, "groups": {}, "minutes": 0, "rpe": 1})
            newlog = sorted([e for e in LOG if e["date"] != iso_c] + [ex_c], key=lambda e: e["date"])
            commit(dict(STATE, log=newlog), f"{cur_date:%d.%m}: записан день без тренировки")

        if saved_items and CL and _sig(CL) == _sig(ex_s["items"]):
            st.success("✅ Записано в дневник. На экране ровно то, что записано.")
        elif saved_items:
            st.warning("⚠️ В дневнике этот день записан иначе, чем на экране. Нажмите «Обновить запись», чтобы заменить.")
        elif ex_s and ex_s.get("rest"):
            st.info("😴 День записан как отдых. Если тренировка всё-таки была — запишите её, отметка отдыха заменится.")
        elif ex_s:
            st.info("День записан в журнале без упражнений (Дневник → Запись дня). Запись из конструктора заменит его состав, чек-ин и заметка сохранятся.")
        elif CL:
            st.info("● Пока не записано. Собранная тренировка живёт, только пока открыта страница: нажмите «Записать тренировку».")
        else:
            st.caption("Соберите тренировку выше и нажмите «Записать тренировку» — или отметьте день без тренировки.")
        if CL:
            q1, q2, q3 = st.columns(3)
            q1.metric("Время", f"{A['minutes']:.0f} мин")
            q2.metric("Нагрузка (мин×RPE)", f"{A['AU']:.0f}", f"RPE ≈ {A['AU'] / max(A['minutes'], 1):.1f}")
            q3.metric("Допуск по самочувствию", f"{100 * A['readiness_factor']:.0f} %", f"сон {sleep}/10, боль {sore}/10")
        bb1, bb2 = st.columns(2)
        if bb1.button(("💾 Обновить запись" if saved_items else "✅ Записать тренировку"), type="primary", disabled=(not CL or not can_rec), key=f"csave_{iso_c}", use_container_width=True):
            _record_training()
        if bb2.button("😴 День без тренировки", disabled=not can_rec, key=f"crest_{iso_c}", use_container_width=True):
            _record_rest()
        if not can_rec:
            st.caption("Записать можно только сегодняшний и прошедшие дни. Для будущих дней конструктор работает как проверка плана (анализ ниже).")

        # ── анализ
        if CL:
            st.markdown("### 4 · Анализ тренировки")
            st.metric("Доза ЦНС / метаб. / механ.", " / ".join(f"{x:.1f}" for x in A["dose"]))
            st.caption(f"Разминка: {A.get('warmup_min', 0):.0f} мин · затронуты виды: " + (", ".join(ds.RU_EV.get(e, e) for e in A["events_touched"]) or "—")
                       + ". Сон и боль берутся из блока «Что по плану» выше.")
            for w_ in A["warnings"]:
                (st.success if w_.startswith("Конфликтов") else st.warning)(w_)
            if len(A["results"]):
                st.markdown("**Результаты на тренировке**")
                st.dataframe(A["results"], use_container_width=True, hide_index=True)
                for n_ in A["result_notes"]:
                    (st.success if "новый лучший" in n_ else st.info if "стартовую точку" in n_ else st.warning)(n_)
                st.caption("Интенсивность = ваше среднее / ваш лучший результат этого упражнения (по журналу). Множитель дозы: 1,0 при 97 % от лучшего, "
                           "±2 за каждую единицу отклонения, в пределах 0,75–1,15 — это мой допуск, он меняет учтённую нагрузку и утомление. Пороги потери результата — практические правила, не нормативы.")
            _log_d = [e for e in LOG if e["date"] != iso_c] + [dict(date=iso_c, items=[{k: v for k, v in x.items() if k not in ("uid", "res_txt")} for x in CL])]
            _pc = ds.pr_candidates(_log_d, res)
            _pc = _pc[_pc["date"] == iso_c]
            if len(_pc):
                st.markdown("**🏅 Лучше результата в вашем профиле**")
                render_pr(_pc, "td")
                st.caption("Тренировочный результат — не рекорд (ветер, замер, правила вида). Нажмите «Принять», только если уверены; тогда очки, профиль и прогноз пересчитаются.")
            if A["suggest"]:
                st.markdown("**Что можно добавить** (подбор по моим правилам: слабые виды, недобранные за неделю каналы, шлюзы):")
                for sg in A["suggest"][:6]:
                    s1, s2 = st.columns([6, 1])
                    s1.markdown(f"• {sg['name']} — <small>{sg['why']}</small>", unsafe_allow_html=True)
                    s2.button("➕", on_click=_add, args=(sg["id"],), key=f"sg_{sg['id']}_{len(CL)}")
            with st.expander("Подробности: вес, пулы, нагрузка по упражнениям, сравнение с планом", expanded=False):
                if len(A["loads"]):
                    st.markdown("**Вес: штанга и снаряды**")
                    st.dataframe(A["loads"], use_container_width=True, hide_index=True)
                    st.caption("Вес штанги и тоннаж влияют на модель только как нагрузка (утомление и фитнес). Связи «присед / жим → результат вида» в модели нет: она не выведена из ваших данных. "
                               "Прогресс силы смотрите в «Тестах СФП» и по результатам самих видов.")
                if "pools_now" in A and ST_REAL is not None:
                    thr = ds.BioGates(ds.GateConfig(), model.const).cns_thr
                    st.dataframe(pd.DataFrame({"Пул": ["ЦНС", "Метаболизм", "Механика"], "Сейчас": np.round(A["pools_now"], 2), "После тренировки": np.round(A["pools_after"], 2)})
                                 .assign(**{"Порог ЦНС для «нервных» блоков": [round(float(thr), 2), None, None]}), use_container_width=True, hide_index=True)
                else:
                    st.caption("Состояние пулов неизвестно (мало записей в журнале) — шлюзы по текущему утомлению не проверялись.")
                show = A["items"][["name", "cat", "dose", "minutes", "AU", "cns", "met", "mech", "gate"]].rename(columns={
                    "name": "Упражнение", "cat": "Раздел", "dose": "Доза", "minutes": "Мин", "AU": "AU", "cns": "ЦНС", "met": "Метаб", "mech": "Механика", "gate": "Шлюз"})
                st.dataframe(show, use_container_width=True, hide_index=True)
                if planned_c:
                    gsum = {}
                    for x in CL:
                        cc_, phi_ = ds.item_dose(x)
                        if cc_.group:
                            gsum[cc_.group] = gsum.get(cc_.group, 0.0) + cc_.log_volume(phi_)
                    cmpr = [dict(Средство=ds.LOG_GROUPS[g][0], Единица=ds.LOG_GROUPS[g][1], План=round(float(v), 1), Выбрано=round(gsum.get(g, 0.0), 1)) for g, v in planned_c.items() if g in ds.LOG_GROUPS]
                    if cmpr:
                        st.markdown("**Сравнение с планом дня**")
                        st.dataframe(pd.DataFrame(cmpr), use_container_width=True, hide_index=True)


# ───────────────────────── сезон ─────────────────────────
with tabs[3]:
    if df is None:
        st.error("Нет плана.")
    else:
        sel = st.selectbox("Этап:", ["Все"] + list(df["phase"].unique()))
        show = df if sel == "Все" else df[df["phase"] == sel]
        st.dataframe(show[["date", "weekday", "phase", "volume_factor", "session_key", "plan", "gate_note"]], use_container_width=True, hide_index=True)
        st.download_button("Скачать план (CSV)", df.to_csv(index=False).encode("utf-8-sig"), "season_plan.csv", "text/csv")
        if "groups" in df:
            rows = []
            for _, x in df[df["groups"] != ""].iterrows():
                for k in x["groups"].split(","):
                    rows.append((x["date"].strftime("%Y-%m"), k))
            if rows:
                piv = pd.DataFrame(rows, columns=["Месяц", "Группа"]).value_counts().unstack(0).fillna(0).astype(int)
                st.markdown("#### Занятий по группам в месяц (знак ~ — поддерживающие; test — контроль)")
                st.dataframe(piv, use_container_width=True)
        st.caption("Состав и объём — из таблицы 3 Грузенкина (средняя группа мастеров спорта СССР, 7859 очк.). Поддерживающие занятия, волна, подводка и тест-дни — мои.")

# ───────────────────────── прогноз модели ─────────────────────────
with tabs[4]:
    st.subheader("Прямая задача: результаты + характеристики модели + план → прогноз")
    if not F["feasible"]:
        st.warning("План нарушает хотя бы один шлюз модели (пороги экспертные) — для прогноза это не критично, в «Сегодня» шлюзы только советуют.")
    TYP_PTS = ds.pts_table(F["res_typical"])
    rows = []
    shr = F["shrink"].set_index("event")
    for e in EV:
        fr = F["res"][e]
        rows.append({"Вид": RU[e], "Сейчас": round(res[e], 2), "Очки сейчас": pts[e], "Прогноз к старту": round(fr, 2),
                     "Очки прогноза": F["pts_forecast"][e], "Δ очков": F["pts_forecast"][e] - pts[e],
                     "Типовая неделя вместо плана": TYP_PTS[e],
                     "Профиль после усадки": int(shr.loc[e, "pts_post"]), "Флаг": shr.loc[e, "flag"]})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Прогноз по плану", f"{F['sum_forecast']:.0f}")
    c2.metric("Если вместо плана — «типовая неделя» (заглушка)", f"{F['sum_typical']:.0f}", f"{F['sum_typical'] - F['sum_forecast']:+.0f} к плану")
    c3.metric("Совместимость профиля с популяцией (p)", f"{F['pval']:.2f}", f"d²={F['d2']:.1f}")
    st.caption("Прогноз = ваши результаты × (1 ± отклик фитнеса и усталости на выбранный план). Отклик зависит от характеристик модели («⚙️ Данные и модель → 🧠 Модель»); "
               "поскольку K и полураспады пока заложены по умолчанию, а не измерены на вашем журнале, это сценарий, а не измерение.")

    st.markdown("#### 🛌 Что если пауза в подготовке")
    st.caption("Те же дни плана, но на выбранный срок — полный отдых. Показано, как меняется форма в дни стартов. Фитнес в модели «живёт» 36–46 суток (настройка τ), поэтому ранняя пауза "
               "почти забывается, а пауза за 1–2 месяца до старта — нет. Долгосрочную потерю базы модель не описывает: это ограничение модели, не вывод о физиологии.")
    if df is None or not seq_dates:
        st.info("Нужен план, чтобы считать сценарий.")
    else:
        _lo, _hi = max(real_today, pd.Timestamp(seq_dates[0])), pd.Timestamp(seq_dates[-1])
        pz1, pz2 = st.columns(2)
        pz_from = pz1.date_input("Пауза с", value=min(max(real_today + pd.Timedelta(days=1), _lo), _hi).date(), min_value=_lo.date(), max_value=_hi.date(), key="pz_from")
        pz_days = pz2.number_input("Дней без тренировок", min_value=1, max_value=200, value=60, step=1, key="pz_days")
        _sd = set(pd.Timestamp(x).normalize() for x in seq_dates)
        _tgt = {}
        for _nm, _d in (("Старт семиборья", d_hept), ("Старт десятиборья", d_dec)):
            _dd = _d.normalize() if _d.normalize() in _sd else (_d - pd.Timedelta(days=1)).normalize()
            if _dd in _sd:
                _tgt[_nm] = _dd
        PZ = ds.pause_scenario(df, real_today, pd.Timestamp(pz_from), int(pz_days), _tgt, RES_F, float(ath_mass), model, ST_REAL, LAST if ST_REAL is not None else None)
        if len(PZ):
            PZ = PZ.rename(columns={"target": "Старт", "date": "Дата", "plan": "Форма по плану, очки", "pause": "Форма с паузой, очки", "delta": "Δ очков"})
            st.dataframe(PZ, use_container_width=True, hide_index=True)
            st.caption("Очки — по шкале десятиборья на этот день: для даты семиборья это «форма на эту дату», а не прогноз семиборья (в модели нет его видов). "
                       "Главное — разность Δ: сколько формы стоит такая пауза.")

    st.markdown("#### Монте-Карло турнира (2 дня, попытки, заступы, планка)")
    cc = st.columns(6)
    cc[0].metric("Медиана", f"{mt['median']:.0f}")
    cc[1].metric("P05 … P95", f"{mt['p05']:.0f} … {mt['p95']:.0f}")
    for i, T in enumerate((6500, 7000, 7500)):
        cc[2 + i].metric(f"P(≥{T})", f"{mt[f'P(>={T})']:.0%}")
    cc[5].metric("P(≥8000)", f"{mt['P(>=8000)']:.1%}")
    hist, edges = np.histogram(F["mc"].total, bins=40)
    st.bar_chart(pd.DataFrame({"розыгрышей": hist}, index=[int(x) for x in edges[:-1]]))
    st.caption(f"Монте-Карло учитывает случайность дня, попыток и планки (потеря относительно суммы по видам ≈ {F['sum_forecast'] - mt['median']:.0f} очк.). "
               f"Нормативная потеря «личные результаты − турнир» у десятиборцев 295±90 заметно больше — потому что личные результаты установлены в лучшие дни. "
               f"Поэтому честная «осторожная» оценка = прогноз − {loss_n} = {F['sum_forecast'] - loss_n:.0f}.")
    ms = F["mc_summary"].rename(columns={"event": "Вид", "median_result": "Медиана результата", "median_pts": "Медиана очков", "p05": "P05", "p95": "P95", "P_zero": "P(0 очков)"})
    ms["Вид"] = ms["Вид"].map(RU)
    st.dataframe(ms, use_container_width=True, hide_index=True)
    st.markdown("#### Где риск потерять очки (чувствительность)")
    sn = F["sens"].rename(columns={"event": "Вид", "var_share": "Доля дисперсии", "ES10": "Недобор в худших 10 %", "P_zero": "P(0)",
                                   "protect": "Прирост P при защите от провала", "upside": "Прирост P при хорошем дне"})
    sn["Вид"] = sn["Вид"].map(RU)
    st.dataframe(sn, use_container_width=True, hide_index=True)
    st.markdown("#### Очков за +1 % результата (куда выгоднее давить)")
    mp = ds.marginal_points(res, 1.0)
    mp["event"] = mp["event"].map(RU)
    st.dataframe(mp.rename(columns={"event": "Вид", "now": "Сейчас", "after": "После +1 %", "d_pts": "Очков"}), use_container_width=True, hide_index=True)

# ───────────────────────── обратная задача ─────────────────────────
with tabs[5]:
    st.subheader("Обратная задача: цель → какие результаты нужны")
    cA, cB = st.columns(2)
    basis = cA.radio("Что задаёте", ["Результат турнира", "Сумму личных результатов"], index=0)
    target = cB.number_input("Нужная сумма (очки)", value=7500, step=50)
    basis_key = "tournament" if basis == "Результат турнира" else "pr"
    with st.expander("«Потолочные» результаты по видам (мои допущения — измените на свои)"):
        sc = st.columns(5)
        stretch = {}
        for i, e in enumerate(EV):
            with sc[i % 5]:
                stretch[e] = X.number_input(RU[e], value=float(ds.STRETCH[e]), step=0.05 if e not in ("1500m",) else 0.5, format="%.2f", key=f"st_{e}")
    inv = ds.required_for_target(res, float(target), basis_key, float(loss_n), stretch, max(horizon, 1))
    a1_, a2_, a3_, a4_ = st.columns(4)
    a1_.metric("Нужная сумма личных результатов", f"{inv['need_sum']:.0f}", f"+{max(inv['gain'], 0):.0f} к текущим {inv['base']:.0f}")
    a2_.metric("Запас до «потолка»", f"{inv['capacity']:.0f}")
    a3_.metric("Доля пути к «потолку» в каждом виде", f"{inv['fraction']:.0%}")
    a4_.metric("Достижимо в сумме", f"{inv['reachable_sum']:.0f}")
    if inv["gain"] <= 0:
        st.success("Эта цель не выше вашей текущей суммы личных результатов.")
    elif inv["shortfall"] > 0:
        st.error(f"Даже если все виды выйдут на «потолок», не хватает {inv['shortfall']:.0f} очков. Либо цель выше реалистичной, либо «потолки» занижены.")
    t = inv["table"].copy()
    t["Вид"] = t["event"].map(RU)
    show = t[["Вид", "current", "pts_now", "required", "pts_required", "d_pts", "stretch"]].rename(
        columns={"current": "Сейчас", "pts_now": "Очки сейчас", "required": "Нужный результат", "pts_required": "Очки", "d_pts": "+ очков", "stretch": "Потолок"})
    st.dataframe(show, use_container_width=True, hide_index=True)
    if inv.get("heinla"):
        g = inv["heinla"]
        lo_, hi_ = g["cap_for_horizon"]
        need_gain = inv["reachable_sum"] - inv["base"]
        verdict = ("в пределах нормы" if need_gain <= hi_ else f"выше верхней границы нормы на {need_gain - hi_:.0f}")
        st.info(f"Нужный рост личных результатов {need_gain:.0f} очк. за {horizon} дн.; норма Хейнлы для вашего уровня на этот срок {lo_:.0f}…{hi_:.0f} — {verdict}.")
    # проверка прямой моделью: если выйти на эти результаты, что даст Монте-Карло
    mc2 = ds.TournamentMC(corr=CORR_M).simulate({e: float(inv["table"].set_index("event").loc[e, "required"]) for e in EV}, n=sims, seed=1)
    ts2 = mc2.total_stats()
    st.caption(f"Проверка Монте-Карло: если в день турнира выйти ровно на эти результаты, медиана суммы ≈ {ts2['median']:.0f} (P05…P95 = {ts2['p05']:.0f}…{ts2['p95']:.0f}); "
               f"это без нормативной потери {loss_n} (она отражает, что личные результаты ставятся в лучшие дни). "
               "Распределение по видам — моё допущение: каждый вид проходит одинаковую долю пути от текущего результата до «потолка»; это не оптимум и не предсказание, а способ показать масштаб.")

# ───────────────────────── Хейнла ─────────────────────────
with tabs[6]:
    st.subheader(f"Что допускают нормы Хейнлы к {d_dec:%d.%m.%Y}")
    lo2, hi2 = ds.heinla_cap(sum_cur, 2.0)
    g_lo, g_hi = lo2 * horizon / 730, hi2 * horizon / 730
    scen = [("Без роста личных результатов", 0), ("Рост по нижней границе нормы", g_lo), ("Рост по верхней границе нормы (потолок)", g_hi)]
    tbl = [{"Сценарий": n, "Сумма личных результатов к старту": round(sum_cur + g), "Ожидаемый результат турнира": round(sum_cur + g - loss_n),
            "Диапазон (±90)": f"{round(sum_cur + g - loss_n - 90)}…{round(sum_cur + g - loss_n + 90)}"} for n, g in scen]
    tbl.append({"Сценарий": "Прогноз вашей модели (с поправкой на потери)", "Сумма личных результатов к старту": round(F["sum_forecast"]),
                "Ожидаемый результат турнира": round(F["sum_forecast"] - loss_n), "Диапазон (±90)": f"{round(F['sum_forecast'] - loss_n - 90)}…{round(F['sum_forecast'] - loss_n + 90)}"})
    st.dataframe(pd.DataFrame(tbl), use_container_width=True, hide_index=True)
    over = F["sum_forecast"] - (sum_cur + g_hi)
    st.caption(f"До старта {horizon} дн.; допустимый прирост за 2 сезона для вашего уровня {lo2:.0f}…{hi2:.0f} очков, пересчитан линейно на период. "
               f"Потери «сумма личных результатов − турнир» в среднем 295±90 (обычно 200–400). Это нормы мировых выборок, а не прогноз по вашим данным. "
               + (f"Модель при текущих настройках выше потолка нормы на {over:.0f} очк. — оптимистичный сценарий." if over > 0 else "Модель при текущих настройках не выходит за потолок нормы."))

# ───────────────────────── семиборье ─────────────────────────
with tabs[7]:
    st.subheader(f"Семиборье {d_hept:%d.%m.%Y} (проверочный старт)")
    names = {"60m": "60 м", "LJ": "Длина", "SP": "Ядро", "HJ": "Высота", "60mH": "60 м с/б", "PV": "Шест", "1000m": "1000 м"}
    st.dataframe(pd.DataFrame([{"Вид": names[e], "Результат": hep[e], "Очки": hep_pts[e]} for e in hep]), use_container_width=True, hide_index=True)
    st.info(f"Сумма личных результатов в манеже: **{sum_hep}** очков. На Универсиаде в марте 2026 вы набрали 5117 ({sum_hep - 5117:+d}). "
            "Норм потерь для семиборья у меня нет (Хейнла — для десятиборья), поэтому прогноза модели для семиборья здесь нет: модель для 60 м / 60 м с/б / 1000 м не построена.")
    st.caption("Манежные формулы (60 м, 60 м с/б, 1000 м) проверены точно на протоколе Универсиады 2026.")

# ───────────────────────── профили ─────────────────────────
with tabs[8]:
    REF7500 = dict(ds.REF_PROFILE)
    REF7850 = {"100m": 11.20, "LJ": 7.25, "SP": 14.20, "HJ": 2.02, "400m": 50.20, "110mH": 14.90, "DT": 43.50, "PV": 4.60, "JT": 59.00, "1500m": 281.0}
    p75 = {e: calc_pts(e, v) for e, v in REF7500.items()}
    p78 = {e: calc_pts(e, v) for e, v in REF7850.items()}
    st.subheader(f"Разница с ориентирными профилями ≈{sum(p75.values())} и ≈{sum(p78.values())} очков")
    out = [{"Вид": RU[e], "Ваш результат": res[e], "Ваши очки": pts[e], "Профиль ≈7500": REF7500[e], "Очки профиля 7500": p75[e], "Разница с 7500": f"{pts[e] - p75[e]:+d}",
            "Профиль ≈7850": REF7850[e], "Очки профиля 7850": p78[e], "Разница с 7850": f"{pts[e] - p78[e]:+d}"} for e in EV]
    st.dataframe(pd.DataFrame(out), use_container_width=True, hide_index=True)
    thr = sum(pts[e] - p75[e] for e in ("DT", "SP", "JT"))
    st.info(f"В трёх метаниях разница с профилем ≈7500: **{thr}** очков; общая разница до 7500 по сумме: {max(0, 7500 - sum_cur)}.")
    st.caption("Профили — условные ориентиры равномерной подготовки, не из литературы. Группа Грузенкина (мастера спорта СССР) имела в среднем 7859.")

# ───────────────────────── тесты ─────────────────────────
with tabs[9]:
    st.subheader("Тесты СФП и потенциал")
    st.info("На этой вкладке сверху вниз: ① какие значения тестов соответствуют уровню · ② оценка ваших тестов по Нариманову · ③ функция «тест → результат вида» (Грузенкин) · "
            "④ **«Что если»: введите рывок, жим, тройной, ядро, диск, 30 м, 1000 м — получите результаты по десяти видам и сумму очков (связи тестов с видами — из литературы, с пометкой откуда)** · ⑤ потенциал к старту (рост по Хейнла, пауза, доля плана) · "
            "⑥ ориентиры тестов на контрольные дни · ⑦ динамика тестов у группы Грузенкина.")
    st.caption("Тесты берутся из журнала сами (лучший за 120 дней) и сопоставляются с таблицей Нариманова; из профиля — только если отмечено «Тесты замерены лично». "
               "Ядро учитывается только весом 7,26 кг; Vmax = 30 м / время с хода; жим — оценка 1ПМ по формуле Эпли (вес × (1 + повторы/30)).")
    LV = [7000, int(sum_cur), 7500, 8000]
    ref_rows = []
    for k_, sp_ in ds.SFP_TESTS.items():
        arr_ = ds.NAR[k_][0]
        ref_rows.append({"Тест": sp_["label"], **{f"≈{L_}": round(float(np.interp(L_, ds.NAR_LEVELS, arr_)), 2) for L_ in LV}})
    ref_rows.append({"Тест": "КД-2", **{f"≈{L_}": round(float(np.interp(L_, ds.NAR_LEVELS, ds.NAR["KD2"][0])), 2) for L_ in LV}})
    st.markdown(f"**Какие значения соответствуют уровню** (колонка ≈{int(sum_cur)} — ваша сумма личных результатов; реальный турнир обычно ниже на ≈{loss_n})")
    st.dataframe(pd.DataFrame(ref_rows), use_container_width=True, hide_index=True)
    if not len(TEST_ASSESS):
        st.info("Ваших тестов Нариманова пока нет. Внесите их в «Сегодня → Моя тренировка»: «Тест: ядро 7,26 кг назад через голову», «Тест: тройной прыжок с места», «Тест: бег 30 м с хода», "
                "«Контрольный бег 1000 м», «Жим лёжа» (вес и повторы) — отсюда они подхватятся сами. Либо вручную: «⚙️ Данные и модель → 👤 Профиль» → «Тесты замерены лично».")
    else:
        rows = [{"Тест": r_.label, "Замер": round(r_.value, 2), "Источник": TEST_SRC.get(r_.test, ""), "Уровень по таблице": round(r_.level_test),
                 "Уровень связанных видов": round(r_.level_events), "Разница": f"{r_.gap:+.0f}", "Вывод": r_.verdict} for r_ in TEST_ASSESS.itertuples()]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        txt_ = []
        if kd2v is not None:
            txt_.append(f"КД-2 = **{kd2v:.2f}** (уровень 7500 = 45,50), по интерполяции таблицы ≈ **{interp_level(KD2_TAB, kd2v):.0f}**.")
        wk_ = TEST_ASSESS.sort_values("gap").iloc[0]
        txt_.append(f"Самый отстающий от результата тест: **{wk_.label}** ({wk_.gap:+.0f}).")
        if T_MULT:
            txt_.append("В плане из-за этого изменены акценты: " + ", ".join(f"{GL.get(g, g)} ×{m:.2f}" for g, m in T_MULT.items()) + ".")
        elif not use_tests_emph:
            txt_.append("Влияние на акценты плана выключено в настройках.")
        else:
            txt_.append("Расхождений ≥ 500 очков нет — акценты плана по тестам не менялись.")
        st.info(" ".join(txt_))
        st.caption("«Уровень связанных видов» = 7500 × очки вида / очки вида в моём условном профиле 7500 (мой перевод, не из литературы). Разница ≤ −500: рост результата упирается в физическую базу — "
                   "тянет сила/СФП; ≥ +500: база опережает результат — резерв в технике и реализации. Это ориентир для выбора акцентов, а не измерение потолка. "
                   "Тесты скорости и силы не говорят о технике.")

    st.markdown("#### Функция «тест → результат вида» (Грузенкин)")
    GR_ = ds.grz_ratios()
    LBL_ = {"SP_bwd": "Ядро: бросок назад → толчок", "SP_stand": "Ядро: толчок с места → толчок", "DT_stand": "Диск: бросок с места → диск"}
    st.dataframe(pd.DataFrame([{"Связь": LBL_[r_.pair], "До эксперимента": round(r_.ratio_before, 3), "После": round(r_.ratio_after, 3), "Среднее": round(r_.ratio, 3),
                                "Коридор": f"{r_.lo:.3f}…{r_.hi:.3f}"} for r_ in GR_.itertuples()]), use_container_width=True, hide_index=True)
    st.caption("Откуда: автореферат диссертации Грузенкина (1990). Табл. 1 — средние тесты группы (до эксперимента и лучшие за период), табл. 4 — личные рекорды шести спортсменов до эксперимента и лучшие результаты 1988 г. "
               "(суммы сверены с табл. 5: 7647,5 → 7859,0). Отношение = средний результат вида / средний тест, «лучший против лучшего» до и после. Оно почти не меняется "
               "(у ядра назад 0,89 → 0,91, у диска 1,12 → 1,13), поэтому это устойчивая связь для мастеров ≈7650–7860. Коридор ±4 % — МОЁ допущение: по двум точкам и шести людям разброс не оценить. "
               "Это групповые средние, а не ваша индивидуальная зависимость; вес снарядов в тестах в источнике не указан (принято 7,26 кг и 2 кг).")
    if len(TEST_RES):
        trr_ = [{"Тест": ds.THROW_TESTS[r_.test]["label"] if r_.test in ds.THROW_TESTS else ds.SFP_TESTS[r_.test]["label"], "Замер": round(r_.value, 2), "Ожидаемый результат вида": f"{r_.exp_lo:.2f}…{r_.exp_hi:.2f} (≈{r_.exp_mid:.2f})",
                 "Ваш результат": round(r_.current, 2), "Разница, м": f"{r_.d_m:+.2f}", "Разница, очков": f"{r_.d_pts:+.0f}",
                 "Вывод": ("результат ниже теста: резерв в технике и реализации" if r_.d_pts >= 50 else ("результат выше теста: упираетесь в силу" if r_.d_pts <= -50 else "соответствует"))}
                for r_ in TEST_RES.itertuples()]
        st.dataframe(pd.DataFrame(trr_), use_container_width=True, hide_index=True)
    else:
        st.info("Нужен хотя бы один замер из трёх: бросок ядра назад 7,26 кг, толчок ядра с места, бросок диска с места (категория «Контроль» / «Ядро» / «Диск» в «Моя тренировка»).")
    # ───── «Что если»: профиль силы и прыжков → результаты видов и сумма очков ─────
    st.markdown("#### «Что если»: профиль силы и прыжков → результаты видов и очки")
    st.caption("Введите значения тестов (0 — не учитывать). Модель покажет, какие результаты по десяти видам и какую сумму очков такой профиль обычно даёт, и сравнит с вашими. "
               "Чем больше тестов введёте, тем меньше в ответе «додуманного»: по видам без теста результат берётся из общего уровня и связей между видами.")
    WF_SPEC = {"snatch": (0.0, 160.0, 2.5), "bench": (0.0, 220.0, 2.5), "triple_sj": (0.0, 11.5, 0.05), "shot_bwd": (0.0, 22.0, 0.1),
               "stand_put": (0.0, 20.0, 0.1), "stand_dt": (0.0, 65.0, 0.5), "fly30": (0.0, 4.5, 0.01), "t1000": (0.0, 400.0, 1.0)}

    def _wf_fill():
        tv_ = dict(TEST_VALS)
        if tv_.get("vmax"):
            tv_["fly30"] = 30.0 / float(tv_["vmax"])
        for k_ in WF_SPEC:
            st.session_state["wf_" + k_] = float(tv_.get(k_, 0.0))

    def _wf_clear():
        for k_ in WF_SPEC:
            st.session_state["wf_" + k_] = 0.0

    wb1, wb2, _wb3 = st.columns([1, 1, 2])
    wb1.button("Подставить мои замеры из журнала", on_click=_wf_fill, key="wf_fill", disabled=not TEST_VALS)
    wb2.button("Очистить", on_click=_wf_clear, key="wf_clear")
    wcols_ = st.columns(4)
    WI = {}
    for i_, (k_, (lo_, hi_, stp_)) in enumerate(WF_SPEC.items()):
        WI[k_] = wcols_[i_ % 4].number_input(ds.PROFILE_TESTS[k_]["label"], lo_, hi_, 0.0, stp_, key="wf_" + k_)
    st.caption("Присед, прыжок с места и прыжок вверх есть только в матрице связей ниже: уровня «какому значению соответствует сколько очков» для них в доступных источниках нет "
               "(ни у Нариманова, ни у Грузенкина, ни у Рудских), поэтому в расчёт суммы они не входят. Рывок есть только у Грузенкина (групповые средние 82,6 → 87,6 кг), у Нариманова его нет.")
    WF = ds.profile_whatif({k_: v_ for k_, v_ in WI.items() if v_ > 0}, res, float(loss_n))
    for n_ in WF["notes"]:
        st.warning(n_)
    if not len(WF["events"]):
        _ex = ds.profile_whatif({"snatch": 90.0, "bench": 110.0, "triple_sj": 9.40}, res, float(loss_n))
        st.info(f"Пример: рывок 90 кг, жим 110 кг, тройной с места 9,40 м → уровень ≈ {_ex['level']:.0f}, сумма очков ≈ {_ex['sum_pts']:.0f} "
                f"(коридор {_ex['sum_lo']:.0f}…{_ex['sum_hi']:.0f}). Введите свои значения выше.")
    else:
        wm1, wm2, wm3, wm4 = st.columns(4)
        wm1.metric("Уровень по тестам", f"{WF['level']:.0f}", "средний по протестированным видам")
        wm2.metric("Сумма очков по профилю", f"{WF['sum_pts']:.0f}", f"коридор {WF['sum_lo']:.0f}…{WF['sum_hi']:.0f}", delta_color="off")
        wm3.metric("Ваш турнир (сумма личных − потери)", f"{WF['cur_tour']:.0f}", f"профиль {WF['d_vs_tour']:+.0f}")
        wm4.metric("Видов с тестом", f"{len(WF['tested'])} из 10")
        wt_ = WF["tests"]
        st.dataframe(pd.DataFrame([{"Тест": r_.label, "Значение": r_.value,
                                    "Уровень по Нариманову": ("—" if pd.isna(r_.nar) else f"{r_.nar:.0f}"), "Уровень по Грузенкину": ("—" if pd.isna(r_.grz) else f"{r_.grz:.0f}"),
                                    "Принятый уровень": ("прямая функция" if pd.isna(r_.level) else f"{r_.level:.0f}"),
                                    "Замечание": (f"вне диапазона источника ({r_.outside}) — экстраполяция" if r_.outside else "")} for r_ in wt_.itertuples()]),
                     use_container_width=True, hide_index=True)
        rows_ = []
        for r_ in WF["events"].itertuples():
            f_ = (lambda x: fmt_t(x)) if r_.event == "1500m" else (lambda x: f"{x:.2f}")
            rows_.append({"Вид": RU[r_.event], "Основа": ("тест" if r_.tested else "связи/уровень"), "Ожидаемый результат": f_(r_.result),
                          "Коридор": f"{f_(r_.res_lo)}…{f_(r_.res_hi)}", "Очки": int(r_.pts), "Ваш ЛР": f_(r_.current), "Очки ЛР": int(r_.cur_pts),
                          "Профиль − ЛР, очков": f"{r_.d_pts:+.0f}", "Лучшая r теста с видом": ("—" if pd.isna(r_.r_best) else f"{r_.r_best:.2f}"),
                          "Надёжность вида (R²)": ("—" if pd.isna(r_.K) else f"{r_.K:.2f}"), "На чём основано": r_.basis})
        st.dataframe(pd.DataFrame(rows_), use_container_width=True, hide_index=True)
        evw_ = WF["events"].set_index("event")
        tt_ = evw_[evw_.tested].sort_values("d_pts", ascending=False)
        if len(tt_):
            res_ = [f"{RU[e_]} {d_:+.0f}" for e_, d_ in tt_.d_pts.items() if d_ >= 50]
            low_ = [f"{RU[e_]} {d_:+.0f}" for e_, d_ in tt_.d_pts.items() if d_ <= -50]
            msg_ = []
            if res_:
                msg_.append("Результат ниже, чем такой тест «поддерживает» (резерв в технике и реализации, не в силе): " + ", ".join(res_) + ".")
            if low_:
                msg_.append("Результат выше, чем этот тест: " + ", ".join(low_) + " (упираетесь в тест — здесь прибавка силы или скорости нужна).")
            if not msg_:
                msg_.append("По видам с тестом ваши личные результаты совпадают с профилем в пределах ±50 очков.")
            msg_.append("По видам «связи/уровень» сравнение с ЛР — это сравнение с уровнем «сбалансированного» спортсмена: тестом оно не подтверждено.")
            st.info(" ".join(msg_))
        sens_ = ds.profile_sensitivity({k_: v_ for k_, v_ in WI.items() if v_ > 0}, res)
        if len(sens_):
            st.markdown("**Что даёт шаг в каждом тесте** (остальные без изменений)")
            st.dataframe(pd.DataFrame([{"Тест": r_.label, "Шаг": f"{r_.step:+g}", "Сумма очков": f"{r_.d_sum:+.0f}", "Уровень": f"{r_.d_level:+.0f}", "Главное по видам": r_.main}
                                       for r_ in sens_.itertuples()]), use_container_width=True, hide_index=True)
        st.caption("Результаты — «типичные для такого профиля», а не прогноз именно для вас: это среднее по спортсменам; ваш личный разрыв (техника, реализация, травмы) сюда не входит. "
                   "Сумма — турнирного типа (профиль 7500 в моей шкале), поэтому сравнивается с вашей суммой личных минус потери. Очки ЛР у вида обычно выше турнирных на ≈30, учитывайте это при сравнении.")

    with st.expander("Матрица связей «тест × вид»: что нашлось в литературе и как я это свёл"):
        MX_, KX_ = ds.test_event_matrix()
        RSUM_ = ds.test_sum_corr(MX_)
        mrows_ = []
        for t_ in MX_.index:
            row_ = {"Тест": ds.TEST_LABEL[t_]}
            for e_ in ds.EVENTS:
                v_, o_ = float(MX_.loc[t_, e_]), KX_.loc[t_, e_]
                row_[RU[e_]] = f"{v_:.2f}" if o_ == "прямая" else (f"≈{v_:.2f}" if o_ == "по аналогии" else f"({v_:.2f})")
            row_["с суммой"] = f"{RSUM_[t_]:.2f}"
            row_["В расчёте суммы"] = "да" if t_ in ds.PROFILE_TESTS else "только здесь"
            mrows_.append(row_)
        st.dataframe(pd.DataFrame(mrows_), use_container_width=True, hide_index=True)
        st.caption("Корреляция r: «выше результат теста → лучше результат вида» (у времён знак приведён к этой ориентации). "
                   "**0,73** — клетка опирается на найденное исследование (после усреднения нескольких работ и стягивания к слабой r = 0,30 с весом 15 наблюдений); "
                   "**≈0,50** — по аналогии (другой снаряд или другая группа: например, бросок ядра назад ← бросок назад у молота, бег 1000 м ← VO₂max); "
                   "**(0,38)** — моё вычисление «через вид»: r теста с его основным видом × связь этих двух видов у десятиборцев (arXiv 2026). Исследований «тест ↔ вид» "
                   "именно у десятиборцев найти не удалось: все прямые клетки — у метателей, прыгунов, спринтеров, студентов; у них диапазон результатов шире, чем у десятиборцев, "
                   "и r завышена (я ослабил её весом «пригодности», он мой). Колонка «с суммой» = Σ r по видам / √ΣΣ связей видов (моё допущение: одинаковый разброс очков).")
        SRC_ = ds.test_event_sources()
        PL_ = ds.test_event_pooled().set_index(["test", "event"])
        st.markdown("**Исходные корреляции по строкам**")
        st.dataframe(pd.DataFrame([{"Тест": r_.test_label, "Вид": RU[r_.event], "r в источнике": round(r_.r, 2), "Выборка, n": int(r_.n), "Пригодность (моя)": r_.rel,
                                    "Тип": r_.origin, "Итог клетки": round(float(PL_.loc[(r_.test, r_.event), "r"]), 2), "Источник": r_.src} for r_ in SRC_.itertuples()]),
                     use_container_width=True, hide_index=True)
        st.caption("Не прочитано: PLOS ONE 0295083 (сервер отказал по лимиту запросов), PMC и PubMed (капча) — из них ничего не взято. В статье Maeda 2024 текст и таблица расходятся для приседа "
                   "(.59 и .69), взята таблица. Присед → длина .906 — одна небольшая выборка студентов, значение сомнительно высокое (в итоге стянуто до ≈.65). "
                   "Единственные данные именно о десятиборцах — Романов 2018 (100 сильнейших мира): масса тела ↔ ядро .42, диск .36, сумма .29; это не тест, показано для справки.")
    with st.expander("Как связаны виды между собой (arXiv 2026, 1007 десятиборцев)"):
        cp_ = ds.corr_top_pairs(10)
        st.dataframe(pd.DataFrame([{"Пара": f"{RU[r_.a]} – {RU[r_.b]}", "Корреляция очков": round(r_.r, 2), "Прежний мой синтез": round(r_.r_old, 2)} for r_ in cp_.itertuples()]),
                     use_container_width=True, hide_index=True)
        st.caption(f"Таблица A7 препринта arXiv 2602.17043 (2026): {ds.LIT3_N_PERF} десятиборий {ds.LIT3_N_ATHLETES} спортсменов 2001–2022, не ниже 6400 очков. Это связи МЕЖДУ спортсменами "
                   "(у кого сильнее ядро, у того в среднем сильнее диск), а не то, как меняется один спортсмен; разброс по уровню класса в них смешан с «профилем». Эти связи используются "
                   "только для видов без теста и для клеток «через вид». Прежний синтез (Юшкевич–Романов, Олимпиады 1952–60, экспертная матрица) оказался слабее — он оставлен для Монте-Карло.")
    with st.expander("Проверка: как расчёт ведёт себя на группе Грузенкина"):
        PV_ = ds.profile_validation()
        tb_ = PV_["table"]
        st.dataframe(pd.DataFrame([{"Момент": r_.period, "Сумма группы (факт)": round(r_.sum_fact), "Расчёт без стягивания": round(r_.sum_raw), "Ошибка без стягивания": f"{r_.err_raw:+.0f}",
                                    "Расчёт со стягиванием": round(r_.sum_model), "Ошибка со стягиванием": f"{r_.err:+.0f}",
                                    "Ядро: факт / модель": f"{r_.SP_fact:.0f} / {r_.SP_model:.0f}", "Длина: факт / модель": f"{r_.LJ_fact:.0f} / {r_.LJ_model:.0f}",
                                    "Диск: факт / модель": f"{r_.DT_fact:.0f} / {r_.DT_model:.0f}", "Копьё: факт / модель": f"{r_.JT_fact:.0f} / {r_.JT_model:.0f}"} for r_ in tb_.itertuples()]),
                     use_container_width=True, hide_index=True)
        st.caption(f"Тесты группы (жим, ядро назад, тройной) подставлены в расчёт, как если бы это были ваши; только таблица Нариманова — линия Грузенкина по построению проходит через сумму группы и проверкой не служит. "
                   f"Без стягивания расчёт завышает отклонение от среднего (ошибки {tb_.err_raw.iloc[0]:+.0f} и {tb_.err_raw.iloc[1]:+.0f}): таблица Нариманова читается «в обратную сторону». "
                   f"Со стягиванием к 7500 ошибки {tb_.err.iloc[0]:+.0f} и {tb_.err.iloc[1]:+.0f} — немного занижено. Коэффициент стягивания, нужный группе (по двум точкам): {PV_['k_emp']:.2f}; "
                   f"получающийся из литературных корреляций: {PV_['k_lit']:.2f}. Они одного порядка, но не совпадают: это проверка порядка величины, а не подгонка (две точки одной группы из шести человек). "
                   f"Среднеквадратичная ошибка одного теста по Нариманову ≈ {PV_['sig_level']:.0f} очков суммы — отсюда ширина коридора. Разброс очков одного вида внутри группы ≈ {PV_['sig_event']:.0f} "
                   f"(шесть спортсменов, четыре вида).")
    with st.expander("Что здесь из источников, а что моё"):
        st.markdown(
            "**Из источников:** таблица Нариманова (какое значение жима, тройного, ядра назад, 1000 м соответствует какой сумме очков); Грузенкин (1990) — средние тесты группы мастеров в двух точках "
            "(рывок, жим, ядро назад, толчок и диск с места, тройной, 30 м с хода ↔ сумма 7647,5 → 7859,0) и отношения «результат вида / тест»; связи между видами — arXiv 2026 (1007 десятиборцев); "
            "связи тестов с результатом вида — Judge & Bellar 2012 (ядро), Maeda 2024 (молот), Abd Rahim 2020, Reis 2006, Zhou & Kang (длина), Loturco 2015, Mackala 2015, Almuzaini 2008, Healy 2022 (100 м), "
            "Guthrie 2021 (связь тестов между собой).\n\n"
            "**Моё:** (1) профиль «7500» как способ перевести уровень в результат вида; (2) линия Грузенкина как «значение теста → уровень» (это динамика одной группы за сезон, не срез по спортсменам — "
            "вне диапазона группы, для рывка 82,6…87,6 кг, это экстраполяция); (3) вес пригодности выборки, стягивание клеток к r = 0,30, корреляция между тестами 0,55, порог связи r ≥ 0,35; "
            "(4) клетки «по аналогии» и «через вид»; (5) стягивание оценки вида к уровню 7500 с коэффициентом R² (доля дисперсии вида, объяснимая тестами); (6) перенос отклонений на виды без теста "
            "при одинаковом разбросе очков; (7) ширина коридора.\n\n"
            "**Чего нет:** исследования «рывок/жим/прыжок → результат вида» у десятиборцев и индивидуальной зависимости. У Грузенкина тесты даны только средними по группе (6–8 человек), в таблице Рудских силовых тестов нет, "
            "у Нариманова нет рывка и приседа. Поэтому рывок здесь — показатель общего уровня силы с корреляцией, перенесённой от метателей, а не доказанный предсказатель конкретного вида.\n\n"
            "Бег 30 м с хода берётся только по Грузенкину: «Vmax» Нариманова измерен иначе (30 м / время даёт по его таблице для той же группы ≈6000–6800 вместо 7650–7860).")

    st.markdown("#### Потенциал к старту десятиборья")
    H_G = max(0, (d_dec - real_today).days)
    cg1, cg2 = st.columns(2)
    pause_d = cg1.number_input("Дней без тренировок до старта", min_value=0, max_value=max(H_G, 1), value=int(min(GAP_DAYS, H_G)), step=1, key="pot_pause",
                               help="По умолчанию — текущая пауза в журнале. Двигайте: сколько ещё дней вы не будете тренироваться.")
    plan_sh = cg2.slider("Доля плана, которую выполню", 0.5, 1.0, 1.0, 0.05, key="pot_share")
    GF = ds.growth_forecast(res, H_G, int(pause_d), float(plan_sh), float(loss_n), STRETCH_UI, TEST_ASSESS if len(TEST_ASSESS) else None, FGAP)
    GF0 = ds.growth_forecast(res, H_G, 0, float(plan_sh), float(loss_n), STRETCH_UI, TEST_ASSESS if len(TEST_ASSESS) else None, FGAP)
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Норма роста на срок (Хейнла)", f"{GF['budget'][0]:.0f}…{GF['budget'][1]:.0f}", f"{H_G} дн. до старта")
    m2.metric("Сумма личных к старту", f"{GF['sum_lo']:.0f}…{GF['sum_hi']:.0f}", f"сейчас {GF['sum_pr']:.0f}")
    m3.metric("Турнир (с потерей)", f"{GF['tour_lo']:.0f}…{GF['tour_hi']:.0f}", f"−{loss_n}")
    m4.metric("Недобор из-за пауз/невыполнения", f"{GF0['sum_hi'] - GF['sum_hi']:.0f}", f"доля времени {GF['share']:.0%}")
    tg = GF["table"].copy()
    tg["Вид"] = tg["event"].map(RU)
    tg["Ожидаемый результат"] = [(f"{min(a_, b_):.2f}" if f"{a_:.2f}" == f"{b_:.2f}" else f"{min(a_, b_):.2f}…{max(a_, b_):.2f}") for a_, b_ in zip(tg["result_lo"], tg["result_hi"])]
    tg["+ очков"] = [f"{a_:.0f}…{b_:.0f}" for a_, b_ in zip(tg["gain_lo"], tg["gain_hi"])]
    st.dataframe(tg.rename(columns={"current": "Сейчас", "pts_now": "Очки", "reserve": "Запас до потолка"})[["Вид", "Сейчас", "Очки", "Ожидаемый результат", "+ очков", "Запас до потолка"]],
                 use_container_width=True, hide_index=True)
    st.caption("Величина роста — норма Хейнлы для вашего класса, пересчитанная на срок (≈300–325 очков за сезон до 7300), и она подходит только при нормальной подготовке. "
               "МОИ допущения: рост уменьшается линейно с долей времени без тренировок и невыполненной частью плана; распределяется по видам пропорционально запасу до «потолка» "
               "(потолки — мои, правятся в «Цель → результаты»), у видов с запасом в технике вес ×1,15, где сила отстаёт — ×0,85. Это ожидание, а не гарантия: разброс между спортсменами большой "
               "(SD прироста в литературе ≈ 190 очков за 2 сезона). От формы на день старта (вкладка «Прогноз») это отдельно: там рассчитана форма относительно рекордов.")
    ctrl_ = [d_ for d_ in ("2026-10-14", "2026-12-23", "2027-02-17", "2027-05-12") if pd.Timestamp(d_) >= real_today]
    if TEST_VALS and ctrl_:
        TT = ds.test_targets(TEST_VALS, real_today, real_today, d_hept, ctrl_)
        if len(TT):
            piv = TT.pivot(index="label", columns="date", values="target").round(2)
            piv.insert(0, "Сейчас", TT.groupby("label")["now"].first().round(2))
            piv.columns = [(f"{c_[8:10]}.{c_[5:7]}" if c_ != "Сейчас" else c_) for c_ in piv.columns]
            st.markdown("**Ориентиры на контрольные дни**")
            st.dataframe(piv, use_container_width=True)
            st.caption("Рост от сегодняшнего замера до конца подготовительного периода (старт семиборья) на ту же долю, на какую выросла средняя группа Грузенкина "
                       "(ядро назад +5,3 %, тройной +4,2 %, жим +12 %, Vmax +3,3 %); линейность — моё допущение. Группа сильнее вас — для вас это ориентир, не норма.")
    st.markdown("#### Динамика тестов за подготовительный период (Грузенкин, средняя группа МС СССР)")
    gz = pd.DataFrame([
        {"Тест": "Рывок штанги (кг)", "Начало": 80.3, "Конец": 87.2}, {"Тест": "Жим лёжа (кг)", "Начало": 106.8, "Конец": 119.6},
        {"Тест": "Бросок ядра назад (м)", "Начало": 15.18, "Конец": 15.99}, {"Тест": "Тройной с места (м)", "Начало": 9.07, "Конец": 9.45},
        {"Тест": "Прыжок с места (м)", "Начало": 2.94, "Конец": 3.07}, {"Тест": "Бег 30 м с хода (с)", "Начало": 3.16, "Конец": 3.06},
        {"Тест": "Толчок ядра с места (м)", "Начало": 12.82, "Конец": 13.45}, {"Тест": "Бросок диска с места (м)", "Начало": 36.85, "Конец": 38.34}])
    gz["Рост, %"] = ((gz["Конец"] / gz["Начало"] - 1) * 100).round(1)
    st.dataframe(gz, use_container_width=True, hide_index=True)
    st.caption("В плане по умолчанию четыре дня контрольных испытаний: 14.10.2026, 23.12.2026, 17.02.2027, 12.05.2027 (идея Грузенкина: тесты в начале и конце этапов). "
               "Единицы «толчок ядра с места» и «диск с места» в скане неоднозначны — ориентируйтесь на проценты, не на абсолюты.")

# ───────────────────────── модель ─────────────────────────
with tabs[10]:
    st.subheader("Что от чего зависит (как устроен пересчёт)")
    st.dataframe(pd.DataFrame([
        ["Результаты по видам", "очки, прогноз модели, Монте-Карло, чувствительность, обратная задача, профили; АКЦЕНТЫ ПЛАНА (какие средства получают больше занятий и объёма)"],
        ["«Потолки» (вкладка «Цель → результаты»)", "обратная задача и акценты плана"],
        ["Индивидуализация, потолок кросса, масштаб объёма, даты", "состав, число занятий и объём плана"],
        ["Полураспады, τ, K1, K2, насыщение", "прогноз по плану и пометки шлюзов в плане; состав плана не меняют"],
        ["Вес", "штраф 1500 м во 2-й день в модели; КД-2"],
        ["Давность замеров", "усадка профиля (сколько данные «пробивают» популяцию)"],
        ["Сон и боль (вкладка «Сегодня»)", "облегчённый вариант на сегодня и допуск при анализе тренировки (пороги условные)"],
        ["Записи журнала", "нагрузка, ACWR, выполнение плана; ДОБОР недовыполненного в плане; состояние пулов и шлюзы от реального журнала; прогноз стартует от реального состояния; подбор T½ ЦНС"],
        ["Потери «личные − турнир»", "консервативная оценка, вкладки «Хейнла» и «Цель → результаты»"]], columns=["Параметр", "Что пересчитывается"]),
        use_container_width=True, hide_index=True)
    with st.expander("Как в модели считается «утомление ЦНС»"):
        st.markdown(ds.CNS_EXPLAIN)
    st.subheader("Характеристики модели (то, что вы двигаете в «⚙️ Данные и модель → 🧠 Модель»)")
    hl = pd.DataFrame({"Пул": list(F["half_life"].keys()), "T½ сейчас, ч": [round(v, 1) for v in F["half_life"].values()],
                       "T½ старт, ч": [round(v, 1) for v in F0["half_life"].values()]})
    st.dataframe(hl, use_container_width=True, hide_index=True)
    tau = pd.DataFrame({"Канал": [tau_names[c] for c in F["tau1"]], "τ фитнеса сейчас, сут.": [round(v, 1) for v in F["tau1"].values()],
                        "τ старт, сут.": [round(v, 1) for v in F0["tau1"].values()]})
    st.dataframe(tau, use_container_width=True, hide_index=True)
    st.metric("Порог ЦНС для шлюза (условный)", f"{F['cns_thr']:.3f}", f"{F['cns_thr'] - F0['cns_thr']:+.3f} к старту")
    st.markdown("#### Сколько усталости остаётся через N часов")
    rt = ds.recovery_table(model.const, (12, 24, 48, 72, 96))
    st.dataframe(rt, use_container_width=True)
    st.caption("Эти числа не измерены на вас. Диапазоны полураспадов взяты из ТЗ; K1, K2, насыщение — заглушки. Как только будет журнал тренировок с тестами, "
               "decathlon_suite.py умеет подогнать их (Calibrator), и тогда эта вкладка станет вашей, а не условной.")

# ───────────────────────── история срезов ─────────────────────────
with tabs[11]:
    st.subheader("История результатов")
    st.caption("Нажмите кнопку, когда обновили результаты: срез запомнится с датой (повторное нажатие в тот же день заменяет срез). "
               "По срезам видно, как менялась сумма личных результатов.")
    if st.button("📌 Сохранить срез результатов сегодня", type="primary"):
        snap = {"date": str(real_today.date()), "res": {e: float(res[e]) for e in EV}, "hep": {k: float(v) for k, v in hep.items()},
                "sum": int(sum_cur), "sum_hep": int(sum_hep), "mass": float(ath_mass), "forecast": float(F["sum_forecast"])}
        hist = [h for h in STATE.get("history", []) if h.get("date") != snap["date"]] + [snap]
        commit(dict(STATE, history=sorted(hist, key=lambda h: h["date"])), f"Срез результатов за {real_today:%d.%m} сохранён")
    H = STATE.get("history", [])
    if not H:
        st.info("Срезов пока нет.")
    else:
        hdf = pd.DataFrame([{"Дата": h["date"], "Сумма 10 видов": h["sum"], "Прогноз модели": h.get("forecast"), "Семиборье": h.get("sum_hep"),
                             "Вес": h.get("mass"), **{RU[e]: h["res"].get(e) for e in EV}} for h in H])
        st.dataframe(hdf, use_container_width=True, hide_index=True)
        if len(hdf) > 1:
            st.line_chart(hdf.set_index("Дата")[["Сумма 10 видов"]])
        if st.button("Удалить последний срез"):
            commit(dict(STATE, history=H[:-1]), "Последний срез удалён")

# ───────────────────────── журнал ─────────────────────────
with tabs[1]:
    st.subheader(f"📝 Запись дня: {cur_date:%d.%m.%Y}")
    st.caption("День выбирается вверху страницы (стрелки «◀ день» / «день ▶» или календарь). Здесь — чек-ин (сон, пульс, HRV…), заметка и запись без каталога упражнений. "
               "Тренировку с упражнениями и результатами удобнее собирать во вкладке «🏠 Сегодня». Дни без записи модель считает днями отдыха.")
    _rec_days = sorted({e["date"] for e in LOG}, reverse=True)[:7]
    if _rec_days:
        st.markdown("**Недавно записанные дни** — нажмите, чтобы открыть:")
        _rc = st.columns(len(_rec_days))
        for _i, _d in enumerate(_rec_days):
            _rc[_i].button(("✅ " if _d != str(cur_date.date()) else "▶ ") + pd.Timestamp(_d).strftime("%d.%m"), on_click=_goto, args=(_d,), key=f"j_open_{_d}", use_container_width=True)
    jd = cur_date.date()
    j_can = cur_date <= real_today
    if not j_can:
        st.info("Это будущий день: записать можно только сегодняшний и прошедшие дни.")
    iso = str(jd)
    ex = LOGD.get(iso)
    prow = df_base[df_base["date"] == pd.Timestamp(jd)] if df_base is not None else pd.DataFrame()
    planned = ds._json_loads(prow.iloc[0]["vol"]) if len(prow) else {}
    if len(prow):
        st.info("План на этот день: " + str(prow.iloc[0]["plan"]))
    rest_ = st.checkbox("День без тренировки (отдых / пропуск)", value=bool(ex and ex.get("rest")), key=f"j_rest_{iso}")
    dflt = list((ex or {}).get("groups", {}).keys()) or [g for g in planned if g in ds.LOG_GROUPS]
    sel = st.multiselect("Что делали", list(ds.LOG_GROUPS), default=dflt, key=f"j_sel_{iso}",
                         format_func=lambda g: f"{ds.LOG_GROUPS[g][0]} ({ds.LOG_GROUPS[g][1]})")
    vols = {}
    if not rest_:
        cols = st.columns(3)
        for i_, g in enumerate(sel):
            lab, unit, _k = ds.LOG_GROUPS[g]
            dv = float((ex or {}).get("groups", {}).get(g, planned.get(g, 0.0)) or 0.0)
            vols[g] = cols[i_ % 3].number_input(f"{lab}, {unit}", min_value=0.0, value=dv, step=(0.5 if unit in ("км", "т") else 1.0), key=f"j_{iso}_{g}")
    c1, c2 = st.columns(2)
    minutes = c1.number_input("Длительность всего, мин", min_value=0, value=int((ex or {}).get("minutes") or 90), step=5, key=f"j_min_{iso}")
    rpe = c2.slider("Тяжесть занятия (RPE 1–10)", 1, 10, int((ex or {}).get("rpe") or 6), key=f"j_rpe_{iso}")
    with st.expander("Утренний чек-ин (необязательно, но нужен для оценки готовности и подбора T½ ЦНС)"):
        g1, g2, g3 = st.columns(3)
        sleep_h = g1.number_input("Сон, ч", min_value=0.0, max_value=14.0, value=float((ex or {}).get("sleep_h") or 0.0), step=0.1, key=f"j_sl_{iso}", help="0 = не вводил")
        sore_ = g2.number_input("Боль в мышцах/связках 1–10", min_value=0, max_value=10, value=int((ex or {}).get("sore") or 0), key=f"j_so_{iso}", help="0 = не вводил")
        rhr_ = g3.number_input("Пульс покоя, уд/мин", min_value=0, max_value=120, value=int((ex or {}).get("rhr") or 0), key=f"j_rh_{iso}")
        h1, h2, h3 = st.columns(3)
        hrv_ = h1.number_input("HRV (RMSSD), мс", min_value=0.0, max_value=300.0, value=float((ex or {}).get("hrv") or 0.0), step=1.0, key=f"j_hv_{iso}")
        cmj_ = h2.number_input("Прыжок CMJ, см", min_value=0.0, max_value=100.0, value=float((ex or {}).get("cmj") or 0.0), step=0.5, key=f"j_cm_{iso}")
        mass_ = h3.number_input("Вес утром, кг", min_value=0.0, max_value=150.0, value=float((ex or {}).get("mass") or 0.0), step=0.1, key=f"j_ms_{iso}")
    note_ = st.text_input("Заметка", value=(ex or {}).get("notes", ""), key=f"j_nt_{iso}")
    b1, b2 = st.columns(2)
    if ex and ex.get("items"):
        st.info("Этот день записан через конструктор («🏠 Сегодня»): упражнения и результаты редактируйте там. Здесь сохраняются только чек-ин и заметка.")
    if b1.button("💾 Сохранить день", type="primary", disabled=not j_can, key="j_save", use_container_width=True):
        rec = {"date": iso, "rest": bool(rest_), "groups": {g: float(v) for g, v in vols.items() if v > 0}, "minutes": int(minutes), "rpe": int(rpe),
               "sleep_h": sleep_h or None, "sore": sore_ or None, "rhr": rhr_ or None, "hrv": hrv_ or None, "cmj": cmj_ or None, "mass": mass_ or None, "notes": note_}
        if ex and ex.get("items"):
            rec.update({k: ex[k] for k in ("items", "groups", "minutes", "rpe", "rpe_ovr", "rest") if k in ex})
        newlog = sorted([e for e in LOG if e["date"] != iso] + [rec], key=lambda e: e["date"])
        commit(dict(STATE, log=newlog), f"День {jd:%d.%m} сохранён")
    if ex:
        _conf = b2.checkbox("Подтверждаю удаление записи", key=f"j_delok_{iso}")
        if b2.button("🗑 Удалить запись за эту дату", disabled=not _conf, key="j_del", use_container_width=True):
            commit(dict(STATE, log=[e for e in LOG if e["date"] != iso]), f"Запись за {jd:%d.%m} удалена")
    if LOG:
        st.markdown("#### Последние записи")
        auby = ds.log_daily_au(LOG, real_today).assign(d=lambda x: x["date"].dt.strftime("%Y-%m-%d")).set_index("d")["AU"]
        rows = [{"Дата": e["date"], "Что делали": ("отдых" if e.get("rest") else ", ".join(f"{ds.LOG_GROUPS[g][0]} {v:g} {ds.LOG_GROUPS[g][1]}" for g, v in (e.get("groups") or {}).items() if g in ds.LOG_GROUPS)),
                 "Мин": e.get("minutes"), "RPE": e.get("rpe"), "Нагрузка (мин×RPE)": round(float(auby.get(e["date"], 0.0))), "Сон": e.get("sleep_h"), "CMJ": e.get("cmj"),
                 "Заметка": e.get("notes", "")} for e in sorted(LOG, key=lambda e: e["date"], reverse=True)[:20]]
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

# ───────────────────────── факт и анализ ─────────────────────────
with tabs[2]:
    st.subheader("📊 Что вы проделали и что это значит")
    with st.expander("Как в модели считается «утомление ЦНС» (честно: что это и чем не является)"):
        st.markdown(ds.CNS_EXPLAIN)
    if not LOG or "daily" not in AN:
        st.info("Журнал пуст. Внесите хотя бы несколько дней: «🏠 Сегодня» или «📓 Дневник → Запись дня». Анализ нагрузки заработает с 14 дней, состояние по модели — с 7, подбор T½ ЦНС — с 21 дня чек-инов.")
    else:
        a1_, a2_, a3_, a4_ = st.columns(4)
        a1_.metric("Нагрузка за 7 дней (мин×RPE)", f"{AN.get('week_au', 0):.0f}")
        a2_.metric("Острая/хроническая (ACWR)", f"{AN['acwr']:.2f}" if AN.get("acwr") else "нужно ≥ 14 дней", "привычная зона 0,8–1,3 — ориентир")
        a3_.metric("Монотонность недели", f"{AN['monotony']:.1f}" if AN.get("monotony") else "—", "> 2 — однообразно (Фостер)")
        a4_.metric("Записанных дней", f"{len(LOG)}")
        for t_ in AN.get("advice", []):
            st.markdown(f"- {t_}")
        day = AN["daily"].copy()
        if len(day) > 1:
            day["острая (7 дн.)"] = pd.Series(day["AU"]).ewm(alpha=2 / 8, adjust=False).mean().values
            day["хроническая (28 дн.)"] = pd.Series(day["AU"]).ewm(alpha=2 / 29, adjust=False).mean().values
            st.line_chart(day.set_index("date")[["острая (7 дн.)", "хроническая (28 дн.)"]])
        wk = AN["weekly"]
        if len(wk):
            st.bar_chart(wk.set_index("week")[["AU"]])
        st.markdown("#### План и факт (только по дням, где есть запись)")
        comp = AN.get("compliance")
        if comp is not None and len(comp):
            cshow = comp.rename(columns={"label": "Средство", "unit": "Ед.", "planned_sessions": "Занятий по плану", "done_sessions": "Сделано занятий",
                                         "planned_vol": "Объём по плану", "done_vol": "Сделано", "pct_vol": "% объёма"})
            st.dataframe(cshow[["Средство", "Ед.", "Занятий по плану", "Сделано занятий", "Объём по плану", "Сделано", "% объёма"]].round(1), use_container_width=True, hide_index=True)
            if DEF:
                st.info("Добор включён: " + ", ".join(f"{ds.LOG_GROUPS.get(g, (g,))[0]} ×{v:.2f}" for g, v in sorted(DEF.items()) if v > 1.0) + " — будущий план усилен по этим средствам.")
        else:
            st.caption("Совпадений записанных дней с планом пока нет.")
        st.markdown("#### Состояние по журналу (модельный индекс, не измерение)")
        if AN.get("pools"):
            pc1, pc2, pc3 = st.columns(3)
            pc1.metric("Индекс ЦНС", f"{AN['pools']['CNS']:.2f}", f"порог шлюза {AN['cns_thr']:.2f}")
            pc2.metric("Метаболический индекс", f"{AN['pools']['MET']:.2f}")
            pc3.metric("Механический индекс", f"{AN['pools']['MECH']:.2f}")
            g_ = AN["gates"][["title", "allowed", "reason"]].rename(columns={"title": "Сессия", "allowed": "Можно сегодня", "reason": "Почему нет"})
            st.dataframe(g_, use_container_width=True, hide_index=True)
        else:
            st.caption("Нужно ≥ 7 дней записей с тренировками.")
        st.markdown("#### Самочувствие против вашей нормы")
        st.dataframe(AN["readiness"].rename(columns={"metric": "Показатель", "last": "Последний", "base": "Ваша норма (28 зап.)", "z": "z", "signal": "Сигнал"}),
                     use_container_width=True, hide_index=True)
        fit = AN.get("cns_fit")
        st.markdown("#### Подбор периода полураспада ЦНС по вашим данным")
        if fit is None:
            st.caption("Нужно ≥ 21 день, где есть CMJ, HRV или пульс покоя. Пока T½ ЦНС — заглушка (36 ч).")
        else:
            best = fit.loc[fit["r"].idxmax()]
            st.line_chart(fit.set_index("half_life_h")[["r"]])
            st.info(f"Лучшее совпадение индекса ЦНС с самочувствием при T½ ≈ {best.half_life_h:.0f} ч (r = {best.r:.2f}, n = {int(best.n)}). "
                    + ("Корреляция слабая (|r| < 0,3): не подгоняйте ползунок под это значение." if abs(best.r) < 0.3 else "Корреляция заметная, но это не причинность; можно сдвинуть ползунок T½ ЦНС к этому значению и посмотреть эффект."))

with tabs[2]:
    st.markdown("---")
    st.markdown("#### 📈 Форма по видам (по результатам тренировок)")
    if not len(EF):
        st.caption("Нужно ≥ 3 занятий с результатами одного упражнения за последние 8 недель (разброс по датам ≥ 10 дней). Тогда здесь появится тренд по видам.")
    else:
        st.dataframe(EF.drop(columns=["event"]), use_container_width=True, hide_index=True)
        with st.expander("Тренды по упражнениям"):
            st.dataframe(TR.drop(columns=["key", "id", "tags", "kind_cat"]), use_container_width=True, hide_index=True)
        notes_ = []
        if FORMX:
            notes_.append("в плане усилены: " + ", ".join(f"{GL.get(g, g)} ×{v:.2f}" for g, v in FORMX.items()))
        if use_adj:
            notes_.append("прогноз сдвинут по тренду (k=" + f"{adj_k:g}" + "): " + ", ".join(f"{e} {res[e]:.2f}→{RES_F[e]:.2f}" for e in res if abs(RES_F[e] - res[e]) > 1e-9))
        st.caption("Тренд — наклон Тейла–Сена лучших результатов за 8 недель, % за 4 недели (плюс = лучше). Порог «растёт/падает» ±0,8 % — мой; при 3–4 точках это ориентир, не вывод. "
                   + ("; ".join(notes_) if notes_ else "Учёт в плане и прогнозе отключён в «⚙️ Данные и модель → 🔧 Правила плана»."))
    st.markdown("#### 🏅 Результаты на тренировках (секунды, метры, высота)")
    _pcs = ds.pr_candidates(LOG, res)
    if len(_pcs):
        st.markdown("**Лучше результата в вашем профиле** — только полные виды с соревновательным снарядом (метание с места, тесты и силовые сюда не входят):")
        render_pr(_pcs, "dn")
    else:
        st.caption("Результатов полных видов, лучших вашего профиля, в журнале нет. Сюда попадают только полные броски/прыжки/забеги с соревновательным снарядом; "
                   "метание с места, тесты и силовые связаны с результатом вида иначе — их прямо в профиль не переношу.")
    _pr = ds.pair_ratio(LOG, res)
    if len(_pr):
        st.markdown("**С места → полный: ваши пары** (простое упражнение и полный бросок/прыжок того же вида в пределах 3 дней)")
        _pt = _pr.copy()
        _pt["Вид"] = _pt["event"].map(RU)
        _pt["Вывод"] = [("нужно ≥ 2 пар с полным результатом — перевести нельзя" if e_ is None or pd.isna(e_) else
                         (f"≈ {e_:.2f} м полного — выше вашего {p_:.2f}: проверьте полным броском" if e_ > p_ else f"≈ {e_:.2f} м полного — ниже вашего {p_:.2f}"))
                        for e_, p_ in zip(_pt["estimate"], _pt["pr"])]
        st.dataframe(_pt.rename(columns={"exercise": "Упражнение", "pairs": "Пар", "ratio": "Простое / полное", "last_date": "Последний раз", "last_value": "Результат"})
                     [["Вид", "Упражнение", "Пар", "Простое / полное", "Последний раз", "Результат", "Вывод"]].round(3), use_container_width=True, hide_index=True)
        st.caption("Коэффициент считается по ВАШИМ парам (медиана отношений), чужих коэффициентов из литературы я не подставляю: у разных людей он разный. Оценка полного — "
                   "ориентир для решения, в профиль и прогноз сама не попадает.")
    PBT = ds.pb_table(LOG)
    if PBT.empty:
        st.caption("Пока нет записей с результатами: в «Сегодня» → «Моя тренировка» введите время/дальность/высоту по попыткам и запишите день.")
    else:
        st.dataframe(PBT.drop(columns=["key"]), use_container_width=True, hide_index=True)
        pick_k = st.selectbox("Динамика упражнения", list(PBT["key"]), format_func=lambda k: PBT.set_index("key").loc[k, "Упражнение"], key="res_pick")
        H_ = ds.result_history(LOG).get(pick_k, [])
        if H_:
            dfh = pd.DataFrame({"Лучшая попытка": [p_["best"] for p_ in H_], "Среднее": [p_["mean"] for p_ in H_]}, index=[p_["date"] for p_ in H_])
            st.line_chart(dfh)
            unit_ = H_[-1]["unit"]
            st.caption(f"Единица: {unit_}. " + ("Для времени ниже = лучше." if H_[-1]["kind"] == "time" else "Выше = лучше."))
        links = [("Тройной с места (м)", t_triple, ["ts_triple", "lj_triple_st"], "max"), ("Бросок ядра назад (м)", t_shot, ["ts_sp_back", "ts_shot", "sp_back"], "max"),
                 ("Бег 1000 м (с)", t_1000, ["en_test1000"], "min")]
        rows_ = []
        for lab_, cur_, ids_, how_ in links:
            vals_ = [p_["best"] for k_, lst_ in ds.result_history(LOG).items() for p_ in lst_ if lst_[-1]["id"] in ids_]
            if vals_:
                rows_.append({"Тест": lab_, "Сейчас в «Тестах СФП»": cur_, "Лучший в тренировках": (max(vals_) if how_ == "max" else min(vals_))})
        if rows_:
            st.markdown("**Сопоставление с тестами профиля** (в профиль вручную, если это был контрольный максимум, а не рабочий подход)")
            st.dataframe(pd.DataFrame(rows_), use_container_width=True, hide_index=True)

# ───────────────────────── связи между видами и часы подготовки ─────────────────────────
with tabs[12]:
    st.subheader("🔗 Связи между видами")
    st.caption("Источники: (1) Табл. 1 из статьи Юшкевича и Романова (Мир спорта № 3, 2012) — сильнейшие десятиборцы мира, n ≈ 90 (из порога «достоверны выше 0,21»); "
               "(2) Олимпиады 1952–60 (из журнала «Лёгкая атлетика»), n ≈ 57; (3) прежняя матрица из ТЗ и экспертных оценок. Всё переведено в корреляции ОЧКОВ (у времён знак меняется).")
    Csy, wts = ds.synth_corr(return_weights=True)
    st.info(f"Синтез: вес 2012 = {wts[0]:.0%}, 1952–60 = {wts[1]:.0%}, ТЗ/эксперт = {wts[2]:.0%}. Вес = (n − 3) × пригодность; пригодность 1952–60 принята 0,25 (моё допущение): "
            "у них R сочетаний с суммой очков 0,77–0,98 против 0,55–0,75 в 2012 — разброс уровня шире, а для разброса результатов ОДНОГО спортсмена в турнире нужна корреляция при узком диапазоне уровня. "
            f"Сейчас в Монте-Карло: **{corr_label}**.")
    cm = pd.DataFrame(CORR_M, index=EV, columns=EV).round(2)
    st.markdown("**Матрица, используемая сейчас (корреляции очков)**")
    st.dataframe(cm, use_container_width=True)
    CC = ds.corr_compare()
    allp = st.checkbox("Показать все 45 пар", value=False, key="cc_all")
    st.markdown("**Где источники расходятся сильнее всего**")
    st.dataframe((CC if allp else CC.head(12)).drop(columns=["a", "b"]), use_container_width=True, hide_index=True)
    st.markdown("#### Сверка: опубликованные R (Табл. 2 и 3) против матриц")
    CR = ds.corr_report()
    st.dataframe(CR.drop(columns=["event"]), use_container_width=True, hide_index=True)
    st.caption("R — множественная корреляция суммы очков с результатами в сочетании видов. «Табл. 1 (2012)» воспроизводит опубликованные R с точностью ±0,005 для 11 из 12 сочетаний — это независимая проверка "
               "моей переписи таблицы (проверены ≈20 клеток). Старая матрица даёт R 0,77–0,98: это другая, гораздо более разнородная выборка. Синтез завышает R на 0,03–0,07 — цена смешивания.")
    st.warning("Расхождение в последней строке Табл. 3: «400 м – высота – шест» = 0,568 по тексту, но по Табл. 1 это сочетание даёт 0,44, а «400 м – высота – диск» даёт 0,566. "
               "Скорее всего, в подписи вместо «диск» стоит «шест» — проверьте по оригиналу. В плане это сочетание не используется.")
    with st.expander("Какие пары и тройки видов использует план (веса из R)"):
        rows_pt = [{"Сочетание": " + ".join(GL.get(g, g) for g in k), "R": R, "Вес": ds.GRZ_PAIRS[frozenset(k)]} for k, R in ds.GRZ_R_PAIRS.items()]
        rows_pt += [{"Сочетание": " + ".join(GL.get(g, g) for g in k), "R": R, "Вес": [w for ks, w in ds.GRZ_TRIPLES if ks == frozenset(k)][0]} for k, R in ds.GRZ_R_TRIPLES.items()]
        st.dataframe(pd.DataFrame(rows_pt), use_container_width=True, hide_index=True)
    st.markdown("---")
    st.subheader("⏱ Часы подготовки и закон убывающей отдачи")
    st.caption("Рис. 1 той же статьи: прирост результата в прыжках у десятиборцев от часов подготовки в этом виде, y = α·ᵏ√t. На графике подписано y = 60·t^0,4, но это расходится с текстом статьи "
               "(первые 50 ч ≈ 250 очков, следующие 50 ч ≈ +100) и с точками графика; поэтому по умолчанию — подгонка по тексту и точкам.")
    st.dataframe(ds.hours_fit_check(), use_container_width=True, hide_index=True)
    tt = np.arange(0, 221, 5)
    curve = pd.DataFrame({"подписанная 60·t^0,4": ds.hours_gain(tt, **ds.HOURS_PUBLISHED), f"подгонка {ds.HOURS_LAW['a']:.0f}·t^{ds.HOURS_LAW['p']:.2f}": ds.hours_gain(tt, **ds.HOURS_LAW)}, index=tt)
    st.line_chart(curve)
    st.info("Почему закон не применён в абсолютном виде: он даёт 150–490 очков на вид за 50–200 часов, а нормы Хейнлы для вашего класса — около 600–650 очков за 2 сезона на ВСЕ виды (в 4–8 раз меньше). "
            "Статья описывает рост «с низкого уровня» в прыжках; для других видов и вашего уровня масштаб не опубликован. Поэтому из закона я беру только ФОРМУ (показатель ≈ 0,47: отдача от часа падает), "
            "а масштаб задают ваши «потолки» (запас до рекорда). При такой постановке оптимум часов по видам ∝ (запас)^(1/(1−p)).")
    src = st.radio("Откуда считать часы", ["По плану (сезон)", "По журналу (факт)"], horizontal=True, key="hrs_src")
    if src.startswith("По журналу"):
        HH = ds.hours_by_event_from_log(LOG) if LOG else {e: 0.0 for e in ds.EVENTS}
    else:
        HH = HB if HB is not None else (ds.hours_by_event_from_plan(df_base) if df_base is not None else {e: 0.0 for e in ds.EVENTS})
    if sum(HH.values()) <= 0:
        st.caption("Нет данных о часах: в журнале пока нет записей.")
    else:
        HT = ds.hours_table(res, HH, STRETCH_UI)
        HT["Вид"] = HT["event"].map(lambda e: ds.RU_EV.get(e, e))
        st.dataframe(HT.drop(columns=["event"]).set_index("Вид"), use_container_width=True)
        st.caption("Часы по плану — оценка из объёмов Грузенкина и МОИХ минут на типовую позицию каталога, это не измерение. Профильные разделы идут в свой вид с весом 1, сила, плиометрика и ОФП — по тегам с весом 0,5. "
                   "«Оптимум» математически крайний (запас в степени 1,9), поэтому в плане множитель ограничен 0,9–1,25 и смешан с индивидуализацией по запасу роста.")
    if EMPH_LAWD:
        gl = sorted(set(EMPH_HEAD) | set(EMPH_LAWD))
        st.markdown("**Множители объёма по средствам**")
        st.dataframe(pd.DataFrame({"Средство": [GL.get(g, g) for g in gl], "По запасу роста": [round(EMPH_HEAD.get(g, 1.0), 2) for g in gl],
                                   "По закону убывающей отдачи": [round(EMPH_LAWD.get(g, 1.0), 2) for g in gl], "Слабое звено (тесты)": [round(T_MULT.get(g, 1.0), 2) for g in gl],
                                   f"Итог (вес закона {w_hours:g})": [round(EMPH.get(g, 1.0), 2) for g in gl]}),
                     use_container_width=True, hide_index=True)

persist_now()

# ───────────────────────── боковая панель: что с сохранением ─────────────────────────
with SIDE_STATUS:
    st.markdown("### 🏃 Самандык Д.В.")
    _cloud = bool(_secret("SUPABASE_URL") and _secret("SUPABASE_KEY"))
    _err = st.session_state.get("_save_err")
    if _cloud and not _err:
        st.success("☁️ Облако настроено: записи переживают перезапуск")
    elif _cloud:
        st.warning("☁️ Облако настроено, но последняя запись туда не прошла: " + str(_err))
    else:
        st.error("☁️ Облако не подключено: после перезапуска сервера записи пропадут. Подключите (DEPLOY.md) или скачивайте копию.")
    _sa = STATE.get("saved_at")
    st.caption("Последнее сохранение: " + (pd.Timestamp(_sa).strftime("%d.%m %H:%M") if _sa else "ещё не было"))
    st.caption(f"Дней в дневнике: {len(LOG)}" + (f" · последний {max(e['date'] for e in LOG)}" if LOG else ""))
    st.caption("Настройки сохраняются сами при каждом изменении. Записи дневника — по кнопкам «Записать».")
    st.caption(f"Сборка от {APP_BUILD}")
    st.markdown("---")
    st.metric("Прогноз модели к старту", f"{F['sum_forecast']:.0f}", f"{F['sum_forecast'] - F0['sum_forecast']:+.0f} к эталону")
    st.caption(f"До семиборья {max(0, (d_hept - real_today).days)} дн. · до десятиборья {max(0, (d_dec - real_today).days)} дн.")
