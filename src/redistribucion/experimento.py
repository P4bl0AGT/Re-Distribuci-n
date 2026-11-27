"""
Runner unificado del experimento de redistribución quirúrgica.

Modos disponibles:
  base          – 18 combinaciones por ventana (3 heurísticas × 2 estrategias × 3 exclusividades)
  preasignacion – 30 combinaciones por ventana (3 heurísticas × 2 estrategias × 5 tasas)

El orden de las semanas se lee desde data/semanas_ordenadas.csv.

Uso:
    python src/redistribucion/experimento.py --modo base
    python src/redistribucion/experimento.py --modo preasignacion
    python src/redistribucion/experimento.py --modo base --max-iter 20
    python src/redistribucion/experimento.py --modo preasignacion --max-iter 20
"""

import argparse
import sys
import time
from pathlib import Path
import pandas as pd

from simulation_engine import (
    CSV_PATH, SIM_START,
    loadAndIndexData,
    runSimulation,
    runPreAssignSimulation,
)

ROOT = Path(__file__).parent.parent.parent

WEEKS_CSV        = ROOT / 'data' / 'semanas_ordenadas.csv'
OUTPUT_BASE      = ROOT / 'data' / 'resultados_experimento.csv'
OUTPUT_PREASSIGN = ROOT / 'data' / 'resultados_experimento_preasignacion.csv'
OUTPUT_SPTA      = ROOT / 'data' / 'resultados_spta_completo.csv'
WINDOW_SIZE      = 5

HEURISTICS      = ['FIFO', 'SPT', 'LPT']
STRATEGIES      = ['VERTICAL', 'HORIZONTAL']
EXCLUSIVE_ORS   = [0, 1, 2]
INSERTION_RATES = [0.00, 0.05, 0.10, 0.15, 0.20]
SPTA_RATES      = [round(i * 0.1, 1) for i in range(11)]   # 0.0 → 1.0 en pasos de 0.1
SEED            = 42

STRATEGY_SHORT = {'VERTICAL': 'A', 'HORIZONTAL': 'S'}
STRATEGY_LONG  = {'VERTICAL': 'Alternado', 'HORIZONTAL': 'Secuencial'}


# ── Utilidades ────────────────────────────────────────────────────────────────
def _eta(elapsed, done, total):
    if done == 0:
        return '?'
    rem = (elapsed / done) * (total - done)
    h, m = divmod(int(rem), 3600)
    m, s = divmod(m, 60)
    if h:
        return f'{h}h {m:02d}m'
    if m:
        return f'{m}m {s:02d}s'
    return f'{s}s'


def _bar(done, total, width=20):
    filled = int(width * done / total) if total else 0
    return f"[{'#' * filled}{'-' * (width - filled)}]"


def _loadWeekDates():
    if not WEEKS_CSV.is_file():
        print(f"\n  ERROR: '{WEEKS_CSV}' no existe.")
        sys.exit(1)
    return pd.read_csv(WEEKS_CSV)['weekStart'].tolist()


def _loadData():
    print(f"\n  Cargando dataset '{CSV_PATH}'...", end=' ', flush=True)
    t0 = time.perf_counter()
    weekIdx = loadAndIndexData(CSV_PATH)
    print(f"listo en {time.perf_counter() - t0:.1f}s  ({len(weekIdx)} semanas indexadas)")
    return weekIdx


def _validateWindows(weekDates, maxIter):
    totalWeeks = len(weekDates)
    if totalWeeks < WINDOW_SIZE:
        print(f"  ERROR: solo {totalWeeks} semanas disponibles (mínimo {WINDOW_SIZE}).")
        sys.exit(1)
    totalWindows = totalWeeks - WINDOW_SIZE + 1
    if maxIter is not None:
        totalWindows = min(totalWindows, maxIter)
    return totalWeeks, totalWindows


def _save(results, outputPath, tStart):
    dfOut = pd.DataFrame(results)
    outputPath.parent.mkdir(parents=True, exist_ok=True)
    dfOut.to_csv(outputPath, index=False)
    elapsed = time.perf_counter() - tStart
    print(f"\n  Experimento completado en {elapsed/60:.1f} min.")
    print(f"  Resultados guardados en '{outputPath}'  ({len(dfOut)} filas)\n")
    return dfOut


def _summary(df, groupCols, metricCols):
    print("  --- Resumen por combinación (media sobre todas las iteraciones) ---")
    print(df.groupby(groupCols)[metricCols].mean().round(2).to_string())
    print()


# ── Modo base ─────────────────────────────────────────────────────────────────
def runBase(weekIdx, maxIter=None):
    print("=" * 65)
    print("  EXPERIMENTO BASE – REDISTRIBUCIÓN QUIRÚRGICA (ventana deslizante)")
    print("=" * 65)

    weekDates             = _loadWeekDates()
    totalWeeks, totalWindows = _validateWindows(weekDates, maxIter)

    scenariosPerWindow = len(HEURISTICS) * len(STRATEGIES) * len(EXCLUSIVE_ORS)
    totalScenarios     = totalWindows * scenariosPerWindow

    print(f"\n  Semanas disponibles : {totalWeeks}")
    if maxIter is not None:
        print(f"  Ventanas a correr  : {totalWindows}  (limitado a --max-iter {maxIter})")
    else:
        print(f"  Ventanas a correr  : {totalWindows}")
    print(f"  Escenarios/ventana : {scenariosPerWindow}  "
          f"(3 heurísticas × 2 estrategias × 3 exclusividades)")
    print(f"  Total escenarios   : {totalScenarios}\n")

    results = []
    count   = 0
    tStart  = time.perf_counter()

    for w in range(totalWindows):
        window = weekDates[w : w + WINDOW_SIZE]

        for exclusive in EXCLUSIVE_ORS:
            for heuristic in HEURISTICS:
                for strategy in STRATEGIES:
                    count += 1
                    metrics = runSimulation(
                        weekIndex       = weekIdx,
                        weekStartDates  = window,
                        simStart        = SIM_START,
                        heuristic       = heuristic,
                        strategy        = strategy,
                        exclusiveUrgOrs = exclusive,
                    )
                    results.append({
                        'iteracion'       : w + 1,
                        'semana_1'        : window[0],
                        'semana_2'        : window[1],
                        'semana_3'        : window[2],
                        'semana_4'        : window[3],
                        'semana_5'        : window[4],
                        'heuristica'      : heuristic,
                        'estrategia'      : STRATEGY_SHORT[strategy],
                        'tipo_asignacion' : f"{heuristic} + {STRATEGY_LONG[strategy]}",
                        'salas_exclusivas': exclusive,
                        'throughput'      : metrics['Throughput'],
                        'lead_time'       : metrics['LeadTimeDays'],
                        'util_urg_pct'    : metrics['UtilUrg%'],
                        'util_elec_pct'   : metrics['UtilElec%'],
                        'util_global_pct' : metrics['UtilGlobal%'],
                    })
                    elapsed = time.perf_counter() - tStart
                    print(
                        f"\r  {_bar(count, totalScenarios)} {count:5d}/{totalScenarios} | "
                        f"iter {w+1:03d} | {heuristic}+{STRATEGY_SHORT[strategy]} "
                        f"excl={exclusive} | "
                        f"TP={metrics['Throughput']:4d} LT={metrics['LeadTimeDays']:.2f}d | "
                        f"ETA {_eta(elapsed, count, totalScenarios)}   ",
                        end='', flush=True,
                    )

    print()
    df = _save(results, OUTPUT_BASE, tStart)
    _summary(df, ['heuristica', 'estrategia', 'salas_exclusivas'],
             ['throughput', 'lead_time', 'util_elec_pct', 'util_global_pct'])


# ── Modo pre-asignación ───────────────────────────────────────────────────────
def runPreassign(weekIdx, maxIter=None):
    print("=" * 70)
    print("  EXPERIMENTO PRE-ASIGNACIÓN – REDISTRIBUCIÓN QUIRÚRGICA (ventana deslizante)")
    print("=" * 70)

    weekDates             = _loadWeekDates()
    totalWeeks, totalWindows = _validateWindows(weekDates, maxIter)

    scenariosPerWindow = len(HEURISTICS) * len(STRATEGIES) * len(INSERTION_RATES)
    totalScenarios     = totalWindows * scenariosPerWindow

    print(f"\n  Semanas disponibles : {totalWeeks}")
    if maxIter is not None:
        print(f"  Ventanas a correr  : {totalWindows}  (limitado a --max-iter {maxIter})")
    else:
        print(f"  Ventanas a correr  : {totalWindows}")
    print(f"  Escenarios/ventana : {scenariosPerWindow}  "
          f"(3 heurísticas × 2 estrategias × {len(INSERTION_RATES)} tasas)")
    print(f"  Tasas de inserción : {INSERTION_RATES}")
    print(f"  Total escenarios   : {totalScenarios}\n")

    results = []
    count   = 0
    tStart  = time.perf_counter()

    for w in range(totalWindows):
        window = weekDates[w : w + WINDOW_SIZE]

        for rate in INSERTION_RATES:
            for heuristic in HEURISTICS:
                for strategy in STRATEGIES:
                    count += 1
                    metrics = runPreAssignSimulation(
                        weekIndex           = weekIdx,
                        weekStartDates      = window,
                        simStart            = SIM_START,
                        heuristic           = heuristic,
                        strategy            = strategy,
                        randomInsertionRate = rate,
                        randomInsertionSeed = SEED,
                    )
                    results.append({
                        'iteracion'         : w + 1,
                        'semana_1'          : window[0],
                        'semana_2'          : window[1],
                        'semana_3'          : window[2],
                        'semana_4'          : window[3],
                        'semana_5'          : window[4],
                        'heuristica'        : heuristic,
                        'estrategia'        : STRATEGY_SHORT[strategy],
                        'tipo_asignacion'   : f"{heuristic} + {STRATEGY_LONG[strategy]}",
                        'random_rate'       : rate,
                        'throughput'        : metrics['Throughput'],
                        'lead_time'         : metrics['LeadTimeDays'],
                        'util_urg_pct'      : metrics['UtilUrg%'],
                        'util_elec_pct'     : metrics['UtilElec%'],
                        'util_global_pct'   : metrics['UtilGlobal%'],
                        'pre_assigned_count': metrics['PreAssigned'],
                        'pre_assigned_pct'  : metrics['PreAssignedPct'],
                    })
                    elapsed = time.perf_counter() - tStart
                    print(
                        f"\r  {_bar(count, totalScenarios)} {count:5d}/{totalScenarios} | "
                        f"iter {w+1:03d} | {heuristic}+{STRATEGY_SHORT[strategy]} "
                        f"rate={rate:.0%} | "
                        f"TP={metrics['Throughput']:4d} LT={metrics['LeadTimeDays']:.2f}d "
                        f"Pre={metrics['PreAssigned']:3d} | "
                        f"ETA {_eta(elapsed, count, totalScenarios)}   ",
                        end='', flush=True,
                    )

    print()
    df = _save(results, OUTPUT_PREASSIGN, tStart)
    _summary(df, ['heuristica', 'estrategia', 'random_rate'],
             ['throughput', 'lead_time', 'util_elec_pct', 'util_global_pct',
              'pre_assigned_count', 'pre_assigned_pct'])


# ── Modo SPT + Alternado completo (0 % – 100 %) ───────────────────────────────
def runSPTA(weekIdx, maxIter=None):
    print("=" * 70)
    print("  EXPERIMENTO SPT + ALTERNADO – PRE-ASIGNACIÓN 0 % a 100 %")
    print("=" * 70)

    weekDates              = _loadWeekDates()
    totalWeeks, totalWindows = _validateWindows(weekDates, maxIter)
    totalScenarios         = totalWindows * len(SPTA_RATES)

    print(f"\n  Semanas disponibles : {totalWeeks}")
    if maxIter is not None:
        print(f"  Ventanas a correr  : {totalWindows}  (limitado a --max-iter {maxIter})")
    else:
        print(f"  Ventanas a correr  : {totalWindows}")
    print(f"  Configuración      : SPT + Alternado (VERTICAL), excl=2")
    print(f"  Tasas de inserción : {SPTA_RATES}")
    print(f"  Total escenarios   : {totalScenarios}\n")

    results = []
    count   = 0
    tStart  = time.perf_counter()

    for w in range(totalWindows):
        window = weekDates[w : w + WINDOW_SIZE]

        for rate in SPTA_RATES:
            count += 1
            metrics = runPreAssignSimulation(
                weekIndex           = weekIdx,
                weekStartDates      = window,
                simStart            = SIM_START,
                heuristic           = 'SPT',
                strategy            = 'VERTICAL',
                randomInsertionRate = rate,
                randomInsertionSeed = SEED,
            )
            results.append({
                'iteracion'         : w + 1,
                'semana_1'          : window[0],
                'semana_2'          : window[1],
                'semana_3'          : window[2],
                'semana_4'          : window[3],
                'semana_5'          : window[4],
                'heuristica'        : 'SPT',
                'estrategia'        : 'A',
                'random_rate'       : rate,
                'throughput'        : metrics['Throughput'],
                'lead_time'         : metrics['LeadTimeDays'],
                'util_urg_pct'      : metrics['UtilUrg%'],
                'util_elec_pct'     : metrics['UtilElec%'],
                'util_global_pct'   : metrics['UtilGlobal%'],
                'pre_assigned_count': metrics['PreAssigned'],
                'pre_assigned_pct'  : metrics['PreAssignedPct'],
            })
            elapsed = time.perf_counter() - tStart
            print(
                f"\r  {_bar(count, totalScenarios)} {count:5d}/{totalScenarios} | "
                f"iter {w+1:03d} | SPT+A rate={rate:.0%} | "
                f"TP={metrics['Throughput']:4d} LT={metrics['LeadTimeDays']:.2f}d "
                f"Pre={metrics['PreAssigned']:3d} | "
                f"ETA {_eta(elapsed, count, totalScenarios)}   ",
                end='', flush=True,
            )

    print()
    df = _save(results, OUTPUT_SPTA, tStart)
    _summary(df, ['random_rate'],
             ['throughput', 'lead_time', 'util_elec_pct', 'util_global_pct',
              'pre_assigned_count', 'pre_assigned_pct'])


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Experimento de redistribución quirúrgica con ventana deslizante.'
    )
    parser.add_argument(
        '--modo', choices=['base', 'preasignacion', 'spta'], default='base',
        help='"base": 18 combinaciones (excl. salas). "preasignacion": 30 combinaciones (tasas). '
             '"spta": SPT+Alternado+excl=2 con tasas 0%%–100%% (pasos de 10%%). Default: base',
    )
    parser.add_argument(
        '--max-iter', type=int, default=None, metavar='N',
        help='Limitar a las primeras N ventanas (útil para pruebas rápidas).',
    )
    args = parser.parse_args()

    weekIdx = _loadData()

    if args.modo == 'base':
        runBase(weekIdx, maxIter=args.max_iter)
    elif args.modo == 'preasignacion':
        runPreassign(weekIdx, maxIter=args.max_iter)
    else:
        runSPTA(weekIdx, maxIter=args.max_iter)
