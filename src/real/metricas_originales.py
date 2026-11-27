"""
Extrae métricas reales semana a semana del dataset original,
sin simulación. Genera data/resultados_baseline.csv para
comparar contra los escenarios redistribuidos de experimento.py.

Columnas de salida:
  weekStart, isoWeek, year, rank_vol,
  totalSurgeries, urgSurgeries, elecSurgeries,
  avgLeadTimeDays, avgLeadTimeUrg, avgLeadTimeElec,
  urgMinutes, elecMinutes,
  utilUrg_pct, utilElec_pct, utilGlobal_pct

Uso:
    python src/metricas_originales.py
"""

import pandas as pd
import numpy as np
from pathlib import Path

ROOT     = Path(__file__).parent.parent.parent
CSV_PATH = ROOT / 'data' / '2_dataset_procesado_actualizado.csv'
OUT_PATH = ROOT / 'data' / 'resultados_baseline.csv'

# Capacidad estructural idéntica a simulation_engine.py para comparación justa
SIM_DAYS       = 5
URG_ORS        = 1
ELEC_ORS       = 9
URG_HOURS_DAY  = 24
ELEC_HOURS_DAY = 9   # 08:00–16:00 + 1h overflow = 9h


def _loadData() -> pd.DataFrame:
    df = pd.read_csv(CSV_PATH, low_memory=False)

    for col in ('Ingreso Quirófano', 'Ingreso Pabellón', 'Fecha de Pabellón'):
        if col in df.columns:
            colIngreso = col
            break
    else:
        raise KeyError("No se encontró columna de fecha de ingreso.")

    renameMap = {
        colIngreso:            'ingresoQuirofano',
        'Tiempo Intervención': 'tiempoIntervencion',
        'Tipo Atención':       'tipoAtencion',
    }
    renameMap = {k: v for k, v in renameMap.items() if k in df.columns}
    df.rename(columns=renameMap, inplace=True)

    if isinstance(df.get('ingresoQuirofano'), pd.DataFrame):
        df['ingresoQuirofano'] = df['ingresoQuirofano'].iloc[:, 0]

    df['ingresoQuirofano']   = pd.to_datetime(df['ingresoQuirofano'],   errors='coerce')
    df['tiempoIntervencion'] = pd.to_numeric(df['tiempoIntervencion'],  errors='coerce')
    df.dropna(subset=['ingresoQuirofano', 'tiempoIntervencion'], inplace=True)
    df = df[df['tiempoIntervencion'] > 0].copy()

    df['isUrgency'] = (
        df['tipoAtencion'].astype(str).str.upper().str.contains('URGENCIA', na=False)
    )

    # weekStart = lunes de cada semana
    df['weekStart'] = (
        df['ingresoQuirofano']
        - pd.to_timedelta(df['ingresoQuirofano'].dt.weekday, unit='D')
    ).dt.normalize()

    return df


def _calcWeekMetrics(group: pd.DataFrame, weekStart: pd.Timestamp) -> dict:
    isUrg    = group['isUrgency']
    urgMins  = group.loc[isUrg,  'tiempoIntervencion'].sum()
    elecMins = group.loc[~isUrg, 'tiempoIntervencion'].sum()

    # Lead time: (hora_fin - lunes_semana) en días, igual que real.py
    endTimes = group['ingresoQuirofano'] + pd.to_timedelta(group['tiempoIntervencion'], unit='m')
    leadAll  = (endTimes - weekStart).dt.total_seconds() / 86400.0
    leadUrg  = leadAll[isUrg.values]
    leadElec = leadAll[~isUrg.values]

    urgCapMins   = URG_ORS  * URG_HOURS_DAY  * SIM_DAYS * 60
    elecCapMins  = ELEC_ORS * ELEC_HOURS_DAY * SIM_DAYS * 60
    totalCapMins = urgCapMins + elecCapMins

    utilUrg    = urgMins  / urgCapMins   * 100 if urgCapMins   > 0 else 0.0
    utilElec   = elecMins / elecCapMins  * 100 if elecCapMins  > 0 else 0.0
    utilGlobal = (urgMins + elecMins) / totalCapMins * 100 if totalCapMins > 0 else 0.0

    return {
        'totalSurgeries'  : len(group),
        'urgSurgeries'    : int(isUrg.sum()),
        'elecSurgeries'   : int((~isUrg).sum()),
        'avgLeadTimeDays' : round(float(leadAll.mean()),  4),
        'avgLeadTimeUrg'  : round(float(leadUrg.mean()),  4) if len(leadUrg)  > 0 else None,
        'avgLeadTimeElec' : round(float(leadElec.mean()), 4) if len(leadElec) > 0 else None,
        'urgMinutes'      : round(float(urgMins),  1),
        'elecMinutes'     : round(float(elecMins), 1),
        'utilUrg_pct'     : round(utilUrg,    2),
        'utilElec_pct'    : round(utilElec,   2),
        'utilGlobal_pct'  : round(utilGlobal, 2),
    }


def main():
    print("Cargando dataset...")
    df = _loadData()
    print(f"  {len(df):,} cirugías cargadas | {df['weekStart'].nunique()} semanas")

    rows = []
    for weekStart, group in df.groupby('weekStart'):
        metrics = _calcWeekMetrics(group, weekStart)
        rows.append({
            'weekStart' : weekStart.strftime('%Y-%m-%d'),
            'isoWeek'   : int(weekStart.isocalendar()[1]),
            'year'      : int(weekStart.year),
            **metrics,
        })

    result = (
        pd.DataFrame(rows)
        .sort_values('weekStart')
        .reset_index(drop=True)
    )
    result.insert(
        0, 'rank_vol',
        result['totalSurgeries'].rank(ascending=False, method='min').astype(int)
    )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(OUT_PATH, index=False)

    print(f"\n  {len(result)} semanas guardadas en '{OUT_PATH}'")
    print(f"\n  --- Estadísticas globales ---")
    print(f"  Throughput:   media={result['totalSurgeries'].mean():.1f}"
          f"  min={result['totalSurgeries'].min()}"
          f"  max={result['totalSurgeries'].max()}")
    print(f"  Lead Time:    media={result['avgLeadTimeDays'].mean():.3f}d"
          f"  min={result['avgLeadTimeDays'].min():.3f}d"
          f"  max={result['avgLeadTimeDays'].max():.3f}d")
    print(f"  Util Global:  media={result['utilGlobal_pct'].mean():.1f}%"
          f"  min={result['utilGlobal_pct'].min():.1f}%"
          f"  max={result['utilGlobal_pct'].max():.1f}%")
    print(f"  Util Elec:    media={result['utilElec_pct'].mean():.1f}%")
    print(f"  Util Urg:     media={result['utilUrg_pct'].mean():.1f}%")


if __name__ == '__main__':
    main()
