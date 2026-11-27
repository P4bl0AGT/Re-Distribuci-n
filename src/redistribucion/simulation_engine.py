"""
Motor de simulación unificado y parametrizable.

Contiene dos motores:
  SimEngine       – motor base (heurística + estrategia + exclusividad)
  PreAssignEngine – extiende el base con pre-asignación aleatoria de electivas

Parámetros de runSimulation():
  weekIndex        – índice pre-cargado con loadAndIndexData()
  weekStartDates   – lista de 5 fechas lunes (str 'YYYY-MM-DD' o date)
  simStart         – datetime de inicio de la semana simulada
  heuristic        – "FIFO" | "SPT" | "LPT"
  strategy         – "VERTICAL" | "HORIZONTAL"
  exclusiveUrgOrs  – 0 | 1 | 2
      0 → sin exclusivos: urgencias usan cualquier pabellón libre
      1 → 1 pabellón exclusivo (OR 0) para urgencias
      2 → 2 pabellones exclusivos (OR 0,1) para urgencias

Parámetros adicionales de runPreAssignSimulation():
  randomInsertionRate – fracción de electivas a pre-asignar (0.0–1.0)
  randomInsertionSeed – semilla para reproducibilidad

Capacidad presupuestada:
  exclusiveUrgOrs 0 o 1 → 1 pabellón urgencias (24h) + 9 electivos (9h)
  exclusiveUrgOrs 2     → 2 pabellones urgencias (24h) + 8 electivos (9h)
"""

import random
import heapq
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings("ignore")

# ── Rutas ────────────────────────────────────────────────────────────────────
ROOT     = Path(__file__).parent.parent.parent
CSV_PATH = ROOT / 'data' / '2_dataset_procesado_actualizado.csv'

# ── Constantes globales ──────────────────────────────────────────────────────
SIM_START            = datetime(2023, 4, 17, 0, 0, 0)   # anchor date (día de semana fijo)
SIM_DAYS             = 5
TOTAL_ORS            = 10

URG_START_HOUR       = 0
URG_END_HOUR         = 24   # urgencias 24/7

ELEC_START_HOUR      = 8
ELEC_END_HOUR        = 16
SHIFT_OVERFLOW_DELTA = 1    # 1h extra → turno electivo efectivo 8–17

PREASSIGN_EXCL_ORS   = 2    # pabellones exclusivos fijos en el modelo de pre-asignación


# ── Modelos de datos ─────────────────────────────────────────────────────────
class SimSurgery:
    __slots__ = [
        'surgeryId', 'isUrgency', 'durationMins',
        'arrivalTime', 'requestDate', 'originWeekIdx',
        'assignedOr', 'startTime', 'endTime', 'leadTimeDays',
    ]

    def __init__(self, surgeryId, isUrgency, durationMins,
                 arrivalTime, requestDate, originWeekIdx):
        self.surgeryId     = surgeryId
        self.isUrgency     = isUrgency
        self.durationMins  = durationMins
        self.arrivalTime   = arrivalTime
        self.requestDate   = requestDate
        self.originWeekIdx = originWeekIdx
        self.assignedOr    = None
        self.startTime     = None
        self.endTime       = None
        self.leadTimeDays  = 0.0


class SimEvent:
    __slots__ = ['time', 'eventType', 'payload']

    def __init__(self, time, eventType, payload=None):
        self.time      = time
        self.eventType = eventType
        self.payload   = payload

    def __lt__(self, other):
        return self.time < other.time


# ── Carga única del CSV ──────────────────────────────────────────────────────
def loadAndIndexData(csvPath: Path = CSV_PATH) -> dict:
    """Lee el CSV una sola vez y construye un índice weekStartDate → list[dict]."""
    df = pd.read_csv(csvPath, low_memory=False)

    for col in ('Ingreso Quirófano', 'Ingreso Pabellón', 'Fecha de Pabellón'):
        if col in df.columns:
            colIngreso = col
            break
    else:
        raise KeyError("No se encontró columna de fecha de ingreso.")

    colId = 'Número' if 'Número' in df.columns else 'idOperacion'

    renameMap = {
        colIngreso:            'ingresoQuirofano',
        'Tiempo Intervención': 'tiempoIntervencion',
        'Tipo Atención':       'tipoAtencion',
        colId:                 'idOperacion',
    }
    renameMap = {k: v for k, v in renameMap.items() if k in df.columns}
    df.rename(columns=renameMap, inplace=True)

    if isinstance(df.get('ingresoQuirofano'), pd.DataFrame):
        df['ingresoQuirofano'] = df['ingresoQuirofano'].iloc[:, 0]

    df['ingresoQuirofano']   = pd.to_datetime(df['ingresoQuirofano'],  errors='coerce')
    df['tiempoIntervencion'] = pd.to_numeric(df['tiempoIntervencion'], errors='coerce')
    df.dropna(subset=['ingresoQuirofano', 'tiempoIntervencion'], inplace=True)
    df = df[df['tiempoIntervencion'] > 0].copy()

    df['weekStartDate'] = (
        df['ingresoQuirofano']
        - pd.to_timedelta(df['ingresoQuirofano'].dt.weekday, unit='D')
    ).dt.normalize().dt.date

    df['isUrgency'] = (
        df['tipoAtencion'].astype(str).str.upper().str.contains('URGENCIA', na=False)
    )
    df['surgeryId'] = df.get('idOperacion', pd.RangeIndex(len(df))).astype(str)

    keep = ['weekStartDate', 'surgeryId', 'isUrgency', 'tiempoIntervencion', 'ingresoQuirofano']
    df = df[keep].rename(columns={
        'tiempoIntervencion': 'durationMins',
        'ingresoQuirofano':   'arrival',
    })

    weekIndex: dict = {}
    for wsDate, group in df.groupby('weekStartDate'):
        weekIndex[wsDate] = group.drop(columns='weekStartDate').to_dict('records')

    return weekIndex


# ── Construcción de datos por ventana ────────────────────────────────────────
def buildSimData(weekIndex: dict, weekStartDates: list, simStart: datetime) -> tuple:
    """Colapsa varias semanas sobre una semana de simulación preservando día-de-semana y hora."""
    urgencies: list = []
    electives: list = []

    for weekIdx, weekDate in enumerate(weekStartDates):
        wsDate    = pd.to_datetime(weekDate).date()
        surgeries = weekIndex.get(wsDate, [])

        for s in surgeries:
            arrival   = s['arrival']
            dayOfWeek = arrival.weekday()

            collapsedArrival = (
                simStart + timedelta(days=dayOfWeek)
            ).replace(hour=arrival.hour, minute=arrival.minute, second=0)

            surg = SimSurgery(
                surgeryId     = f"{weekIdx}_{s['surgeryId']}",
                isUrgency     = bool(s['isUrgency']),
                durationMins  = float(s['durationMins']),
                arrivalTime   = collapsedArrival,
                requestDate   = collapsedArrival,
                originWeekIdx = weekIdx,
            )

            if surg.isUrgency:
                urgencies.append(surg)
            else:
                surg.arrivalTime = simStart   # electivas disponibles al inicio
                electives.append(surg)

    return urgencies, electives


# ─────────────────────────────────────────────────────────────────────────────
# MOTOR BASE
# ─────────────────────────────────────────────────────────────────────────────
class SimEngine:
    def __init__(self, urgencies, electives, simStart, heuristic, strategy, exclusiveUrgOrs):
        self.simStart        = simStart
        self.simEnd          = simStart + timedelta(days=SIM_DAYS)
        self.heuristic       = heuristic
        self.strategy        = strategy
        self.exclusiveUrgOrs = exclusiveUrgOrs

        self.pendingUrgencies   = sorted(urgencies, key=lambda x: (x.arrivalTime, x.originWeekIdx))
        self.pendingElectives   = list(electives)
        self._sortElectives()

        self.activeUrgencies    = []
        self.completedSurgeries = []
        self.orFreeTime         = {i: simStart for i in range(TOTAL_ORS)}
        self.events             = []

        for u in self.pendingUrgencies:
            heapq.heappush(self.events, SimEvent(u.arrivalTime, 'URG_ARRIVAL', u))
        for i in range(TOTAL_ORS):
            heapq.heappush(self.events, SimEvent(simStart, 'OR_FREE', i))

    def _sortElectives(self):
        if self.heuristic == 'FIFO':
            self.pendingElectives.sort(key=lambda x: (x.requestDate, x.originWeekIdx))
        elif self.heuristic == 'SPT':
            self.pendingElectives.sort(key=lambda x: (x.durationMins, x.originWeekIdx))
        elif self.heuristic == 'LPT':
            self.pendingElectives.sort(key=lambda x: (-x.durationMins, x.originWeekIdx))

    def _assign(self, surg, orId, startTime):
        surg.assignedOr   = orId
        surg.startTime    = startTime
        surg.endTime      = startTime + timedelta(minutes=surg.durationMins)
        surg.leadTimeDays = (surg.endTime - surg.arrivalTime).total_seconds() / 86400.0
        self.orFreeTime[orId] = surg.endTime
        self.completedSurgeries.append(surg)
        heapq.heappush(self.events, SimEvent(surg.endTime, 'OR_FREE', orId))

    def run(self) -> list:
        ELEC_LIMIT_HOUR = ELEC_END_HOUR + SHIFT_OVERFLOW_DELTA

        while self.events:
            e = heapq.heappop(self.events)
            t = e.time

            if t >= self.simEnd:
                break

            if e.eventType == 'URG_ARRIVAL':
                self.activeUrgencies.append(e.payload)

            # 1. Urgencias
            urgPool = list(range(TOTAL_ORS)) if self.exclusiveUrgOrs == 0 \
                      else list(range(self.exclusiveUrgOrs))
            freeUrgOrs = [i for i in urgPool if self.orFreeTime[i] <= t]
            while freeUrgOrs and self.activeUrgencies:
                orId = freeUrgOrs.pop(0)
                surg = self.activeUrgencies.pop(0)
                self._assign(surg, orId, max(t, self.orFreeTime[orId]))

            # 2. Electivas
            if self.exclusiveUrgOrs == 0:
                freeElecOrs = [i for i in range(TOTAL_ORS) if self.orFreeTime[i] <= t]
            else:
                freeElecOrs = [
                    i for i in range(self.exclusiveUrgOrs, TOTAL_ORS)
                    if self.orFreeTime[i] <= t
                ]

            isElecTime = ELEC_START_HOUR <= t.hour < ELEC_LIMIT_HOUR

            if not isElecTime:
                if self.pendingElectives:
                    nextOpen = t.replace(hour=ELEC_START_HOUR, minute=0, second=0)
                    if t.hour >= ELEC_LIMIT_HOUR:
                        nextOpen += timedelta(days=1)
                    if self.exclusiveUrgOrs == 0:
                        heapq.heappush(self.events, SimEvent(nextOpen, 'WAKE_UP'))
                    else:
                        for orId in freeElecOrs:
                            if self.orFreeTime[orId] < nextOpen:
                                self.orFreeTime[orId] = nextOpen
                                heapq.heappush(self.events, SimEvent(nextOpen, 'OR_FREE', orId))
                continue

            if self.strategy == 'VERTICAL':
                for orId in freeElecOrs:
                    if not self.pendingElectives:
                        break
                    propStart = max(t, self.orFreeTime[orId])
                    surg      = self.pendingElectives[0]
                    estEnd    = propStart + timedelta(minutes=surg.durationMins)
                    limitTime = propStart.replace(
                        hour=ELEC_END_HOUR, minute=0, second=0
                    ) + timedelta(hours=SHIFT_OVERFLOW_DELTA)

                    if estEnd <= limitTime:
                        self.pendingElectives.pop(0)
                        self._assign(surg, orId, propStart)
                    else:
                        nextOpen = propStart.replace(
                            hour=ELEC_START_HOUR, minute=0, second=0
                        ) + timedelta(days=1)
                        if self.exclusiveUrgOrs == 0:
                            heapq.heappush(self.events, SimEvent(nextOpen, 'WAKE_UP'))
                        else:
                            self.orFreeTime[orId] = nextOpen
                            heapq.heappush(self.events, SimEvent(nextOpen, 'OR_FREE', orId))

            elif self.strategy == 'HORIZONTAL':
                for orId in freeElecOrs:
                    while self.pendingElectives:
                        propStart = max(t, self.orFreeTime[orId])
                        surg      = self.pendingElectives[0]
                        estEnd    = propStart + timedelta(minutes=surg.durationMins)
                        limitTime = propStart.replace(
                            hour=ELEC_END_HOUR, minute=0, second=0
                        ) + timedelta(hours=SHIFT_OVERFLOW_DELTA)

                        if estEnd > limitTime:
                            nextOpen = propStart.replace(
                                hour=ELEC_START_HOUR, minute=0, second=0
                            ) + timedelta(days=1)
                            if self.exclusiveUrgOrs == 0:
                                heapq.heappush(self.events, SimEvent(nextOpen, 'WAKE_UP'))
                            else:
                                self.orFreeTime[orId] = nextOpen
                                heapq.heappush(self.events, SimEvent(nextOpen, 'OR_FREE', orId))
                            break
                        self.pendingElectives.pop(0)
                        self._assign(surg, orId, propStart)

        return self.completedSurgeries


# ── Métricas base ────────────────────────────────────────────────────────────
def calculateMetrics(surgeries: list, simStart: datetime, exclusiveUrgOrs: int) -> dict:
    simEnd    = simStart + timedelta(days=SIM_DAYS)
    completed = [s for s in surgeries if s.endTime is not None and s.endTime <= simEnd]

    throughput = len(completed)
    if throughput == 0:
        return {'Throughput': 0, 'LeadTimeDays': 0.0,
                'UtilUrg%': 0.0, 'UtilElec%': 0.0, 'UtilGlobal%': 0.0}

    urgMins  = sum(s.durationMins for s in completed if s.isUrgency)
    elecMins = sum(s.durationMins for s in completed if not s.isUrgency)

    budgetUrgOrs    = max(1, exclusiveUrgOrs)
    budgetElecOrs   = TOTAL_ORS - budgetUrgOrs
    elecHoursPerDay = (ELEC_END_HOUR - ELEC_START_HOUR) + SHIFT_OVERFLOW_DELTA
    urgCapMins      = budgetUrgOrs  * 24 * SIM_DAYS * 60
    elecCapMins     = budgetElecOrs * elecHoursPerDay * SIM_DAYS * 60
    totalCapMins    = urgCapMins + elecCapMins

    return {
        'Throughput':   throughput,
        'LeadTimeDays': round(float(np.mean([s.leadTimeDays for s in completed])), 4),
        'UtilUrg%':     round(urgMins  / urgCapMins   * 100 if urgCapMins   > 0 else 0.0, 2),
        'UtilElec%':    round(elecMins / elecCapMins  * 100 if elecCapMins  > 0 else 0.0, 2),
        'UtilGlobal%':  round(
            (urgMins + elecMins) / totalCapMins * 100 if totalCapMins > 0 else 0.0, 2
        ),
    }


def runSimulation(
    weekIndex: dict, weekStartDates: list, simStart: datetime,
    heuristic: str, strategy: str, exclusiveUrgOrs: int,
) -> dict:
    urgencies, electives = buildSimData(weekIndex, weekStartDates, simStart)
    engine  = SimEngine(urgencies, electives, simStart, heuristic, strategy, exclusiveUrgOrs)
    results = engine.run()
    return calculateMetrics(results, simStart, exclusiveUrgOrs)


# ─────────────────────────────────────────────────────────────────────────────
# MOTOR DE PRE-ASIGNACIÓN
# ─────────────────────────────────────────────────────────────────────────────
class PreAssignEngine:
    def __init__(self, urgencies, electives, simStart, heuristic, strategy,
                 randomInsertionRate=0.15, randomInsertionSeed=42):
        self.simStart = simStart
        self.simEnd   = simStart + timedelta(days=SIM_DAYS)
        self.heuristic = heuristic
        self.strategy  = strategy
        self.rng       = random.Random(randomInsertionSeed)

        self.pendingUrgencies   = sorted(urgencies, key=lambda x: (x.arrivalTime, x.originWeekIdx))
        self.pendingElectives   = list(electives)
        self._sortElectives()

        self.activeUrgencies    = []
        self.completedSurgeries = []
        self.preAssignedIds     = set()
        self.orReservations     = {i: [] for i in range(TOTAL_ORS)}

        self._prePlaceRandom(randomInsertionRate)

        self.orFreeTime = {i: simStart for i in range(TOTAL_ORS)}
        self.events     = []

        for u in self.pendingUrgencies:
            heapq.heappush(self.events, SimEvent(u.arrivalTime, 'URG_ARRIVAL', u))
        for i in range(TOTAL_ORS):
            heapq.heappush(self.events, SimEvent(simStart, 'OR_FREE', i))

    def _sortElectives(self):
        if self.heuristic == 'FIFO':
            self.pendingElectives.sort(key=lambda x: (x.requestDate, x.originWeekIdx))
        elif self.heuristic == 'SPT':
            self.pendingElectives.sort(key=lambda x: (x.durationMins, x.originWeekIdx))
        elif self.heuristic == 'LPT':
            self.pendingElectives.sort(key=lambda x: (-x.durationMins, x.originWeekIdx))

    def _prePlaceRandom(self, rate):
        n = int(len(self.pendingElectives) * rate)
        if n == 0:
            return

        indices  = self.rng.sample(range(len(self.pendingElectives)), n)
        selected = []
        for idx in sorted(indices, reverse=True):
            selected.append(self.pendingElectives.pop(idx))

        limitHour  = ELEC_END_HOUR + SHIFT_OVERFLOW_DELTA
        maxRetries = 100

        for surg in selected:
            placed = False
            for _ in range(maxRetries):
                orId = self.rng.randint(PREASSIGN_EXCL_ORS, TOTAL_ORS - 1)
                day  = self.rng.randint(0, SIM_DAYS - 1)

                dayStart  = self.simStart + timedelta(days=day)
                elecStart = dayStart.replace(hour=ELEC_START_HOUR, minute=0, second=0)
                elecEnd   = dayStart.replace(hour=limitHour, minute=0, second=0)

                windowMins    = int((elecEnd - elecStart).total_seconds() / 60)
                availableMins = windowMins - int(surg.durationMins)
                if availableMins < 0:
                    continue

                propStart = elecStart + timedelta(minutes=self.rng.randint(0, availableMins))
                propEnd   = propStart + timedelta(minutes=surg.durationMins)

                if any(propStart < rEnd and propEnd > rStart
                       for rStart, rEnd in self.orReservations[orId]):
                    continue

                surg.assignedOr   = orId
                surg.startTime    = propStart
                surg.endTime      = propEnd
                surg.leadTimeDays = (propEnd - surg.arrivalTime).total_seconds() / 86400.0

                self.orReservations[orId].append((propStart, propEnd))
                self.orReservations[orId].sort()
                self.completedSurgeries.append(surg)
                self.preAssignedIds.add(surg.surgeryId)
                placed = True
                break

            if not placed:
                self.pendingElectives.append(surg)

        self._sortElectives()

    def _findNextSlot(self, orId, afterTime, durationMins, limitTime):
        proposed = afterTime
        for resStart, resEnd in self.orReservations[orId]:
            if proposed >= resEnd:
                continue
            if proposed + timedelta(minutes=durationMins) <= resStart:
                break
            proposed = resEnd
        if proposed + timedelta(minutes=durationMins) <= limitTime:
            return proposed
        return None

    def _assign(self, surg, orId, startTime):
        surg.assignedOr   = orId
        surg.startTime    = startTime
        surg.endTime      = startTime + timedelta(minutes=surg.durationMins)
        surg.leadTimeDays = (surg.endTime - surg.arrivalTime).total_seconds() / 86400.0
        self.orFreeTime[orId] = surg.endTime
        self.completedSurgeries.append(surg)
        heapq.heappush(self.events, SimEvent(surg.endTime, 'OR_FREE', orId))

    def run(self) -> list:
        ELEC_LIMIT_HOUR = ELEC_END_HOUR + SHIFT_OVERFLOW_DELTA

        while self.events:
            e = heapq.heappop(self.events)
            t = e.time

            if t >= self.simEnd:
                break

            if e.eventType == 'URG_ARRIVAL':
                self.activeUrgencies.append(e.payload)

            # 1. Urgencias — pabellones exclusivos
            freeUrgOrs = [i for i in range(PREASSIGN_EXCL_ORS) if self.orFreeTime[i] <= t]
            while freeUrgOrs and self.activeUrgencies:
                orId = freeUrgOrs.pop(0)
                surg = self.activeUrgencies.pop(0)
                self._assign(surg, orId, max(t, self.orFreeTime[orId]))

            # 2. Electivas — pabellones electivos
            freeElecOrs = [
                i for i in range(PREASSIGN_EXCL_ORS, TOTAL_ORS)
                if self.orFreeTime[i] <= t
            ]
            isElecTime = ELEC_START_HOUR <= t.hour < ELEC_LIMIT_HOUR

            if not isElecTime:
                if self.pendingElectives:
                    nextOpen = t.replace(hour=ELEC_START_HOUR, minute=0, second=0)
                    if t.hour >= ELEC_LIMIT_HOUR:
                        nextOpen += timedelta(days=1)
                    for orId in freeElecOrs:
                        if self.orFreeTime[orId] < nextOpen:
                            self.orFreeTime[orId] = nextOpen
                            heapq.heappush(self.events, SimEvent(nextOpen, 'OR_FREE', orId))
                continue

            if self.strategy == 'VERTICAL':
                for orId in freeElecOrs:
                    if not self.pendingElectives:
                        break
                    afterTime    = max(t, self.orFreeTime[orId])
                    surg         = self.pendingElectives[0]
                    dayLimitTime = afterTime.replace(
                        hour=ELEC_END_HOUR, minute=0, second=0
                    ) + timedelta(hours=SHIFT_OVERFLOW_DELTA)
                    slot = self._findNextSlot(orId, afterTime, surg.durationMins, dayLimitTime)
                    if slot is not None:
                        self.pendingElectives.pop(0)
                        self._assign(surg, orId, slot)
                    else:
                        nextOpen = afterTime.replace(
                            hour=ELEC_START_HOUR, minute=0, second=0
                        ) + timedelta(days=1)
                        self.orFreeTime[orId] = nextOpen
                        heapq.heappush(self.events, SimEvent(nextOpen, 'OR_FREE', orId))

            elif self.strategy == 'HORIZONTAL':
                for orId in freeElecOrs:
                    while self.pendingElectives:
                        afterTime    = max(t, self.orFreeTime[orId])
                        surg         = self.pendingElectives[0]
                        dayLimitTime = afterTime.replace(
                            hour=ELEC_END_HOUR, minute=0, second=0
                        ) + timedelta(hours=SHIFT_OVERFLOW_DELTA)
                        slot = self._findNextSlot(orId, afterTime, surg.durationMins, dayLimitTime)
                        if slot is None:
                            nextOpen = afterTime.replace(
                                hour=ELEC_START_HOUR, minute=0, second=0
                            ) + timedelta(days=1)
                            self.orFreeTime[orId] = nextOpen
                            heapq.heappush(self.events, SimEvent(nextOpen, 'OR_FREE', orId))
                            break
                        self.pendingElectives.pop(0)
                        self._assign(surg, orId, slot)

        return self.completedSurgeries


# ── Métricas pre-asignación ──────────────────────────────────────────────────
def calculatePreAssignMetrics(surgeries: list, preAssignedIds: set, simStart: datetime) -> dict:
    simEnd    = simStart + timedelta(days=SIM_DAYS)
    completed = [s for s in surgeries if s.endTime is not None and s.endTime <= simEnd]

    throughput = len(completed)
    preCount   = sum(1 for s in completed if s.surgeryId in preAssignedIds)

    if throughput == 0:
        return {'Throughput': 0, 'LeadTimeDays': 0.0,
                'UtilUrg%': 0.0, 'UtilElec%': 0.0, 'UtilGlobal%': 0.0,
                'PreAssigned': 0, 'PreAssignedPct': 0.0}

    urgMins       = sum(s.durationMins for s in completed if s.isUrgency)
    elecMins      = sum(s.durationMins for s in completed if not s.isUrgency)
    budgetUrgOrs  = PREASSIGN_EXCL_ORS
    budgetElecOrs = TOTAL_ORS - budgetUrgOrs
    elecHoursPerDay = (ELEC_END_HOUR - ELEC_START_HOUR) + SHIFT_OVERFLOW_DELTA
    urgCapMins    = budgetUrgOrs  * 24 * SIM_DAYS * 60
    elecCapMins   = budgetElecOrs * elecHoursPerDay * SIM_DAYS * 60
    totalCapMins  = urgCapMins + elecCapMins
    totalElec     = sum(1 for s in completed if not s.isUrgency)

    return {
        'Throughput':     throughput,
        'LeadTimeDays':   round(float(np.mean([s.leadTimeDays for s in completed])), 4),
        'UtilUrg%':       round(urgMins  / urgCapMins   * 100 if urgCapMins   > 0 else 0.0, 2),
        'UtilElec%':      round(elecMins / elecCapMins  * 100 if elecCapMins  > 0 else 0.0, 2),
        'UtilGlobal%':    round(
            (urgMins + elecMins) / totalCapMins * 100 if totalCapMins > 0 else 0.0, 2
        ),
        'PreAssigned':    preCount,
        'PreAssignedPct': round(preCount / totalElec * 100 if totalElec > 0 else 0.0, 2),
    }


def runPreAssignSimulation(
    weekIndex: dict, weekStartDates: list, simStart: datetime,
    heuristic: str, strategy: str,
    randomInsertionRate: float = 0.15, randomInsertionSeed: int = 42,
) -> dict:
    urgencies, electives = buildSimData(weekIndex, weekStartDates, simStart)
    engine  = PreAssignEngine(
        urgencies, electives, simStart, heuristic, strategy,
        randomInsertionRate, randomInsertionSeed,
    )
    results = engine.run()
    return calculatePreAssignMetrics(results, engine.preAssignedIds, simStart)
