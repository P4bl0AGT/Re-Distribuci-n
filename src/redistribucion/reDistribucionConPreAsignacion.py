"""
Redistribucion quirurgica con pre-asignacion aleatoria.

Flujo:
  1. Ordena las electivas segun la heuristica (FIFO/SPT/LPT)
  2. Selecciona un porcentaje al azar (randomInsertionRate)
  3. Coloca esas cirugias en pabellones y horarios aleatorios dentro del Gantt
  4. Rellena los huecos restantes con las electivas sobrantes usando
     la heuristica y estrategia correspondiente (VERTICAL/HORIZONTAL)

Uso:
    python src/reDistribucionConPreAsignacion.py
"""

import pandas as pd
import numpy as np
import heapq
import copy
import random
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent.parent
DATA_DIR   = ROOT / 'data'
OUTPUT_DIR = ROOT / 'output' / 'reDistribucionConPreAsignacion'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --- CONFIGURATION ---
class RedistConfig:
    csvPath = DATA_DIR / '2_dataset_procesado_actualizado.csv'
    targetWeekStart = '2023-04-17 00:00:00'

    weeksToCollapse = 5
    simDays = 5

    totalOrs = 10
    urgencyOrs = 2
    electiveOrs = 8

    urgencyStartHour = 0
    urgencyEndHour = 24
    shiftOverflowDelta = 1

    electiveStartHour = 8
    electiveEndHour = 16

    heuristics = ["FIFO", "SPT", "LPT"]
    strategies = ["VERTICAL", "HORIZONTAL"]

    randomInsertionRate = 0.15
    randomInsertionSeed = 42


# --- DATA MODELS ---
class RedistSurgery:
    def __init__(self, surgeryId, isUrgency, durationMins, arrivalTime, requestDate, originWeek):
        self.surgeryId = surgeryId
        self.isUrgency = isUrgency
        self.durationMins = durationMins
        self.arrivalTime = arrivalTime
        self.requestDate = requestDate
        self.originWeek = originWeek

        self.assignedOr = None
        self.startTime = None
        self.endTime = None
        self.leadTimeDays = 0.0
        self.preAssigned = False


class SimEvent:
    def __init__(self, time, eventType, payload=None):
        self.time = time
        self.eventType = eventType
        self.payload = payload

    def __lt__(self, other):
        return self.time < other.time


# --- DATA PROCESSING ---
def loadRedistData(config):
    df = pd.read_csv(config.csvPath, low_memory=False)

    if 'Ingreso Quirófano' in df.columns:
        colIngreso = 'Ingreso Quirófano'
    elif 'Ingreso Pabellón' in df.columns:
        colIngreso = 'Ingreso Pabellón'
    else:
        colIngreso = 'Fecha de Pabellón'

    colId = 'Número' if 'Número' in df.columns else 'idOperacion'
    colPabellon = 'Pabellón' if 'Pabellón' in df.columns else 'pabellon'

    columnMap = {
        colIngreso: 'ingresoQuirofano',
        'Tiempo Intervención': 'tiempoIntervencion',
        'Tipo Atención': 'tipoAtencion',
        colId: 'idOperacion',
        colPabellon: 'pabellon'
    }

    if 'fechaSolicitud' in df.columns:
        columnMap['fechaSolicitud'] = 'fechaSolicitud'

    columnMap = {k: v for k, v in columnMap.items() if k in df.columns}
    df.rename(columns=columnMap, inplace=True)

    if isinstance(df.get('ingresoQuirofano'), pd.DataFrame):
        df['ingresoQuirofano'] = df['ingresoQuirofano'].iloc[:, 0]

    df['ingresoQuirofano'] = pd.to_datetime(df['ingresoQuirofano'], errors='coerce')

    if 'fechaSolicitud' in df.columns:
        df['fechaSolicitud'] = pd.to_datetime(df['fechaSolicitud'], errors='coerce')
    else:
        df['fechaSolicitud'] = df['ingresoQuirofano']

    df.dropna(subset=['ingresoQuirofano', 'tiempoIntervencion'], inplace=True)

    simStart = pd.to_datetime(config.targetWeekStart)
    baseWeekStart = df['ingresoQuirofano'].min().replace(hour=0, minute=0, second=0)

    urgencies = []
    electives = []

    for i, row in df.iterrows():
        arrival = row['ingresoQuirofano']

        daysDiff = (arrival.date() - baseWeekStart.date()).days
        originWeek = daysDiff // 7

        if originWeek >= config.weeksToCollapse:
            continue

        collapsedDays = daysDiff % 7
        collapsedArrival = simStart + timedelta(days=collapsedDays)
        collapsedArrival = collapsedArrival.replace(hour=arrival.hour, minute=arrival.minute, second=0)

        reqDate = row.get('fechaSolicitud', collapsedArrival)
        if pd.isna(reqDate):
            reqDate = collapsedArrival

        tipo = str(row.get('tipoAtencion', '')).upper()
        isUrg = 'URGENCIA' in tipo

        surg = RedistSurgery(
            surgeryId=str(row.get('idOperacion', i)),
            isUrgency=isUrg,
            durationMins=float(row['tiempoIntervencion']),
            arrivalTime=collapsedArrival,
            requestDate=reqDate,
            originWeek=originWeek
        )

        if isUrg:
            urgencies.append(surg)
        else:
            surg.arrivalTime = simStart
            electives.append(surg)

    return urgencies, electives, simStart


# --- DISCRETE EVENT ENGINE WITH PRE-ASSIGNMENT ---
class RedistEngine:
    def __init__(self, urgencies, electives, simStart, heuristic, strategy, config):
        self.config = config
        self.simStart = simStart
        self.simEnd = simStart + timedelta(days=config.simDays)
        self.heuristic = heuristic
        self.strategy = strategy
        self.rng = random.Random(config.randomInsertionSeed)

        self.pendingUrgencies = sorted(urgencies, key=lambda x: (x.arrivalTime, x.originWeek))
        self.pendingElectives = list(electives)
        self._sortElectives()

        self.activeUrgencies = []
        self.completedSurgeries = []
        self.preAssignedSurgeries = []

        self.orReservations = {i: [] for i in range(config.totalOrs)}

        self._prePlaceRandom()

        self.orFreeTime = {i: simStart for i in range(config.totalOrs)}
        self.events = []

        for u in self.pendingUrgencies:
            heapq.heappush(self.events, SimEvent(u.arrivalTime, 'URG_ARRIVAL', u))

        for i in range(config.totalOrs):
            heapq.heappush(self.events, SimEvent(simStart, 'OR_FREE', i))

    def _sortElectives(self):
        if self.heuristic == "FIFO":
            self.pendingElectives.sort(key=lambda x: (x.requestDate, x.originWeek))
        elif self.heuristic == "SPT":
            self.pendingElectives.sort(key=lambda x: (x.durationMins, x.originWeek))
        elif self.heuristic == "LPT":
            self.pendingElectives.sort(key=lambda x: (-x.durationMins, x.originWeek))

    def _prePlaceRandom(self):
        n = int(len(self.pendingElectives) * self.config.randomInsertionRate)
        if n == 0:
            return

        indices = self.rng.sample(range(len(self.pendingElectives)), n)
        selected = []
        for idx in sorted(indices, reverse=True):
            selected.append(self.pendingElectives.pop(idx))

        limitHour = self.config.electiveEndHour + self.config.shiftOverflowDelta
        maxRetries = 100

        for surg in selected:
            placed = False
            for _ in range(maxRetries):
                orId = self.rng.randint(self.config.urgencyOrs, self.config.totalOrs - 1)
                day = self.rng.randint(0, self.config.simDays - 1)

                dayStart = self.simStart + timedelta(days=day)
                elecStart = dayStart.replace(hour=self.config.electiveStartHour, minute=0, second=0)
                elecEnd = dayStart.replace(hour=limitHour, minute=0, second=0)

                windowMins = int((elecEnd - elecStart).total_seconds() / 60)
                availableMins = windowMins - int(surg.durationMins)
                if availableMins < 0:
                    continue

                offsetMins = self.rng.randint(0, availableMins)
                propStart = elecStart + timedelta(minutes=offsetMins)
                propEnd = propStart + timedelta(minutes=surg.durationMins)

                conflict = False
                for resStart, resEnd in self.orReservations[orId]:
                    if propStart < resEnd and propEnd > resStart:
                        conflict = True
                        break

                if not conflict:
                    surg.preAssigned = True
                    surg.assignedOr = orId
                    surg.startTime = propStart
                    surg.endTime = propEnd
                    surg.leadTimeDays = (propEnd - surg.arrivalTime).total_seconds() / 86400.0

                    self.orReservations[orId].append((propStart, propEnd))
                    self.orReservations[orId].sort()
                    self.completedSurgeries.append(surg)
                    self.preAssignedSurgeries.append(surg)
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
        surg.assignedOr = orId
        surg.startTime = startTime
        surg.endTime = startTime + timedelta(minutes=surg.durationMins)
        surg.leadTimeDays = (surg.endTime - surg.arrivalTime).total_seconds() / 86400.0

        self.orFreeTime[orId] = surg.endTime
        self.completedSurgeries.append(surg)
        heapq.heappush(self.events, SimEvent(surg.endTime, 'OR_FREE', orId))

    def run(self):
        limitDelta = self.config.electiveEndHour + self.config.shiftOverflowDelta

        while self.events:
            e = heapq.heappop(self.events)
            t = e.time

            if t > self.simEnd:
                break

            if e.eventType == 'URG_ARRIVAL':
                self.activeUrgencies.append(e.payload)

            # 1. Urgencies
            freeUrgOrs = [i for i in range(self.config.urgencyOrs) if self.orFreeTime[i] <= t]
            validUrgOrs = [
                i for i in freeUrgOrs
                if self.config.urgencyStartHour <= t.hour < (self.config.urgencyEndHour + self.config.shiftOverflowDelta)
            ]

            while validUrgOrs and self.activeUrgencies:
                orId = validUrgOrs.pop(0)
                surg = self.activeUrgencies.pop(0)
                self._assign(surg, orId, max(t, self.orFreeTime[orId]))

            # 2. Electives
            freeElecOrs = [
                i for i in range(self.config.urgencyOrs, self.config.totalOrs)
                if self.orFreeTime[i] <= t
            ]

            if t.hour >= limitDelta or t.hour < self.config.electiveStartHour:
                for orId in freeElecOrs:
                    nextOpen = t.replace(hour=self.config.electiveStartHour, minute=0, second=0)
                    if t.hour >= limitDelta:
                        nextOpen += timedelta(days=1)
                    if self.orFreeTime[orId] < nextOpen:
                        self.orFreeTime[orId] = nextOpen
                        heapq.heappush(self.events, SimEvent(nextOpen, 'OR_FREE', orId))
                continue

            if self.strategy == "VERTICAL":
                for orId in freeElecOrs:
                    if not self.pendingElectives:
                        break
                    afterTime = max(t, self.orFreeTime[orId])
                    surg = self.pendingElectives[0]

                    dayLimitTime = afterTime.replace(
                        hour=self.config.electiveEndHour, minute=0, second=0
                    ) + timedelta(hours=self.config.shiftOverflowDelta)

                    slot = self._findNextSlot(orId, afterTime, surg.durationMins, dayLimitTime)
                    if slot is not None:
                        self.pendingElectives.pop(0)
                        self._assign(surg, orId, slot)
                    else:
                        nextOpen = afterTime.replace(
                            hour=self.config.electiveStartHour, minute=0, second=0
                        ) + timedelta(days=1)
                        self.orFreeTime[orId] = nextOpen
                        heapq.heappush(self.events, SimEvent(nextOpen, 'OR_FREE', orId))

            elif self.strategy == "HORIZONTAL":
                for orId in freeElecOrs:
                    while self.pendingElectives:
                        afterTime = max(t, self.orFreeTime[orId])
                        surg = self.pendingElectives[0]

                        dayLimitTime = afterTime.replace(
                            hour=self.config.electiveEndHour, minute=0, second=0
                        ) + timedelta(hours=self.config.shiftOverflowDelta)

                        slot = self._findNextSlot(orId, afterTime, surg.durationMins, dayLimitTime)
                        if slot is None:
                            nextOpen = afterTime.replace(
                                hour=self.config.electiveStartHour, minute=0, second=0
                            ) + timedelta(days=1)
                            self.orFreeTime[orId] = nextOpen
                            heapq.heappush(self.events, SimEvent(nextOpen, 'OR_FREE', orId))
                            break

                        self.pendingElectives.pop(0)
                        self._assign(surg, orId, slot)

        return self.completedSurgeries


# --- METRICS ---
def calculateRedistMetrics(completed, preAssigned, config):
    simEnd = pd.to_datetime(config.targetWeekStart) + timedelta(days=config.simDays)
    completedInHorizon = [s for s in completed if s.endTime <= simEnd]

    throughput = len(completedInHorizon)
    preAssignedCount = len(preAssigned)

    if throughput == 0:
        return {
            "Throughput": 0,
            "LeadTimeDays": 0.0,
            "UtilUrg%": 0.0,
            "UtilElec%": 0.0,
            "ElecHoursBase": 0,
            "PreAssigned": preAssignedCount,
            "PreAssignedRate%": 0.0,
        }

    leadTimeDays = np.mean([s.leadTimeDays for s in completedInHorizon])

    urgMins = sum(s.durationMins for s in completedInHorizon if s.isUrgency)
    elecMins = sum(s.durationMins for s in completedInHorizon if not s.isUrgency)

    urgCapMins = config.urgencyOrs * (config.urgencyEndHour - config.urgencyStartHour) * config.simDays * 60
    elecHoursPerDay = (config.electiveEndHour - config.electiveStartHour) + config.shiftOverflowDelta
    elecCapMins = config.electiveOrs * elecHoursPerDay * config.simDays * 60

    utilUrg = (urgMins / urgCapMins * 100) if urgCapMins > 0 else 0.0
    utilElec = (elecMins / elecCapMins * 100) if elecCapMins > 0 else 0.0

    totalElec = len([s for s in completedInHorizon if not s.isUrgency])
    actualRate = (preAssignedCount / (preAssignedCount + totalElec - preAssignedCount) * 100) if totalElec > 0 else 0.0

    return {
        "Throughput": throughput,
        "LeadTimeDays": round(leadTimeDays, 2),
        "UtilUrg%": round(utilUrg, 2),
        "UtilElec%": round(utilElec, 2),
        "ElecHoursBase": elecHoursPerDay,
        "PreAssigned": preAssignedCount,
        "PreAssignedRate%": round(actualRate, 2),
    }


# --- GANTT CHART ---
def drawRedistGantt(completed, preAssigned, config, scenarioName):
    if not completed:
        print(f"No surgeries to plot for scenario {scenarioName}.")
        return

    fig, ax = plt.subplots(figsize=(24, 10))
    ax.set_facecolor("#F8F9FA")

    simStart = pd.to_datetime(config.targetWeekStart)

    for s in completed:
        if s.startTime is None:
            continue

        orIndex = s.assignedOr
        startHours = (s.startTime - simStart).total_seconds() / 3600.0
        durationHrs = s.durationMins / 60.0

        if s.isUrgency:
            ax.barh(
                orIndex, durationHrs, left=startHours,
                color='#D32F2F', edgecolor='white', linewidth=1.2, alpha=0.9, zorder=3
            )
        elif s.preAssigned:
            ax.barh(
                orIndex, durationHrs, left=startHours,
                color='#F57C00', edgecolor='white', linewidth=1.2, alpha=0.9, zorder=3
            )
        else:
            ax.barh(
                orIndex, durationHrs, left=startHours,
                color='#1565C0', edgecolor='white', linewidth=1.2, alpha=0.9, zorder=3
            )

    ax.set_yticks(range(config.totalOrs))
    yLabels = [
        f"OR {i} (Urg)" if i < config.urgencyOrs else f"OR {i} (Elec)"
        for i in range(config.totalOrs)
    ]
    ax.set_yticklabels(yLabels, fontweight='bold')
    ax.set_ylabel("Operating Room", fontweight='bold', labelpad=10)

    totalHours = config.simDays * 24
    xTicks = np.arange(0, totalHours + 1, 2)
    xLabels = [f"{(int(h) % 24):02d}:00" for h in xTicks]
    ax.set_xticks(xTicks)
    ax.set_xticklabels(xLabels, fontsize=8, rotation=45)
    ax.set_xlim(0, totalHours)
    ax.set_xlabel("Time of Day", fontweight='bold', labelpad=10)

    ax.grid(axis='x', color='#E0E0E0', linestyle='-', linewidth=0.5, zorder=0)
    ax.grid(axis='y', color='#E0E0E0', linestyle='-', linewidth=0.5, zorder=0)

    dayNames = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    monthNames = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    yMax = config.totalOrs - 0.4

    for d in range(config.simDays):
        dayStartHour = d * 24
        if d > 0:
            ax.axvline(x=dayStartHour, color='#A0A0A0', linestyle='--', alpha=0.7, zorder=1)

        currentDate = simStart + timedelta(days=d)
        dayString = f"{dayNames[currentDate.weekday()]} {currentDate.day} {monthNames[currentDate.month - 1]}"
        ax.text(
            dayStartHour + 12, yMax, dayString,
            ha='center', va='bottom', fontweight='bold', fontsize=11, color='#333333'
        )

    endDate = simStart + timedelta(days=config.simDays - 1)
    startStr = f"{simStart.day} {monthNames[simStart.month - 1]}"
    endStr = f"{endDate.day} {monthNames[endDate.month - 1]}"

    preCount = len(preAssigned)
    ax.set_title(
        f"Redistribution + Random Pre-assignment ({scenarioName}) — {config.weeksToCollapse} Weeks Collapsed\n"
        f"({startStr} - {endStr})  |  Pre-assigned: {config.randomInsertionRate*100:.0f}%  |  Pre-assigned slots: {preCount}",
        fontsize=14, fontweight='bold', pad=35
    )

    legendPatches = [
        mpatches.Patch(color='#D32F2F', label='Urgency'),
        mpatches.Patch(color='#F57C00', label='Elective (pre-assigned randomly)'),
        mpatches.Patch(color='#1565C0', label='Elective (heuristic-assigned)'),
    ]
    ax.legend(handles=legendPatches, loc='lower right', framealpha=1.0)

    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_color("#CCCCCC")

    plt.tight_layout()
    outPath = OUTPUT_DIR / f"gantt_PREASSIGN_{scenarioName}.png"
    plt.savefig(outPath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Gantt exported -> '{outPath}'")


# --- MAIN ---
def main():
    config = RedistConfig()
    print(f"Loading and collapsing data ({config.weeksToCollapse} weeks)...")
    print(f"Random insertion rate: {config.randomInsertionRate*100:.0f}%  |  seed: {config.randomInsertionSeed}")
    baseUrgencies, baseElectives, simStart = loadRedistData(config)
    print(f"  {len(baseUrgencies)} urgencies, {len(baseElectives)} electives loaded\n")

    print("--- RUNNING REDISTRIBUTION + RANDOM PRE-ASSIGNMENT SCENARIOS ---")

    for strategy in config.strategies:
        for heuristic in config.heuristics:
            urgs = copy.deepcopy(baseUrgencies)
            elecs = copy.deepcopy(baseElectives)

            engine = RedistEngine(urgs, elecs, simStart, heuristic, strategy, config)
            completed = engine.run()
            preAssigned = engine.preAssignedSurgeries

            metrics = calculateRedistMetrics(completed, preAssigned, config)
            scenarioStr = f"{strategy}_{heuristic}"

            print(
                f"[{scenarioStr.ljust(20)}] "
                f"Throughput: {metrics['Throughput']:4d} | "
                f"Lead Time: {metrics['LeadTimeDays']}d | "
                f"Util(Elec): {metrics['UtilElec%']}% (Base {metrics['ElecHoursBase']}h) | "
                f"Util(Urg): {metrics['UtilUrg%']}% | "
                f"PreAssigned: {metrics['PreAssigned']} ({metrics['PreAssignedRate%']}%)"
            )

            drawRedistGantt(completed, preAssigned, config, scenarioStr)

    print("\nScenarios evaluated and Gantts exported successfully.")


main()
