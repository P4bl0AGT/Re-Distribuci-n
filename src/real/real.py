import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent.parent
DATA_DIR   = ROOT / 'data'
OUTPUT_DIR = ROOT / 'output' / 'real'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# --- CONFIGURATION MODULE ---
class BaselineConfig:
    csvPath = DATA_DIR / '2_dataset_procesado_actualizado.csv'
    targetWeekStart = '2023-04-17 00:00:00'
    simDays = 5

    # Capacidad Real (Ajusta esto según la realidad de tu hospital)
    urgencyOrsCount = 1             # Ej: El Q-8
    urgencyActiveHoursPerDay = 24   # 24/7

    electiveOrsCount = 11           # El resto de los pabellones
    electiveActiveHoursPerDay = 9   # Ej: 08:00 a 17:00 (9 horas)

# --- DATA MODELS ---
class BaselineSurgery:
    def __init__(self, surgeryId, isUrgency, orId, startTime, durationMins, simStart):
        self.surgeryId = surgeryId
        self.isUrgency = isUrgency
        self.orId = orId
        self.startTime = startTime
        self.durationMins = durationMins
        self.endTime = startTime + timedelta(minutes=durationMins)

        # Lead time: From the beginning of the simulated horizon to surgery completion
        self.leadTimeDays = (self.endTime - simStart).total_seconds() / 86400.0

# --- DATA PROCESSING ---
def loadBaselineData(config: BaselineConfig) -> list[BaselineSurgery]:
    df = pd.read_csv(config.csvPath, low_memory=False)

    # 1. Identificamos dinámicamente cuál es la columna correcta de ingreso
    if 'Ingreso Quirófano' in df.columns:
        colIngreso = 'Ingreso Quirófano'
    elif 'Ingreso Pabellón' in df.columns:
        colIngreso = 'Ingreso Pabellón'
    else:
        colIngreso = 'Fecha de Pabellón' # Último respaldo posible

    # 2. Identificamos las otras columnas dinámicamente
    colId = 'Número' if 'Número' in df.columns else 'idOperacion'
    colPabellon = 'Pabellón' if 'Pabellón' in df.columns else 'pabellon'

    # 3. Construimos el mapa solo con las columnas confirmadas
    columnMap = {
        colIngreso: 'ingresoQuirofano',
        'Tiempo Intervención': 'tiempoIntervencion',
        'Tipo Atención': 'tipoAtencion',
        colId: 'idOperacion',
        colPabellon: 'pabellon'
    }

    # Filtramos para renombrar solo lo que realmente existe en el dataframe
    columnMap = {k: v for k, v in columnMap.items() if k in df.columns}
    df.rename(columns=columnMap, inplace=True)

    # Si por algún motivo extraño aún hay columnas duplicadas, nos quedamos solo con la primera
    if isinstance(df['ingresoQuirofano'], pd.DataFrame):
        df['ingresoQuirofano'] = df['ingresoQuirofano'].iloc[:, 0]

    # Ahora sí parseamos la fecha con seguridad
    df['ingresoQuirofano'] = pd.to_datetime(df['ingresoQuirofano'], errors='coerce')
    df.dropna(subset=['ingresoQuirofano', 'tiempoIntervencion'], inplace=True)

    simStart = pd.to_datetime(config.targetWeekStart)
    simEnd = simStart + timedelta(days=config.simDays)

    mask = (df['ingresoQuirofano'] >= simStart) & (df['ingresoQuirofano'] < simEnd)
    weekData = df[mask]

    surgeries = []
    for i, row in weekData.iterrows():
        tipo = str(row.get('tipoAtencion', '')).upper()
        isUrg = 'URGENCIA' in tipo

        surgery = BaselineSurgery(
            surgeryId=str(row.get('idOperacion', i)),
            isUrgency=isUrg,
            orId=str(row.get('pabellon', 'Desconocido')),
            startTime=row['ingresoQuirofano'],
            durationMins=float(row['tiempoIntervencion']),
            simStart=simStart
        )
        surgeries.append(surgery)

    return surgeries

# --- METRICS CALCULATION ---
def calculateBaselineMetrics(surgeries: list[BaselineSurgery], config: BaselineConfig) -> dict:
    throughput = len(surgeries)

    if throughput == 0:
        return {"Throughput": 0, "LeadTimeDays": 0.0, "UtilUrg%": 0.0, "UtilElec%": 0.0, "UtilTotal%": 0.0}

    leadTimeDays = np.mean([s.leadTimeDays for s in surgeries])

    urgMins = sum(s.durationMins for s in surgeries if s.isUrgency)
    elecMins = sum(s.durationMins for s in surgeries if not s.isUrgency)

    # Capacidades ajustadas a la realidad
    urgCapMins = config.urgencyOrsCount * config.urgencyActiveHoursPerDay * config.simDays * 60
    elecCapMins = config.electiveOrsCount * config.electiveActiveHoursPerDay * config.simDays * 60

    utilUrg = (urgMins / urgCapMins * 100) if urgCapMins > 0 else 0.0
    utilElec = (elecMins / elecCapMins * 100) if elecCapMins > 0 else 0.0

    totalCapMins = urgCapMins + elecCapMins
    utilTotal = ((urgMins + elecMins) / totalCapMins * 100) if totalCapMins > 0 else 0.0

    return {
        "Throughput": throughput,
        "LeadTimeDays": round(leadTimeDays, 5),
        "UtilUrg%": round(utilUrg, 5),
        "UtilElec%": round(utilElec, 5),
        "UtilTotal%": round(utilTotal, 5)
    }

# --- GANTT CHART GENERATION ---
def drawBaselineGantt(surgeries: list[BaselineSurgery], config: BaselineConfig):
    if not surgeries:
        print("No surgeries to plot.")
        return

    fig, ax = plt.subplots(figsize=(24, 10))
    ax.set_facecolor("#F8F9FA")

    simStart = pd.to_datetime(config.targetWeekStart)

    # Extract and sort unique ORs
    uniqueOrs = sorted(list(set(s.orId for s in surgeries)))
    orIndexMap = {orId: idx for idx, orId in enumerate(uniqueOrs)}

    colors = {True: '#D32F2F', False: '#1565C0'}

    for s in surgeries:
        orIndex = orIndexMap[s.orId]

        # Calculate X position as total continuous hours from simStart
        startHours = (s.startTime - simStart).total_seconds() / 3600.0
        durationHrs = s.durationMins / 60.0

        ax.barh(
            orIndex, durationHrs, left=startHours,
            color=colors[s.isUrgency], edgecolor='none', alpha=0.9, zorder=3
        )

    # --- Y-Axis Formatting (OR Labels) ---
    ax.set_yticks(range(len(uniqueOrs)))
    ax.set_yticklabels([f"OR {orId}" for orId in uniqueOrs], fontweight='bold')
    ax.set_ylabel("Operating Room", fontweight='bold', labelpad=10)

    # --- X-Axis Formatting (Time Slots) ---
    totalHours = config.simDays * 24
    xTicks = np.arange(0, totalHours + 1, 2)

    xLabels = [f"{(int(h) % 24):02d}:00" for h in xTicks]

    ax.set_xticks(xTicks)
    ax.set_xticklabels(xLabels, fontsize=8, rotation=45)
    ax.set_xlim(0, totalHours)
    ax.set_xlabel("Time of Day", fontweight='bold', labelpad=10)

    # Grid configuration
    ax.grid(axis='x', color='#E0E0E0', linestyle='-', linewidth=0.5, zorder=0)
    ax.grid(axis='y', color='#E0E0E0', linestyle='-', linewidth=0.5, zorder=0)

    # --- Day Separators & Headers ---
    dayNames = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    monthNames = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]

    yMax = len(uniqueOrs) - 0.4

    for d in range(config.simDays):
        dayStartHour = d * 24

        if d > 0:
            ax.axvline(x=dayStartHour, color='#A0A0A0', linestyle='--', alpha=0.7, zorder=1)

        currentDate = simStart + timedelta(days=d)
        dayString = f"{dayNames[currentDate.weekday()]} {currentDate.day} {monthNames[currentDate.month - 1]}"
        midPoint = dayStartHour + 12

        ax.text(
            midPoint, yMax, dayString,
            ha='center', va='bottom', fontweight='bold', fontsize=11, color='#333333'
        )

    # --- Global Title ---
    endDate = simStart + timedelta(days=config.simDays - 1)
    startStr = f"{simStart.day} {monthNames[simStart.month - 1]}"
    endStr = f"{endDate.day} {monthNames[endDate.month - 1]}"

    ax.set_title(f"Reality Baseline Gantt Chart — ({startStr} - {endStr})", fontsize=15, fontweight='bold', pad=35)

    # --- Legend ---
    legendPatches = [
        mpatches.Patch(color='#D32F2F', label='Urgency'),
        mpatches.Patch(color='#1565C0', label='Elective')
    ]
    ax.legend(handles=legendPatches, loc='lower right', framealpha=1.0)

    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_color("#CCCCCC")

    plt.tight_layout()
    outPath = OUTPUT_DIR / 'baselineGantt_continuous.png'
    plt.savefig(outPath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Baseline Gantt chart ({config.simDays} days) exported as '{outPath}'")

# --- MAIN EXECUTION ---
def main():
    config = BaselineConfig()

    print("Running Baseline Reality Simulation...")
    surgeries = loadBaselineData(config)

    metrics = calculateBaselineMetrics(surgeries, config)

    print("\n--- BASELINE METRICS ---")
    print(f"Throughput:    {metrics['Throughput']} surgeries")
    print(f"Lead Time:     {metrics['LeadTimeDays']} days")
    print(f"Util(Elec):    {metrics['UtilElec%']}% (Base {config.electiveActiveHoursPerDay}h)")
    print(f"Util(Urg):     {metrics['UtilUrg%']}% (Base {config.urgencyActiveHoursPerDay}h)")
    print(f"Util(Total):   {metrics['UtilTotal%']}% (Adjusted)")

    drawBaselineGantt(surgeries, config)

main()
