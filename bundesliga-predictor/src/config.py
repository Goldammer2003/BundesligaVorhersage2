from pathlib import Path

# ------------------------------------------------------------------------
# Basisverzeichnisse automatisch relativ zum Projektverzeichnis bestimmen
# ------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent  # Projektstammverzeichnis

DATA_DIR  = ROOT / "data"
RAW_DIR   = DATA_DIR / "raw"
PROC_DIR  = DATA_DIR / "processed"
MODEL_DIR = ROOT / "models"

# Alle Ordner anlegen, wenn sie noch nicht existieren
for d in (DATA_DIR, RAW_DIR, PROC_DIR, MODEL_DIR):
    d.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------------
# Saisonkonfiguration und externe Datenquellen
# ------------------------------------------------------------------------

# Alle verfügbaren Bundesliga-Saisons von 1993/94 bis einschließlich 2024/25
SEASONS = list(range(1993, 2025))  # z. B. 1993 → Saison 1993/94

# URL-Vorlage für football-data.co.uk CSV-Dateien
# Beispiel: https://www.football-data.co.uk/mmz4281/9394/D1.csv
FD_URL = "https://www.football-data.co.uk/mmz4281/{y1:02d}{y2:02d}/D1.csv"