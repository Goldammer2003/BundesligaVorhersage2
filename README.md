# ⚽ BundesligaVorhersage2

Ein datengetriebenes Machine-Learning-Projekt zur Vorhersage von Bundesliga-Ergebnissen.  
Basierend auf historischen Daten werden verschiedene Modelle trainiert, um den Ausgang zukünftiger Spiele vorherzusagen – optional mit Meisterschaftssimulation.

---

## 📦 Funktionsumfang

- Automatischer Download historischer Bundesliga-Daten (ab Saison 1993/94)  
- Feature Engineering (Wettquoten, Form, Tor- und xG-Differenzen, optional H2H)  
- Training mehrerer ML-Modelle (RandomForest, LogisticRegression, SVC)  
- Evaluation anhand zahlreicher Metriken (Accuracy, LogLoss, RMSE)  
- Speicherung des besten Modells (`models/best_model.pkl`)  
- Erstellung einer Korrelationsmatrix (`reports/correlation_matrix.png`)  
- Erzeugung der finalen Inputdaten als CSV (`data/final/season_summary.csv`)  
- **Flask-Frontend** zur interaktiven Meister-Simulation ab fiktivem Spieltag  
  (Formular unter `/` → Ausgabe unter `/simulate`)  

---

## 🛠️ Installation & Setup

### Voraussetzungen

- Python 3.11 oder höher  
- Git  

### 1. Repository klonen

```bash
git clone https://github.com/Goldammer2003/BundesligaVorhersage2.git
cd bundesliga-predictor


Virtuelle Umgebung erstellen: 

macOS/Linux: 
python3 -m venv venv
source venv/bin/activate

Windows (CMD):
python -m venv venv
venv\Scripts\activate

Abhängigkeiten installieren: 
pip install --upgrade pip
pip install -r requirements.txt


Projekt ausführen 

1. Modelltraining starten 
python -m src.train


2. Reports & Modelle 
	•	📊 Korrelationsmatrix: reports/correlation_matrix.png
	•	🧠 Modell-Metriken: reports/model_metrics.csv
	•	🔐 Bestes Modell: models/best_model.pkl


    🔒 Hinweis zu .gitignore

3. Flask App starten
python run.py

``` Falls der Port von einem vorherigen Laden noch in Benutzung ist: 
1. Finde den Prozess, der Port 8000 blockiert: 
lsof -i :8000

Beispielausgabe:
COMMAND     PID       USER   FD   TYPE             DEVICE SIZE/OFF NODE NAME
python3.1 23175 prometheus    6u  IPv4 ...      0t0  TCP *:irdmi (LISTEN)

2. Beende den Prozess 
kill -9 23175 
-> Ersetze 23175 durch die tatsächliche PID aus deinem Befehl oben 

3. starte Flask danach erneut: 
python run.py 

```

Die virtuelle Umgebung (venv/) und Outputs wie models/ oder reports/ werden nicht in Git getrackt – siehe .gitignore.


''' 