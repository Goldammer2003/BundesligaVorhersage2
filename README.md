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
cd BundesligaVorhersage2