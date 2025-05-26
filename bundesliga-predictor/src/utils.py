"""
Hilfsfunktionen zum Cachen, Parsen und Scrapen
"""
from pathlib import Path
import pandas as pd
import requests
import json
import time
from bs4 import BeautifulSoup


def read_csv_cached(url: str, path: Path, delay: float = 1.0) -> pd.DataFrame:
    """
    Lädt eine CSV-Datei aus dem Cache (falls vorhanden), andernfalls aus dem Web und speichert sie.
    """
    if path.exists():
        return pd.read_csv(path, encoding="ISO-8859-1", sep=",", on_bad_lines="skip")

    time.sleep(delay)  # um Server nicht zu überlasten
    df = pd.read_csv(url, encoding="ISO-8859-1", sep=",", on_bad_lines="skip")
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, encoding="ISO-8859-1")
    return df


def parse_understat_match(match_id: int) -> dict:
    """
    Holt die xG-Werte aus dem Understat-Spiel anhand der Match-ID.
    Gibt ein Dictionary mit Teamnamen und ihren xG-Werten zurück.
    """
    url = f"https://understat.com/match/{match_id}"
    r = requests.get(url, timeout=30)
    soup = BeautifulSoup(r.content, "html.parser")
    scripts = soup.find_all("script")

    for script in scripts:
        if "shotsData" in script.text:
            text = script.text.split("('")[1].split("')")[0]
            json_text = bytes(text, "utf-8").decode("unicode_escape")
            data = json.loads(json_text)
            break
    else:
        return {}

    xg_totals = {}
    for team in ["h", "a"]:
        xg = sum(float(entry["xG"]) for entry in data[team])
        team_name = data[team][0]["team"] if data[team] else f"Team_{team}"
        xg_totals[team_name] = xg
    return xg_totals


def scrape_bulibox_h2h(home: str, away: str) -> dict:
    """
    Holt die historische Siegquote des Heimteams aus der Bulibox-Seite.
    Gibt einen Dictionary-Eintrag wie {'h2h_home_winrate': 0.55} zurück.
    """
    url = f"https://www.bulibox.de/stats/{home}_vs_{away}"
    try:
        r = requests.get(url, timeout=10)
        soup = BeautifulSoup(r.content, "html.parser")
        rate_text = soup.select_one(".winrate-home")
        if rate_text:
            rate = float(rate_text.text.strip("%")) / 100
        else:
            rate = 0.5  # fallback
    except Exception:
        rate = 0.5  # fallback bei Fehler
    return {"h2h_home_winrate": rate}