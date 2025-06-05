# src/fbref_ingest.py
from __future__ import annotations
from pathlib import Path
from typing import List, Dict
import time
import pandas as pd
import requests
from tqdm import tqdm

# Cache-Ordner für FBref-Downloads
_CACHE = Path("data/raw/fbref_cache")
_CACHE.mkdir(parents=True, exist_ok=True)

def _download_json(url: str, path: Path, delay: float = 0.3) -> Path | None:
    """
    Lädt eine JSON-Datei und cached sie.
    Bei 404 (Saison nicht vorhanden) wird None zurückgegeben.
    """
    if path.exists():
        return path
    time.sleep(delay)
    r = requests.get(url, timeout=20)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    path.write_bytes(r.content)
    return path

def _season_games(season: int) -> pd.DataFrame:
    """
    Liest das StatsBomb-Feed für eine Saison und extrahiert Home/Away-xG.
    """
    url = (
        f"https://raw.githubusercontent.com/"
        f"statsbomb/open-data/master/data/matches/{season}/9.json"
    )
    cache_file = _CACHE / f"matches_{season}.json"
    js_path = _download_json(url, cache_file)
    if js_path is None:
        return pd.DataFrame()  # keine Daten für diese Saison

    games = pd.read_json(js_path, orient="records")
    rows: List[Dict] = []
    for g in games:
        rows.append({
            "Season":   f"{season}/{str(season+1)[-2:]}",
            "date":     pd.to_datetime(g["match_date"]),
            "home_team": g["home_team"]["home_team_name"],
            "away_team": g["away_team"]["away_team_name"],
            "xg_home":   g["home_xg"],
            "xg_away":   g["away_xg"],
        })
    return pd.DataFrame(rows)

def load_fbref_xg(seasons: List[int]) -> pd.DataFrame:
    """
    Aggregiert die einzelnen Saison-DataFrames zu einem großen DataFrame.
    """
    dfs: List[pd.DataFrame] = []
    for s in tqdm(seasons, desc="FBref-xG"):
        df_s = _season_games(s)
        if not df_s.empty:
            dfs.append(df_s)
    if not dfs:
        return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)

def add_fbref_xg(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fügt die Spalten xg_home/xg_away zum football-data DataFrame hinzu.
    Fehlende Werte werden mit dem Saisondurchschnitt aufgefüllt.
    """
    seasons = sorted(df["Season"].str[:4].astype(int).unique())
    xg_df = load_fbref_xg(seasons)
    if xg_df.empty:
        # Wenn komplett leer, lege Spalten an und gib zurück
        df["xg_home"] = df["xg_away"] = pd.NA
        return df

    # Merge auf Season, date, home_team, away_team
    merged = df.merge(
        xg_df,
        how="left",
        on=["Season", "date", "home_team", "away_team"],
    )

    # Fehlende xG-Werte mit Saisondurchschnitt füllen
    merged["xg_home"] = merged["xg_home"].fillna(
        merged.groupby("Season")["xg_home"].transform("mean")
    )
    merged["xg_away"] = merged["xg_away"].fillna(
        merged.groupby("Season")["xg_away"].transform("mean")
    )

    # Typkonvertierung
    merged["xg_home"] = merged["xg_home"].astype(float)
    merged["xg_away"] = merged["xg_away"].astype(float)

    return merged