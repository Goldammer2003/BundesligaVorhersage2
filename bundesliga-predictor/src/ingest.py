"""
Laden & Cachen aller Datenquellen (football-data.co.uk, OpenLigaDB, Understat, Bulibox)
"""
from pathlib import Path
from typing import List

import pandas as pd
import requests
from tqdm import tqdm
from dateutil import tz

from .config import SEASONS, FD_URL, RAW_DIR
from .utils import read_csv_cached, parse_understat_match, scrape_bulibox_h2h

# -----------------------------------------------------------------------------
# football-data.co.uk  (historische Ergebnisse + Quoten)
# -----------------------------------------------------------------------------
def load_fd(seasons: List[int] = SEASONS) -> pd.DataFrame:
    frames = []
    for s in tqdm(seasons, desc="football-data"):
        url = FD_URL.format(y1=s % 100, y2=(s + 1) % 100)
        csv_path = RAW_DIR / f"fd_D1_{s}.csv"
        print(f"Lade Saison {s} von {url} ...")
        try:
            df = read_csv_cached(url, csv_path)
            print(f"✅ Daten geladen: {df.shape}")
        except pd.errors.EmptyDataError:
            print(f"⚠️ Leere Datei für Saison {s}")
            continue
        df["Season"] = f"{s}/{str(s+1)[-2:]}"
        frames.append(df)

    if not frames:
        print("❌ Keine Daten geladen.")
        return pd.DataFrame()

    df_all = pd.concat(frames, ignore_index=True)

    rename = {
        "Date": "date",
        "HomeTeam": "home_team",
        "AwayTeam": "away_team",
        "FTHG": "home_goals",
        "FTAG": "away_goals",
        "FTR": "result",
        "B365H": "odds_home",
        "B365D": "odds_draw",
        "B365A": "odds_away",
    }
    df_all = df_all.rename(columns=rename)[list(rename.values()) + ["Season"]]

    try:
        df_all["date"] = pd.to_datetime(df_all["date"], format="%d/%m/%Y", errors="raise")
    except Exception:
        df_all["date"] = pd.to_datetime(df_all["date"], dayfirst=True, errors="coerce")

    return df_all.dropna(subset=["date"])


# -----------------------------------------------------------------------------
# OpenLigaDB  (aktuelle/kommende Spieltage)
# -----------------------------------------------------------------------------
def openliga_request(path: str):
    url = f"https://www.openligadb.de/api/{path}"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()

def fetch_fixtures(season: int) -> pd.DataFrame:
    js = openliga_request(f"getmatchdata/bl1/{season}")
    rows = []
    for m in js:
        rows.append(
            {
                "match_id": m["MatchID"],
                "date": m["MatchDateTimeUTC"],
                "home_team": m["Team1"]["TeamName"],
                "away_team": m["Team2"]["TeamName"],
                "matchday": m["Group"]["GroupOrderID"],
            }
        )
    df = pd.DataFrame(rows)
    berlin = tz.gettz("Europe/Berlin")
    df["date"] = pd.to_datetime(df["date"], utc=True).dt.tz_convert(berlin).dt.tz_localize(None)
    return df

# -----------------------------------------------------------------------------
# Enrichment – Understat (xG)  &  Bulibox (H2H)
# -----------------------------------------------------------------------------
def add_understat_xg(df: pd.DataFrame) -> pd.DataFrame:
    xg_home, xg_away = [], []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Understat"):
        try:
            d = parse_understat_match(r.match_id)
            xg_home.append(d.get(r.home_team, float("nan")))
            xg_away.append(d.get(r.away_team, float("nan")))
        except Exception:
            xg_home.append(float("nan"))
            xg_away.append(float("nan"))
    df["xg_home"] = xg_home
    df["xg_away"] = xg_away
    return df

def add_h2h(df: pd.DataFrame) -> pd.DataFrame:
    winrates = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Bulibox H2H"):
        winrates.append(scrape_bulibox_h2h(r.home_team, r.away_team))
    wr_df = pd.DataFrame(winrates)
    return pd.concat([df.reset_index(drop=True), wr_df], axis=1)