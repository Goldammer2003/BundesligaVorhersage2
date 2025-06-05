# src/ingest.py
from pathlib import Path
from typing import List
import pandas as pd
import requests
from dateutil import tz
from tqdm import tqdm

from .config import SEASONS, FD_URL, RAW_DIR
from .utils import read_csv_cached, scrape_bulibox_h2h
from .fbref_ingest import add_fbref_xg


def load_fd(seasons: List[int] = SEASONS) -> pd.DataFrame:
    """
    Lädt alle historischen Bundesliga-Daten von football-data.co.uk,
    cached sie lokal und formatiert die DataFrame-Spalten.
    """
    frames = []
    for s in tqdm(seasons, desc="football-data"):
        url = FD_URL.format(y1=s % 100, y2=(s + 1) % 100)
        csv_path = RAW_DIR / f"fd_D1_{s}.csv"
        try:
            df = read_csv_cached(url, csv_path)
        except pd.errors.EmptyDataError:
            continue
        df["Season"] = f"{s}/{str(s+1)[-2:]}"
        frames.append(df)

    if not frames:
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

    # Datum parsen (fallback auf dayfirst)
    try:
        df_all["date"] = pd.to_datetime(df_all["date"], format="%d/%m/%Y", errors="raise")
    except Exception:
        df_all["date"] = pd.to_datetime(df_all["date"], dayfirst=True, errors="coerce")

    return df_all.dropna(subset=["date"])


def fetch_fixtures(season: int) -> pd.DataFrame:
    """
    Holt kommende Spieltage via OpenLigaDB-API.
    """
    url = f"https://www.openligadb.de/api/getmatchdata/bl1/{season}"
    js = requests.get(url, timeout=30).json()
    rows = []
    for m in js:
        rows.append({
            "match_id": m["MatchID"],
            "date": m["MatchDateTimeUTC"],
            "home_team": m["Team1"]["TeamName"],
            "away_team": m["Team2"]["TeamName"],
            "matchday": m["Group"]["GroupOrderID"],
        })
    df = pd.DataFrame(rows)
    berlin = tz.gettz("Europe/Berlin")
    df["date"] = pd.to_datetime(df["date"], utc=True) \
                  .dt.tz_convert(berlin) \
                  .dt.tz_localize(None)
    return df


def add_h2h(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reicht H2H-Win-Rate vom Bulibox-Scraper nach.
    """
    winrates = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc="Bulibox H2H"):
        winrates.append(scrape_bulibox_h2h(r.home_team, r.away_team))
    wr_df = pd.DataFrame(winrates)
    return pd.concat([df.reset_index(drop=True), wr_df], axis=1)