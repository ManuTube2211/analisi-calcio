
# analysis.py – toolkit per Football Analysis
# Versione riscritta: fix modello Dixon-Coles, groupby più robusto, fallback xG coerente
 
from __future__ import annotations
 
import io
import math
import html
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
from email.utils import parsedate_to_datetime
from typing import Dict, Tuple
from urllib.parse import urlencode
from xml.etree import ElementTree as ET

import numpy as np
import pandas as pd
import requests
 
# ============================================================
# Normalizzazione dati
# ============================================================
 
# Colonne che, se assenti, vengono create a 0 (statistiche opzionali)
OPTIONAL_NUMERIC_COLS = [
    "home_possession", "away_possession",
    "home_shots", "away_shots",
    "home_shots_on_target", "away_shots_on_target",
    "home_corners", "away_corners",
    "home_yellow", "away_yellow",
    "home_red", "away_red",
]
 
REQUIRED_COLS = ["date", "home_team", "away_team", "home_score", "away_score"]
 
RENAME_MAP = {
    "xG": "home_xg",
    "xG.1": "away_xg",
    "home_xG": "home_xg",
    "away_xG": "away_xg",
    "home_exp_goals": "home_xg",
    "away_exp_goals": "away_xg",
    "away_possesion": "away_possession",  # typo comune nei CSV
}
 
 
def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizza colonne e tipi. Gli xG sono mantenuti per il solo motore predittivo.
 
    Solleva ValueError con un messaggio chiaro se mancano colonne obbligatorie,
    così l'app puo' mostrare l'errore invece di un traceback grezzo di pandas.
    """
    df = df.copy()
 
    # 1) Pulizia nomi colonna
    df.columns = [c.strip() for c in df.columns]
    df.rename(columns=RENAME_MAP, inplace=True)
 
    # 2) Controllo colonne obbligatorie
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Mancano colonne obbligatorie: {missing}. "
            f"Colonne trovate nel file: {list(df.columns)}"
        )
 
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
 
    # 2b) Nomi squadra: rimuovo spazi extra, altrimenti "Milan" e "Milan " (con uno
    # spazio in più per un errore di battitura nel CSV) verrebbero trattate come due
    # squadre diverse, con statistiche sparse su due righe invece che una sola.
    df["home_team"] = df["home_team"].astype(str).str.strip()
    df["away_team"] = df["away_team"].astype(str).str.strip()
 
    # 3) Pulizia possesso palla: "55%" -> 55.0 ; "55,3%" -> 55.3
    for col in ["home_possession", "away_possession"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace("%", "", regex=False)
                .str.replace(",", ".", regex=False)
            )
 
    # 4) Conversione numerica delle colonne opzionali (create a 0 se assenti)
    for col in OPTIONAL_NUMERIC_COLS:
        if col not in df.columns:
            df[col] = 0.0
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
 
    # 5) Gol: numerici, poi Int64 nullable
    for col in ["home_score", "away_score"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
 
    # 6) xG: conversione se presenti, altrimenti NaN; in assenza di dato si
    # usa il gol reale della singola partita come fallback per il motore.
    for col in ["home_xg", "away_xg"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan
    df["home_xg"] = df["home_xg"].where(
        df["home_xg"].notna() & (df["home_xg"] != 0), df["home_score"]
    )
    df["away_xg"] = df["away_xg"].where(
        df["away_xg"].notna() & (df["away_xg"] != 0), df["away_score"]
    )

    # 7) Righe non valide fuori (gol mancanti = partita non giocata/errata)
    df = df.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"])
    df["home_score"] = df["home_score"].astype("Int64")
    df["away_score"] = df["away_score"].astype("Int64")
 
    # 8) Colonne opzionali di contesto
    for c in ["competition", "round", "season"]:
        if c not in df.columns:
            df[c] = pd.NA
 
    df = df.sort_values("date").reset_index(drop=True)
    return df
 
 
# ============================================================
# Classifica
# ============================================================
 
@dataclass
class PointSchema:
    win: int = 3
    draw: int = 1
    loss: int = 0
 
 
def compute_table(df: pd.DataFrame, schema: PointSchema = PointSchema()) -> pd.DataFrame:
    """Classifica totale con statistiche dettagliate."""
 
    col_map_home = {
        "home_team": "team", "home_score": "gf", "away_score": "ga",
        "home_possession": "poss_f", "away_possession": "poss_a",
        "home_shots": "shots_f", "away_shots": "shots_a",
        "home_shots_on_target": "st_f", "away_shots_on_target": "st_a",
        "home_corners": "corners_f", "away_corners": "corners_a",
        "home_yellow": "yellow_f", "home_red": "red_f",
    }
    col_map_away = {
        "away_team": "team", "away_score": "gf", "home_score": "ga",
        "away_possession": "poss_f", "home_possession": "poss_a",
        "away_shots": "shots_f", "home_shots": "shots_a",
        "away_shots_on_target": "st_f", "home_shots_on_target": "st_a",
        "away_corners": "corners_f", "home_corners": "corners_a",
        "away_yellow": "yellow_f", "away_red": "red_f",
    }
 
    h = df[list(col_map_home.keys())].rename(columns=col_map_home)
    a = df[list(col_map_away.keys())].rename(columns=col_map_away)
    long = pd.concat([h, a], ignore_index=True)
 
    long["W"] = (long["gf"] > long["ga"]).astype(int)
    long["D"] = (long["gf"] == long["ga"]).astype(int)
    long["L"] = (long["gf"] < long["ga"]).astype(int)
 
    g = long.groupby("team", as_index=False).agg(
        GP=("gf", "count"),
        W=("W", "sum"), D=("D", "sum"), L=("L", "sum"),
        GF=("gf", "sum"), GA=("ga", "sum"),
        ShotsF=("shots_f", "sum"), ShotsA=("shots_a", "sum"),
        StF=("st_f", "sum"), StA=("st_a", "sum"),
        CornersF=("corners_f", "sum"), CornersA=("corners_a", "sum"),
        Yellow=("yellow_f", "sum"), Red=("red_f", "sum"),
        PossAvg=("poss_f", "mean"),
    )
 
    g["GD"] = g["GF"] - g["GA"]
    g["Pts"] = g["W"] * schema.win + g["D"] * schema.draw + g["L"] * schema.loss
    gp_safe = g["GP"].replace(0, 1)
    g["PPG"] = g["Pts"] / gp_safe
    g["WinRate"] = (g["W"] / gp_safe) * 100.0
 
    for src, dst in [
        ("ShotsF", "ShotsF_pG"), ("ShotsA", "ShotsA_pG"),
        ("StF", "StF_pG"), ("StA", "StA_pG"),
        ("CornersF", "CornersF_pG"), ("CornersA", "CornersA_pG"),
    ]:
        g[dst] = g[src] / gp_safe
 
    g = g.sort_values(["Pts", "GD", "GF"], ascending=[False, False, False]).reset_index(drop=True)
    g.insert(0, "Pos", range(1, len(g) + 1))
    return g
 
 
# ============================================================
# Modello Gol (Poisson + Dixon-Coles)
# ============================================================
 
def _league_baselines(df: pd.DataFrame) -> Dict[str, float]:
    """Media gol e xG per squadra in campionato, per il solo motore interno."""
    gpt = float((df["home_score"] + df["away_score"]).mean() / 2.0)
    xgpt = float((df["home_xg"] + df["away_xg"]).mean() / 2.0)
    return {"gpt": gpt, "xgpt": xgpt}
 
 
def _weighted_group_stats(df_w: pd.DataFrame, team_col: str, cols: Dict[str, str]) -> pd.DataFrame:
    """Somma pesata di più colonne per squadra, senza usare .apply(pd.Series).
 
    cols: mappa {nome_colonna_output: nome_colonna_input}. Ogni colonna di
    input viene moltiplicata per il peso e sommata; evita il pattern
    groupby().apply(lambda g: pd.Series({...})) che è più lento e soggetto
    a deprecazioni nelle versioni recenti di pandas.
    """
    tmp = pd.DataFrame({team_col: df_w[team_col]})
    for out_name, in_name in cols.items():
        tmp[out_name] = df_w[in_name] * df_w["weight"]
    tmp["weight_sum"] = df_w["weight"]
    return tmp.groupby(team_col, as_index=False).sum(numeric_only=True)
 
 
def _team_strengths(df: pd.DataFrame, half_life_days: int = 60) -> pd.DataFrame:
    """Parametri di forza delle squadre (attacco/difesa, casa/trasferta)."""
    base = _league_baselines(df)
    gpt = base.get("gpt") or 1.3
    xgpt = base.get("xgpt") or 1.3
 
    df_w = df.copy()
    latest = df_w["date"].max()
    days_past = (latest - df_w["date"]).dt.days
    if half_life_days > 0:
        alpha = np.log(2) / half_life_days
        df_w["weight"] = np.exp(-alpha * days_past)
    else:
        df_w["weight"] = 1.0
 
    home_stats = _weighted_group_stats(
        df_w, "home_team",
        {"GF_H": "home_score", "GA_H": "away_score", "XGF_H": "home_xg", "XGA_H": "away_xg"},
    ).rename(columns={"home_team": "team", "weight_sum": "weight_H"})
 
    away_stats = _weighted_group_stats(
        df_w, "away_team",
        {"GF_A": "away_score", "GA_A": "home_score", "XGF_A": "away_xg", "XGA_A": "home_xg"},
    ).rename(columns={"away_team": "team", "weight_sum": "weight_A"})
 
    stats = pd.merge(home_stats, away_stats, on="team", how="outer").fillna(0)
 
    stats["weight_H"] = stats["weight_H"].replace(0, 1)
    stats["weight_A"] = stats["weight_A"].replace(0, 1)
 
    stats["att_home_G"] = (stats["GF_H"] / stats["weight_H"]) / gpt
    stats["def_home_G"] = (stats["GA_H"] / stats["weight_H"]) / gpt
    stats["att_away_G"] = (stats["GF_A"] / stats["weight_A"]) / gpt
    stats["def_away_G"] = (stats["GA_A"] / stats["weight_A"]) / gpt
    stats["att_home_XG"] = (stats["XGF_H"] / stats["weight_H"]) / xgpt
    stats["def_home_XG"] = (stats["XGA_H"] / stats["weight_H"]) / xgpt
    stats["att_away_XG"] = (stats["XGF_A"] / stats["weight_A"]) / xgpt
    stats["def_away_XG"] = (stats["XGA_A"] / stats["weight_A"]) / xgpt
 
    # Quante partite (pesate) ha giocato ogni squadra in totale: usato per
    # scalare lo shrinkage verso la media di lega su campioni piccoli.
    stats["sample_weight"] = stats["weight_H"] + stats["weight_A"]
 
    cols = [
        "team", "att_home_G", "def_home_G", "att_away_G", "def_away_G",
        "att_home_XG", "def_home_XG", "att_away_XG", "def_away_XG", "sample_weight",
    ]
    result = stats[cols].fillna(1.0)
    result = result.replace([np.inf, -np.inf], 1.0)
    return result
 
 
def expected_goals(
    df: pd.DataFrame,
    home_team: str,
    away_team: str,
    blend: float = 0.20,
    mode: str = "hybrid",
) -> Tuple[float, float]:
    """Stima lambda (gol attesi) per casa e trasferta.
 
    `blend` è lo shrinkage minimo verso la media di lega; viene aumentato
    automaticamente per squadre con poche partite giocate (campione piccolo),
    per evitare stime estreme a inizio stagione.
    """
    if mode == "hybrid":
        goals_h, goals_a = expected_goals(df, home_team, away_team, blend=blend, mode="goals")
        xg_h, xg_a = expected_goals(df, home_team, away_team, blend=blend, mode="xg")
        return 0.55 * goals_h + 0.45 * xg_h, 0.55 * goals_a + 0.45 * xg_a

    base = _league_baselines(df)
    td = _team_strengths(df)
    th = td[td["team"] == home_team]
    ta = td[td["team"] == away_team]
 
    fallback = base.get("xgpt", 1.2) if mode == "xg" else base.get("gpt", 1.2)
    if th.empty or ta.empty:
        return fallback, fallback
 
    if mode == "xg":
        gpt = base.get("xgpt", 1.3)
        att_h, def_h = float(th["att_home_XG"].iloc[0]), float(th["def_home_XG"].iloc[0])
        att_a, def_a = float(ta["att_away_XG"].iloc[0]), float(ta["def_away_XG"].iloc[0])
    else:
        gpt = base.get("gpt", 1.3)
        att_h, def_h = float(th["att_home_G"].iloc[0]), float(th["def_home_G"].iloc[0])
        att_a, def_a = float(ta["att_away_G"].iloc[0]), float(ta["def_away_G"].iloc[0])
 
    lam_h = gpt * att_h * def_a
    lam_a = gpt * att_a * def_h
 
    # Shrinkage extra se una delle due squadre ha giocato poche partite
    # (sample_weight basso -> più incertezza -> più peso alla media di lega)
    min_sample = min(float(th["sample_weight"].iloc[0]), float(ta["sample_weight"].iloc[0]))
    extra_shrink = max(0.0, (5.0 - min_sample) / 5.0) * 0.3 if min_sample < 5.0 else 0.0
    eff_blend = min(0.8, blend + extra_shrink)
 
    lam_h = (1 - eff_blend) * lam_h + eff_blend * gpt
    lam_a = (1 - eff_blend) * lam_a + eff_blend * gpt
 
    lam_h = float(np.clip(lam_h, 0.2, 3.5))
    lam_a = float(np.clip(lam_a, 0.2, 3.5))
    return lam_h, lam_a
 
 
def dixon_coles_matrix(
    lam_h: float,
    lam_a: float,
    rho: float = 0.05,
    max_goals: int = 8,
) -> np.ndarray:
    """Matrice congiunta P(gol_casa=i, gol_trasferta=j) con correzione Dixon-Coles.
 
    Correzione applicata secondo Dixon & Coles (1997):
        tau(0,0) = 1 - lam_h*lam_a*rho
        tau(0,1) = 1 + lam_h*rho     (i=0 casa, j=1 trasferta -> dipende da lam_h)
        tau(1,0) = 1 + lam_a*rho     (i=1 casa, j=0 trasferta -> dipende da lam_a)
        tau(1,1) = 1 - rho
    """
    ks = np.arange(0, max_goals + 1)
    factorials = np.array([math.factorial(k) for k in ks], dtype=float)
 
    px = np.exp(-lam_h) * np.power(lam_h, ks) / factorials
    py = np.exp(-lam_a) * np.power(lam_a, ks) / factorials
    P = np.outer(px, py)
 
    corr = np.ones_like(P)
    if 0 <= rho <= 0.25:
        corr[0, 0] = 1 - (lam_h * lam_a) * rho
        if P.shape[1] > 1:
            corr[0, 1] = 1 + lam_h * rho
        if P.shape[0] > 1:
            corr[1, 0] = 1 + lam_a * rho
        if P.shape[0] > 1 and P.shape[1] > 1:
            corr[1, 1] = 1 - rho
 
    P = P * corr
    P = np.clip(P, 0, None)  # rho alto in teoria potrebbe rendere negativa qualche cella
    P = P / P.sum()
    return P
 
 
def goal_market_probs(P: np.ndarray) -> Dict[str, object]:
    max_g = P.shape[0] - 1
    tot = np.zeros(max_g * 2 + 1)
    for i in range(P.shape[0]):
        for j in range(P.shape[1]):
            tot[i + j] += P[i, j]
 
    home_goals = P.sum(axis=1)
    away_goals = P.sum(axis=0)
    btts = 1.0 - (home_goals[0] + away_goals[0] - P[0, 0])
 
    def ou(k: float) -> float:
        thresh = int(np.floor(k))
        return float(tot[thresh + 1:].sum())
 
    ou_dict = {}
    for line in [0.5, 1.5, 2.5, 3.5, 4.5]:
        over = ou(line)
        ou_dict[f"Under {line}"] = 1 - over
        ou_dict[f"Over {line}"] = over
 
    return {
        "total_goals_dist": tot,
        "home_goals_dist": home_goals,
        "away_goals_dist": away_goals,
        "btts": btts,
        "ou": ou_dict,
        "joint": P,
    }
 
 
def predict_goals_distribution(
    df: pd.DataFrame,
    home_team: str,
    away_team: str,
    rho: float = 0.05,
    max_goals: int = 8,
    mode: str = "hybrid",
) -> Dict[str, object]:
    lam_h, lam_a = expected_goals(df, home_team, away_team, mode=mode)
    P = dixon_coles_matrix(lam_h, lam_a, rho=rho, max_goals=max_goals)
    out = goal_market_probs(P)
    out["lambda_home"] = lam_h
    out["lambda_away"] = lam_a
    return out
 
 
def predict_1x2_from_matrix(P: np.ndarray) -> Dict[str, float]:
    home_win = np.tril(P, -1).sum()
    away_win = np.triu(P, 1).sum()
    draw = np.trace(P)
    return {"1": float(home_win), "X": float(draw), "2": float(away_win)}


def match_recommendations(
    one_x_two: Dict[str, float],
    markets: Dict[str, object],
) -> Dict[str, Dict[str, object]]:
    """Restituisce tre indicazioni leggibili, basate solo sui mercati utili.

    Non viene scelto il mercato matematicamente più lontano dal 50% fra tutte
    le soglie: quel criterio favoriva quasi sempre Under 4.5 e produceva un
    consiglio visivamente identico per qualsiasi incontro.
    """
    outcome_key, outcome_probability = max(one_x_two.items(), key=lambda item: item[1])
    outcome_names = {"1": "Vittoria casa (1)", "X": "Pareggio (X)", "2": "Vittoria trasferta (2)"}

    over_25 = float(markets["Over 2.5"])
    goals_label, goals_probability = (
        ("Over 2.5", over_25) if over_25 >= 0.5 else ("Under 2.5", 1 - over_25)
    )

    btts_yes = float(markets["btts"])
    btts_label, btts_probability = (
        ("Gol (entrambe segnano)", btts_yes)
        if btts_yes >= 0.5
        else ("No Gol", 1 - btts_yes)
    )

    def confidence(probability: float) -> str:
        if probability >= 0.68:
            return "Alta"
        if probability >= 0.58:
            return "Media"
        return "Bassa"

    return {
        "esito": {
            "label": outcome_names[outcome_key],
            "probability": outcome_probability,
            "confidence": confidence(outcome_probability),
        },
        "gol": {
            "label": goals_label,
            "probability": goals_probability,
            "confidence": confidence(goals_probability),
        },
        "btts": {
            "label": btts_label,
            "probability": btts_probability,
            "confidence": confidence(btts_probability),
        },
    }


# ============================================================
# Notizie live (RSS)
# ============================================================

def fetch_match_news(home_team: str, away_team: str, limit: int = 10) -> list[Dict[str, str]]:
    """Recupera gli ultimi titoli sul match da un feed RSS pubblico.

    Il feed è una ricerca italiana Google News: non effettua scraping delle
    pagine degli editori e conserva soltanto titolo, data, fonte e link.
    """
    teams = [team.strip() for team in (home_team, away_team) if team and team.strip()]
    if not teams:
        raise ValueError("Inserisci almeno una squadra o un argomento da cercare.")
    query = " ".join(f'"{team}"' for team in teams) + " calcio"
    params = {"q": query, "hl": "it", "gl": "IT", "ceid": "IT:it"}
    url = f"https://news.google.com/rss/search?{urlencode(params)}"
    response = requests.get(
        url,
        timeout=8,
        headers={"User-Agent": "MatchScope/1.0 RSS reader"},
    )
    response.raise_for_status()

    root = ET.fromstring(response.content)
    news: list[Dict[str, str]] = []
    for item in root.findall("./channel/item")[:limit]:
        title = html.unescape(item.findtext("title", default="Titolo non disponibile")).strip()
        link = item.findtext("link", default="").strip()
        source = html.unescape(item.findtext("source", default="Google News")).strip()
        pub_date = item.findtext("pubDate", default="").strip()
        try:
            published = parsedate_to_datetime(pub_date).astimezone().strftime("%d/%m/%Y %H:%M")
        except (TypeError, ValueError, IndexError):
            published = pub_date or "Data non disponibile"

        if link:
            news.append({"title": title, "link": link, "source": source, "published": published})
    return news


# ============================================================
# Classifiche live
# ============================================================

LIVE_LEAGUES = {
    "serie_a": {"name": "Serie A", "country": "Italia", "espn_code": "ita.1", "limit": 5},
    "serie_b": {"name": "Serie B", "country": "Italia", "espn_code": "ita.2", "limit": 5},
    "laliga": {"name": "LaLiga", "country": "Spagna", "espn_code": "esp.1", "limit": 5},
    "bundesliga": {"name": "Bundesliga", "country": "Germania", "espn_code": "ger.1", "limit": 5},
    "ligue_1": {"name": "Ligue 1", "country": "Francia", "espn_code": "fra.1", "limit": 5},
    "premier_league": {"name": "Premier League", "country": "Inghilterra", "espn_code": "eng.1", "limit": 5},
    "champions_league": {"name": "Champions League", "country": "UEFA", "espn_code": "uefa.champions", "limit": 8},
    "europa_league": {"name": "Europa League", "country": "UEFA", "espn_code": "uefa.europa", "limit": 8},
    "conference_league": {"name": "Conference League", "country": "UEFA", "espn_code": "uefa.europa.conf", "limit": 8},
}


def _fetch_one_live_standing(league_key: str, limit: int) -> Dict[str, object]:
    league = LIVE_LEAGUES[league_key]
    url = f"https://site.api.espn.com/apis/v2/sports/soccer/{league['espn_code']}/standings"
    try:
        # Il feed pubblico risponde senza credenziali: non inviare un User-Agent
        # applicativo, perché alcuni edge ESPN lo rifiutano con 403.
        response = requests.get(url, timeout=8)
        response.raise_for_status()
        payload = response.json()
        entries = payload["children"][0]["standings"]["entries"]
        rows = []
        for entry in entries[:limit]:
            stats = {stat["name"]: stat.get("displayValue", "-") for stat in entry.get("stats", [])}
            team = entry.get("team", {})
            rows.append({
                "pos": stats.get("rank", str(len(rows) + 1)),
                "team": team.get("shortDisplayName") or team.get("displayName", "Squadra"),
                "played": stats.get("gamesPlayed", "-"),
                "points": stats.get("points", "-"),
            })
        if not rows:
            raise ValueError("Classifica non disponibile")
        return {**league, "rows": rows, "error": None}
    except (requests.RequestException, ValueError, KeyError, IndexError, TypeError) as exc:
        return {**league, "rows": [], "error": str(exc)}


def fetch_live_standings() -> Dict[str, Dict[str, object]]:
    """Recupera in parallelo le prime posizioni, con limiti per competizione."""
    results: Dict[str, Dict[str, object]] = {}
    with ThreadPoolExecutor(max_workers=len(LIVE_LEAGUES)) as executor:
        futures = {
            executor.submit(_fetch_one_live_standing, key, int(league["limit"])): key
            for key, league in LIVE_LEAGUES.items()
        }
        for future in as_completed(futures):
            key = futures[future]
            results[key] = future.result()
    return results
 
 
# ============================================================
# Export Excel
# ============================================================
 
def export_excel_bytes(**tables: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
        for name, table in tables.items():
            if isinstance(table, pd.DataFrame) and not table.empty:
                sheet = str(name)[:31]
                table.to_excel(xw, index=False, sheet_name=sheet)
    buf.seek(0)
    return buf.getvalue()
 
