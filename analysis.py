# analysis.py – toolkit per Football Analysis
# Versione Finale (include modello GOL + modello XG)

from __future__ import annotations
import io
import math
from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd

# -------------------------- Normalizzazione dati --------------------------

def normalize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Normalizza colonne e tipi (ora include xG con fallback e possesso %)."""
    df = df.copy()

    # 1) Sistema spazi nei nomi colonna
    df.columns = [c.strip() for c in df.columns]

    # 2) Dizionario RENAME per xG (es. da FBref e dal tuo CSV)
    rename_xg = {
        "xG": "home_xg",
        "xG.1": "away_xg",
        "home_xG": "home_xg",     # tuo CSV
        "away_xG": "away_xg",     # tuo CSV
        "home_exp_goals": "home_xg",
        "away_exp_goals": "away_xg",
    }
    df.rename(columns=rename_xg, inplace=True)

    # 3) Typo possesso
    if "away_possesion" in df.columns:
        df.rename(columns={"away_possesion": "away_possession"}, inplace=True)

    # 4) Colonne obbligatorie
    needed = ["date", "home_team", "away_team", "home_score", "away_score"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise ValueError(f"Mancano colonne obbligatorie: {missing}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    # 5) Pulizia possesso palla:
    #    "55%" -> "55" -> 55.0 ; "55,3%" -> "55.3" -> 55.3
    for col in ["home_possession", "away_possession"]:
        if col in df.columns:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace("%", "", regex=False)
                .str.replace(",", ".", regex=False)
            )

    # 6) Conversione numerica
    numeric_cols = [
        "home_score", "away_score",
        "home_possession", "away_possession",
        "home_shots", "away_shots",
        "home_shots_on_target", "away_shots_on_target",
        "home_corners", "away_corners",
        "home_yellow", "away_yellow",
        "home_xg", "away_xg",
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            df[col] = df[col].fillna(0)
            if col in ["home_score", "away_score"]:
                df[col] = df[col].astype("Int64")
        else:
            # se non esiste (statistica opzionale), la creo a 0
            if col not in needed and col not in ["home_xg", "away_xg"]:
                df[col] = 0

    # 7) Fallback per xG: se non ci sono o sono tutti 0, uso i gol reali
    if "home_xg" not in df.columns or df["home_xg"].sum() == 0:
        df["home_xg"] = df["home_score"].astype(float)
    if "away_xg" not in df.columns or df["away_xg"].sum() == 0:
        df["away_xg"] = df["away_score"].astype(float)

    # 8) Colonne opzionali di contesto
    for c in ["competition", "round", "season"]:
        if c not in df.columns:
            df[c] = pd.NA

    df = df.dropna(subset=["date", "home_team", "away_team", "home_score", "away_score"])
    df = df.sort_values("date").reset_index(drop=True)
    return df

# -------------------------- Classifica --------------------------
@dataclass
class PointSchema:
    win: int = 3
    draw: int = 1
    loss: int = 0

def compute_table(df: pd.DataFrame, schema: PointSchema = PointSchema()) -> pd.DataFrame:
    """Classifica totale, ora include tutte le statistiche dettagliate (incluso xG, MA senza rossi)."""
    
    # Partite in casa
    h = pd.DataFrame()
    h["team"] = df["home_team"]
    h["gf"] = df["home_score"]
    h["ga"] = df["away_score"]
    h["xgf"] = df["home_xg"]
    h["xga"] = df["away_xg"]
    h["poss_f"] = df["home_possession"]
    h["poss_a"] = df["away_possession"]
    h["shots_f"] = df["home_shots"]
    h["shots_a"] = df["away_shots"]
    h["st_f"] = df["home_shots_on_target"]
    h["st_a"] = df["away_shots_on_target"]
    h["corners_f"] = df["home_corners"]
    h["corners_a"] = df["away_corners"]
    h["yellow_f"] = df["home_yellow"]
    h["yellow_a"] = df["away_yellow"]

    # Partite in trasferta
    a = pd.DataFrame()
    a["team"] = df["away_team"]
    a["gf"] = df["away_score"]
    a["ga"] = df["home_score"]
    a["xgf"] = df["away_xg"]
    a["xga"] = df["home_xg"]
    a["poss_f"] = df["away_possession"]
    a["poss_a"] = df["home_possession"]
    a["shots_f"] = df["away_shots"]
    a["shots_a"] = df["home_shots"]
    a["st_f"] = df["away_shots_on_target"]
    a["st_a"] = df["home_shots_on_target"]
    a["corners_f"] = df["away_corners"]
    a["corners_a"] = df["home_corners"]
    a["yellow_f"] = df["away_yellow"]
    a["yellow_a"] = df["home_yellow"]

    # Uniamo casa+trasferta
    long = pd.concat([h, a], ignore_index=True)

    # Esiti
    long["W"] = (long["gf"] > long["ga"]).astype(int)
    long["D"] = (long["gf"] == long["ga"]).astype(int)
    long["L"] = (long["gf"] < long["ga"]).astype(int)

    # Aggregazione per squadra
    g = long.groupby("team", as_index=False).agg(
        GP=("gf", "count"),
        W=("W", "sum"),
        D=("D", "sum"),
        L=("L", "sum"),
        GF=("gf", "sum"),
        GA=("ga", "sum"),
        XGF=("xgf", "sum"),
        XGA=("xga", "sum"),
        ShotsF=("shots_f", "sum"),
        ShotsA=("shots_a", "sum"),
        StF=("st_f", "sum"),
        StA=("st_a", "sum"),
        CornersF=("corners_f", "sum"),
        CornersA=("corners_a", "sum"),
        Yellow=("yellow_f", "sum"),
        PossAvg=("poss_f", "mean"),
    )
    
    # Statistiche derivate
    g["GD"] = g["GF"] - g["GA"]
    g["XGD"] = g["XGF"] - g["XGA"]
    g["Pts"] = g["W"] * schema.win + g["D"] * schema.draw + g["L"] * schema.loss
    g["PPG"] = g["Pts"] / g["GP"]
    g["WinRate"] = (g["W"] / g["GP"]) * 100.0

    # Medie per partita (evita divisione per 0)
    gp_safe = g["GP"].replace(0, 1)
    g["XGF_pG"] = g["XGF"] / gp_safe
    g["XGA_pG"] = g["XGA"] / gp_safe
    g["ShotsF_pG"] = g["ShotsF"] / gp_safe
    g["StF_pG"] = g["StF"] / gp_safe
    g["CornersF_pG"] = g["CornersF"] / gp_safe
    g["ShotsA_pG"] = g["ShotsA"] / gp_safe
    g["StA_pG"] = g["StA"] / gp_safe
    g["CornersA_pG"] = g["CornersA"] / gp_safe
    
    # Ordinamento classifica
    g = g.sort_values(["Pts", "GD", "GF"], ascending=[False, False, False]).reset_index(drop=True)
    g.insert(0, "Pos", range(1, len(g) + 1))
    return g
# -------------------------- Modello Gol (Poisson + Dixon-Coles) --------------------------

def _league_baselines(df: pd.DataFrame) -> Dict[str, float]:
    """Calcola gpt (Gol Per Team) e xgpt (xG Per Team) medi di lega."""
    gpm = (df["home_score"] + df["away_score"]).mean()
    gpt = float(gpm / 2.0)
    
    xgpm = (df["home_xg"] + df["away_xg"]).mean()
    xgpt = float(xgpm / 2.0)
    
    return {"gpt": gpt, "xgpt": xgpt}


def _team_strengths(df: pd.DataFrame, half_life_days: int = 60) -> pd.DataFrame:
    """
    Calcola i parametri di forza delle squadre (attacco/difesa casa/trasferta)
    Versione 100% compatibile con Streamlit + Pandas recenti
    """
    base = _league_baselines(df)
    gpt = base.get("gpt", 1.3)
    xgpt = base.get("xgpt", 1.3)

    df_w = df.copy()
    latest = df_w['date'].max()
    df_w['days_past'] = (latest - df_w['date']).dt.days
    df_w['weight'] = 1.0
    if half_life_days > 0:
        alpha = np.log(2) / half_life_days
        df_w['weight'] = np.exp(-alpha * df_w['days_past'])

    # === Usa aggregazioni dirette (niente .apply su Series) ===
    home_stats = df_w.groupby("home_team").apply(
        lambda g: pd.Series({
            "GF_H": (g["home_score"] * g["weight"]).sum(),
            "GA_H": (g["away_score"] * g["weight"]).sum(),
            "XGF_H": (g["home_xg"] * g["weight"]).sum(),
            "XGA_H": (g["away_xg"] * g["weight"]).sum(),
            "weight_H": g["weight"].sum(),
        })
    ).reset_index()

    away_stats = df_w.groupby("away_team").apply(
        lambda g: pd.Series({
            "GF_A": (g["away_score"] * g["weight"]).sum(),
            "GA_A": (g["home_score"] * g["weight"]).sum(),
            "XGF_A": (g["away_xg"] * g["weight"]).sum(),
            "XGA_A": (g["home_xg"] * g["weight"]).sum(),
            "weight_A": g["weight"].sum(),
        })
    ).reset_index()

    # Rinominiamo per merge
    home_stats = home_stats.rename(columns={"home_team": "team"})
    away_stats = away_stats.rename(columns={"away_team": "team"})

    # Merge
    stats = pd.merge(home_stats, away_stats, on="team", how="outer").fillna(0)

    # Evitiamo divisione per zero
    stats["weight_H"] = stats["weight_H"].replace(0, 1)
    stats["weight_A"] = stats["weight_A"].replace(0, 1)

    # Calcolo forze (Gol reali)
    stats["att_home_G"]  = (stats["GF_H"]  / stats["weight_H"]) / gpt
    stats["def_home_G"]  = (stats["GA_H"]  / stats["weight_H"]) / gpt
    stats["att_away_G"]  = (stats["GF_A"]  / stats["weight_A"]) / gpt
    stats["def_away_G"]  = (stats["GA_A"]  / stats["weight_A"]) / gpt

    # Calcolo forze (xG)
    stats["att_home_XG"] = (stats["XGF_H"] / stats["weight_H"]) / xgpt
    stats["def_home_XG"] = (stats["XGA_H"] / stats["weight_H"]) / xgpt
    stats["att_away_XG"] = (stats["XGF_A"] / stats["weight_A"]) / xgpt
    stats["def_away_XG"] = (stats["XGA_A"] / stats["weight_A"]) / xgpt

    # Pulizia finale
    cols = ["team", "att_home_G", "def_home_G", "att_away_G", "def_away_G",
            "att_home_XG", "def_home_XG", "att_away_XG", "def_away_XG"]
    result = stats[cols].fillna(1.0)
    result = result.replace([np.inf, -np.inf], 1.0)

    return result


def expected_goals(
    df: pd.DataFrame,
    home_team: str,
    away_team: str,
    blend: float = 0.20,
    mode: str = 'goals'  # 'goals' o 'xg'
) -> Tuple[float, float]:
    
    base = _league_baselines(df)
    td = _team_strengths(df) 
    th = td[td["team"] == home_team]
    ta = td[td["team"] == away_team]
    
    if th.empty or ta.empty:
        return base.get("gpt", 1.2), base.get("gpt", 1.2)

    if mode == 'xg':
        gpt = base.get("xgpt", 1.3)
        att_h = float(th["att_home_XG"])
        def_h = float(th["def_home_XG"]) 
        att_a = float(ta["att_away_XG"])
        def_a = float(ta["def_away_XG"])
    else:
        gpt = base.get("gpt", 1.3)
        att_h = float(th["att_home_G"])
        def_h = float(th["def_home_G"]) 
        att_a = float(ta["att_away_G"])
        def_a = float(ta["def_away_G"])

    lam_h = gpt * att_h * def_a 
    lam_a = gpt * att_a * def_h

    lam_h = (1 - blend) * lam_h + blend * gpt
    lam_a = (1 - blend) * lam_a + blend * gpt
    
    lam_h = float(np.clip(lam_h, 0.2, 3.5))
    lam_a = float(np.clip(lam_a, 0.2, 3.5))
    return lam_h, lam_a

def dixon_coles_matrix(
    lam_h: float,
    lam_a: float,
    rho: float = 0.05,
    max_goals: int = 8,
) -> np.ndarray:
    
    xs = np.arange(0, max_goals + 1)
    ys = np.arange(0, max_goals + 1)

    px = np.exp(-lam_h) * np.power(lam_h, xs) / np.array([math.factorial(k) for k in xs])
    py = np.exp(-lam_a) * np.power(lam_a, ys) / np.array([math.factorial(k) for k in xs])
    P = np.outer(px, py)

    corr = np.ones_like(P)
    if 0 <= rho <= 0.25:
        corr[0, 0] = 1 - (lam_h * lam_a) * rho
        if P.shape[0] > 1:
            corr[1, 0] = 1 + lam_h * rho
        if P.shape[1] > 1:
            corr[0, 1] = 1 + lam_a * rho
        if P.shape[0] > 1 and P.shape[1] > 1:
            corr[1, 1] = 1 - rho

    P = P * corr
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
        return float(tot[thresh + 1 :].sum())

    ou_dict = {
        "Under 0.5": 1 - ou(0.5), "Over 0.5": ou(0.5),
        "Under 1.5": 1 - ou(1.5), "Over 1.5": ou(1.5),
        "Under 2.5": 1 - ou(2.5), "Over 2.5": ou(2.5),
        "Under 3.5": 1 - ou(3.5), "Over 3.5": ou(3.5),
        "Under 4.5": 1 - ou(4.5), "Over 4.5": ou(4.5),
    }

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
    mode: str = 'goals'  # 'goals' o 'xg'
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

# -------------------------- Export Excel --------------------------

def export_excel_bytes(**tables: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
        for name, df in tables.items():
            if isinstance(df, pd.DataFrame) and not df.empty:
                sheet = str(name)[:31]
                df.to_excel(xw, index=False, sheet_name=sheet)
    buf.seek(0)
    return buf.getvalue()