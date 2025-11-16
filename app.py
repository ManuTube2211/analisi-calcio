# app.py — Analisi Calcio (Gol + xG) - VERSIONE ITALIANA STABILE

from datetime import datetime
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from analysis import (
    normalize_df,
    compute_table,
    PointSchema,
    export_excel_bytes,
    predict_goals_distribution,
    predict_1x2_from_matrix,
    expected_goals,
)

# ----------------- CONFIGURAZIONE PAGINA -----------------
st.set_page_config(page_title="Analisi Calcio - 4 Sezioni", page_icon="⚽", layout="wide")
st.title("⚽ Analisi Calcio")
st.caption("CSV richiesto: date, home_team, away_team, home_score, away_score, xG, stats...")

# ----------------- FUNZIONE DI CARICAMENTO DATI (con cache) -----------------
@st.cache_data
def load_data(uploaded_file):
    """
    Carica i dati.
    Priorità 1: File caricato dall'utente.
    Priorità 2: File locale 'dati_default.csv'.
    """
    data_source = None

    if uploaded_file is not None:
        data_source = uploaded_file
        st.sidebar.info("Dati caricati da file utente.")
    else:
        try:
            data_source = "dati_default.csv"
            with open(data_source, "r"):
                pass
            st.sidebar.info("Dati caricati da file locale.")
        except FileNotFoundError:
            st.sidebar.warning("Nessun file caricato. 'dati_default.csv' non trovato.")
            return None

    try:
        df_raw = pd.read_csv(
            data_source,
            delimiter=";",
            decimal=",",
            header=1,  # Salta la prima riga ("Season 25-26")
        )
        return normalize_df(df_raw)
    except Exception as e:
        st.error(f"Errore lettura/normalizzazione: {e}")
        return None

# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Dati")
    up = st.file_uploader(
        "Carica CSV (opzionale)",
        type=["csv"],
        help="Se non carichi un file, verrà usato il 'dati_default.csv' locale.",
    )
    st.code(
        "date;home_team;away_team;home_score;away_score;home_xg;away_xg\n"
        "2024-08-17;Juventus;Roma;3;0;2,1;0,4",
        language="csv",
    )
    st.divider()
    win = st.number_input("Punti per vittoria", 0, 5, 3)
    draw = st.number_input("Punti per pareggio", 0, 3, 1)

    st.divider()
    st.header("Filtri dati")
    days_window = st.slider(
        "Usa solo dati degli ultimi N giorni",
        min_value=60,
        max_value=1000,
        value=365,
        step=30,
        help="Filtra i dati per calcolare le statistiche solo sulle partite più recenti.",
    )

# ---------------- Lettura & normalizzazione ----------------
df_normalized = load_data(up)

if df_normalized is None:
    st.info("👆 Carica un CSV o aggiungi 'dati_default.csv' alla cartella per iniziare.")
    st.stop()

if df_normalized.empty:
    st.error("Errore: il CSV è vuoto o nessun dato è valido.")
    st.stop()

# --- Filtro temporale ---
max_date = df_normalized["date"].max()
min_date = max_date - pd.to_timedelta(days_window, unit="D")
df = df_normalized[df_normalized["date"] >= min_date].copy()

if df.empty:
    st.warning(
        f"Nessuna partita trovata negli ultimi {days_window} giorni. "
        "Prova ad allargare il filtro 'Usa solo dati degli ultimi N giorni' nella sidebar."
    )
    st.stop()

st.sidebar.caption(
    f"Statistiche calcolate su {len(df)} partite (dal {min_date.date()})"
)

# ---------------- Classifica ----------------
schema = PointSchema(win=win, draw=draw, loss=0)
table = compute_table(df, schema=schema)

# ---------------- NAVIGAZIONE SEZIONI ----------------
sezione = st.radio(
    "Sezione",
    ["📊 Classifica", "🔮 Previsioni (1X2 + Gol)", "👤 Scheda squadra", "⚖️ Confronto squadre (H2H)"],
    horizontal=True,
)

# ========== SEZIONE 1: Classifica ==========
if sezione == "📊 Classifica":
    st.subheader("Classifica totale e statistiche")

    table_it = table.rename(columns={
        "Pos": "Posizione",
        "team": "Squadra",
        "GP": "Partite",
        "W": "Vittorie",
        "D": "Pareggi",
        "L": "Sconfitte",
        "GF": "Gol fatti",
        "GA": "Gol subiti",
        "XGF": "xG fatti",
        "XGA": "xG subiti",
        "ShotsF": "Tiri",
        "ShotsA": "Tiri subiti",
        "StF": "Tiri in porta",
        "StA": "Tiri in porta subiti",
        "CornersF": "Corner",
        "CornersA": "Corner subiti",
        "Yellow": "Cartellini gialli",
        "PossAvg": "Possesso medio (%)",
        "GD": "Differenza reti",
        "XGD": "Differenza xG",
        "Pts": "Punti",
        "PPG": "Punti/partita (PPG)",
        "WinRate": "Percentuale vittorie",
    })

    st.dataframe(table_it, use_container_width=True, hide_index=True)
    st.caption(
        "PPG = punti per partita. Scorri orizzontalmente per vedere tiri, corner, xG, ecc."
    )

# ========== SEZIONE 2: Previsioni ==========
elif sezione == "🔮 Previsioni (1X2 + Gol)":
    st.subheader("🔮 Previsioni (1X2 + Gol)")
    st.info("Il modello di previsione si basa sui GOL (punteggi) storici delle squadre selezionate.")

    mode = st.radio(
        "Modalità algoritmo",
        ["Conservativa", "Standard", "Aggressiva"],
        index=1,
        horizontal=True,
    )

    if mode == "Conservativa":
        btts_yes_thr, btts_no_thr, ou_min_gap = 0.65, 0.35, 0.15
    elif mode == "Aggressiva":
        btts_yes_thr, btts_no_thr, ou_min_gap = 0.55, 0.45, 0.05
    else:  # Standard
        btts_yes_thr, btts_no_thr, ou_min_gap = 0.60, 0.40, 0.10

    teams = sorted(pd.unique(pd.concat([df["home_team"], df["away_team"]])))
    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        teamH = st.selectbox("Squadra di casa", teams)
    with c2:
        teamA = st.selectbox("Squadra in trasferta", teams)
    with c3:
        rho = st.slider("Correlazione (ρ)", 0.00, 0.25, 0.05, 0.01)
        st.caption(
            "ρ regola quanto i gol delle due squadre sono legati tra loro.\n"
            "- 0.00 ≈ modello base (gol indipendenti)\n"
            "- Valori bassi (0.03–0.07) per partite normali\n"
            "- Valori più alti (0.10–0.20) se ti aspetti molti risultati stretti (0–0, 1–0, 1–1)."
        )

    if not teamH or not teamA:
        st.warning("Seleziona entrambe le squadre.")
    elif teamH == teamA:
        st.warning("Scegli due squadre diverse.")
    else:
        gd = predict_goals_distribution(df, teamH, teamA, rho=rho, max_goals=8)
        P = gd["joint"]

        oneXtwo = predict_1x2_from_matrix(P)
        oneXtwo_pct = {k: round(v * 100, 1) for k, v in oneXtwo.items()}

        colm = st.columns(3)
        colm[0].metric("Probabilità 1 (casa)", f"{oneXtwo_pct['1']}%")
        colm[1].metric("Probabilità X (pareggio)", f"{oneXtwo_pct['X']}%")
        colm[2].metric("Probabilità 2 (trasferta)", f"{oneXtwo_pct['2']}%")

        fig = go.Figure(
            go.Pie(
                labels=[f"{teamH} (1)", "Pareggio (X)", f"{teamA} (2)"],
                values=[oneXtwo["1"], oneXtwo["X"], oneXtwo["2"]],
                textinfo="label+percent",
            )
        )
        fig.update_layout(height=360, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)

        st.divider()

        sub = st.columns(3)
        sub[0].metric(f"{teamH} λ (gol attesi)", f"{gd['lambda_home']:.2f}")
        sub[1].metric(f"{teamA} λ (gol attesi)", f"{gd['lambda_away']:.2f}")
        sub[2].metric("BTTS (entrambe segnano)", f"{gd['btts'] * 100:.1f}%")

        tot = gd["total_goals_dist"]
        tot_labels = [str(i) for i in range(len(tot))]
        st.subheader("Distribuzione gol totali previsti")
        st.bar_chart(
            pd.DataFrame({"Probabilità %": (tot * 100).round(2)}, index=tot_labels)
        )

        st.subheader("📊 Mercati Gol (Over/Under)")
        ou_dict = gd["ou"]
        ou_df = pd.DataFrame(
            {
                "Mercato": list(ou_dict.keys()),
                "Probabilità": [f"{p * 100:.1f}%" for p in ou_dict.values()],
            }
        )
        st.dataframe(ou_df, use_container_width=True, hide_index=True)

        st.subheader("⚽ Gol / No Gol (BTTS)")
        btts = float(gd.get("btts", 0.0))
        btts_pct = btts * 100
        st.metric("Probabilità Gol (entrambe segnano)", f"{btts_pct:.1f}%")

        if btts > btts_yes_thr:
            st.success(f"Consiglio ({mode}): **Gol (BTTS Sì)** — {btts_pct:.1f}%")
        elif btts < btts_no_thr:
            st.warning(
                f"Consiglio ({mode}): **No Gol (BTTS No)** — {100 - btts_pct:.1f}%"
            )
        else:
            st.info(f"({mode}) Nessun chiaro favorito tra Gol / No Gol.")

        best_key, best_prob = max(
            ou_dict.items(),
            key=lambda kv: abs(kv[1] - 0.5),
        )
        gap = abs(best_prob - 0.5)

        if gap >= ou_min_gap:
            st.success(
                f"Consiglio ({mode}) Over/Under più forte: **{best_key}** — {best_prob * 100:.1f}%"
            )
        else:
            st.info(
                f"({mode}) Nessun mercato Over/Under con margine chiaro rispetto al 50%."
            )

        imax, jmax = np.unravel_index(np.argmax(P), P.shape)
        best_score_prob = P[imax, jmax] * 100.0
        st.info(
            f"Risultato esatto più probabile: "
            f"{teamH} {imax}–{jmax} {teamA} — {best_score_prob:.1f}%"
        )

        flat = [
            (i, j, P[i, j])
            for i in range(P.shape[0])
            for j in range(P.shape[1])
        ]
        flat.sort(key=lambda x: x[2], reverse=True)
        top5 = flat[:5]
        top5_df = pd.DataFrame(
            {
                "Risultato": [f"{teamH} {i}-{j} {teamA}" for i, j, _ in top5],
                "Probabilità": [f"{p * 100:.1f}%" for _, _, p in top5],
            }
        )
        st.table(top5_df)

# ========== SEZIONE 3: Scheda Squadra ==========
elif sezione == "👤 Scheda squadra":
    st.header("👤 Scheda squadra")
    teams_list = sorted(pd.unique(pd.concat([df["home_team"], df["away_team"]])))
    team_sel = st.selectbox("Seleziona una squadra", teams_list)

    if not team_sel:
        st.info("Seleziona una squadra per vedere i dettagli.")
    else:
        team_matches = df[
            (df["home_team"] == team_sel) | (df["away_team"] == team_sel)
        ].copy()

        st.subheader("Partite giocate (periodo filtrato)")
        if team_matches.empty:
            st.info("Nessuna partita trovata per questa squadra nel periodo selezionato.")
        else:
            cols_exclude = ["competition", "round", "season"]
            display_cols = [c for c in df.columns if c not in cols_exclude]
            tm = (
                team_matches
                .sort_values("date")[display_cols]
                .copy()
            )

            rename_matches = {
                "date": "Data",
                "home_team": "Squadra casa",
                "away_team": "Squadra trasferta",
                "home_score": "Gol casa",
                "away_score": "Gol trasferta",
                "home_xg": "xG casa",
                "away_xg": "xG trasferta",
                "home_possession": "Possesso casa (%)",
                "away_possession": "Possesso trasferta (%)",
                "home_shots": "Tiri casa",
                "away_shots": "Tiri trasferta",
                "home_shots_on_target": "Tiri in porta casa",
                "away_shots_on_target": "Tiri in porta trasferta",
                "home_corners": "Corner casa",
                "away_corners": "Corner trasferta",
                "home_yellow": "Gialli casa",
                "away_yellow": "Gialli trasferta",
            }
            tm = tm.rename(columns=rename_matches)

            st.dataframe(tm, use_container_width=True, hide_index=True)
            st.caption(
                f"Partite mostrate (basate sul filtro di {days_window} giorni): "
                f"{len(team_matches)}"
            )

        row = table[table["team"] == team_sel]

        if row.empty:
            st.info("Squadra non trovata in classifica.")
        else:
            def get_val(r: pd.DataFrame, col_name: str, default=0.0):
                return float(r[col_name].iloc[0]) if col_name in r.columns else default

            st.subheader("Statistiche complessive (classifica)")

            c_tot1, c_tot2, c_tot3 = st.columns(3)
            with c_tot1:
                st.metric("Partite giocate", int(get_val(row, "GP", 0)))
                st.metric("Gol fatti (GF)", int(get_val(row, "GF", 0)))
                st.metric("Gol subiti (GA)", int(get_val(row, "GA", 0)))
                st.metric("Punti totali", int(get_val(row, "Pts", 0)))
            with c_tot2:
                st.metric("Tiri totali fatti", int(get_val(row, "ShotsF", 0)))
                st.metric("Tiri in porta fatti", int(get_val(row, "StF", 0)))
                st.metric("Corner totali fatti", int(get_val(row, "CornersF", 0)))
                st.metric("Cartellini gialli", int(get_val(row, "Yellow", 0)))
            with c_tot3:
                st.metric("Tiri totali subiti", int(get_val(row, "ShotsA", 0)))
                st.metric("Tiri in porta subiti", int(get_val(row, "StA", 0)))
                st.metric("Corner totali subiti", int(get_val(row, "CornersA", 0)))
                st.metric("Differenza reti (GD)", int(get_val(row, "GD", 0)))

            st.subheader("Statistiche medie per partita")
            c_avg1, c_avg2 = st.columns(2)
            with c_avg1:
                st.metric("Possesso medio", f"{get_val(row, 'PossAvg', 0):.1f}%")
                st.metric("Tiri fatti / gara", f"{get_val(row, 'ShotsF_pG', 0):.1f}")
                st.metric("Tiri in porta fatti / gara", f"{get_val(row, 'StF_pG', 0):.1f}")
                st.metric("Corner fatti / gara", f"{get_val(row, 'CornersF_pG', 0):.1f}")
            with c_avg2:
                st.metric("Tiri subiti / gara", f"{get_val(row, 'ShotsA_pG', 0):.1f}")
                st.metric("Tiri in porta subiti / gara", f"{get_val(row, 'StA_pG', 0):.1f}")
                st.metric("Corner subiti / gara", f"{get_val(row, 'CornersA_pG', 0):.1f}")
                st.metric("PPG (punti per gara)", f"{get_val(row, 'PPG', 0):.2f}")

            GP = int(get_val(row, "GP", 0))
            st.subheader("Ripartizione esiti delle partite")
            if GP > 0:
                W = int(get_val(row, "W", 0))
                D = int(get_val(row, "D", 0))
                L = int(get_val(row, "L", 0))

                fig_pie = go.Figure(go.Pie(
                    labels=["Vittorie (W)", "Pareggi (D)", "Sconfitte (L)"],
                    values=[W, D, L],
                    textinfo="label+percent",
                    hole=0.25,
                ))
                fig_pie.update_layout(height=320, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_pie, use_container_width=True)

                c1, c2, c3, c4 = st.columns(4)
                with c1: st.metric("Partite", GP)
                with c2: st.metric("Win %", f"{(W/GP*100):.1f}%")
                with c3: st.metric("Draw %", f"{(D/GP*100):.1f}%")
                with c4: st.metric("Loss %", f"{(L/GP*100):.1f}%")
            else:
                st.info("Nessuna partita giocata per calcolare le percentuali.")

# ========== SEZIONE 4: Confronto squadre (H2H) ==========
elif sezione == "⚖️ Confronto squadre (H2H)":
    st.header("⚖️ Confronto squadre (H2H) – statistiche medie")

    teams_list = sorted(table["team"].dropna().unique().astype(str))

    col1, col2 = st.columns(2)
    with col1:
        team_a = st.selectbox("Squadra A", options=teams_list, key="h2h_a")
    with col2:
        team_b = st.selectbox("Squadra B", options=teams_list, key="h2h_b")

    if team_a == team_b:
        st.warning("Seleziona due squadre diverse.")
    else:
        if team_a not in table["team"].values or team_b not in table["team"].values:
            st.error("Errore nei dati: una delle squadre non è presente in classifica.")
        else:
            a = table[table["team"] == team_a].iloc[0]
            b = table[table["team"] == team_b].iloc[0]

            df_h2h = pd.DataFrame({
                "Statistica": [
                    "Partite giocate", "Punti totali", "PPG",
                    "Gol fatti/partita", "Gol subiti/partita",
                    "xG fatti/partita", "xG subiti/partita",
                    "Possesso medio %", "Differenza reti", "Percentuale vittorie",
                ],
                team_a: [
                    int(a.GP), int(a.Pts), f"{a.PPG:.2f}",
                    f"{a.GF/a.GP:.2f}" if a.GP>0 else "0",
                    f"{a.GA/a.GP:.2f}" if a.GP>0 else "0",
                    f"{a.get('XGF_pG',0):.2f}", f"{a.get('XGA_pG',0):.2f}",
                    f"{a.get('PossAvg',0):.1f}", f"{a.GD:+}", f"{a.WinRate:.1f}",
                ],
                team_b: [
                    int(b.GP), int(b.Pts), f"{b.PPG:.2f}",
                    f"{b.GF/b.GP:.2f}" if b.GP>0 else "0",
                    f"{b.GA/b.GP:.2f}" if b.GP>0 else "0",
                    f"{b.get('XGF_pG',0):.2f}", f"{b.get('XGA_pG',0):.2f}",
                    f"{b.get('PossAvg',0):.1f}", f"{b.GD:+}", f"{b.WinRate:.1f}",
                ]
            })

            st.dataframe(df_h2h, use_container_width=True, hide_index=True)

# ---------------- Export Excel ----------------
st.subheader("📥 Esporta report")
if st.button("Genera file Excel"):
    excel = export_excel_bytes(Classifica=table, Partite_Filtrate=df)
    st.download_button(
        "Scarica report Excel",
        data=excel,
        file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )