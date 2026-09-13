# app.py — Streamlit Football Analysis
# Versione riscritta: caching corretto, validazione CSV più chiara, niente variabili duplicate tra tab

from datetime import datetime
from html import escape
import hmac
import os

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st

from analysis import (
    normalize_df,
    compute_table,
    PointSchema,
    export_excel_bytes,
    predict_goals_distribution,
    adaptive_goal_calibration,
    predict_1x2_from_matrix,
    match_recommendations,
    fetch_match_news,
    fetch_live_standings,
    FOOTBALL_DATA_COMPETITIONS,
    fetch_football_data_results,
)

st.set_page_config(page_title="MatchScope", page_icon="⚽", layout="wide")


def require_private_access():
    """Attiva un accesso privato solo quando MATCHSCOPE_PASSWORD è configurata.

    In locale non è richiesta alcuna password; sul servizio di hosting la
    variabile viene salvata come segreto e protegge l'app dal link pubblico.
    """
    configured_password = os.getenv("MATCHSCOPE_PASSWORD")
    if not configured_password or st.session_state.get("matchscope_authenticated"):
        return

    st.markdown("<div class='hero'><h1>⚽ MatchScope</h1><p>Accesso privato</p></div>", unsafe_allow_html=True)
    candidate = st.text_input("Password di accesso", type="password")
    if st.button("Accedi", type="primary"):
        if hmac.compare_digest(candidate, configured_password):
            st.session_state.matchscope_authenticated = True
            st.rerun()
        st.error("Password non corretta.")
    st.stop()


require_private_access()
st.markdown("""
<link rel="manifest" href="./app/static/manifest.json">
<meta name="theme-color" content="#07111f">
<meta name="apple-mobile-web-app-capable" content="yes">
<meta name="apple-mobile-web-app-status-bar-style" content="default">
<meta name="apple-mobile-web-app-title" content="MatchScope">
<link rel="apple-touch-icon" href="./app/static/icon.svg">
""", unsafe_allow_html=True)
st.markdown("""
<style>
    :root { --ink: #fff4ae; --muted: #a8b4c2; --brand: #ffd54a; --accent: #ffb300; --surface: #0d1b2a; --line: #25445c; }
    .stApp { background: radial-gradient(circle at 50% -15%, #183d58 0, #091522 38%, #050b14 78%); color: var(--ink); font-family: "Avenir Next", Avenir, "Segoe UI", sans-serif; }
    [data-testid="stHeader"] { background: rgba(5, 11, 20, .88); }
    [data-testid="stSidebar"] { background: #08131f; border-right: 1px solid var(--line); }
    [data-testid="stSidebar"] * { color: var(--ink); }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] { background:#102333; border:1px dashed #456d82; border-radius:12px; }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button { background:#d89b00; color:#07111f; border:0; font-weight:800; }
    [data-testid="stSidebar"] [data-baseweb="input"], [data-testid="stSidebar"] [data-baseweb="base-input"], [data-testid="stSidebar"] input { background:#102333 !important; color:#fff4ae !important; }
    [data-testid="stSidebar"] [data-baseweb="input"] { border:1px solid #456d82; border-radius:9px; }
    [data-testid="stSidebar"] pre, [data-testid="stSidebar"] code { background:#102333 !important; color:#ffe780 !important; border-color:#31546b !important; }
    [data-testid="stSidebar"] [data-testid="stCode"] { background:#102333 !important; border:1px solid #31546b; border-radius:10px; }
    h1, h2, h3, [data-testid="stMetricLabel"] { font-family: "Avenir Next", "Trebuchet MS", sans-serif; color: var(--ink); letter-spacing: .03em; }
    .hero { padding: 1.55rem 1.8rem; border-radius: 20px; margin-bottom: 1.4rem;
        background: linear-gradient(105deg, rgba(21, 58, 83, .92), rgba(9, 21, 34, .92) 65%); border: 1px solid #35627b;
        box-shadow: 0 0 34px rgba(255, 202, 46, .10), inset 0 1px rgba(255,255,255,.05); }
    .hero h1 { margin: 0; color: #ffe780; font-size: 2.5rem; letter-spacing: .02em; text-shadow: 0 0 15px rgba(255, 213, 74, .36); }
    .hero p { margin: .4rem 0 0; color: #d1dae4; font-size: 1.02rem; }
    .hero-kicker { color:#ffd54a !important; font-size:.72rem !important; font-weight:800; letter-spacing:.16em; margin:0 0 .3rem !important; }
    .control-note { color:#a8b4c2; font-size:.84rem; margin:-.3rem 0 .8rem; }
    .home-tab-title { color:#ffe780; font-size:1.15rem; font-weight:800; letter-spacing:.045em; margin:.45rem 0 .15rem; }
    .home-tab-subtitle { color:#9fb0c0; font-size:.82rem; margin:0 0 .9rem; }
    [data-testid="stMetric"] { background: rgba(13, 27, 42, .92); border: 1px solid var(--line);
        padding: .9rem; border-radius: 14px; box-shadow: inset 0 1px rgba(255,255,255,.04); }
    [data-testid="stMetricValue"] { color: #ffe780; font-family: "Avenir Next", Avenir, sans-serif; }
    .pick { background: rgba(13, 27, 42, .92); border: 1px solid #31546b; border-top: 4px solid #ffd54a;
        border-radius: 14px; padding: .85rem 1rem; min-height: 118px; box-shadow: 0 0 16px rgba(255, 213, 74, .06); }
    .pick-title { color: #a8b4c2; font-size: .78rem; text-transform: uppercase; letter-spacing: .08em; font-weight: 700; }
    .pick-value { color: #fff4ae; font-size: 1.08rem; font-weight: 700; margin: .35rem 0; }
    .pick-meta { color: #c8d3de; font-size: .84rem; }
    .news-card { height: 172px; padding: .85rem; border-radius: 14px; background: rgba(13, 27, 42, .92);
        border: 1px solid #31546b; border-top: 3px solid #ffd54a; box-sizing: border-box;
        box-shadow: inset 0 1px rgba(255,255,255,.04); display: flex; flex-direction: column; }
    .news-card-title { color: #fff4ae; font-size: .94rem; font-weight: 700; line-height: 1.3;
        overflow: hidden; display: -webkit-box; -webkit-line-clamp: 3; -webkit-box-orient: vertical; }
    .news-card-meta { color: #a8b4c2; font-size: .75rem; margin-top: auto; padding-top: .65rem; }
    .news-card-link { color: #ffe780; font-size: .8rem; font-weight: 700; text-decoration: none; margin-top: .25rem; }
    .stTabs [data-baseweb="tab-list"] { gap: .35rem; border-bottom: 1px solid var(--line); }
    /* Streamlit applica il colore al nodo interno della tab: lo imponiamo sia
       sul bottone sia sui suoi figli, anche quando la tab non è selezionata. */
    .stTabs [data-testid="stTab"], .stTabs [data-testid="stTab"] *,
    .stTabs [data-baseweb="tab"], .stTabs [data-baseweb="tab"] *,
    .stTabs button[role="tab"], .stTabs button[role="tab"] * {
        color: #ffe780 !important;
        font-weight: 650;
    }
    .stTabs [data-testid="stTab"], .stTabs [data-baseweb="tab"], .stTabs button[role="tab"] { padding: .6rem 1rem; }
    .stTabs [aria-selected="true"] { border-bottom-color: #ffd54a !important; }
    [data-testid="stDataFrame"], [data-testid="stTable"] { border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }
    [data-testid="stTable"] * { color:#fff4ae !important; }
    div[data-testid="stAlert"] { border-radius: 12px; background:#102333; border:1px solid #456d82; }
    div[data-testid="stAlert"] * { color:#ffe780 !important; }
    .stButton > button, .stDownloadButton > button { background: #d89b00; color: #07111f; border: 0; border-radius: 9px; font-weight: 800; }
    .stButton > button:hover, .stDownloadButton > button:hover { background: #ffe780; color: #07111f; }
    .jarvis-orb-wrap { display:flex; justify-content:center; margin:-.25rem 0 1rem; }
    .jarvis-orb { width:86px; height:86px; border-radius:50%; position:relative; overflow:hidden; border:2px solid #ffe780;
        background: radial-gradient(circle at 30% 28%, #fff8bb 0 4%, #ffd54a 6%, #d89100 28%, #152838 64%, #050b14 100%);
        box-shadow: 0 0 10px #ffd54a, 0 0 35px rgba(255,213,74,.75), inset -14px -12px 24px rgba(0,0,0,.65); animation: orb-float 3.5s ease-in-out infinite; }
    .jarvis-orb:before { content:""; position:absolute; inset:12px; border-radius:50%; border:1px dashed rgba(7,17,31,.75); animation: orb-spin 4s linear infinite; }
    .jarvis-orb:after { content:"⚽"; display:grid; place-items:center; position:absolute; inset:0; font-size:46px; filter:drop-shadow(0 0 4px #fff4ae); animation: orb-spin 7s linear infinite reverse; }
    @keyframes orb-spin { to { transform:rotate(360deg); } }
    @keyframes orb-float { 50% { transform:translateY(-7px) scale(1.035); } }
    .standing-panel { margin:0 0 .7rem; padding:.65rem .7rem; background:rgba(8,19,31,.88); border:1px solid #31546b; border-left:3px solid #ffd54a; border-radius:10px; }
    .standing-title { color:#ffe780; font-size:.87rem; font-weight:800; letter-spacing:.035em; }.standing-country{ color:#92a4b5; font-size:.68rem; }
    .standing-row { display:grid; grid-template-columns:20px 1fr 24px 27px; gap:.25rem; padding:.22rem 0; border-top:1px solid rgba(71,104,126,.38); color:#e7edf3; font-size:.74rem; }.standing-head{color:#8ea3b5;font-size:.64rem;border-top:0;padding-top:.42rem}.standing-team{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.standing-pts{color:#ffe780;font-weight:800;text-align:right}
    @media (max-width: 640px) {
        .block-container { padding: 0.8rem 0.8rem 2rem; }
        .hero { padding: 1.1rem; border-radius: 16px; margin-bottom: 1rem; }
        .hero h1 { font-size: 2rem; }
        .hero p { font-size: .92rem; }
        .news-card { height: 152px; padding: .72rem; }
        .news-card-title { font-size: .86rem; -webkit-line-clamp: 3; }
        .news-card-meta, .news-card-link { font-size: .72rem; }
        [data-testid="stMetric"] { padding: .7rem; }
        [data-testid="stMetricValue"] { font-size: 1.45rem; }
        .stTabs [data-baseweb="tab"] { font-size: .78rem; padding: .55rem .35rem; }
        .jarvis-orb { width:72px; height:72px; } .jarvis-orb:after { font-size:38px; }
    }
</style>
""", unsafe_allow_html=True)
st.markdown("""
<div class="hero">
  <p class="hero-kicker">MATCH INTELLIGENCE · LIVE DATA</p>
  <h1>⚽ MatchScope</h1>
  <p>Analisi squadra per squadra, probabilità trasparenti e pronostici costruiti sui dati della sfida.</p>
</div>
""", unsafe_allow_html=True)
st.markdown('<div class="jarvis-orb-wrap"><div class="jarvis-orb" aria-label="Palla animata"></div></div>', unsafe_allow_html=True)
st.caption("CSV richiesto: date, home_team, away_team, home_score, away_score, statistiche opzionali...")

# ---------------- Notizie live sempre disponibili ----------------
@st.cache_data(ttl=600, show_spinner=False)
def cached_match_news(home_team: str, away_team: str):
    """Cache di dieci minuti: aggiornabile dalla homepage."""
    return fetch_match_news(home_team, away_team)


@st.cache_data(ttl=600, show_spinner=False)
def cached_live_standings():
    """Cache di dieci minuti per evitare chiamate live a ogni interazione."""
    return fetch_live_standings()


def render_standing(standing):
    """Rende una mini classifica compatta, pensata per i pannelli laterali."""
    if standing.get("error"):
        return (
            f'<div class="standing-panel"><div class="standing-title">{escape(standing["name"])}</div>'
            '<div class="standing-country">Dati momentaneamente non disponibili</div></div>'
        )
    rows_html = ''.join(
        f'<div class="standing-row"><span>{escape(str(row["pos"]))}</span>'
        f'<span class="standing-team">{escape(str(row["team"]))}</span>'
        f'<span>{escape(str(row["played"]))}</span>'
        f'<span class="standing-pts">{escape(str(row["points"]))}</span></div>'
        for row in standing["rows"]
    )
    return (
        f'<div class="standing-panel"><div class="standing-title">{escape(standing["name"])}</div>'
        f'<div class="standing-country">{escape(standing["country"])} · live</div>'
        '<div class="standing-row standing-head"><span>#</span><span>Squadra</span><span>G</span><span>Pt</span></div>'
        f'{rows_html}</div>'
    )


st.subheader("◈ Centro operativo live")
st.markdown('<div class="control-note">Scegli la vista che ti serve: aggiornamenti editoriali oppure gerarchie dei campionati.</div>', unsafe_allow_html=True)

home_news_tab, home_standings_tab = st.tabs(["📰 Notizie", "🏆 Classifiche"])

with home_news_tab:
    st.markdown('<div class="home-tab-title">Notizie in primo piano</div><div class="home-tab-subtitle">Cerca una squadra o un argomento e segui gli aggiornamenti più recenti.</div>', unsafe_allow_html=True)
    news_input, refresh_news_col = st.columns([4, 1])
    with news_input:
        homepage_topic = st.text_input("Squadra o argomento", value="Serie A", key="homepage_news_topic")
    with refresh_news_col:
        st.write("")
        refresh_homepage_news = st.button("↻ Aggiorna", use_container_width=True, key="refresh_homepage_news")

    if refresh_homepage_news:
        cached_match_news.clear()
    try:
        with st.spinner("Aggiorno le notizie…"):
            homepage_news = cached_match_news(homepage_topic, "")
    except (requests.RequestException, ValueError):
        homepage_news = []

    if homepage_news:
        news_columns = st.columns(3, gap="medium")
        for column, item in zip(news_columns, homepage_news[:3]):
            safe_title = escape(item["title"])
            safe_source = escape(item["source"])
            safe_date = escape(item["published"])
            safe_link = escape(item["link"], quote=True)
            with column:
                st.markdown(
                    f'<div class="news-card">'
                    f'<div class="news-card-title">{safe_title}</div>'
                    f'<div class="news-card-meta">{safe_source} · {safe_date}</div>'
                    f'<a class="news-card-link" href="{safe_link}" target="_blank" rel="noopener noreferrer">Leggi articolo ↗</a>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
    else:
        st.info("Le notizie non sono raggiungibili al momento.")

with home_standings_tab:
    st.markdown('<div class="home-tab-title">Classifiche live</div><div class="home-tab-subtitle">Campionati nazionali e competizioni UEFA, con partite giocate e punti aggiornati.</div>', unsafe_allow_html=True)
    refresh_homepage_tables = st.button("↻ Aggiorna classifiche", key="refresh_homepage_tables")
    if refresh_homepage_tables:
        cached_live_standings.clear()
    try:
        with st.spinner("Aggiorno le classifiche…"):
            live_standings = cached_live_standings()
    except requests.RequestException:
        live_standings = {}

    league_keys = (
        "serie_a", "serie_b", "laliga",
        "bundesliga", "ligue_1", "premier_league",
        "champions_league", "europa_league", "conference_league",
    )
    for row_keys in (league_keys[:3], league_keys[3:6], league_keys[6:]):
        standing_columns = st.columns(3, gap="medium")
        for column, league_key in zip(standing_columns, row_keys):
            standing = live_standings.get(league_key)
            with column:
                if standing:
                    st.markdown(render_standing(standing), unsafe_allow_html=True)
    st.caption("Dati live ESPN · prime 5 dei campionati nazionali e prime 8 delle competizioni UEFA.")

st.divider()

# ----------------- Caricamento dati -----------------

def _find_header_row(raw_bytes: bytes, delimiter: str = ";") -> int:
    """Trova l'indice (0-based) della riga che contiene davvero l'intestazione
    delle colonne, cercando 'date' e 'home_team' tra i primi valori della riga.
    Così l'app funziona sia con 0, 1 o più righe decorative sopra (es. 'Tabella 1',
    'Season 25-26'), senza dover contare le righe a mano ogni volta."""
    text = raw_bytes.decode("utf-8", errors="replace")
    lines = text.splitlines()
    for i, line in enumerate(lines[:10]):  # cerca solo nelle prime righe
        cells = [c.strip().lower() for c in line.split(delimiter)]
        if "date" in cells and "home_team" in cells:
            return i
    # fallback: nessuna riga trovata, assume che l'intestazione sia la prima riga
    return 0


@st.cache_data(show_spinner=False)
def _read_and_normalize(raw_bytes: bytes, source_label: str):
    """Legge e normalizza il CSV. Non chiama st.* qui dentro: i messaggi UI
    vengono mostrati fuori, così funzionano sempre anche in cache hit."""
    import io
    header_row = _find_header_row(raw_bytes)
    df_raw = pd.read_csv(
        io.BytesIO(raw_bytes),
        delimiter=";",
        decimal=",",
        header=header_row,  # rilevata automaticamente, non più fissa a 1
    )
    df = normalize_df(df_raw)
    return df


def load_data(uploaded_file, online_data=None, online_label=None):
    """Priorità 1: file caricato dall'utente. Priorità 2: dati API in sessione.
    Priorità 3: dati_default.csv locale.

    Ritorna (df, source_label, error_message). error_message è None se ok.
    """
    if uploaded_file is not None:
        raw_bytes = uploaded_file.getvalue()
        source_label = "file caricato"
    elif online_data is not None:
        try:
            return normalize_df(online_data), online_label or "football-data.org", None
        except ValueError as e:
            return None, online_label, f"Dati online non validi: {e}"
    else:
        try:
            with open("dati_default.csv", "rb") as f:
                raw_bytes = f.read()
            source_label = "dati_default.csv locale"
        except FileNotFoundError:
            return None, None, "Nessun file caricato e 'dati_default.csv' non trovato."

    try:
        df = _read_and_normalize(raw_bytes, source_label)
        return df, source_label, None
    except ValueError as e:
        # Colonne mancanti o simili: messaggio parlante invece del traceback pandas
        return None, source_label, f"CSV non valido: {e}"
    except Exception as e:
        return None, source_label, (
            f"Errore lettura/normalizzazione ({source_label}): {e}. "
            "Controlla che il file usi ';' come separatore e ',' come decimale, "
            "e che la prima riga sia un'intestazione stagione da saltare."
        )


@st.cache_data(show_spinner=False)
def cached_compute_table(df: pd.DataFrame, win: int, draw: int) -> pd.DataFrame:
    schema = PointSchema(win=win, draw=draw, loss=0)
    return compute_table(df, schema=schema)


@st.cache_data(show_spinner=False)
def cached_adaptive_calibration(df: pd.DataFrame):
    """Ricalcola l'adattamento quando il file o la finestra dati cambiano."""
    return adaptive_goal_calibration(df)


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Dati")
    st.caption("Aggiornamento online")
    # Su Render la chiave vive nella variabile d'ambiente. Nell'app desktop non
    # esiste invece un ambiente Render: consentiamo quindi di incollarla solo
    # nella sessione corrente, senza scriverla nel repository o in un CSV.
    configured_api_key = os.getenv("FOOTBALL_DATA_API_KEY")
    if configured_api_key:
        api_key = configured_api_key
    else:
        api_key = st.text_input(
            "Chiave football-data.org (solo app Mac)",
            type="password",
            key="local_football_data_api_key",
            help="Incollala qui nell'app desktop. Resta solo nella sessione aperta e non viene salvata nei file.",
        ).strip()
    online_competition = st.selectbox(
        "Campionato football-data.org",
        options=list(FOOTBALL_DATA_COMPETITIONS),
        key="football_data_competition",
    )
    online_col, clear_online_col = st.columns(2)
    with online_col:
        update_from_api = st.button("↻ Aggiorna online", use_container_width=True)
    with clear_online_col:
        clear_online = st.button("Usa CSV", use_container_width=True)

    if clear_online:
        st.session_state.pop("football_data_matches", None)
        st.session_state.pop("football_data_label", None)
        st.rerun()

    if update_from_api:
        try:
            with st.spinner("Scarico i risultati ufficiali…"):
                api_results = fetch_football_data_results(
                    api_key,
                    FOOTBALL_DATA_COMPETITIONS[online_competition],
                )
            st.session_state["football_data_matches"] = api_results
            st.session_state["football_data_label"] = f"football-data.org · {online_competition}"
            st.success(f"Aggiornati {len(api_results)} risultati di {online_competition}.")
        except (requests.RequestException, ValueError) as exc:
            st.error(f"Aggiornamento online non riuscito: {exc}")

    st.caption("Un aggiornamento usa una richiesta API. I dati restano attivi finché non scegli “Usa CSV”.")
    st.divider()
    up = st.file_uploader(
        "Carica CSV (Opzionale)",
        type=["csv"],
        help="Se non carichi un file, verrà usato il 'dati_default.csv' locale.",
    )
    st.code(
        "date;home_team;away_team;home_score;away_score\n"
        "2024-08-17;Juventus;Roma;3;0",
        language="csv",
    )
    st.divider()
    win = st.number_input("Punti vittoria", 0, 5, 3)
    draw = st.number_input("Punti pareggio", 0, 3, 1)

    st.divider()
    st.header("Filtri Dati")
    days_window = st.slider(
        "Usa solo dati ultimi N giorni",
        min_value=60,
        max_value=1000,
        value=365,
        step=30,
        help="Filtra i dati per calcolare le statistiche solo sulle partite più recenti.",
    )

# ---------------- Lettura & normalizzazione ----------------
df_normalized, source_label, error_message = load_data(
    up,
    st.session_state.get("football_data_matches"),
    st.session_state.get("football_data_label"),
)

if source_label:
    st.sidebar.info(f"Dati caricati da: {source_label}")
if error_message:
    st.sidebar.warning(error_message)

if df_normalized is None:
    st.info("👆 Carica un CSV valido o aggiungi 'dati_default.csv' alla cartella per iniziare.")
    st.stop()

if df_normalized.empty:
    st.error("Errore: il CSV è vuoto o nessun dato è valido dopo la normalizzazione.")
    st.stop()

# --- Filtro temporale ---
max_date = df_normalized["date"].max()
min_date = max_date - pd.to_timedelta(days_window, unit="D")
df = df_normalized[df_normalized["date"] >= min_date].copy()

if df.empty:
    st.warning(
        f"Nessuna partita trovata negli ultimi {days_window} giorni. "
        "Prova ad allargare il filtro 'Usa solo dati ultimi N giorni' nella sidebar."
    )
    st.stop()

st.sidebar.caption(f"Statistiche calcolate su {len(df)} partite (dal {min_date.date()})")

# Lista squadre calcolata una sola volta e riusata in tutti i tab
all_teams = sorted(pd.unique(pd.concat([df["home_team"], df["away_team"]])))
ADVANCED_STATS_COLUMNS = [
    "home_possession", "away_possession", "home_shots", "away_shots",
    "home_corners", "away_corners", "home_yellow", "away_yellow",
]
advanced_stats_available = any(
    column in df and pd.to_numeric(df[column], errors="coerce").fillna(0).ne(0).any()
    for column in ADVANCED_STATS_COLUMNS
)

# ---------------- Classifica (cachata: dipende solo da df, win, draw) ----------------
table = cached_compute_table(df, win, draw)
calibration = cached_adaptive_calibration(df)

# ---------------- Tabs ----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏆 Classifica", "🔮 Pronostico match", "📄 Scheda squadra", "🧠 Apprendimento", "📈 Scalata"
])

# ========== TAB 1: Classifica ==========
with tab1:
    st.subheader("Classifica totale e Statistiche")
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.caption("PPG = punti per partita, WinRate = % vittorie. Scorri per vedere tiri, corner e altre statistiche.")

# ========== TAB 2: Previsioni ==========
with tab2:
    st.subheader("🔮 Previsioni (1X2 + Gol)")
    st.info("Il modello usa gol reali, fattore casa e maggiore peso alle partite recenti.")
    if calibration["enabled"]:
        adjustment = (float(calibration["factor"]) - 1) * 100
        direction = "aumenta" if adjustment > 0 else "riduce"
        st.caption(
            f"🧠 Calibrazione adattiva attiva: {direction} la stima gol del "
            f"{abs(adjustment):.1f}% sulla base di {calibration['sample_size']} partite concluse."
        )
    else:
        st.caption("🧠 Calibrazione adattiva in attesa di uno storico più ampio.")

    c1, c2, c3 = st.columns([1, 1, 1])
    with c1:
        pred_home = st.selectbox("Squadra di casa", all_teams, key="pred_home")
    with c2:
        pred_away = st.selectbox(
            "Squadra in trasferta", all_teams,
            index=1 if len(all_teams) > 1 else 0,
            key="pred_away",
        )
    with c3:
        rho = st.slider("Correlazione (ρ)", 0.00, 0.25, 0.05, 0.01)

    if not pred_home or not pred_away:
        st.warning("Seleziona entrambe le squadre.")
    elif pred_home == pred_away:
        st.warning("Scegli due squadre diverse.")
    else:
        gd = predict_goals_distribution(
            df, pred_home, pred_away, rho=rho, max_goals=8,
            calibration_factor=float(calibration["factor"]),
        )
        P = gd["joint"]

        oneXtwo = predict_1x2_from_matrix(P)
        oneXtwo_pct = {k: round(v * 100, 1) for k, v in oneXtwo.items()}

        colm = st.columns(3)
        colm[0].metric("Probabilità 1", f"{oneXtwo_pct['1']}%")
        colm[1].metric("Probabilità X", f"{oneXtwo_pct['X']}%")
        colm[2].metric("Probabilità 2", f"{oneXtwo_pct['2']}%")

        suggestions = match_recommendations(oneXtwo, {**gd["ou"], "btts": gd["btts"]})
        st.subheader("Indicazioni per questa partita")
        st.caption("Non sono certezze: ogni indicazione cambia con la coppia casa/trasferta e con lo storico selezionato.")
        pick_columns = st.columns(3)
        pick_titles = {"esito": "Esito più probabile", "gol": "Linea gol 2.5", "btts": "Entrambe segnano"}
        for column, key in zip(pick_columns, ("esito", "gol", "btts")):
            pick = suggestions[key]
            with column:
                st.markdown(
                    f'<div class="pick"><div class="pick-title">{pick_titles[key]}</div>'
                    f'<div class="pick-value">{pick["label"]}</div>'
                    f'<div class="pick-meta">{pick["probability"] * 100:.1f}% · confidenza {pick["confidence"]}</div></div>',
                    unsafe_allow_html=True,
                )

        fig = go.Figure(
            go.Pie(
                labels=[f"{pred_home} (1)", "Pareggio (X)", f"{pred_away} (2)"],
                values=[oneXtwo["1"], oneXtwo["X"], oneXtwo["2"]],
                textinfo="label+percent",
            )
        )
        fig.update_layout(height=360, margin=dict(l=0, r=0, t=30, b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Probabilità 1X2 ottenute da gol reali, fattore casa e forma recente.")

        st.divider()
        st.subheader("Motore del pronostico")
        sub = st.columns(2)
        sub[0].metric(f"{pred_home} gol stimati", f"{gd['lambda_home']:.2f}")
        sub[1].metric(f"{pred_away} gol stimati", f"{gd['lambda_away']:.2f}")

        tot = gd["total_goals_dist"]
        tot_labels = [str(i) for i in range(len(tot))]
        st.subheader("Distribuzione dei gol attesi")
        st.bar_chart(pd.DataFrame({"TotGoals%": (tot * 100).round(2)}, index=tot_labels))

        st.subheader("📊 Probabilità mercati gol")
        ou_dict = gd["ou"]
        ou_df = pd.DataFrame({
            "Mercato": list(ou_dict.keys()),
            "Probabilità": [f"{p * 100:.1f}%" for p in ou_dict.values()],
        })
        st.dataframe(ou_df, use_container_width=True, hide_index=True)

        st.subheader("⚽ Entrambe le squadre segnano")
        btts = float(gd.get("btts", 0.0))
        btts_pct = btts * 100
        st.metric("Probabilità BTTS (entrambe segnano)", f"{btts_pct:.1f}%")

        imax, jmax = np.unravel_index(np.argmax(P), P.shape)
        best_score_prob = P[imax, jmax] * 100.0
        st.info(
            f"Risultato esatto più probabile (modello combinato): "
            f"{pred_home} {imax}–{jmax} {pred_away} — {best_score_prob:.1f}%"
        )

        flat = [(i, j, P[i, j]) for i in range(P.shape[0]) for j in range(P.shape[1])]
        flat.sort(key=lambda x: x[2], reverse=True)
        top5 = flat[:5]
        top5_df = pd.DataFrame({
            "Score": [f"{pred_home} {i}-{j} {pred_away}" for i, j, _ in top5],
            "Probabilità": [f"{p * 100:.1f}%" for _, _, p in top5],
        })
        st.table(top5_df)

# ========== TAB 3: Scheda Squadra ==========
with tab3:
    st.header("📄 Scheda Squadra")
    team_sel = st.selectbox("Seleziona Squadra", all_teams, key="scheda_team")

    if team_sel:
        team_matches = df[(df["home_team"] == team_sel) | (df["away_team"] == team_sel)].copy()

        st.subheader("Partite giocate (filtrate)")
        team_matches_view = (
            team_matches
            .sort_values("date")
            [[c for c in df.columns if c not in ["competition", "round", "season", "home_xg", "away_xg"]]]
        )
        st.dataframe(team_matches_view, use_container_width=True, hide_index=True)
        st.caption(
            f"Partite mostrate (basate sul filtro di {days_window} giorni): {len(team_matches_view)}"
        )

        row = table[table["team"] == team_sel]
        if row.empty:
            st.info("Squadra non trovata in classifica.")
        else:
            st.subheader("Statistiche totali (dal DataFrame classifica)")
            st.dataframe(row.T, use_container_width=True)

            def get_stat(r, col_name, default=0.0):
                return r[col_name].iloc[0] if col_name in r.columns else default

            GP = int(get_stat(row, "GP", 0))

            # Queste medie si possono calcolare sempre dai risultati ufficiali
            # e sono perciò disponibili anche con la sorgente football-data.
            GF = float(get_stat(row, "GF", 0))
            GA = float(get_stat(row, "GA", 0))
            clean_sheets = int(((team_matches["home_team"] == team_sel) & (team_matches["away_score"] == 0)).sum())
            clean_sheets += int(((team_matches["away_team"] == team_sel) & (team_matches["home_score"] == 0)).sum())
            both_score = (
                (team_matches["home_score"] > 0) & (team_matches["away_score"] > 0)
            ).mean() if GP else 0.0
            over_25 = (
                (team_matches["home_score"] + team_matches["away_score"] >= 3)
            ).mean() if GP else 0.0

            st.subheader("Medie dai risultati")
            core_a, core_b, core_c, core_d = st.columns(4)
            core_a.metric("Gol fatti / gara", f"{GF / GP:.2f}" if GP else "–")
            core_b.metric("Gol subiti / gara", f"{GA / GP:.2f}" if GP else "–")
            core_c.metric("Clean sheet", f"{clean_sheets / GP * 100:.0f}%" if GP else "–")
            core_d.metric("Over 2.5 / BTTS", f"{over_25 * 100:.0f}% / {both_score * 100:.0f}%" if GP else "–")

            st.subheader("Statistiche medie (per partita)")
            advanced_available = advanced_stats_available
            if not advanced_available:
                st.info("Possesso, tiri, corner e cartellini non sono inclusi nei risultati di football-data.org. Per queste metriche avanzate carica il tuo CSV.")
            else:
                # Inserimento per righe, non per colonne: così ogni riquadro
                # riempie il primo spazio libero e la griglia resta compatta.
                advanced_metrics = [
                    ("Possesso palla medio", f"{get_stat(row, 'PossAvg', 0):.1f}%"),
                    ("Tiri subiti / gara", f"{get_stat(row, 'ShotsA_pG', 0):.1f}"),
                    ("Corner subiti / gara", f"{get_stat(row, 'CornersA_pG', 0):.1f}"),
                    ("Tiri fatti / gara", f"{get_stat(row, 'ShotsF_pG', 0):.1f}"),
                    ("Tiri in porta subiti / gara", f"{get_stat(row, 'StA_pG', 0):.1f}"),
                    ("Cartellini gialli (totali)", f"{int(get_stat(row, 'Yellow', 0))}"),
                    ("Tiri in porta fatti / gara", f"{get_stat(row, 'StF_pG', 0):.1f}"),
                    ("Corner fatti / gara", f"{get_stat(row, 'CornersF_pG', 0):.1f}"),
                    ("Cartellini rossi (totali)", f"{int(get_stat(row, 'Red', 0))}"),
                ]
                for start in range(0, len(advanced_metrics), 3):
                    metric_columns = st.columns(3)
                    for metric_column, (label, value) in zip(metric_columns, advanced_metrics[start:start + 3]):
                        with metric_column:
                            st.metric(label, value)

            st.subheader("Ripartizione esiti stagione")
            if GP > 0:
                W = int(get_stat(row, "W", 0))
                D = int(get_stat(row, "D", 0))
                L = int(get_stat(row, "L", 0))

                fig_pie = go.Figure(go.Pie(
                    labels=["Vittorie (W)", "Pareggi (D)", "Sconfitte (L)"],
                    values=[W, D, L],
                    textinfo="label+percent",
                    hole=0.25,
                ))
                fig_pie.update_layout(height=320, margin=dict(l=0, r=0, t=30, b=0))
                st.plotly_chart(fig_pie, use_container_width=True)

                c1_p, c2_p, c3_p, c4_p = st.columns(4)
                with c1_p: st.metric("Partite", GP)
                with c2_p: st.metric("Win %", f"{(W / GP * 100):.1f}%")
                with c3_p: st.metric("Draw %", f"{(D / GP * 100):.1f}%")
                with c4_p: st.metric("Loss %", f"{(L / GP * 100):.1f}%")
            else:
                st.info("Nessuna partita giocata per calcolare le percentuali.")

# ========== TAB 4: Apprendimento ==========
with tab4:
    st.subheader("🧠 Apprendimento controllato")
    st.caption(
        "L'app verifica il modello sui risultati già presenti nel CSV, in ordine temporale, "
        "e applica solo una correzione prudente alle stime dei gol."
    )
    if not calibration["enabled"]:
        st.info(str(calibration["message"]))
    else:
        adjustment = (float(calibration["factor"]) - 1) * 100
        metric_a, metric_b, metric_c, metric_d = st.columns(4)
        metric_a.metric("Correzione gol", f"{adjustment:+.1f}%")
        metric_b.metric("Partite verificate", int(calibration["sample_size"]))
        metric_c.metric("Errore medio gol", f"{float(calibration['goal_mae']):.2f}")
        metric_d.metric("Accuratezza O/U 2.5", f"{float(calibration['ou_accuracy']) * 100:.1f}%")

        comparison = pd.DataFrame({
            "Misura": ["Gol attesi dal modello", "Gol realmente segnati"],
            "Media per partita": [
                round(float(calibration["predicted_goals"]), 2),
                round(float(calibration["actual_goals"]), 2),
            ],
        })
        st.subheader("Verifica sugli ultimi risultati")
        st.dataframe(comparison, use_container_width=True, hide_index=True)
        st.success(str(calibration["message"]))
        st.caption(
            "Aggiorna il CSV con i risultati conclusi: la calibrazione verrà ricalcolata automaticamente. "
            "La correzione è limitata a ±8% per evitare che una breve serie alteri il modello."
        )

# ========== TAB 5: Scalata ==========
with tab5:
    st.subheader("📈 Pianificatore di scalata")
    st.caption(
        "Calcola la quota media necessaria a raggiungere un obiettivo reinvestendo l'intero budget a ogni step. "
        "È una simulazione matematica: non include quote reali né garantisce risultati."
    )

    budget_col, target_col, steps_col = st.columns(3)
    with budget_col:
        initial_budget = st.number_input("Budget iniziale (€)", min_value=1.0, value=10.0, step=1.0, key="growth_initial")
    with target_col:
        target_budget = st.number_input("Budget obiettivo (€)", min_value=1.0, value=100.0, step=1.0, key="growth_target")
    with steps_col:
        growth_steps = st.select_slider(
            "Numero di step", options=list(range(2, 13)), value=5,
            format_func=lambda value: f"x{value}", key="growth_steps",
        )

    if target_budget <= initial_budget:
        st.warning("Il budget obiettivo deve essere superiore al budget iniziale.")
    else:
        multiplier = target_budget / initial_budget
        ideal_quote = multiplier ** (1 / growth_steps)
        quote_tolerance = st.slider(
            "Tolleranza fascia quota", min_value=0.02, max_value=0.25, value=0.10,
            step=0.01, key="growth_tolerance",
            help="Quanto può discostarsi una quota teorica dalla quota media ideale dello step.",
        )
        range_low = max(1.01, ideal_quote - quote_tolerance)
        range_high = ideal_quote + quote_tolerance

        metric_a, metric_b, metric_c = st.columns(3)
        metric_a.metric("Moltiplicatore obiettivo", f"x{multiplier:.2f}")
        metric_b.metric("Quota media ideale", f"{ideal_quote:.2f}")
        metric_c.metric("Fascia per step", f"{range_low:.2f} – {range_high:.2f}")

        plan_rows = []
        current_budget = float(initial_budget)
        for step_number in range(1, growth_steps + 1):
            next_budget = current_budget * ideal_quote
            plan_rows.append({
                "Step": f"{step_number}/{growth_steps}",
                "Budget prima dello step": f"€ {current_budget:.2f}",
                "Quota media": f"{ideal_quote:.2f}",
                "Budget dopo lo step": f"€ {next_budget:.2f}",
            })
            current_budget = next_budget
        st.subheader("Percorso simulato")
        st.dataframe(pd.DataFrame(plan_rows), use_container_width=True, hide_index=True)

        st.subheader("Mercati teoricamente compatibili con lo step")
        if pred_home == pred_away:
            st.info("Seleziona due squadre diverse nel tab Pronostico match per confrontare i mercati teorici.")
        else:
            growth_gd = predict_goals_distribution(
                df, pred_home, pred_away, rho=rho, max_goals=8,
                calibration_factor=float(calibration["factor"]),
            )
            growth_1x2 = predict_1x2_from_matrix(growth_gd["joint"])
            growth_markets = [
                (f"{pred_home} vincente (1)", growth_1x2["1"]),
                ("Pareggio (X)", growth_1x2["X"]),
                (f"{pred_away} vincente (2)", growth_1x2["2"]),
                ("Over 2.5", float(growth_gd["ou"]["Over 2.5"])),
                ("Under 2.5", float(growth_gd["ou"]["Under 2.5"])),
                ("Gol (entrambe segnano)", float(growth_gd["btts"])),
                ("No Gol", 1 - float(growth_gd["btts"])),
            ]
            compatible = []
            for market, probability in growth_markets:
                fair_quote = 1 / probability if probability > 0 else np.nan
                if range_low <= fair_quote <= range_high:
                    compatible.append({
                        "Mercato": market,
                        "Probabilità modello": f"{probability * 100:.1f}%",
                        "Quota teorica equa": f"{fair_quote:.2f}",
                        "Compatibilità": "Nella fascia dello step",
                    })
            if compatible:
                st.dataframe(pd.DataFrame(compatible), use_container_width=True, hide_index=True)
            else:
                st.info("Nessun mercato teorico del match selezionato rientra nella fascia di questo step.")
            st.caption(
                "La quota teorica equa deriva dalla probabilità del modello (1 / probabilità) e non è una quota reale. "
                "Confrontala sempre con le quote effettivamente disponibili prima di prendere decisioni."
            )

# ========== Confronto integrato nel pronostico ==========
with tab2:
    st.divider()
    st.header("⚖️ Confronto della sfida")
    st.caption("Confronta le medie delle due squadre del match selezionato sopra.")

    h2h_teamA, h2h_teamB = pred_home, pred_away

    if h2h_teamA == h2h_teamB:
        st.warning("Seleziona due squadre diverse per il confronto.")
    else:
        rowA = table[table["team"] == h2h_teamA]
        rowB = table[table["team"] == h2h_teamB]

        if rowA.empty or rowB.empty:
            st.error("Una delle due squadre non è presente in classifica.")
        else:
            rA, rB = rowA.iloc[0], rowB.iloc[0]

            gpA = rA["GP"] if rA["GP"] > 0 else 1
            gpB = rB["GP"] if rB["GP"] > 0 else 1
            gfpgA, gfpgB = rA["GF"] / gpA, rB["GF"] / gpB
            gapgA, gapgB = rA["GA"] / gpA, rB["GA"] / gpB

            colA, colB = h2h_teamA, h2h_teamB
            rows = [
                {"Statistica": "Partite giocate (GP)", colA: f"{rA['GP']:.0f}", colB: f"{rB['GP']:.0f}"},
                {"Statistica": "Punti per gara (PPG)", colA: f"{rA['PPG']:.2f}", colB: f"{rB['PPG']:.2f}"},
                {"Statistica": "Gol fatti / gara", colA: f"{gfpgA:.2f}", colB: f"{gfpgB:.2f}"},
                {"Statistica": "Gol subiti / gara", colA: f"{gapgA:.2f}", colB: f"{gapgB:.2f}"},
                {"Statistica": "Differenza reti (GD)", colA: f"{rA['GD']:.0f}", colB: f"{rB['GD']:.0f}"},
            ]
            if advanced_stats_available:
                rows[4:4] = [
                    {"Statistica": "Tiri fatti / gara", colA: f"{rA.get('ShotsF_pG', 0):.1f}", colB: f"{rB.get('ShotsF_pG', 0):.1f}"},
                    {"Statistica": "Tiri subiti / gara", colA: f"{rA.get('ShotsA_pG', 0):.1f}", colB: f"{rB.get('ShotsA_pG', 0):.1f}"},
                    {"Statistica": "Tiri in porta fatti / gara", colA: f"{rA.get('StF_pG', 0):.1f}", colB: f"{rB.get('StA_pG', 0):.1f}"},
                    {"Statistica": "Possesso medio %", colA: f"{rA.get('PossAvg', 0):.1f}%", colB: f"{rB.get('PossAvg', 0):.1f}%"},
                ]

            h2h_df = pd.DataFrame(rows)
            st.subheader("Confronto statistico")
            st.dataframe(h2h_df, use_container_width=True, hide_index=True)

# ---------------- Export Excel ----------------
if st.button("Scarica Report Excel"):
    excel = export_excel_bytes(Classifica=table, Partite_Filtrate=df)
    st.download_button(
        "Report.xlsx",
        data=excel,
        file_name=f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )
# Fine app.py
