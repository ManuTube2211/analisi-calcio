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
    /* I controlli e le tabelle del pianificatore devono restare leggibili
       anche con il tema scuro di Streamlit. */
    [data-testid="stNumberInput"] [data-baseweb="input"],
    [data-testid="stNumberInput"] input {
        background:#102333 !important; color:#fff4ae !important;
        -webkit-text-fill-color:#fff4ae !important;
    }
    [data-testid="stNumberInput"] label,
    [data-testid="stSlider"] label,
    [data-testid="stSelectSlider"] label { color:#ffe780 !important; }
    div[data-testid="stAlert"] { border-radius: 12px; background:#102333; border:1px solid #456d82; }
    div[data-testid="stAlert"] * { color:#ffe780 !important; }
    .stButton > button, .stDownloadButton > button { background: #d89b00; color: #07111f; border: 0; border-radius: 9px; font-weight: 800; }
    .stButton > button:hover, .stDownloadButton > button:hover { background: #ffe780; color: #07111f; }
    .matchscope-title { margin:.75rem 0 .35rem; color:#ffe780; font-family:"Avenir Next", "Trebuchet MS", sans-serif;
        font-size:2.55rem; font-weight:800; letter-spacing:.035em; text-shadow:0 0 18px rgba(255,213,74,.3); }
    .match-core-wrap { display:flex; justify-content:center; margin:.25rem 0 1.1rem; height:116px; align-items:center; }
    .match-core { width:82px; height:82px; position:relative; display:grid; place-items:center; transform:rotate(30deg);
        border:1px solid #ffd54a; background:linear-gradient(145deg, rgba(37,120,158,.62), rgba(5,11,20,.96));
        box-shadow:0 0 12px rgba(255,213,74,.8), 0 0 34px rgba(47,190,255,.32), inset 0 0 24px rgba(31,178,235,.2);
        animation:core-float 3.8s ease-in-out infinite; }
    .match-core:before, .match-core:after { content:""; position:absolute; inset:-18px; border:1px solid rgba(84,205,255,.55); transform:rotate(15deg); }
    .match-core:after { inset:-31px; border-color:rgba(255,213,74,.28); animation:core-spin 8s linear infinite; }
    .match-core-ring { position:absolute; width:55px; height:55px; border:2px solid rgba(255,231,128,.82); border-radius:50%;
        box-shadow:0 0 13px rgba(255,213,74,.55); animation:core-spin 4.5s linear infinite reverse; }
    .match-core-ring:before { content:""; position:absolute; width:8px; height:8px; top:-5px; left:50%; border-radius:50%; background:#fff4ae; box-shadow:0 0 12px #ffd54a; }
    .match-core-mark { position:relative; transform:rotate(-30deg); color:#fff4ae; font-size:31px; font-weight:900; line-height:1;
        text-shadow:0 0 12px #ffd54a; }
    @keyframes core-spin { to { transform:rotate(360deg); } }
    @keyframes core-float { 50% { transform:rotate(30deg) translateY(-7px) scale(1.04); } }
    .standing-panel { margin:0 0 .7rem; padding:.65rem .7rem; background:rgba(8,19,31,.88); border:1px solid #31546b; border-left:3px solid #ffd54a; border-radius:10px; }
    .standing-title { color:#ffe780; font-size:.87rem; font-weight:800; letter-spacing:.035em; }.standing-country{ color:#92a4b5; font-size:.68rem; }
    .standing-row { display:grid; grid-template-columns:20px 1fr 24px 27px; gap:.25rem; padding:.22rem 0; border-top:1px solid rgba(71,104,126,.38); color:#e7edf3; font-size:.74rem; }.standing-head{color:#8ea3b5;font-size:.64rem;border-top:0;padding-top:.42rem}.standing-team{white-space:nowrap;overflow:hidden;text-overflow:ellipsis}.standing-pts{color:#ffe780;font-weight:800;text-align:right}
    /* Tema chiaro */
    :root { --ink:#18283b; --muted:#64748b; --brand:#b77900; --accent:#d99400; --surface:#ffffff; --line:#d6e0ea; }
    .stApp { background:radial-gradient(circle at 50% -20%, #fff4c9 0, #f7fafc 35%, #edf3f8 100%); color:var(--ink); }
    [data-testid="stHeader"] { background:rgba(255,255,255,.88); }
    [data-testid="stSidebar"] { background:#f8fbfe; border-right:1px solid var(--line); }
    [data-testid="stSidebar"] *, p, label, [data-testid="stCaptionContainer"] { color:var(--ink); }
    [data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"], [data-testid="stFileUploaderDropzone"] { background:#fff; border-color:#b7c9d9; }
    [data-testid="stSidebar"] [data-baseweb="input"], [data-testid="stSidebar"] input, [data-baseweb="input"], input { background:#fff !important; color:var(--ink) !important; -webkit-text-fill-color:var(--ink) !important; }
    [data-testid="stNumberInput"] [data-baseweb="input"], [data-testid="stNumberInput"] input { background:#fff !important; color:var(--ink) !important; -webkit-text-fill-color:var(--ink) !important; }
    [data-testid="stNumberInput"] label, [data-testid="stSlider"] label, [data-testid="stSelectSlider"] label { color:var(--ink) !important; }
    [data-testid="stSidebar"] pre, [data-testid="stSidebar"] code, [data-testid="stSidebar"] [data-testid="stCode"] { background:#fff8dc !important; color:#493400 !important; border-color:#e6cb79 !important; }
    h1, h2, h3, [data-testid="stMetricLabel"] { color:var(--ink); }
    [data-testid="stMetric"], .pick, .news-card { background:#fff; border-color:#d4dee8; box-shadow:0 6px 18px rgba(22,40,59,.07); }
    [data-testid="stMetricValue"], .pick-value, .news-card-title { color:#172a3d; }
    .pick-title, .pick-meta, .news-card-meta, .control-note, .home-tab-subtitle { color:#5f7082; }
    .home-tab-title, .news-card-link, .matchscope-title { color:#9a6500; text-shadow:none; }
    .stTabs [data-testid="stTab"], .stTabs [data-testid="stTab"] *, .stTabs [data-baseweb="tab"], .stTabs [data-baseweb="tab"] *, .stTabs button[role="tab"], .stTabs button[role="tab"] * { color:#384b5d !important; }
    .stTabs [aria-selected="true"], .stTabs [aria-selected="true"] * { color:#9a6500 !important; }
    [data-testid="stTable"] * { color:#18283b !important; }
    div[data-testid="stAlert"] { background:#fffdf3; border-color:#ecd58b; }
    div[data-testid="stAlert"] * { color:#4b3a13 !important; }
    .standing-panel { background:#fff; border-color:#d4dee8; border-left-color:#c88900; }
    .standing-title, .standing-pts { color:#3b2b00; }.standing-country,.standing-head { color:#64748b; }.standing-row { color:#24384b; border-top-color:#e4ebf1; }
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
        .matchscope-title { font-size:2.05rem; margin-top:.35rem; }
        .match-core-wrap { height:100px; }
        .match-core { width:66px; height:66px; }
    }
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="matchscope-title">MatchScope</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="match-core-wrap" aria-label="Nucleo dati animato">'
    '<div class="match-core"><span class="match-core-ring"></span>'
    '<span class="match-core-mark">⌁</span></div></div>',
    unsafe_allow_html=True,
)

# ---------------- Notizie live sempre disponibili ----------------
@st.cache_data(ttl=600, show_spinner=False)
def cached_match_news(home_team: str, away_team: str):
    """Cache di dieci minuti: aggiornabile dalla homepage."""
    return fetch_match_news(home_team, away_team)


@st.cache_data(ttl=120, show_spinner=False)
def cached_live_standings(refresh_token: str):
    """Cache breve; il token cambia quando l'utente richiede un refresh."""
    return fetch_live_standings(refresh_token)


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
    if "standings_refresh_token" not in st.session_state:
        st.session_state["standings_refresh_token"] = "initial"
    if refresh_homepage_tables:
        st.session_state["standings_refresh_token"] = datetime.now().isoformat(timespec="microseconds")
        st.session_state["standings_refreshed_at"] = datetime.now().strftime("%H:%M:%S")
    try:
        with st.spinner("Aggiorno le classifiche…"):
            live_standings = cached_live_standings(st.session_state["standings_refresh_token"])
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
    refreshed_at = st.session_state.get("standings_refreshed_at")
    refresh_note = f" · ultimo aggiornamento richiesto alle {refreshed_at}" if refreshed_at else ""
    st.caption(f"Dati live ESPN · prime 5 dei campionati nazionali e prime 8 delle competizioni UEFA{refresh_note}.")

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


def load_data(uploaded_file):
    """Legge e normalizza il CSV selezionato dall'utente nella sessione corrente."""
    if uploaded_file is None:
        return None, None, "Nessun CSV caricato."
    try:
        return _read_and_normalize(uploaded_file.getvalue(), "file caricato"), "file caricato", None
    except ValueError as exc:
        return None, "file caricato", f"CSV non valido: {exc}"
    except Exception as exc:
        return None, "file caricato", f"Errore durante la lettura del CSV: {exc}"


def clear_analysis_cache() -> None:
    _read_and_normalize.clear()
    cached_compute_table.clear()
    cached_adaptive_calibration.clear()


@st.cache_data(show_spinner=False)
def cached_compute_table(df: pd.DataFrame, win: int, draw: int) -> pd.DataFrame:
    schema = PointSchema(win=win, draw=draw, loss=0)
    return compute_table(df, schema=schema)


@st.cache_data(show_spinner=False)
def cached_adaptive_calibration(df: pd.DataFrame):
    """Ricalcola l'adattamento quando il file o la finestra dati cambiano."""
    return adaptive_goal_calibration(df)


def refresh_csv_analysis():
    """Forza il ricalcolo non appena l'utente sceglie un nuovo CSV."""
    clear_analysis_cache()


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("Dati")
    up = st.file_uploader(
        "Carica il tuo CSV",
        type=["csv"],
        key="matchscope_csv_upload",
        on_change=refresh_csv_analysis,
        help="Il file viene analizzato subito dopo la selezione.",
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

# ---------------- Lettura CSV ----------------
df_normalized, source_label, error_message = load_data(up)

if source_label:
    st.sidebar.success("CSV caricato correttamente.")
if error_message:
    st.sidebar.error(error_message)

if df_normalized is None:
    st.info("👆 Carica un CSV nella barra laterale per iniziare.")
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
    "home_shots_on_target", "away_shots_on_target",
    "home_corners", "away_corners", "home_yellow", "away_yellow",
]
advanced_stats_available = any(
    column in df and pd.to_numeric(df[column], errors="coerce").fillna(0).ne(0).any()
    for column in ADVANCED_STATS_COLUMNS
)

# ---------------- Classifica (cachata: dipende solo da df, win, draw) ----------------
table = cached_compute_table(df, win, draw)
calibration = cached_adaptive_calibration(df)


def _series_average(values: pd.Series) -> float | None:
    """Restituisce una media numerica, oppure None se lo storico non esiste."""
    numeric_values = pd.to_numeric(values, errors="coerce").dropna()
    return float(numeric_values.mean()) if not numeric_values.empty else None


def _venue_average(primary: pd.Series, fallback: pd.Series) -> float:
    """Privilegia casa/trasferta; con pochi dati usa la media stagionale."""
    primary_average = _series_average(primary)
    return primary_average if primary_average is not None else float(_series_average(fallback) or 0.0)


def estimate_match_volume(
    matches: pd.DataFrame, home_team: str, away_team: str, home_column: str, away_column: str,
) -> tuple[float, float]:
    """Stima una statistica di volume combinando produzione e dato concesso.

    Per la squadra di casa, ad esempio, media i suoi tiri prodotti in casa
    con i tiri concessi in trasferta dall'avversaria. Il ragionamento è
    speculare per la squadra ospite.
    """
    home_at_home = matches["home_team"].eq(home_team)
    home_at_away = matches["away_team"].eq(home_team)
    away_at_home = matches["home_team"].eq(away_team)
    away_at_away = matches["away_team"].eq(away_team)

    home_for = _venue_average(
        matches.loc[home_at_home, home_column],
        pd.concat([matches.loc[home_at_home, home_column], matches.loc[home_at_away, away_column]]),
    )
    home_against = _venue_average(
        matches.loc[home_at_home, away_column],
        pd.concat([matches.loc[home_at_home, away_column], matches.loc[home_at_away, home_column]]),
    )
    away_for = _venue_average(
        matches.loc[away_at_away, away_column],
        pd.concat([matches.loc[away_at_home, home_column], matches.loc[away_at_away, away_column]]),
    )
    away_against = _venue_average(
        matches.loc[away_at_away, home_column],
        pd.concat([matches.loc[away_at_home, away_column], matches.loc[away_at_away, home_column]]),
    )
    return (home_for + away_against) / 2, (away_for + home_against) / 2

# ---------------- Tabs ----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🏆 Classifica", "🔮 Pronostico match", "📄 Scheda squadra", "🧠 Apprendimento", "📈 Scalata"
])

# ========== TAB 1: Classifica ==========
with tab1:
    st.subheader("Classifica totale e Statistiche")
    primary_columns = ["Pos", "team", "Pts", "GP", "W", "D", "L", "GF", "GA", "GD", "PPG", "WinRate"]
    ordered_columns = [column for column in primary_columns if column in table.columns]
    ordered_columns.extend(
        column for column in table.columns
        if column not in ordered_columns and "red" not in column.casefold() and "ross" not in column.casefold()
    )
    standings_view = table[ordered_columns].rename(columns={
        "Pos": "Pos.", "team": "Squadra", "Pts": "Punti", "GP": "PG",
        "W": "V", "D": "N", "L": "P", "GF": "GF", "GA": "GS",
        "GD": "DR", "PPG": "PPG", "WinRate": "Vittorie %",
    })
    standings_style = standings_view.style.format({
        "PPG": "{:.2f}", "Vittorie %": "{:.1f}%",
    }).set_properties(
        subset=["Punti"], **{"font-weight": "700", "color": "#8a5b00"}
    )
    st.dataframe(standings_style, use_container_width=True, hide_index=True)
    st.caption("Punti in evidenza · PG = partite giocate · DR = differenza reti. Scorri per tiri, corner e altre statistiche.")

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
        fig.update_layout(title="Probabilità esito 1X2", height=360, margin=dict(l=0, r=0, t=45, b=0))
        st.plotly_chart(fig, use_container_width=True)
        st.caption("Probabilità 1X2 ottenute da gol reali, fattore casa e forma recente.")

        st.divider()
        st.subheader("Motore del pronostico")
        home_goals_estimate = float(gd["lambda_home"])
        away_goals_estimate = float(gd["lambda_away"])
        mirror_limit = max(home_goals_estimate, away_goals_estimate) * 1.35
        goals_mirror_fig = go.Figure()
        goals_mirror_fig.add_bar(
            name=pred_home, y=["Gol stimati"], x=[-home_goals_estimate], orientation="h",
            marker_color="#c58b10", text=[f"{home_goals_estimate:.2f}"], textposition="outside", textangle=0,
        )
        goals_mirror_fig.add_bar(
            name=pred_away, y=["Gol stimati"], x=[away_goals_estimate], orientation="h",
            marker_color="#4f7d9d", text=[f"{away_goals_estimate:.2f}"], textposition="outside", textangle=0,
        )
        goals_mirror_fig.update_layout(
            title="Confronto gol stimati", barmode="overlay", height=230,
            margin=dict(l=0, r=25, t=45, b=0),
            xaxis=dict(range=[-mirror_limit, mirror_limit], showticklabels=False, zeroline=True, zerolinecolor="#64748b"),
            yaxis=dict(showticklabels=False, title=None),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
            legend_title_text="Squadra",
        )
        st.plotly_chart(goals_mirror_fig, use_container_width=True)

        volume_metrics = [
            ("Corner previsti", "home_corners", "away_corners"),
            ("Tiri previsti", "home_shots", "away_shots"),
            ("Tiri in porta previsti", "home_shots_on_target", "away_shots_on_target"),
        ]
        available_volume_metrics = [
            metric for metric in volume_metrics
            if any(
                pd.to_numeric(df[column], errors="coerce").fillna(0).ne(0).any()
                for column in metric[1:]
            )
        ]
        st.divider()
        st.subheader("🎯 Statistiche previste della partita")
        if not available_volume_metrics:
            st.info(
                "Per stimare corner e tiri serve un CSV che contenga queste statistiche. "
                "I risultati scaricati online includono solo punteggi e classifiche."
            )
        else:
            st.caption(
                "La stima combina la media prodotta dalla squadra nel proprio campo "
                "con la media concessa dall'avversaria in trasferta."
            )
            forecast_columns = st.columns(len(available_volume_metrics))
            for forecast_column, (label, home_column, away_column) in zip(forecast_columns, available_volume_metrics):
                home_estimate, away_estimate = estimate_match_volume(
                    df, pred_home, pred_away, home_column, away_column,
                )
                with forecast_column:
                    st.metric(label, f"{home_estimate + away_estimate:.1f}")
                    st.caption(
                        f"{pred_home}: **{home_estimate:.1f}**  ·  "
                        f"{pred_away}: **{away_estimate:.1f}**"
                    )

        st.subheader("📊 Probabilità mercati gol")
        ou_dict = gd["ou"]
        ou_labels = list(ou_dict.keys())
        ou_values = [float(probability * 100) for probability in ou_dict.values()]
        market_fig = go.Figure(go.Bar(
            x=ou_values,
            y=ou_labels,
            orientation="h",
            marker_color=["#c58b10" if label.startswith("Over") else "#4f7d9d" for label in ou_labels],
            text=[f"{value:.1f}%" for value in ou_values],
            textposition="auto",
        ))
        market_fig.update_layout(
            height=360,
            margin=dict(l=0, r=25, t=15, b=0),
            xaxis=dict(range=[0, 100], title="Probabilità (%)"),
            yaxis=dict(autorange="reversed", title=None),
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(market_fig, use_container_width=True)

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
        score_values = [float(p * 100) for _, _, p in top5]
        score_fig = go.Figure(go.Bar(
            x=score_values,
            y=top5_df["Score"],
            orientation="h",
            marker_color="#c58b10",
            text=top5_df["Probabilità"],
            textposition="outside",
            textangle=0,
        ))
        score_fig.update_layout(
            title="I 5 risultati esatti più probabili",
            height=300,
            margin=dict(l=0, r=35, t=45, b=0),
            xaxis=dict(range=[0, max(score_values) * 1.25], title="Probabilità (%)"),
            yaxis=dict(autorange="reversed", title=None),
            plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(score_fig, use_container_width=True)

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
            [[
                column for column in df.columns
                if column not in ["competition", "round", "season", "home_xg", "away_xg"]
                and "red" not in column.casefold() and "ross" not in column.casefold()
            ]]
        )
        team_matches_view = team_matches_view.copy()
        team_matches_view["date"] = pd.to_datetime(team_matches_view["date"]).dt.strftime("%d/%m/%Y")
        # Trattandola esplicitamente come testo Streamlit non applica il proprio
        # formato locale (mm/gg/aaaa) alla colonna già formattata in italiano.
        st.dataframe(
            team_matches_view,
            use_container_width=True,
            hide_index=True,
            column_config={"date": st.column_config.TextColumn("Data")},
        )
        st.caption(
            f"Partite mostrate (basate sul filtro di {days_window} giorni): {len(team_matches_view)}"
        )

        row = table[table["team"] == team_sel]
        if row.empty:
            st.info("Squadra non trovata in classifica.")
        else:
            def get_stat(r, col_name, default=0.0):
                return r[col_name].iloc[0] if col_name in r.columns else default

            GP = int(get_stat(row, "GP", 0))
            st.subheader("Statistiche totali")
            total_metrics = [
                ("Partite", f"{GP}"),
                ("Punti", f"{int(get_stat(row, 'Pts', 0))}"),
                ("Punti per gara", f"{get_stat(row, 'PPG', 0):.2f}"),
                ("Vittorie", f"{int(get_stat(row, 'W', 0))}"),
                ("Pareggi", f"{int(get_stat(row, 'D', 0))}"),
                ("Sconfitte", f"{int(get_stat(row, 'L', 0))}"),
                ("Gol fatti", f"{int(get_stat(row, 'GF', 0))}"),
                ("Gol subiti", f"{int(get_stat(row, 'GA', 0))}"),
                ("Differenza reti", f"{int(get_stat(row, 'GD', 0)):+d}"),
                ("Vittorie %", f"{get_stat(row, 'WinRate', 0):.1f}%"),
            ]
            for start in range(0, len(total_metrics), 5):
                total_columns = st.columns(5)
                for column, (label, value) in zip(total_columns, total_metrics[start:start + 5]):
                    with column:
                        st.metric(label, value)

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
            total_shots = float(get_stat(row, "ShotsF", 0))
            shots_on_target = float(get_stat(row, "StF", 0))
            shooting_accuracy = (shots_on_target / total_shots * 100) if total_shots > 0 else None

            st.subheader("Medie dai risultati")
            core_a, core_b, core_c, core_d, core_e = st.columns(5)
            core_a.metric("Gol fatti / gara", f"{GF / GP:.2f}" if GP else "–")
            core_b.metric("Gol subiti / gara", f"{GA / GP:.2f}" if GP else "–")
            core_c.metric("Clean sheet", f"{clean_sheets / GP * 100:.0f}%" if GP else "–")
            core_d.metric("Over 2.5 / BTTS", f"{over_25 * 100:.0f}% / {both_score * 100:.0f}%" if GP else "–")
            core_e.metric("Precisione tiri", f"{shooting_accuracy:.1f}%" if shooting_accuracy is not None else "–")

            st.subheader("Statistiche medie (per partita)")
            advanced_available = advanced_stats_available
            if not advanced_available:
                st.info("Il tuo archivio non contiene ancora possesso, tiri, corner o cartellini per questa squadra.")
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
                ]
                for start in range(0, len(advanced_metrics), 3):
                    metric_columns = st.columns(3)
                    for metric_column, (label, value) in zip(metric_columns, advanced_metrics[start:start + 3]):
                        with metric_column:
                            st.metric(label, value)

            st.subheader("Riepilogo stagione")
            if GP > 0:
                W = int(get_stat(row, "W", 0))
                D = int(get_stat(row, "D", 0))
                L = int(get_stat(row, "L", 0))

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
        st.table(pd.DataFrame(plan_rows))

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
                st.table(pd.DataFrame(compatible))
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
                    {"Statistica": "Tiri in porta fatti / gara", colA: f"{rA.get('StF_pG', 0):.1f}", colB: f"{rB.get('StF_pG', 0):.1f}"},
                    {"Statistica": "Tiri in porta subiti / gara", colA: f"{rA.get('StA_pG', 0):.1f}", colB: f"{rB.get('StA_pG', 0):.1f}"},
                    {"Statistica": "Corner fatti / gara", colA: f"{rA.get('CornersF_pG', 0):.1f}", colB: f"{rB.get('CornersF_pG', 0):.1f}"},
                    {"Statistica": "Corner subiti / gara", colA: f"{rA.get('CornersA_pG', 0):.1f}", colB: f"{rB.get('CornersA_pG', 0):.1f}"},
                    {"Statistica": "Possesso medio %", colA: f"{rA.get('PossAvg', 0):.1f}%", colB: f"{rB.get('PossAvg', 0):.1f}%"},
                ]

            h2h_df = pd.DataFrame(rows)
            st.subheader("Confronto statistico")
            chart_home = pd.to_numeric(
                h2h_df[colA].astype(str).str.replace("%", "", regex=False), errors="coerce"
            ).fillna(0)
            chart_away = pd.to_numeric(
                h2h_df[colB].astype(str).str.replace("%", "", regex=False), errors="coerce"
            ).fillna(0)
            comparison_fig = go.Figure()
            comparison_fig.add_bar(
                name=colA, y=h2h_df["Statistica"], x=chart_home, orientation="h",
                marker_color="#c58b10", text=h2h_df[colA], textposition="outside", textangle=0,
            )
            comparison_fig.add_bar(
                name=colB, y=h2h_df["Statistica"], x=chart_away, orientation="h",
                marker_color="#4f7d9d", text=h2h_df[colB], textposition="outside", textangle=0,
            )
            comparison_fig.update_layout(
                barmode="group", height=max(420, len(h2h_df) * 52),
                margin=dict(l=0, r=25, t=15, b=0),
                xaxis_title="Valore medio / totale",
                yaxis=dict(autorange="reversed", title=None),
                plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
                legend_title_text="Squadra",
            )
            st.plotly_chart(comparison_fig, use_container_width=True)

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
