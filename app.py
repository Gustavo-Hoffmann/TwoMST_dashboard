import os
import re
import io
import pandas as pd
import streamlit as st
import plotly.express as px
from supabase import create_client, Client

# ========== CONFIG ==========

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")
BUCKET = "metrics"

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Defina SUPABASE_URL e SUPABASE_*KEY nas variáveis de ambiente.")
    st.stop()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

st.set_page_config(
    page_title="Dashboard Phone Metrics",
    layout="wide"
)

st.title("📊 Dashboard – Phone Metrics (Supabase Storage)")

# regex para extrair timestamp e subject do nome do arquivo:
# ex: 20251021-150659_predict_phone_S000000000000.xlsx
NAME_RE = re.compile(r"(\d{8})-(\d{6})_.*_(S\d+)\.xlsx")

def parse_meta(folder: str, filename: str):
    m = NAME_RE.match(filename)
    if not m:
        return None
    date_str, time_str, subj = m.groups()
    ts = pd.to_datetime(date_str + time_str, format="%Y%m%d%H%M%S", utc=True)
    return {
        "user_id": folder,
        "subject_code": subj,
        "session_ts": ts,
        "file_path": f"{BUCKET}/{folder}/{filename}",
    }

@st.cache_data(ttl=300)  # 5 minutos
def load_all_metrics():
    dfs = []

    # lista objetos na raiz do bucket (pastas dos usuários)
    root = supabase.storage.from_(BUCKET).list()
    folders = [item["name"] for item in root if item.get("id") is None]  # heurística "pasta"

    for folder in folders:
        files = supabase.storage.from_(BUCKET).list(path=folder)
        for f in files:
            fname = f["name"]
            if not fname.lower().endswith(".xlsx"):
                continue

            meta = parse_meta(folder, fname)
            if meta is None:
                continue

            # baixa o arquivo como bytes
            file_bytes: bytes = supabase.storage.from_(BUCKET).download(f"{folder}/{fname}")
            bio = io.BytesIO(file_bytes)

            try:
                df = pd.read_excel(bio)
            except Exception as e:
                print("Erro lendo", folder, fname, e)
                continue

            if df.empty:
                continue

            row = df.iloc[0].copy()

            # normaliza nomes de colunas se precisar (trocar espaços, etc.)
            row_dict = {
                "n_cycles": row.get("N#", None),
                "strategy": row.get("Strategy", None),
                "cadence_cpm": row.get("Cadence (cycles/min)", None),
                "vel_ini": row.get("Vel ini", None),
                "vel_end": row.get("Vel end", None),
                "slope_deg_s2": row.get("Slope (deg/s²)", None),
                "vel_mean": row.get("Vel mean", None),
                "vel_sd": row.get("Vel SD", None),
                "cv_vel": row.get("CV Vel", None),
                "vel_max": row.get("Vel max", None),
                "vel_min": row.get("Vel min", None),
                "time_mean": row.get("Time mean", None),
                "time_sd": row.get("Time SD", None),
                "cv_time": row.get("CV Time", None),
                "time_max": row.get("Time max", None),
                "time_min": row.get("Time min", None),
            }

            full_row = {**meta, **row_dict}
            dfs.append(full_row)

    if not dfs:
        return pd.DataFrame()

    df_all = pd.DataFrame(dfs)

    # tipos numéricos
    num_cols = [
        "n_cycles", "cadence_cpm", "vel_ini", "vel_end", "slope_deg_s2",
        "vel_mean", "vel_sd", "cv_vel", "vel_max", "vel_min",
        "time_mean", "time_sd", "cv_time", "time_max", "time_min",
    ]
    for c in num_cols:
        if c in df_all.columns:
            df_all[c] = pd.to_numeric(df_all[c], errors="coerce")

    return df_all

df = load_all_metrics()

if df.empty:
    st.warning("Nenhum arquivo .xlsx válido encontrado no bucket 'metrics'.")
    st.stop()

# ========== SIDEBAR – FILTROS ==========
st.sidebar.header("Filtros")

if "user_id" in df.columns:
    users = ["(todos)"] + sorted(df["user_id"].dropna().unique().tolist())
    sel_user = st.sidebar.selectbox("User ID", users)
    if sel_user != "(todos)":
        df = df[df["user_id"] == sel_user]

if "strategy" in df.columns:
    strategies = ["(todas)"] + sorted(df["strategy"].dropna().unique().tolist())
    sel_strat = st.sidebar.selectbox("Strategy", strategies)
    if sel_strat != "(todas)":
        df = df[df["strategy"] == sel_strat]

# filtro por data
if "session_ts" in df.columns:
    min_date = df["session_ts"].min().date()
    max_date = df["session_ts"].max().date()
    start, end = st.sidebar.date_input(
        "Período",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )
    if isinstance(start, tuple):  # streamlit bug às vezes
        start, end = start
    mask = (df["session_ts"].dt.date >= start) & (df["session_ts"].dt.date <= end)
    df = df[mask]

st.sidebar.markdown("---")
st.sidebar.write(f"Registros filtrados: **{len(df)}**")

# ========== RESUMO ==========
st.subheader("Resumo geral")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("N sessões", len(df))
with col2:
    st.metric("Usuários únicos", df["user_id"].nunique() if "user_id" in df.columns else 0)
with col3:
    st.metric("Cadência média (c/min)", f"{df['cadence_cpm'].mean():.1f}")
with col4:
    st.metric("Velocidade média (deg/s)", f"{df['vel_mean'].mean():.1f}")

st.write("### Estatísticas descritivas das métricas principais")
num_cols = [
    c for c in df.columns
    if df[c].dtype.kind in "fi"  # float/int
]
st.dataframe(df[num_cols].describe().T)

# ========== GRÁFICOS ==========
st.write("### Distribuição de cadência e velocidade")

c1, c2 = st.columns(2)

with c1:
    if "cadence_cpm" in df.columns:
        fig = px.histogram(df, x="cadence_cpm", nbins=20, title="Cadence (cycles/min)")
        st.plotly_chart(fig, use_container_width=True)

with c2:
    if "vel_mean" in df.columns:
        fig = px.histogram(df, x="vel_mean", nbins=20, title="Vel mean (deg/s)")
        st.plotly_chart(fig, use_container_width=True)

st.write("### Relação Cadence x Vel mean")

if {"cadence_cpm", "vel_mean"} <= set(df.columns):
    fig_scatter = px.scatter(
        df,
        x="cadence_cpm",
        y="vel_mean",
        color="strategy" if "strategy" in df.columns else None,
        hover_data=["user_id", "subject_code", "session_ts"],
        title="Vel mean vs Cadence"
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

# ========== TABELA ==========
st.write("### Dados brutos (uma linha por arquivo .xlsx)")
st.dataframe(df.sort_values("session_ts", ascending=False))
