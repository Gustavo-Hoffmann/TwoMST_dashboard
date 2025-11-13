import os
import re
import io
import pandas as pd
import streamlit as st
import plotly.express as px
from supabase import create_client, Client

# ========== CONFIG GERAL ==========

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")

BUCKET_METRICS = "metrics"
BUCKET_RAW = "raw"

PRIMARY_GREEN = "#004d40"  # verde escuro
BLACK = "#000000"

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Defina SUPABASE_URL e SUPABASE_*KEY nas variáveis de ambiente.")
    st.stop()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

st.set_page_config(
    page_title="TwoMST Dashboard",
    layout="wide"
)

# ====== MINI TEMA (verde, branco, preto) ======
st.markdown(
    """
    <style>
    .stApp {
        background-color: #ffffff;
        color: #000000;
    }
    .block-container {
        padding-top: 1rem;
        padding-bottom: 2rem;
    }
    .stMetric label, .stMetric span {
        color: #004d40 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📊 TwoMST – Dashboard Phone Metrics")

# regex para METRICS: 20251021-150659_predict_phone_S000000000000.xlsx
METRICS_RE = re.compile(r"(\d{8})-(\d{6})_.*_(S\d+)\.xlsx", re.IGNORECASE)

# regex para RAW: 20251021-150659_dados.csv
RAW_RE = re.compile(r"(\d{8})-(\d{6})_dados\.csv", re.IGNORECASE)


def parse_metrics_meta(folder: str, filename: str):
    m = METRICS_RE.match(filename)
    if not m:
        return None
    date_str, time_str, subj = m.groups()
    ts = pd.to_datetime(date_str + time_str, format="%Y%m%d%H%M%S", utc=True)
    session_key = f"{date_str}-{time_str}"
    return {
        "user_id": folder,
        "subject_code": subj,
        "session_ts": ts,
        "session_key": session_key,
        "file_path": f"{BUCKET_METRICS}/{folder}/{filename}",
        "filename": filename,
    }


def parse_raw_meta(folder: str, filename: str):
    m = RAW_RE.match(filename)
    if not m:
        return None
    date_str, time_str = m.groups()
    session_key = f"{date_str}-{time_str}"
    return {
        "user_id": folder,
        "session_key": session_key,
        "file_path": f"{BUCKET_RAW}/{folder}/{filename}",
        "filename": filename,
    }


@st.cache_data(ttl=300)
def load_all_metrics():
    """Lê todos os .xlsx do bucket metrics e monta um DF de sessões."""
    root = supabase.storage.from_(BUCKET_METRICS).list()
    folders = [item["name"] for item in root]

    rows = []
    for folder in folders:
        files = supabase.storage.from_(BUCKET_METRICS).list(path=folder)
        for f in files:
            fname = f["name"]
            if not fname.lower().endswith(".xlsx"):
                continue

            meta = parse_metrics_meta(folder, fname)
            if meta is None:
                continue

            file_bytes: bytes = supabase.storage.from_(BUCKET_METRICS).download(
                f"{folder}/{fname}"
            )
            bio = io.BytesIO(file_bytes)

            try:
                df_x = pd.read_excel(bio)
            except Exception as e:
                print("Erro lendo metrics", folder, fname, e)
                continue

            if df_x.empty:
                continue

            row = df_x.iloc[0].copy()

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

            rows.append({**meta, **row_dict})

    if not rows:
        return pd.DataFrame()

    df_all = pd.DataFrame(rows)

    num_cols = [
        "n_cycles",
        "cadence_cpm",
        "vel_ini",
        "vel_end",
        "slope_deg_s2",
        "vel_mean",
        "vel_sd",
        "cv_vel",
        "vel_max",
        "vel_min",
        "time_mean",
        "time_sd",
        "cv_time",
        "time_max",
        "time_min",
    ]
    for c in num_cols:
        if c in df_all.columns:
            df_all[c] = pd.to_numeric(df_all[c], errors="coerce")

    return df_all


@st.cache_data(ttl=300)
def index_raw_files():
    """Cria um índice (user_id, session_key) -> caminho do arquivo bruto."""
    root = supabase.storage.from_(BUCKET_RAW).list()
    folders = [item["name"] for item in root]

    mapping = {}
    for folder in folders:
        files = supabase.storage.from_(BUCKET_RAW).list(path=folder)
        for f in files:
            fname = f["name"]
            if not fname.lower().endswith(".csv"):
                continue
            meta = parse_raw_meta(folder, fname)
            if meta is None:
                continue
            key = (str(meta["user_id"]), str(meta["session_key"]))
            mapping[key] = f"{folder}/{fname}"
    return mapping


def load_raw_series(user_id: str, session_key: str, raw_index: dict):
    """Carrega série temporal do gyro X para um par (user_id, session)."""
    key = (str(user_id), str(session_key))
    rel_path = raw_index.get(key)
    if not rel_path:
        return None

    file_bytes: bytes = supabase.storage.from_(BUCKET_RAW).download(rel_path)
    bio = io.BytesIO(file_bytes)

    try:
        df_raw = pd.read_csv(bio)
    except Exception as e:
        print("Erro lendo raw", rel_path, e)
        return None

    if df_raw.empty:
        return None

    # Colunas esperadas:
    # gyroTimestamp_sinceReboot(s), gyroRotationX(rad/s), ...
    time_col = None
    for c in df_raw.columns:
        if "timestamp" in c.lower() or "time" in c.lower():
            time_col = c
            break

    gyro_col = None
    for c in df_raw.columns:
        if "gyrorotationx" in c.lower() or "gyrox" in c.lower():
            gyro_col = c
            break

    if not time_col or not gyro_col:
        return None

    t = pd.to_numeric(df_raw[time_col], errors="coerce")
    t = t - t.min()
    g = pd.to_numeric(df_raw[gyro_col], errors="coerce")

    return pd.DataFrame({"t_s": t, "gyro_x": g})


# ========== CARREGA MÉTRICAS ==========
df = load_all_metrics()

if df.empty:
    st.warning("Nenhum arquivo .xlsx válido encontrado no bucket 'metrics'.")
    st.stop()

# ========== SIDEBAR – FILTROS GERAIS ==========
st.sidebar.header("Filtros")

if "strategy" in df.columns:
    strategies = ["(todas)"] + sorted(df["strategy"].dropna().unique().tolist())
    sel_strat = st.sidebar.selectbox("Strategy", strategies)
    if sel_strat != "(todas)":
        df = df[df["strategy"] == sel_strat]

# filtro por data
if "session_ts" in df.columns and not df["session_ts"].isna().all():
    min_date = df["session_ts"].min().date()
    max_date = df["session_ts"].max().date()
    start, end = st.sidebar.date_input(
        "Período",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
    )
    if isinstance(start, tuple):
        start, end = start
    mask = (df["session_ts"].dt.date >= start) & (df["session_ts"].dt.date <= end)
    df = df[mask]

st.sidebar.markdown("---")
st.sidebar.write(f"Sessões filtradas: **{len(df)}**")

# ========== TABS ==========
tab_overview, tab_subject = st.tabs(["Visão geral", "Análise por sujeito"])

# --------------------------------------------------
# TAB 1 — VISÃO GERAL
# --------------------------------------------------
with tab_overview:
    st.subheader("Overview geral")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("N sessões", len(df))
    with col2:
        st.metric(
            "N sujeitos únicos",
            df["subject_code"].nunique() if "subject_code" in df.columns else 0,
        )
    with col3:
        st.metric("Cadência média (c/min)", f"{df['cadence_cpm'].mean():.1f}")
    with col4:
        st.metric("Velocidade média (deg/s)", f"{df['vel_mean'].mean():.1f}")

    st.markdown("### Distribuição das métricas principais")

    c1, c2 = st.columns(2)

    with c1:
        if "cadence_cpm" in df.columns:
            fig = px.histogram(
                df,
                x="cadence_cpm",
                nbins=20,
                title="Cadence (cycles/min)",
                color_discrete_sequence=[PRIMARY_GREEN],
            )
            st.plotly_chart(fig, width="stretch")

    with c2:
        if "vel_mean" in df.columns:
            fig = px.histogram(
                df,
                x="vel_mean",
                nbins=20,
                title="Vel mean (deg/s)",
                color_discrete_sequence=[PRIMARY_GREEN],
            )
            st.plotly_chart(fig, width="stretch")

    st.markdown("### Relação Cadence × Vel mean")
    if {"cadence_cpm", "vel_mean"} <= set(df.columns):
        fig_scatter = px.scatter(
            df,
            x="cadence_cpm",
            y="vel_mean",
            color="strategy" if "strategy" in df.columns else None,
            hover_data=["subject_code", "session_ts"],
            title="Vel mean vs Cadence",
            color_discrete_sequence=[PRIMARY_GREEN],
        )
        st.plotly_chart(fig_scatter, width="stretch")

# --------------------------------------------------
# TAB 2 — ANÁLISE POR SUJEITO
# --------------------------------------------------
with tab_subject:
    st.subheader("Análise por sujeito")

    if "subject_code" not in df.columns:
        st.info("Não há coluna 'subject_code' nas métricas.")
    else:
        subj_opts = sorted(df["subject_code"].dropna().unique().tolist())
        if not subj_opts:
            st.info("Nenhum sujeito disponível com os filtros atuais.")
        else:
            selected_subj = st.selectbox("Selecione o sujeito", subj_opts)

            df_subj = df[df["subject_code"] == selected_subj].sort_values("session_ts")

            if df_subj.empty:
                st.info("Nenhum dado para esse sujeito com os filtros atuais.")
            else:
                df_subj = df_subj.copy()
                df_subj["session_idx"] = range(1, len(df_subj) + 1)

                # índice de arquivos brutos
                raw_index = index_raw_files()
                df_subj["has_raw"] = df_subj.apply(
                    lambda r: (str(r["user_id"]), str(r["session_key"])) in raw_index,
                    axis=1,
                )

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Sessões do sujeito", len(df_subj))
                with c2:
                    st.metric("Repetições (última)", int(df_subj["n_cycles"].iloc[-1]))
                with c3:
                    st.metric(
                        "Cadência (última)",
                        f"{df_subj['cadence_cpm'].iloc[-1]:.1f} c/min",
                    )

                # eixo X dos gráficos (data ou ordem)
                x_mode = st.radio(
                    "Eixo X do gráfico de evolução",
                    ["Data da coleta", "Ordem da sessão"],
                    horizontal=True,
                )

                if x_mode == "Data da coleta" and "session_ts" in df_subj.columns:
                    x_col = "session_ts"
                    x_label = "Data da coleta"
                else:
                    x_col = "session_idx"
                    x_label = "Sessão (#)"

                # ---- Evolução do nº de repetições ----
                fig_rep = px.line(
                    df_subj,
                    x=x_col,
                    y="n_cycles",
                    markers=True,
                    title=f"Evolução do nº de repetições – {selected_subj}",
                    color_discrete_sequence=[PRIMARY_GREEN],
                )
                fig_rep.update_layout(
                    xaxis_title=x_label, yaxis_title="Nº de repetições"
                )
                st.plotly_chart(fig_rep, width="stretch")

                # ---- Cadência + vel_mean ----
                if {"cadence_cpm", "vel_mean"} <= set(df_subj.columns):
                    fig_multi = px.line(
                        df_subj,
                        x=x_col,
                        y=["cadence_cpm", "vel_mean"],
                        markers=True,
                        title=f"Cadência e velocidade média – {selected_subj}",
                        color_discrete_map={
                            "cadence_cpm": PRIMARY_GREEN,
                            "vel_mean": BLACK,
                        },
                    )
                    fig_multi.update_layout(xaxis_title=x_label, yaxis_title="Valor")
                    st.plotly_chart(fig_multi, width="stretch")

                st.markdown("### Série temporal do gyro X (dados brutos)")

                # selectbox de sessão (para escolher qual série temporal ver)
                def _format_session(i: int) -> str:
                    r = df_subj.iloc[i]
                    base = f"Sessão {int(r['session_idx'])} – {r['session_ts'].strftime('%Y-%m-%d %H:%M')}"
                    if r["has_raw"]:
                        return base + " ✅ bruto"
                    return base + " ⚠️ sem bruto"
                options_idx = list(range(len(df_subj)))
                default_idx = len(df_subj) - 1  # última sessão
                sel_pos = st.selectbox(
                    "Sessão para visualizar o gyro X",
                    options_idx,
                    index=default_idx,
                    format_func=_format_session,
                )
                row_sel = df_subj.iloc[sel_pos]

                if not row_sel["has_raw"]:
                    st.info("Essa sessão não tem arquivo bruto correspondente no bucket 'raw'.")
                else:
                    df_raw_series = load_raw_series(
                        user_id=row_sel["user_id"],
                        session_key=row_sel["session_key"],
                        raw_index=raw_index,
                    )
                    if df_raw_series is None or df_raw_series.empty:
                        st.warning("Não foi possível ler a série temporal do gyro X.")
                    else:
                        fig_raw = px.line(
                            df_raw_series,
                            x="t_s",
                            y="gyro_x",
                            title=f"Gyro X – sessão {int(row_sel['session_idx'])}",
                            color_discrete_sequence=[PRIMARY_GREEN],
                        )
                        fig_raw.update_layout(
                            xaxis_title="Tempo (s)",
                            yaxis_title="gyro X (rad/s)",
                        )
                        st.plotly_chart(fig_raw, width="stretch")

                st.markdown("### Sessões desse sujeito")
                st.dataframe(
                    df_subj[
                        [
                            "session_ts",
                            "session_idx",
                            "n_cycles",
                            "cadence_cpm",
                            "vel_mean",
                            "time_mean",
                            "has_raw",
                        ]
                    ].sort_values("session_ts", ascending=False)
                )
