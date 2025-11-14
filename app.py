import os
import re
import io
import pandas as pd
import streamlit as st
import plotly.express as px
from supabase import create_client, Client

# ================= CONFIG GERAL =================

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_SERVICE_ROLE_KEY") or os.getenv("SUPABASE_ANON_KEY")

BUCKET_METRICS = "metrics"
BUCKET_RAW = "raw"

PRIMARY_GREEN = "#004d40"  # verde escuro
BLACK = "#000000"
LOGO_PATH = "twomst_logo.png"  # coloque o PNG na raiz do repo com esse nome

if not SUPABASE_URL or not SUPABASE_KEY:
    st.error("Defina SUPABASE_URL e SUPABASE_*KEY nas variáveis de ambiente.")
    st.stop()

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

st.set_page_config(
    page_title="TwoMST Dashboard",
    layout="wide"
)

# ================= TEMA / CSS =================

st.markdown(
    f"""
    <style>
    .stApp {{
        background-color: #f5f7fb;
        color: #000000;
        font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        font-size: 17px;
    }}

    section[data-testid="stSidebar"] {{
        background-color: {PRIMARY_GREEN};
        padding-top: 1rem;
    }}
    section[data-testid="stSidebar"] * {{
        color: #ffffff !important;
        font-size: 17px;
    }}

    .sidebar-title-text {{
        font-size: 2.0rem;
        font-weight: 800;
        margin-top: 0.1rem;
    }}

    .sidebar-logo-fallback {{
        width: 40px;
        height: 40px;
        border-radius: 10px;
        background: #ffffff;
        display: flex;
        align-items: center;
        justify-content: center;
        color: {PRIMARY_GREEN};
        font-weight: 900;
        font-size: 1.2rem;
    }}

    /* cards bonitos */
    .card {{
        background-color: #ffffff;
        border-radius: 18px;
        padding: 1.4rem 1.7rem;
        box-shadow: 0 2px 6px rgba(0,0,0,0.06);
        border: 1px solid #e0e3eb;
        margin-bottom: 1rem;
    }}
    .card-title {{
        font-size: 1.1rem;
        color: #666a7a;
        margin-bottom: 0.4rem;
        font-weight: 500;
    }}
    .card-value {{
        font-size: 2.3rem;
        font-weight: 700;
        color: {PRIMARY_GREEN};
    }}

    /* tabelas arredondadas + fonte maior + linhas mais grossas */
    div[data-testid="stDataFrame"] {{
        border-radius: 12px !important;
        overflow: hidden;
        box-shadow: 0 2px 6px rgba(0,0,0,0.04);
    }}
    div[data-testid="stDataFrame"] table {{
        font-size: 16px;
        border-collapse: separate;
        border-spacing: 0;
    }}
    div[data-testid="stDataFrame"] table thead tr th,
    div[data-testid="stDataFrame"] table tbody tr td {{
        padding-top: 0.55rem;
        padding-bottom: 0.55rem;
        border-bottom: 2px solid #e0e3eb;
    }}

    h2, h3, h4 {{
        font-weight: 700;
    }}
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("📊 TwoMST.app – Dashboard")

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

    # tempo
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


# ================= CARREGAR DADOS =================

df = load_all_metrics()

if df.empty:
    st.warning("Nenhum arquivo .xlsx válido encontrado no bucket 'metrics'.")
    st.stop()

raw_index = index_raw_files()

# ================= SIDEBAR / MENU =================

with st.sidebar:
    col_logo, col_txt = st.columns([1, 3])
    with col_logo:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, width=40)
        else:
            st.markdown('<div class="sidebar-logo-fallback">T</div>', unsafe_allow_html=True)
    with col_txt:
        st.markdown('<div class="sidebar-title-text">TwoMST.app</div>', unsafe_allow_html=True)

    menu = st.radio(
        "Menu",
        ["Home", "Pacientes"],
        index=0,
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.caption(f"Sessões carregadas: {len(df)}")


# ================= HOME =================

if menu == "Home":
    total_tests = len(df)
    total_subjects = df["subject_code"].nunique() if "subject_code" in df.columns else 0

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f"""
            <div class="card">
              <div class="card-title">Testes realizados</div>
              <div class="card-value">{total_tests}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f"""
            <div class="card">
              <div class="card-title">Sujeitos únicos</div>
              <div class="card-value">{total_subjects}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if "session_ts" in df.columns:
        st.subheader("Testes por dia")
        df_daily = (
            df.assign(date=df["session_ts"].dt.date)
            .groupby("date")
            .size()
            .reset_index(name="n_tests")
        )
        fig_day = px.bar(
            df_daily,
            x="date",
            y="n_tests",
            title="Número de testes por dia",
            color_discrete_sequence=[PRIMARY_GREEN],
        )
        fig_day.update_traces(marker_line_width=0, marker_line_color="#ffffff")
        fig_day.update_layout(
            xaxis_title="Data",
            yaxis_title="N testes",
            plot_bgcolor="#ffffff",
            font=dict(size=17),
        )
        st.plotly_chart(fig_day, width="stretch")

# ================= PACIENTES =================

if menu == "Pacientes":
    st.subheader("Pacientes")

    # ---- Lista de pacientes ----
    df_pat = (
        df.groupby("subject_code")
        .agg(
            n_tests=("session_key", "nunique"),
            first_test=("session_ts", "min"),
            last_test=("session_ts", "max"),
        )
        .reset_index()
        .sort_values("subject_code")
    )

    st.markdown("#### Lista de pacientes")
    df_pat_view = df_pat.assign(
        first_test=df_pat["first_test"].dt.strftime("%Y-%m-%d %H:%M"),
        last_test=df_pat["last_test"].dt.strftime("%Y-%m-%d %H:%M"),
    ).reset_index(drop=True)
    st.dataframe(df_pat_view, use_container_width=True)

    # ---- Seleção de paciente ----
    subj_opts = df_pat["subject_code"].tolist()
    if not subj_opts:
        st.info("Nenhum paciente disponível.")
        st.stop()

    selected_subj = st.selectbox("Selecionar paciente", subj_opts)

    df_subj = df[df["subject_code"] == selected_subj].sort_values("session_ts")

    if df_subj.empty:
        st.info("Nenhum dado para esse paciente.")
        st.stop()

    st.markdown(f"### Paciente: `{selected_subj}`")

    df_subj = df_subj.copy()
    df_subj["session_idx"] = range(1, len(df_subj) + 1)

    df_subj["has_raw"] = df_subj.apply(
        lambda r: (str(r["user_id"]), str(r["session_key"])) in raw_index,
        axis=1,
    )

    # ---- Métricas rápidas ----
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f"""
            <div class="card">
              <div class="card-title">Sessões do sujeito</div>
              <div class="card-value">{len(df_subj)}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f"""
            <div class="card">
              <div class="card-title">Repetições (última sessão)</div>
              <div class="card-value">{int(df_subj['n_cycles'].iloc[-1])}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(
            f"""
            <div class="card">
              <div class="card-title">Cadência (última sessão, c/min)</div>
              <div class="card-value">{df_subj['cadence_cpm'].iloc[-1]:.1f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # ---- Evolução do nº de repetições ----
    st.markdown("#### Evolução do número de repetições")

    x_mode = st.radio(
        "Eixo X",
        ["Data da sessão", "Número da sessão"],
        horizontal=True,
    )

    if x_mode == "Data da sessão":
        x_col = "session_ts"
        x_label = "Data"
    else:
        x_col = "session_idx"
        x_label = "Sessão (#)"

    fig_rep = px.line(
        df_subj,
        x=x_col,
        y="n_cycles",
        markers=True,
        title=None,
        color_discrete_sequence=[PRIMARY_GREEN],
    )
    fig_rep.update_traces(line=dict(width=6), marker=dict(size=10))
    fig_rep.update_layout(
        xaxis_title=x_label,
        yaxis_title="Nº de repetições",
        plot_bgcolor="#ffffff",
        font=dict(size=17),
    )
    st.plotly_chart(fig_rep, width="stretch")

    # ---- Gráficos de barras das métricas ----
    st.markdown("#### Comparação de métricas entre sessões")

    metrics_to_plot = [
        ("cadence_cpm", "Cadência (c/min)"),
        ("vel_mean", "Velocidade média (deg/s)"),
        ("time_mean", "Tempo médio (s)"),
        ("cv_time", "CV Tempo"),
        ("cv_vel", "CV Vel"),
        ("vel_sd", "SD Vel"),
        ("vel_max", "Vel máx (deg/s)"),
        ("vel_min", "Vel mín (deg/s)"),
    ]
    metrics_to_plot = [m for m in metrics_to_plot if m[0] in df_subj.columns]

    for i in range(0, len(metrics_to_plot), 4):
        cols = st.columns(4)
        for j, (col_name, label) in enumerate(metrics_to_plot[i : i + 4]):
            with cols[j]:
                fig_bar = px.bar(
                    df_subj,
                    x="session_idx",
                    y=col_name,
                    title=label,
                    color_discrete_sequence=[PRIMARY_GREEN],
                )
                fig_bar.update_traces(marker_line_width=0, marker_line_color="#ffffff")
                fig_bar.update_layout(
                    xaxis_title="Sessão",
                    yaxis_title=None,
                    plot_bgcolor="#ffffff",
                    margin=dict(l=10, r=10, t=40, b=20),
                    font=dict(size=16),
                )
                st.plotly_chart(fig_bar, width="stretch")

    # ---- Lista de testes do sujeito ----
    st.markdown("#### Testes desse paciente")

    df_sessions_view = df_subj[
        [
            "session_idx",
            "session_ts",
            "n_cycles",
            "cadence_cpm",
            "vel_mean",
            "time_mean",
            "has_raw",
        ]
    ].copy()
    df_sessions_view["session_ts"] = df_sessions_view["session_ts"].dt.strftime(
        "%Y-%m-%d %H:%M"
    )
    df_sessions_view = df_sessions_view.reset_index(drop=True)
    st.dataframe(df_sessions_view, use_container_width=True)

    # ---- Selecionar um teste específico ----
    st.markdown("### Detalhes de um teste específico")

    def _format_session(row):
        return f"Sessão {int(row['session_idx'])} – {row['session_ts'].strftime('%Y-%m-%d %H:%M')}"

    options_idx = list(range(len(df_subj)))
    default_idx = len(df_subj) - 1
    sel_pos = st.selectbox(
        "Escolha a sessão",
        options_idx,
        index=default_idx,
        format_func=lambda i: _format_session(df_subj.iloc[i]),
    )
    row_sel = df_subj.iloc[sel_pos]

    # ---- Tabela com todas as métricas dessa sessão ----
    st.markdown("#### Métricas da sessão selecionada")

    metrics_cols = [
        "n_cycles",
        "strategy",
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
    metrics_cols = [c for c in metrics_cols if c in df_subj.columns]

    df_one = row_sel[metrics_cols].to_frame().T
    df_one.index = [row_sel["session_ts"].strftime("%Y-%m-%d %H:%M")]
    df_one = df_one.reset_index(drop=True)
    st.dataframe(df_one, use_container_width=True)

    # ---- Série temporal do gyro X ----
    st.markdown("#### Série temporal – gyroRotationX(rad/s)")

    df_raw_series = load_raw_series(
        user_id=row_sel["user_id"],
        session_key=row_sel["session_key"],
        raw_index=raw_index,
    )

    if df_raw_series is None or df_raw_series.empty:
        st.info("Não foi possível encontrar ou ler o arquivo bruto para essa sessão.")
    else:
        fig_raw = px.line(
            df_raw_series,
            x="t_s",
            y="gyro_x",
            title=None,
            color_discrete_sequence=[PRIMARY_GREEN],
        )
        fig_raw.update_traces(line=dict(width=4))
        fig_raw.update_layout(
            xaxis_title="Tempo (s)",
            yaxis_title="gyro X (rad/s)",
            plot_bgcolor="#ffffff",
            font=dict(size=17),
        )
        st.plotly_chart(fig_raw, width="stretch")
