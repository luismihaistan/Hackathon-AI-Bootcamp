import os
import joblib
import pandas as pd
import streamlit as st

# =========================
#       CONFIG PATHS
# =========================
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(ROOT, "models", "fraud_model.pkl")
SCALER_PATH = os.path.join(ROOT, "models", "scaler.pkl")
DATA_PATH = os.path.join(ROOT, "data", "creditcard_mini.csv")

# =========================
#   LOAD MODEL + SCALER
# =========================
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


# =========================
#     LOAD RAW DATA
# =========================
@st.cache_data
def load_raw_data():
    df = pd.read_csv(DATA_PATH)
    return df


# =========================
#     PREPARE DATA
# =========================
def prepare_data(df: pd.DataFrame, model, scaler):
    # 1. Asigurăm consistența cu train_model.py
    if "Class" in df.columns:
        df = df.rename(columns={"Class": "is_fraud"})

    # 2. Filtru micro-tranzacții (la fel ca în train_model.py)
    df = df[df["Amount"] <= 100.0].copy()

    # 3. Alegem EXACT features-urile pe care a fost antrenat scalerul
    feature_cols = getattr(scaler, "feature_names_in_", None)

    if feature_cols is not None:
        # dacă StandardScaler are salvat numele coloanelor, le folosim direct
        features = df[feature_cols]
    else:
        # fallback – ar trebui să nu mai ajungem aici, dar e safe
        cols_to_drop = ["user_id", "id", "Time", "Unnamed: 0", "is_fraud", "Class"]
        features = df.drop(cols_to_drop, axis=1, errors="ignore")

    # 4. Scalăm și facem predicții
    scaled = scaler.transform(features)

    df["fraud_prediction"] = model.predict(scaled)
    df["risk_score"] = model.predict_proba(scaled)[:, 1]

    return df, list(features.columns)


# =========================
#      STYLES / THEME
# =========================
st.set_page_config(
    page_title="Micro-Fraud Detection Dashboard",
    page_icon="🛡️",
    layout="wide",
)

st.markdown(
    """
<style>
/* Main container padding */
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Title */
.big-title {
    font-size: 44px;
    font-weight: 800;
    color: #0b1120;
    margin-bottom: 0.3rem;
}
.subtext {
    font-size: 18px;
    color: #4b5563;
}

/* Hero pill */
.hero-pill {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: 4px 12px;
    border-radius: 999px;
    background: linear-gradient(90deg, #38bdf8, #a855f7);
    color: #f9fafb;
    font-size: 13px;
    font-weight: 600;
}

/* Metric cards */
.metric-card {
    padding: 18px 18px 14px 18px;
    border-radius: 16px;
    background: #0f172a;
    border: 1px solid rgba(148, 163, 184, 0.35);
    color: #e5e7eb;
}
.metric-label {
    font-size: 13px;
    text-transform: uppercase;
    letter-spacing: .08em;
    color: #9ca3af;
}
.metric-value {
    font-size: 26px;
    font-weight: 700;
    margin-top: 4px;
}
.metric-sub {
    font-size: 12px;
    color: #6b7280;
}

/* Risk badges */
.risk-high {
    background-color: #ef4444;
    padding: 4px 10px;
    border-radius: 999px;
    color: white;
    font-weight: 600;
    font-size: 12px;
}
.risk-medium {
    background-color: #f97316;
    padding: 4px 10px;
    border-radius: 999px;
    color: white;
    font-weight: 600;
    font-size: 12px;
}
.risk-low {
    background-color: #22c55e;
    padding: 4px 10px;
    border-radius: 999px;
    color: white;
    font-weight: 600;
    font-size: 12px;
}

/* Table container */
.table-box {
    background: #020617;
    border-radius: 18px;
    border: 1px solid rgba(148, 163, 184, 0.35);
    padding: 16px 16px 4px 16px;
}

/* Section titles */
.section-title {
    font-weight: 700;
    font-size: 20px;
    color: #e5e7eb;
}

/* Transaction inspector */
.inspect-box {
    background: #020617;
    border-radius: 18px;
    border: 1px solid rgba(148, 163, 184, 0.35);
    padding: 18px;
    color: #e5e7eb;
}

/* Small grey text */
.small-muted {
    font-size: 12px;
    color: #9ca3af;
}

/* Override default headers color */
h1, h2, h3, h4 {
    color: #e5e7eb !important;
}
</style>
""",
    unsafe_allow_html=True,
)

# =========================
#          HEADER
# =========================
st.markdown(
    '<div class="hero-pill">🛡️ AI Hackathon · Micro-Fraud Detection</div>',
    unsafe_allow_html=True,
)

st.markdown(
    '<p class="big-title">AI pentru prevenirea fraudelor mici<br/>prin analiza micro-tranzacțiilor</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="subtext">Modelul filtrează micro-tranzacțiile (≤ 100 unități), le '
    'preprocesează și folosește un <strong>Random Forest</strong> antrenat pe date reale pentru '
    'a atribui fiecărei tranzacții un <strong>scor de risc de fraudă</strong>.</p>',
    unsafe_allow_html=True,
)

st.write("")
tag_cols = st.columns(4)
tag_cols[0].markdown("✅ Random Forest classifier")
tag_cols[1].markdown("📊 Feature scaling & preprocessing")
tag_cols[2].markdown("💸 Focus pe micro-tranzacții")
tag_cols[3].markdown("👀 Scor de risc explicabil")

st.markdown("---")

# =========================
#   LOAD MODEL + DATA
# =========================
try:
    model, scaler = load_model_and_scaler()
    raw_df = load_raw_data()
    df, feature_names = prepare_data(raw_df, model, scaler)
except Exception as e:
    st.error(
        "Nu am reușit să încarc modelul sau datele. "
        "Verifică structura folderelor și fișierele .pkl / .csv."
    )
    st.exception(e)
    st.stop()

# =========================
#        TOP METRICS
# =========================
total_tx = len(df)
fraud_rate = df["fraud_prediction"].mean() if total_tx > 0 else 0
avg_amount = df["Amount"].mean() if total_tx > 0 else 0
max_risk = df["risk_score"].max() if total_tx > 0 else 0

m1, m2, m3, m4 = st.columns(4)

m1.markdown(
    f"""
<div class="metric-card">
    <div class="metric-label">Total micro-tranzacții analizate</div>
    <div class="metric-value">{total_tx}</div>
    <div class="metric-sub">filtrate cu limită ≤ 100</div>
</div>
""",
    unsafe_allow_html=True,
)

m2.markdown(
    f"""
<div class="metric-card">
    <div class="metric-label">Rată estimată de tranzacții suspecte</div>
    <div class="metric-value">{fraud_rate*100:.2f}%</div>
    <div class="metric-sub">bazat pe predicția modelului</div>
</div>
""",
    unsafe_allow_html=True,
)

m3.markdown(
    f"""
<div class="metric-card">
    <div class="metric-label">Valoare medie a micro-tranzacțiilor</div>
    <div class="metric-value">{avg_amount:.2f}</div>
    <div class="metric-sub">unități monetare</div>
</div>
""",
    unsafe_allow_html=True,
)

m4.markdown(
    f"""
<div class="metric-card">
    <div class="metric-label">Cel mai mare scor de risc</div>
    <div class="metric-value">{max_risk:.2f}</div>
    <div class="metric-sub">1.00 = foarte probabilă fraudă</div>
</div>
""",
    unsafe_allow_html=True,
)

st.markdown("---")

# =========================
#    HELPER: RISK BADGE
# =========================
def risk_badge(score: float) -> str:
    if score >= 0.85:
        return f"<span class='risk-high'>High risk · {score:.2f}</span>"
    if score >= 0.5:
        return f"<span class='risk-medium'>Medium risk · {score:.2f}</span>"
    return f"<span class='risk-low'>Low risk · {score:.2f}</span>"


# =========================
#    TABLE + INSPECTOR
# =========================
left, right = st.columns([2.2, 1])

with left:
    st.markdown('<div class="section-title">📊 Micro-tranzacții și scorul de risc</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="small-muted">Fiecare rând reprezintă o micro-tranzacție filtrată din setul de date real, '
        'cu scorul de risc calculat de model.</p>',
        unsafe_allow_html=True,
    )

    df_view = df.copy()
    df_view["Risk Level"] = df_view["risk_score"].apply(risk_badge)
    df_view["Fraud (model)"] = df_view["fraud_prediction"].map({0: "Legit", 1: "⚠ Fraud-like"})

    show_cols = ["Amount", "risk_score", "Risk Level", "Fraud (model)"]
    existing = [c for c in show_cols if c in df_view.columns]

    st.markdown(
        "<div class='table-box'>"
        + df_view[existing].head(150).to_html(escape=False, index=False)
        + "</div>",
        unsafe_allow_html=True,
    )

with right:
    st.markdown('<div class="section-title">🔍 Inspector de tranzacții</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="small-muted">Selectează o tranzacție pentru a vedea cum o interpretează modelul AI.</p>',
        unsafe_allow_html=True,
    )

    if total_tx == 0:
        st.warning("Nu există micro-tranzacții după filtrare.")
    else:
        idx = st.slider(
            "Alege indexul tranzacției",
            min_value=0,
            max_value=total_tx - 1,
            value=0,
            key="tx_index_slider",
        )

        tx = df.iloc[idx]

        st.markdown("<div class='inspect-box'>", unsafe_allow_html=True)
        st.markdown(f"**Tranzacția #{idx}**")
        st.write(f"**Amount:** `{tx['Amount']:.2f}`")
        st.write(f"**Scor de risc:** `{tx['risk_score']:.3f}`")
        prediction_label = "⚠️ Probabil fraudă" if tx["fraud_prediction"] == 1 else "✔️ Tranzacție legită"
        st.write(f"**Predicție model:** {prediction_label}")

        if tx["risk_score"] >= 0.85:
            st.markdown("##### 🟥 De ce o considerăm foarte suspectă?")
            st.markdown(
                """
- Valoare mică (micro-tranzacție), un pattern comun în fraudele „invizibile”
- Combinație de features (V1..V28) similară cu tranzacțiile etichetate ca fraudă
- Modelul Random Forest a dat un scor mare pentru această configurație de date
                """
            )
        elif tx["risk_score"] >= 0.5:
            st.markdown("##### 🟧 Activitate potențial suspectă")
            st.markdown(
                """
- Unele caracteristici seamănă cu pattern-uri de fraudă,
  dar nu suficient de puternic pentru a fi 100% fraudă
- Recomandare: verificare manuală de către un analist de risc
                """
            )
        else:
            st.markdown("##### 🟩 Tranzacție în zona „normală”")
            st.markdown(
                """
- Nu prezintă pattern-uri similare cu tranzacțiile frauduloase din setul de antrenare
- Scorul de risc este scăzut, dar sistemul o păstrează în istoric pentru învățare viitoare
                """
            )

        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")

# =========================
#   PIPELINE EXPLANATION
# =========================
st.markdown("### 🔗 Cum funcționează pipeline-ul nostru AI (pe scurt)")

col_a, col_b, col_c, col_d = st.columns(4)

col_a.markdown(
    """
**1. Ingest & Filter**  
• Importăm date reale din `creditcard_mini.csv`  
• Păstrăm doar tranzacțiile cu `Amount ≤ 100`  
• Focus pe fraudele mici, greu de observat manual
"""
)

col_b.markdown(
    """
**2. Preprocesare & Feature Engineering**  
• Eliminăm coloanele care nu ajută modelul (`id`, `Time`, etc.)  
• Scălăm numeric features cu `StandardScaler`  
• Obținem un vector numeric pentru fiecare tranzacție
"""
)

col_c.markdown(
    """
**3. Model Random Forest**  
• Antrenat în `ai_model/train_model.py`  
• Folosește `class_weight='balanced'` pentru a trata dezechilibrul de clase  
• Învață tiparele subtile dintre tranzacții legitime și fraude
"""
)

col_d.markdown(
    """
**4. Scor de risc & Dashboard**  
• Pentru fiecare tranzacție calculăm `fraud_prediction` și `risk_score`  
• Afișăm scorurile, badge-urile de risc și explicații  
• Ușor de integrat într-un sistem de monitorizare real-time
"""
)

st.success("Gus")
