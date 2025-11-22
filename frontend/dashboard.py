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

# =========================
#   LOAD MODEL + SCALER
# =========================
@st.cache_resource
def load_model_and_scaler():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


# =========================
#   ENRICH + PREDICT
# =========================
def enrich_and_predict(df: pd.DataFrame, model, scaler):
    df = df.copy()

    # --- target column ---
    if "is_fraud" not in df.columns:
        if "Class" in df.columns:
            df = df.rename(columns={"Class": "is_fraud"})
        else:
            # dacă nu avem label, punem 0 (necunoscut / non-fraud)
            df["is_fraud"] = 0

    # --- datetime parsing ---
    if "datetime" in df.columns:
        df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce")
        df["hour"] = df["datetime"].dt.hour.fillna(0).astype(int)
        df["date"] = df["datetime"].dt.date
    else:
        df["datetime"] = pd.NaT
        df["hour"] = 0
        df["date"] = None

    # --- micro-filter ---
    if "Amount" not in df.columns:
        raise ValueError("CSV-ul trebuie să conțină o coloană 'Amount'.")

    df = df[df["Amount"] <= 100.0].copy()

    if len(df) == 0:
        raise ValueError("După filtrul Amount <= 100 nu a mai rămas nicio tranzacție.")

    # --- agregări contextuale ---
    if "transaction_id" not in df.columns:
        df["transaction_id"] = [f"T{idx:06d}" for idx in range(len(df))]

    if "user_id" in df.columns:
        df["user_tx_total"] = df.groupby("user_id")["transaction_id"].transform("count")
        df["user_countries_count"] = df.groupby("user_id")["country"].transform("nunique") if "country" in df.columns else 1
    else:
        df["user_tx_total"] = 1
        df["user_countries_count"] = 1

    if "merchant_id" in df.columns:
        df["merchant_tx_total"] = df.groupby("merchant_id")["transaction_id"].transform("count")
        # merchant fraud rate poate fi calculat doar dacă avem label (is_fraud)
        df["merchant_fraud_rate"] = df.groupby("merchant_id")["is_fraud"].transform("mean")
    else:
        df["merchant_tx_total"] = 1
        df["merchant_fraud_rate"] = 0.0

    if "user_id" in df.columns and "merchant_id" in df.columns:
        df["user_merchant_tx_total"] = df.groupby(["user_id", "merchant_id"])["transaction_id"].transform("count")
    else:
        df["user_merchant_tx_total"] = 1

    # --- features pentru model: EXACT ca la antrenare ---
    feature_cols = getattr(scaler, "feature_names_in_", None)
    if feature_cols is None:
        raise ValueError(
            "Scaler-ul nu are feature_names_in_. Asigură-te că l-ai salvat după fit pe un DataFrame cu nume de coloane."
        )

    missing_features = [col for col in feature_cols if col not in df.columns]
    if missing_features:
        raise ValueError(
            f"CSV-ul încărcat NU are toate coloanele necesare pentru model.\n"
            f"Lipsesc coloanele: {missing_features}"
        )

    X = df[feature_cols]

    # --- scaling + predicții ---
    scaled = scaler.transform(X)
    df["fraud_prediction"] = model.predict(scaled)
    df["risk_score"] = model.predict_proba(scaled)[:, 1]

    return df


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
.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
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
.table-box {
    background: #020617;
    border-radius: 18px;
    border: 1px solid rgba(148, 163, 184, 0.35);
    padding: 16px 16px 4px 16px;
}
.section-title {
    font-weight: 700;
    font-size: 20px;
    color: #e5e7eb;
}
.inspect-box {
    background: #020617;
    border-radius: 18px;
    border: 1px solid rgba(148, 163, 184, 0.35);
    padding: 18px;
    color: #e5e7eb;
}
.small-muted {
    font-size: 12px;
    color: #9ca3af;
}
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
    '<p class="subtext">Încarcă propriul tău fișier CSV cu tranzacții, iar modelul antrenat analizează fiecare micro-tranzacție '
    'în context: utilizator, comerciant, țară, oră și istoric.</p>',
    unsafe_allow_html=True,
)

st.write("")
tag_cols = st.columns(4)
tag_cols[0].markdown("✅ Model Random Forest antrenat")
tag_cols[1].markdown("📂 CSV upload direct din browser")
tag_cols[2].markdown("🌍 Context user / merchant / country")
tag_cols[3].markdown("👀 Explicații pentru tranzacțiile suspecte")
st.markdown("---")

# =========================
#   LOAD MODEL
# =========================
try:
    model, scaler = load_model_and_scaler()
except Exception as e:
    st.error("Nu am reușit să încarc modelul sau scaler-ul. Verifică fișierele din folderul 'models/'.")
    st.exception(e)
    st.stop()

# =========================
#   FILE UPLOADER
# =========================
uploaded_file = st.file_uploader(
    "Încarcă un fișier CSV cu tranzacții (trebuie să aibă cel puțin coloanele folosite la antrenare: Amount, V1..V28, etc.)",
    type=["csv"],
)

if uploaded_file is None:
    st.info(
        "👆 Încarcă un fișier CSV ca să vezi analiza. "
        "Ideal, folosește același format ca setul de antrenare: "
        "`transaction_id, user_id, merchant_id, country, channel, datetime, Amount, Class/is_fraud, V1..V28`."
    )
    st.stop()

try:
    raw_df = pd.read_csv(uploaded_file)
except Exception as e:
    st.error("Nu am putut citi fișierul CSV. Verifică dacă este un CSV valid.")
    st.exception(e)
    st.stop()

# =========================
#   PREDICTIONS + ENRICH
# =========================
try:
    df = enrich_and_predict(raw_df, model, scaler)
except Exception as e:
    st.error("A apărut o eroare la procesarea datelor și calculul scorurilor.")
    st.exception(e)
    st.stop()

# =========================
#        METRICS
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
    <div class="metric-sub">tranzacții cu Amount ≤ 100</div>
</div>
""",
    unsafe_allow_html=True,
)
m2.markdown(
    f"""
<div class="metric-card">
    <div class="metric-label">Rată tranzacții suspecte (model)</div>
    <div class="metric-value">{fraud_rate*100:.2f}%</div>
    <div class="metric-sub">procent din micro-tranzacțiile încărcate</div>
</div>
""",
    unsafe_allow_html=True,
)
m3.markdown(
    f"""
<div class="metric-card">
    <div class="metric-label">Valoare medie micro-tranzacții</div>
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
#   TABLE + INSPECTOR
# =========================
left, right = st.columns([2.2, 1])

with left:
    st.markdown('<div class="section-title">📊 Micro-tranzacții și scorul de risc</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="small-muted">Tabelul de mai jos este calculat din fișierul tău CSV încărcat în aplicație.</p>',
        unsafe_allow_html=True,
    )

    df_view = df.copy()
    df_view["Risk Level"] = df_view["risk_score"].apply(risk_badge)
    df_view["Fraud (model)"] = df_view["fraud_prediction"].map({0: "Legit", 1: "⚠ Fraud-like"})

    show_cols = [
        "transaction_id",
        "user_id",
        "merchant_id",
        "country",
        "Amount",
        "risk_score",
        "Risk Level",
        "Fraud (model)",
    ]
    existing = [c for c in show_cols if c in df_view.columns]

    st.markdown(
        "<div class='table-box'>"
        + df_view[existing].head(200).to_html(escape=False, index=False)
        + "</div>",
        unsafe_allow_html=True,
    )

with right:
    st.markdown('<div class="section-title">🔍 Inspector detaliat de tranzacții</div>', unsafe_allow_html=True)
    st.markdown(
        '<p class="small-muted">Alege o tranzacție din setul încărcat pentru a vedea contextul complet.</p>',
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
        st.markdown(f"**Tranzacția #{idx} · ID: `{tx['transaction_id']}`**")
        if "user_id" in tx:
            st.write(f"**User:** `{tx['user_id']}`")
        if "merchant_id" in tx:
            st.write(f"**Merchant:** `{tx['merchant_id']}`")
        if "country" in tx:
            st.write(f"**Țară:** `{tx['country']}`")
        if "channel" in tx:
            st.write(f"**Channel:** `{tx['channel']}`")
        st.write(f"**Data/Ora:** `{tx['datetime']}`")
        st.write(f"**Amount:** `{tx['Amount']:.2f}`")
        st.write(f"**Scor de risc (model):** `{tx['risk_score']:.3f}`")
        prediction_label = "⚠️ Probabil fraudă" if tx["fraud_prediction"] == 1 else "✔️ Tranzacție legită"
        st.write(f"**Predicție model:** {prediction_label}")

        st.write("---")
        st.markdown("#### 📌 Context agregat")
        st.write(f"- Total tranzacții ale user-ului: `{tx.get('user_tx_total', 'n/a')}`")
        st.write(f"- Tranzacții user → acest merchant: `{tx.get('user_merchant_tx_total', 'n/a')}`")
        st.write(f"- Total tranzacții ale merchant-ului: `{tx.get('merchant_tx_total', 'n/a')}`")
        if "merchant_fraud_rate" in tx and not pd.isna(tx["merchant_fraud_rate"]):
            st.write(f"- Rata de fraudă la acest merchant (în datele încărcate): `{tx['merchant_fraud_rate']*100:.1f}%`")
        st.write(f"- Număr de țări diferite folosite de user: `{tx.get('user_countries_count', 'n/a')}`")
        st.write(f"- Ora tranzacției: `{int(tx['hour'])}:00`")

        st.write("---")
        st.markdown("#### 🧠 De ce poate fi considerată suspectă?")

        reasons = []
        if tx.get("user_merchant_tx_total", 0) >= 5 and tx["Amount"] < 20:
            reasons.append(
                f"- User-ul are `{tx['user_merchant_tx_total']}` micro-tranzacții către **același merchant**."
            )
        if tx.get("user_countries_count", 0) >= 3:
            reasons.append(
                f"- User-ul a tranzacționat din `{tx['user_countries_count']}` țări diferite (posibil geo-hopping)."
            )
        if int(tx["hour"]) < 5 or int(tx["hour"]) >= 23:
            reasons.append("- Tranzacție efectuată la o oră atipică (noaptea / foarte târziu).")
        if "merchant_fraud_rate" in tx and not pd.isna(tx["merchant_fraud_rate"]) and tx["merchant_fraud_rate"] >= 0.3:
            reasons.append(
                f"- Merchant cu istoric de fraudă ridicat în datele tale: `{tx['merchant_fraud_rate']*100:.1f}%`."
            )
        if tx["fraud_prediction"] == 1 and not reasons:
            reasons.append(
                "- Pattern numeric (V1..V28 + Amount) foarte similar cu tranzacțiile frauduloase din setul de antrenare."
            )
        if not reasons:
            reasons.append("- Nu există semnale foarte puternice; modelul consideră tranzacția relativ normală.")

        for r in reasons:
            st.write(r)

        st.write("---")
        st.markdown("#### 📜 Istoric user ↔ merchant (primele 20)")

        if "user_id" in df.columns and "merchant_id" in df.columns:
            mask = (df["user_id"] == tx["user_id"]) & (df["merchant_id"] == tx["merchant_id"])
            history = df.loc[mask, ["datetime", "Amount", "risk_score", "fraud_prediction"]].sort_values("datetime").head(20)
            history = history.rename(
                columns={
                    "datetime": "Datetime",
                    "Amount": "Amount",
                    "risk_score": "RiskScore",
                    "fraud_prediction": "Fraud(0/1)",
                }
            )
            st.dataframe(history, use_container_width=True)
        else:
            st.write("Nu pot genera istoricul user ↔ merchant: lipsesc coloanele 'user_id' sau 'merchant_id' în CSV.")

        st.markdown("</div>", unsafe_allow_html=True)

st.markdown("---")
st.success("Dashboard-ul a analizat cu succes fișierul CSV încărcat. 🚀")
