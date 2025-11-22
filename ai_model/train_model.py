import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import os

# --- CONFIGURARE CĂI (PATHS) ---
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)

DATA_PATH = os.path.join(root_dir, "data", "transactions_micro_extended.csv")
MODELS_DIR = os.path.join(root_dir, "models")

# 1. ÎNCĂRCARE DATE DIN CSV
def incarca_date():
    print(f"📂 Căutăm fișierul la: {DATA_PATH}")
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"❌ Nu am găsit fișierul la: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    # Facem coloana target consistentă
    if "is_fraud" not in df.columns:
        if "Class" in df.columns:
            df.rename(columns={"Class": "is_fraud"}, inplace=True)
        else:
            raise ValueError("❌ CSV-ul nu are 'is_fraud' sau 'Class'.")

    LIMITA_MICRO = 100.0
    print(f"📊 Total tranzacții inițiale: {len(df)}")
    print(f"💰 Suma maximă: {df['Amount'].max()}")

    df = df[df["Amount"] <= LIMITA_MICRO].copy()

    print(f"✂️ Filtru micro-tranzacții (<= {LIMITA_MICRO})")
    print(f"✅ Tranzacții rămase: {len(df)}")
    print(f"Fraude în datele filtrate: {df['is_fraud'].mean()*100:.2f}%")
    return df

# 2. PREPROCESARE
def preprocesare_date(df):
    # Eliminăm coloane non-numerice sau de identificare
    cols_to_drop = [
        "transaction_id",
        "user_id",
        "merchant_id",
        "country",
        "channel",
        "datetime",
        "Time",          # dacă nu vrei Time în features
        "Unnamed: 0",
    ]
    X = df.drop(["is_fraud"] + cols_to_drop, axis=1, errors="ignore")
    y = df["is_fraud"]

    print(f"Features folosite pentru antrenare: {list(X.columns)}")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Ne asigurăm că scaler-ul știe numele coloanelor
    if not hasattr(scaler, "feature_names_in_"):
        scaler.feature_names_in_ = np.array(X.columns)

    return X_scaled, y, scaler

# 3. ANTRENARE MODEL
def antreneaza_model():
    print("--- Începere Proces Antrenare (Date Sintetice Extinse) ---")

    try:
        df = incarca_date()
    except Exception as e:
        print(e)
        return

    X, y, scaler = preprocesare_date(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print("🚀 Inițializare Random Forest...")
    model = RandomForestClassifier(
        n_estimators=120,
        max_depth=12,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
    )

    print("⏳ Se antrenează modelul...")
    model.fit(X_train, y_train)

    print("\n--- Rezultate Evaluare (Test) ---")
    y_pred = model.predict(X_test)
    print("Matrice de confuzie:")
    print(confusion_matrix(y_test, y_pred))
    print("\nRaport detaliat:")
    print(classification_report(y_test, y_pred))

    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    model_save_path = os.path.join(MODELS_DIR, "fraud_model.pkl")
    scaler_save_path = os.path.join(MODELS_DIR, "scaler.pkl")

    joblib.dump(model, model_save_path)
    joblib.dump(scaler, scaler_save_path)
    print(f"\n✅ Model salvat în: {model_save_path}")
    print(f"✅ Scaler salvat în: {scaler_save_path}")

if __name__ == "__main__":
    antreneaza_model()
