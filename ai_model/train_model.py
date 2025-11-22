import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import joblib
import os

# --- CONFIGURARE CĂI (PATHS) ---
# Aflăm unde se află acest script (în folderul ai_model)
current_dir = os.path.dirname(os.path.abspath(__file__))
# Mergem un nivel mai sus (în root)
root_dir = os.path.dirname(current_dir)
# Construim calea către CSV și către folderul de modele
DATA_PATH = os.path.join(root_dir, 'data', 'creditcard_mini.csv')
MODELS_DIR = os.path.join(root_dir, 'models')

# 1. ÎNCĂRCARE DATE DIN CSV

def incarca_date():
    print(f"📂 Căutăm fișierul la: {DATA_PATH}")
    
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"❌ Eroare: Nu am găsit fișierul 'creditcard_mini.csv' la calea specificată.")
    
    df = pd.read_csv(DATA_PATH)
    
    # --- FILTRU PENTRU MICRO-TRANZACȚII ---
    # Definim limita (ex: 50.00 unități monetare)
    LIMITA_MICRO = 100.0
    
    print(f"📊 Total tranzacții inițiale: {len(df)}")
    print(f"💰 Suma maximă existentă: {df['Amount'].max()}")
    
    # Păstrăm doar ce e sub limită
    df = df[df['Amount'] <= LIMITA_MICRO]
    
    print(f"✂️  Aplicăm filtru micro-tranzacții (<= {LIMITA_MICRO})")
    print(f"✅ Tranzacții rămase: {len(df)}")
    
    # Verificăm coloana target
    if 'is_fraud' not in df.columns:
        if 'Class' in df.columns:
            df.rename(columns={'Class': 'is_fraud'}, inplace=True)
        else:
            raise ValueError("❌ CSV-ul nu conține o coloană 'is_fraud' sau 'Class'.")
            
    print(f"Procentaj fraude în datele filtrate: {df['is_fraud'].mean()*100:.2f}%")
    return df

# 2. PREPROCESARE
def preprocesare_date(df):
    # Eliminăm coloane care nu ajută modelul (ID-uri, Nume, etc.)
    # errors='ignore' înseamnă că nu dă eroare dacă coloana nu există
    cols_to_drop = ['user_id', 'id', 'Time', 'Unnamed: 0']
    X = df.drop(['is_fraud'] + cols_to_drop, axis=1, errors='ignore')
    
    y = df['is_fraud']
    
    print(f"Features folosite pentru antrenare: {list(X.columns)}")
    
    # Scalarea datelor
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Returnăm scaler-ul ca să îl putem salva
    return X_scaled, y, scaler

# 3. ANTRENARE MODEL
def antreneaza_model():
    print("--- Începere Proces Antrenare (Date Reale) ---")
    
    # A. Încărcare date
    try:
        df = incarca_date()
    except Exception as e:
        print(e)
        return # Opresc execuția dacă nu pot încărca datele
    
    # B. Preprocesare
    X, y, scaler = preprocesare_date(df)
    
    # C. Împărțire Train / Test
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # D. Inițializare Model
    print("🚀 Inițializare Random Forest...")
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        random_state=42, 
        class_weight='balanced', # Important pentru date dezechilibrate (frauda e rară)
        n_jobs=-1
    )
    
    # E. Fitting
    print("⏳ Se antrenează modelul...")
    model.fit(X_train, y_train)
    
    # F. Evaluare
    print("\n--- Rezultate Evaluare (Pe datele de test) ---")
    y_pred = model.predict(X_test)
    
    print("Matrice de confuzie:")
    print(confusion_matrix(y_test, y_pred))
    print("\nRaport Detaliat:")
    print(classification_report(y_test, y_pred))
    
    # G. Salvare Model și Scaler
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        
    model_save_path = os.path.join(MODELS_DIR, 'fraud_model.pkl')
    scaler_save_path = os.path.join(MODELS_DIR, 'scaler.pkl')

    joblib.dump(model, model_save_path)
    joblib.dump(scaler, scaler_save_path)
    print(f"\n✅ Succes! Modelul a fost salvat în: {MODELS_DIR}")

if __name__ == "__main__":
    antreneaza_model()