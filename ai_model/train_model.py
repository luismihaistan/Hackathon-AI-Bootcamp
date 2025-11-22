import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
import joblib
import os

# 1. GENERARE DATE SINTETICE (Simulare Micro-tranzacții)
# În producție, vei înlocui această funcție cu: df = pd.read_csv('tranzactii.csv')
def genereaza_dataset_fictiv(n_rows=10000):
    np.random.seed(42)
    
    # Generăm ID-uri și sume mici (micro-tranzacții între 0.50 RON și 50 RON)
    data = {
        'user_id': np.random.randint(1, 1000, n_rows),
        'amount': np.round(np.random.uniform(0.5, 50.0, n_rows), 2),
        'hour': np.random.randint(0, 24, n_rows),
        'merchant_risk_score': np.random.uniform(0, 1, n_rows), # Scorul de risc al comerciantului
        'is_international': np.random.choice([0, 1], n_rows, p=[0.9, 0.1]), # 1 = Da, 0 = Nu
        'transaction_velocity': np.random.randint(0, 20, n_rows) # Câte tranzacții a făcut userul în ultima oră
    }
    
    df = pd.DataFrame(data)
    
    # Creăm eticheta 'is_fraud' (Target) bazată pe reguli logice pentru a avea un model antrenabil
    # Exemplu: Micro-fraudele apar des noaptea, internațional, cu frecvență mare (velocity)
    def simulate_fraud(row):
        risk = 0
        if row['transaction_velocity'] > 10: risk += 0.4
        if row['merchant_risk_score'] > 0.8: risk += 0.3
        if row['is_international'] == 1: risk += 0.2
        if row['hour'] < 4: risk += 0.2
        
        # Introducem elementul aleatoriu (nu toate tranzacțiile riscante sunt fraude)
        return 1 if (risk + np.random.uniform(0, 0.2)) > 0.8 else 0

    df['is_fraud'] = df.apply(simulate_fraud, axis=1)
    
    print(f"Dataset generat: {n_rows} tranzacții.")
    print(f"Procentaj fraude: {df['is_fraud'].mean()*100:.2f}%")
    return df

# 2. PREPROCESARE
def preprocesare_date(df):
    # Separăm caracteristicile (features) de țintă (target)
    X = df.drop(['is_fraud', 'user_id'], axis=1) # Scoatem user_id, nu e relevant direct pentru model
    y = df['is_fraud']
    
    # Scalarea datelor (importantă pentru anumite modele, bună practică generală)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler

# 3. ANTRENARE MODEL
def antreneaza_model():
    print("--- Începere Proces Antrenare ---")
    
    # A. Încărcare date
    df = genereaza_dataset_fictiv()
    
    # B. Preprocesare
    X, y, scaler = preprocesare_date(df)
    
    # C. Împărțire Train / Test (80% antrenare, 20% testare)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # D. Inițializare Model
    # Folosim class_weight='balanced' pentru a pune mai mult accent pe fraude (care sunt puține)
    model = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        random_state=42, 
        class_weight='balanced',
        n_jobs=-1
    )
    
    # E. Fitting (Antrenarea propriu-zisă)
    print("Se antrenează modelul (Random Forest)...")
    model.fit(X_train, y_train)
    
    # F. Evaluare
    print("\n--- Rezultate Evaluare (Pe datele de test) ---")
    y_pred = model.predict(X_test)
    
    print(confusion_matrix(y_test, y_pred))
    print("\nRaport Detaliat:")
    print(classification_report(y_test, y_pred))
    
    # G. Salvare Model și Scaler
    if not os.path.exists('models'):
        os.makedirs('models')
        
    joblib.dump(model, 'models/fraud_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    print("\nSucces! Modelul a fost salvat în folderul 'models/'.")

if __name__ == "__main__":
    antreneaza_model()