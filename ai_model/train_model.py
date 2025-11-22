import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import StandardScaler
import joblib
import os

# --- 1. GENERARE DATE SINTETICE (Aceeași ca înainte) ---
def genereaza_dataset_fictiv(n_rows=20000):
    print(f"⏳ Generare dataset cu {n_rows} de tranzacții...")
    np.random.seed(42)
    
    data = {
        'amount': np.round(np.random.uniform(0.5, 50.0, n_rows), 2),
        'hour': np.random.randint(0, 24, n_rows),
        'merchant_risk_score': np.random.uniform(0, 1, n_rows),
        'is_international': np.random.choice([0, 1], n_rows, p=[0.9, 0.1]),
        'transaction_velocity': np.random.randint(0, 20, n_rows)
    }
    
    df = pd.DataFrame(data)
    
    # Logică de fraudă (puțin mai complexă)
    def simulate_fraud(row):
        risk = 0
        # Regulile 'secrete' pe care AI-ul trebuie să le ghicească
        if row['transaction_velocity'] > 12: risk += 0.45
        if row['merchant_risk_score'] > 0.75: risk += 0.25
        if row['is_international'] == 1: risk += 0.15
        if row['hour'] < 5: risk += 0.20
        
        return 1 if (risk + np.random.uniform(0, 0.15)) > 0.85 else 0

    df['is_fraud'] = df.apply(simulate_fraud, axis=1)
    print(f"✅ Dataset gata. Rată fraudă: {df['is_fraud'].mean()*100:.2f}%")
    return df

# --- 2. PREPROCESARE ---
def preprocesare_date(df):
    X = df.drop(['is_fraud'], axis=1)
    y = df['is_fraud']
    
    # Păstrăm numele coloanelor pentru raportul final
    feature_names = X.columns.tolist()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, feature_names

# --- 3. ANTRENARE ȘI RAPORTARE DETALIATĂ ---
def antreneaza_model():
    # A. Pregătire
    df = genereaza_dataset_fictiv()
    X, y, scaler, feature_names = preprocesare_date(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # B. Model
    print("\n🧠 Antrenare model Random Forest (optimizat pentru fraude)...")
    model = RandomForestClassifier(
        n_estimators=200, 
        max_depth=12, 
        random_state=42, 
        class_weight='balanced',
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # C. Predicții
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] # Probabilitatea de a fi fraudă (0.0 la 1.0)
    
    print("\n" + "="*40)
    print("      RAPORT DETALIAT DE PERFORMANȚĂ")
    print("="*40)

    # --- ANALIZA 1: Importanța Caracteristicilor ---
    # Ce contează cel mai mult pentru model?
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    print("\n📊 CEI MAI IMPORTANȚI FACTORI DE RISC:")
    print("-" * 40)
    print(f"{'Factor (Feature)':<25} | {'Importanță (%)':<15}")
    print("-" * 40)
    for f in range(X.shape[1]):
        feat_name = feature_names[indices[f]]
        score = importances[indices[f]] * 100
        print(f"{feat_name:<25} | {score:.2f}%")
    print("-" * 40)
    print("Interpretabilitate: Factorul de sus este cel la care AI-ul se uită primul.")

    # --- ANALIZA 2: Performanța Statistică ---
    print("\n📈 METRICI CHEIE:")
    roc_score = roc_auc_score(y_test, y_prob)
    print(f"Scor ROC-AUC: {roc_score:.4f} (1.0 = Perfect, 0.5 = Ghicit)")
    if roc_score > 0.9: print("   -> Calificativ: EXCELENT")
    elif roc_score > 0.8: print("   -> Calificativ: BUN")
    else: print("   -> Calificativ: NECESITĂ ÎMBUNĂTĂȚIRI")

    # --- ANALIZA 3: Confuzie explicată ---
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print("\n🔍 REALITATEA DIN TEREN (Matricea de Confuzie):")
    print(f"✅ Tranzacții OK permise:     {tn} (Clienți fericiți)")
    print(f"❌ Alarme False (Blocaje):    {fp} (Clienți deranjați)")
    print(f"⚠️ Fraude SCĂPATE:            {fn} (Pierdere bani)")
    print(f"🛡️ Fraude OPRITE:             {tp} (Bani salvați)")

    # --- ANALIZA 4: Exemple Concrete ---
    print("\n💡 EXEMPLE DE TRANZACȚII ANALIZATE ACUM:")
    # Luăm 5 exemple din test set
    test_indices = np.random.choice(len(X_test), 5, replace=False)
    print(f"{'Viteză':<10} {'ScorRisk':<10} {'Oră':<5} {'Intl?':<5} | {'REAL':<5} -> {'PREZIS (Prob %)':<15}")
    
    X_test_original = scaler.inverse_transform(X_test) # Revenim la valorile normale pentru afișare
    
    for i in test_indices:
        row = X_test_original[i]
        real_val = "FRAUDĂ" if y_test.iloc[i] == 1 else "OK"
        pred_val = "FRAUDĂ" if y_pred[i] == 1 else "OK"
        prob_fraud = y_prob[i] * 100
        
        # Extragem valorile pentru afișare
        vel = int(row[4]) # velocity e pe index 4 in X_test
        risk = f"{row[2]:.2f}" # merchant risk
        hour = int(row[1])
        intl = "DA" if row[3] > 0.5 else "NU"
        
        print(f"{vel:<10} {risk:<10} {hour:<5} {intl:<5} | {real_val:<5} -> {pred_val} ({prob_fraud:.1f}%)")

    # Salvare
    if not os.path.exists('models'): os.makedirs('models')
    joblib.dump(model, 'models/fraud_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    print("\n💾 Model salvat.")

if __name__ == "__main__":
    antreneaza_model()