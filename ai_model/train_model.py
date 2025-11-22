import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
from sklearn.preprocessing import StandardScaler
import joblib
import os
import time

# --- 1. GENERARE DATE (Motorul de simulare) ---
def genereaza_dataset_fictiv(n_rows=100000):
    np.random.seed(None) # Random diferit la fiecare apel
    
    data = {
        'amount': np.round(np.random.uniform(0.5, 50.0, n_rows), 2),
        'hour': np.random.randint(0, 24, n_rows),
        'merchant_risk_score': np.random.uniform(0, 1, n_rows),
        'is_international': np.random.choice([0, 1], n_rows, p=[0.9, 0.1]),
        'transaction_velocity': np.random.randint(0, 20, n_rows)
    }
    
    df = pd.DataFrame(data)
    
    # Logică fraudă
    def simulate_fraud(row):
        risk = 0
        if row['transaction_velocity'] > 12: risk += 0.45
        if row['merchant_risk_score'] > 0.75: risk += 0.25
        if row['is_international'] == 1: risk += 0.15
        if row['hour'] < 5: risk += 0.20
        return 1 if (risk + np.random.uniform(0, 0.15)) > 0.85 else 0

    df['is_fraud'] = df.apply(simulate_fraud, axis=1)
    return df

# --- 2. PREPROCESARE ---
def preprocesare_date(df):
    X = df.drop(['is_fraud'], axis=1)
    y = df['is_fraud']
    feature_names = X.columns.tolist()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    return X_scaled, y, scaler, feature_names

# --- 3. RAPORT FINAL (Audit) ---
def afiseaza_raport_final(model, X_test, y_test, scaler, feature_names):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    print("\n" + "="*50)
    print("      AUDIT FINAL (DUPĂ TOATE GENERAȚIILE)")
    print("="*50)

    # A. Feature Importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    print("\n📊 CE A ÎNVĂȚAT MODELUL (Factori de Risc):")
    print("-" * 45)
    for f in range(len(feature_names)):
        feat_name = feature_names[indices[f]]
        score = importances[indices[f]] * 100
        print(f"{feat_name:<25} | {score:.2f}%")

    # B. Matrice Confuzie
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print("\n🔍 REZULTATE OPERAȚIONALE:")
    print(f"   ✅ Corecte (Legitime): {tn}")
    print(f"   🛡️ Corecte (Fraude prinse): {tp}")
    print(f"   ❌ Erori (Fraude scăpate): {fn}")
    print(f"   ⚠️ Erori (Alarme false): {fp}")

    # C. Salvare
    if not os.path.exists('models'): os.makedirs('models')
    joblib.dump(model, 'models/fraud_model.pkl')
    joblib.dump(scaler, 'models/scaler.pkl')
    print("\n💾 Modelul antrenat a fost salvat în 'models/'.")

# --- 4. PROCESUL EVOLUTIV (MAIN) ---
def antrenare_evolutiva(nr_generatii=10, date_per_generatie=100000):
    print(f"🚀 Start Antrenare Evolutivă: {nr_generatii} Generații planificate.")
    
    dataset_total = pd.DataFrame()
    model = None
    scaler = None
    
    # Variabile pentru a stoca istoricul (ca să vedem progresul)
    istoric_acuratete = []
    
    for i in range(nr_generatii):
        gen_start = time.time()
        print(f"\n--- GENERAȚIA {i+1}/{nr_generatii} ---")
        
        # 1. Vin date noi (Simulare trafic nou)
        print(f"   📥 Colectare {date_per_generatie} tranzacții noi...")
        date_noi = genereaza_dataset_fictiv(date_per_generatie)
        
        # 2. Se adaugă la memoria totală ("Experience Replay")
        dataset_total = pd.concat([dataset_total, date_noi])
        total_acumulat = len(dataset_total)
        fraude_total = dataset_total['is_fraud'].sum()
        
        print(f"   💾 Baza de cunoștințe: {total_acumulat} tranzacții (din care {fraude_total} fraude).")
        
        # 3. Preprocesare pe TOT setul de date
        X, y, scaler, feature_names = preprocesare_date(dataset_total)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
        
        # 4. Antrenare (Re-learning)
        # Creștem ușor numărul de estimatori pe măsură ce avem mai multe date
        n_estimatori = 100 + (i * 20) 
        
        model = RandomForestClassifier(
            n_estimators=n_estimatori,
            max_depth=12,
            class_weight='balanced',
            n_jobs=-1,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # 5. Evaluare Rapidă a Generației
        acc = model.score(X_test, y_test)
        istoric_acuratete.append(acc)
        timp_gen = time.time() - gen_start
        print(f"   ⏱️  Antrenat în {timp_gen:.2f}s. Acuratețe test: {acc*100:.2f}%")

    # După ce s-au terminat toate generațiile, facem auditul final
    afiseaza_raport_final(model, X_test, y_test, scaler, feature_names)
    
    print("\n📈 Evoluția Acurateței pe parcursul generațiilor:", 
          [f"{x*100:.1f}%" for x in istoric_acuratete])

if _name_ == "_main_":
    antrenare_evolutiva()