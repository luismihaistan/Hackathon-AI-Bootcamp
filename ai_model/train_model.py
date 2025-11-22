import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import pickle
import os

# Setăm calea corectă pentru salvare
current_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(current_dir, 'model.pkl')

print("⏳ Generare date sintetice (Pattern-uri de fraudă mică)...")

# 1. Generăm date "NORMALE" (Oameni obișnuiți)
# Comportament: Sume variate, ore normale, categorii diverse
n_normal = 2000
normal_data = pd.DataFrame({
    'amount': np.random.uniform(5, 50, n_normal),        # Sume între 5 și 50 RON
    'category_id': np.random.choice([1, 2, 3, 4], n_normal), # Diverse categorii
    'hour': np.random.randint(6, 23, n_normal),          # Ziua
    'frequency': np.random.randint(1, 5, n_normal)       # 1-5 tranzacții pe zi
})

# 2. Generăm date de "FRAUDĂ" (Micro-tranzacții rapide)
# Comportament: Sume foarte mici, repetitive, noaptea, frecvență mare
n_fraud = 100
fraud_data = pd.DataFrame({
    'amount': np.random.uniform(0.5, 2.5, n_fraud),      # Sume f. mici (sub 3 RON)
    'category_id': [1] * n_fraud,                        # Doar o categorie (ex: Gaming)
    'hour': np.random.randint(0, 4, n_fraud),            # Noaptea
    'frequency': np.random.randint(20, 50, n_fraud)      # 20-50 tranzacții "mitraliate"
})

# Unim datele
df = pd.concat([normal_data, fraud_data]).reset_index(drop=True)

# 3. Antrenăm Modelul
print("🧠 Antrenare Isolation Forest...")
# Features folosite: Suma, Categoria, Ora, Frecvența (tranzacții/oră)
model = IsolationForest(contamination=0.05, random_state=42)
model.fit(df[['amount', 'category_id', 'hour', 'frequency']])

# 4. Salvăm modelul
with open(model_path, 'wb') as f:
    pickle.dump(model, f)

print(f"✅ Model salvat cu succes la: {model_path}")