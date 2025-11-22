import pandas as pd
import numpy as np

rng = np.random.default_rng(123)

# Config
n_users = 200
n_merchants = 60
countries = ["RO", "DE", "FR", "US", "GB", "IT", "ES", "NL", "PL", "HU"]
channels = ["web", "mobile", "pos"]

users = [f"U{idx:04d}" for idx in range(n_users)]
merchants = [f"M{idx:03d}" for idx in range(n_merchants)]

user_home_country = rng.choice(countries, size=n_users)
merchant_country = rng.choice(countries, size=n_merchants)

base_datetime = np.datetime64("2025-01-01T00:00")


def simulate_block_normal(n):
    rows = []
    for _ in range(n):
        u_idx = rng.integers(0, n_users)
        m_idx = rng.integers(0, n_merchants)
        user_id = users[u_idx]
        merchant_id = merchants[m_idx]
        # timp în primele 14 zile
        seconds_offset = int(rng.integers(0, 14 * 24 * 3600))
        dt = base_datetime + np.timedelta64(seconds_offset, "s")
        time_numeric = seconds_offset
        # majoritatea micro, unele mai mari
        amount = float(np.clip(rng.gamma(shape=2.0, scale=25.0), 0.5, 500.0))
        # țara: de obicei a userului sau a comerciantului
        if rng.random() < 0.7:
            country = user_home_country[u_idx]
        else:
            country = merchant_country[m_idx]
        channel = str(rng.choice(channels))
        feats = {f"V{i}": float(rng.normal()) for i in range(1, 29)}
        rows.append(
            {
                "transaction_id": None,  # completăm mai târziu
                "user_id": user_id,
                "merchant_id": merchant_id,
                "country": country,
                "channel": channel,
                "datetime": dt.astype("datetime64[s]").astype(str),
                "Time": time_numeric,
                "Amount": round(amount, 2),
                "Class": 0,
                **feats,
            }
        )
    return rows


def simulate_micro_burst_fraud(num_users=90, tx_per_user=6):
    rows = []
    chosen_users = rng.choice(users, size=num_users, replace=False)
    for user_id in chosen_users:
        u_idx = int(user_id[1:])  # din "Uxxxx"
        m_idx = rng.integers(0, n_merchants)
        merchant_id = merchants[m_idx]
        base_sec = int(rng.integers(0, 7 * 24 * 3600))
        for _ in range(tx_per_user):
            offset = int(rng.integers(0, 30 * 60))   # 30 minute
            seconds_offset = base_sec + offset
            dt = base_datetime + np.timedelta64(seconds_offset, "s")
            amount = float(rng.uniform(1.0, 15.0))   # micro-fraud
            country = user_home_country[min(u_idx, n_users - 1)]
            channel = str(rng.choice(["web", "mobile"]))
            feats = {f"V{i}": float(rng.normal(loc=2.0, scale=1.2)) for i in range(1, 29)}
            rows.append(
                {
                    "transaction_id": None,
                    "user_id": user_id,
                    "merchant_id": merchant_id,
                    "country": country,
                    "channel": channel,
                    "datetime": dt.astype("datetime64[s]").astype(str),
                    "Time": seconds_offset,
                    "Amount": round(amount, 2),
                    "Class": 1,
                    **feats,
                }
            )
    return rows


def simulate_geo_hop_fraud(num_users=50, tx_per_user=5):
    rows = []
    chosen_users = rng.choice(users, size=num_users, replace=False)
    for user_id in chosen_users:
        base_sec = int(rng.integers(7 * 24 * 3600, 14 * 24 * 3600))
        hop_countries = rng.choice(countries, size=4, replace=False)
        for j in range(tx_per_user):
            seconds_offset = base_sec + int(rng.integers(0, 3 * 3600))  # în 3 ore
            dt = base_datetime + np.timedelta64(seconds_offset, "s")
            amount = float(np.clip(rng.gamma(shape=1.5, scale=40.0), 5.0, 300.0))
            country = str(hop_countries[j % len(hop_countries)])
            m_idx = rng.integers(0, n_merchants)
            merchant_id = merchants[m_idx]
            channel = str(rng.choice(channels))
            feats = {f"V{i}": float(rng.normal(loc=1.5, scale=1.0)) for i in range(1, 29)}
            rows.append(
                {
                    "transaction_id": None,
                    "user_id": user_id,
                    "merchant_id": merchant_id,
                    "country": country,
                    "channel": channel,
                    "datetime": dt.astype("datetime64[s]").astype(str),
                    "Time": seconds_offset,
                    "Amount": round(amount, 2),
                    "Class": 1,
                    **feats,
                }
            )
    return rows


def main():
    normal_rows = simulate_block_normal(3200)
    burst_rows = simulate_micro_burst_fraud(num_users=90, tx_per_user=6)
    geo_rows = simulate_geo_hop_fraud(num_users=50, tx_per_user=5)

    all_rows = normal_rows + burst_rows + geo_rows
    df = pd.DataFrame(all_rows)

    df = df.sample(frac=1.0, random_state=123).reset_index(drop=True)
    df["transaction_id"] = [f"T{idx:06d}" for idx in range(len(df))]

    print("Total tranzacții:", len(df))
    print("Rată fraude:", df["Class"].mean() * 100, "%")

    df.to_csv("transactions_micro_extended.csv", index=False)


if __name__ == "__main__":
    main()
