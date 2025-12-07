#%% Libraries
import requests
import pandas as pd
import numpy as np
import os
import time
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler

#%% Configuration
csv_path = r"C:\path\werte.csv"  # Historical NO2 data
model_path = "omd_no2_weight.npy"  # Save OMD weight
pred_csv_path = "online_no2_predictions.csv"  # Predictions CSV
eta = 0.1  # Learning rate for OMD
api_token = "YOUR_TOKEN_HERE"  # WAQI API token
url = f"https://api.waqi.info/feed/here/?token=[your Token]"  # API endpoint

#%% Function to fetch current NO2 from API
def fetch_no2():
    try:
        resp = requests.get(url)
        resp.raise_for_status()
        data = resp.json()
        if data["status"] != "ok":
            print("API error:", data["data"])
            return None
        iaqi = data["data"].get("iaqi", {})
        no2 = iaqi.get("no2", {}).get("v", None)
        timestamp = data["data"]["time"]["s"]
        return {"timestamp": timestamp, "no2": no2}
    except Exception as e:
        print("Fetch error:", e)
        return None

#%% OMD helper function
def omd_update(x, y_t, eta):
    """Entropic OMD update for a single weight."""
    g = 2 * (x[0] - y_t)
    x = x * np.exp(-eta * g)
    x = np.clip(x, 0, 1)
    return x

#%% Load and preprocess historical data (for initial guess)
df_hist = pd.read_csv(csv_path, encoding='latin1', header=0)

# Ensure NO2 column exists
if 'No2' in df_hist.columns:
    df_hist.rename(columns={'No2': 'no2'}, inplace=True)
elif 'no2' not in df_hist.columns:
    raise ValueError("CSV has no 'No2' or 'no2' column")

# Parse datetime
df_hist['Datum_Zeit'] = pd.to_datetime(df_hist['Datum_Zeit'], errors='coerce', dayfirst=True)
df_hist = df_hist.dropna(subset=['Datum_Zeit', 'no2'])
df_hist['no2'] = pd.to_numeric(df_hist['no2'], errors='coerce')
df_hist = df_hist.dropna(subset=['no2'])
df_hist.set_index('Datum_Zeit', inplace=True)
df_hist.sort_index(inplace=True)

# Normalize NO2 for OMD
scaler = MinMaxScaler()
y_norm = scaler.fit_transform(df_hist['no2'].values.reshape(-1,1)).flatten()

#%% Initialize OMD weight
if os.path.exists(model_path):
    x = np.load(model_path)
    print("Loaded previous OMD weight:", x)
else:
    # Use mean of first 10 historical measurements as initial guess
    x = np.array([y_norm[:10].mean()])
    print("Initialized new OMD weight from historical data:", x)

#%% Online loop: fetch new data, predict, update OMD
print("Starting real-time online updates. Press Ctrl+C to stop.")
try:
    while True:
        rec = fetch_no2()
        if rec is not None and rec['no2'] is not None:
            # Predict before update
            pred_norm = x[0]
            pred_actual = scaler.inverse_transform([[pred_norm]])[0,0]

            # Save prediction to CSV
            df_pred = pd.DataFrame({
                "timestamp": [pd.to_datetime(rec['timestamp'])],
                "prediction": [pred_actual],
                "observed": [rec['no2']]
            })
            df_pred.to_csv(pred_csv_path, mode='a', 
                           header=not os.path.exists(pred_csv_path), index=False)

            # Normalize observed value and update OMD
            y_t_norm = scaler.transform([[rec['no2']]])[0,0]
            x = omd_update(x, y_t_norm, eta)

            # Save updated OMD weight
            np.save(model_path, x)

            print(f"{rec['timestamp']} | Predicted: {pred_actual:.2f} µg/m³ | "
                  f"Observed: {rec['no2']:.2f} µg/m³ | Updated weight: {x[0]:.4f}")
        
        # Wait for next measurement (1 hour)
        time.sleep(3600)

except KeyboardInterrupt:
    print("Online learning stopped by user.")
#%%