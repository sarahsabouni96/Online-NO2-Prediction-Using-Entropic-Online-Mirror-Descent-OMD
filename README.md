#  Online-NO2-Prediction-Using-Entropic-Online-Mirror-Descent-OMD

This project implements a **real-time $NO_2$ (Nitrogen Dioxide) concentration prediction system** using a continuous **Online Learning** algorithm: the **Entropic Online Mirror Descent (OMD)**.

The model operates by continuously receiving live $NO_2$ measurements, predicting the next reading, and iteratively updating its internal prediction parameter every hour, allowing it to adapt to local air quality dynamics and **concept drift**.

The primary API endpoint used to fetch live $NO_2$ values is:
`https://api.waqi.info/feed/here/?token=YOUR_TOKEN`

---

## 🔬 Project Overview

This system is a demonstration of using online convex optimization for environmental time-series data. It is structured to run continuously, bridging the gap between historical batch processing and real-time operational learning.

### Execution Flow

1.  **Initialization:** Load historical $NO_2$ data (batch data) to fit the $\text{MinMaxScaler}$ and provide an initial guess for the OMD weight.
2.  **Normalization:** Scale all $NO_2$ measurements to the constrained domain $[0, 1]$.
3.  **Persistence:** Load the existing OMD weight from `omd_no2_weight.npy` or initialize it from the historical data's mean.
4.  **Online Loop (Runs Hourly):**
    * Fetch the current $NO_2$ concentration from the **WAQI API**.
    * **Predict** the next $NO_2$ level based on the current OMD weight.
    * Log the prediction vs. observed value to `online_no2_predictions.csv`.
    * Update the **OMD weight** using the latest observation.
    * Save the updated weight to `omd_no2_weight.npy`.

---

## 📊 Data Sources

### 1. Historical Data (Offline Initialization & Scaling)

This data is used exclusively to determine the minimum and maximum boundaries for the $\text{MinMaxScaler}$ and to provide a stable initial weight for the OMD algorithm.

* **Source:** WAQI Historical Data / Network Pages (e.g., [aqicn.org/network/de-sachsen/](https://aqicn.org/network/de-sachsen/))
* **File Requirement:** You must place an exported CSV file named `werte2.csv` containing at least the columns: `Datum_Zeit` and `No2` (or `no2`).

### 2. Real-Time Data (Online Learning)

The system fetches live, current-hour $NO_2$ readings to drive the online learning updates.

* **Source:** **World Air Quality Index (WAQI) API**.
* **Sensor Network:** WAQI aggregates data from various global and regional sensor platforms, including networks like **Sensoto** (`https://sensoto.io/en/`). The WAQI API provides the most accessible real-time patch data for the project.
* **API Endpoint:** `https://api.waqi.info/feed/here/?token=YOUR_TOKEN`

---

## 🔑 Algorithm: Entropic Online Mirror Descent (OMD)

OMD is a first-order optimization algorithm that generalizes Gradient Descent to non-Euclidean geometries defined by a **Mirror Map** (or Regularizer). For the entropic case, the Mirror Map is related to the negative entropy, leading to the **Exponentiated Gradient** method.

The single-parameter model uses the OMD weight $x_t$ as the normalized prediction $\hat{y}_t$.

### OMD Update Rule

The update step is calculated as:
$$x_{t+1} = \text{clip}\left( x_t \cdot \exp(-\eta g_t), 0, 1 \right)$$

* $x_t$ is the **current OMD weight** (normalized prediction parameter).
* $\eta$ is the **Learning Rate** (`eta`).
* $g_t$ is the **Gradient of the Squared Loss**, where $L_t(x) = (x - y_t)^2$.
    $$g_t = 2(x_t - y_t)$$
* $y_t$ is the **observed normalized $NO_2$ value**.
* The $\text{clip}(\cdot, 0, 1)$ function ensures the parameter remains in the normalized domain **$[0, 1]$**.

This update rule inherently provides stability and a smooth, adaptive prediction curve, characteristic of algorithms utilizing an entropic regularizer. 

---

## 🚀 Features

* **Online Learning:** Continuous, adaptive prediction using OMD.
* **Minimalistic Model:** Single-parameter estimator, highly efficient for simple time-series prediction.
* **Robust Data Handling:** Utilizes $\text{MinMaxScaler}$ for robust normalization based on historical data.
* **Persistence:** Automatic saving/loading of the learned weight (`omd_no2_weight.npy`).
* **Real-time Logging:** Comprehensive logging of all prediction rounds to CSV.

---

## 💻 Running the Project

### Step 1: Configuration

1.  **Insert your WAQI API token** into the Python script:
    ```python
    api_token = "YOUR_TOKEN_HERE"
    ```
2.  **Verify the `csv_path`** points to your historical data file (`werte2.csv`).

### Step 2: Execute

Run the script from your terminal (assuming the filename is `online_no2.py`):
The script will begin printing its hourly updates:
```bash
Loaded previous OMD weight: [0.5321]
Starting real-time online updates. Press Ctrl+C to stop.
2025-02-14 15:00:00 | Predicted: 21.30 µg/m³ | Observed: 24.00 µg/m³ | Updated weight: 0.5348
...
```
Press Ctrl + C to stop the script gracefully.

---

## Output Files
1. online_no2_predictions.csv


| Column      | Description                           |
|-------------|-----------------------------------------|
| timestamp   | Time of the observation                 |
| prediction  | Predicted NO₂ level (µg/m³)             |
| observed    | Actual NO₂ value from the API (µg/m³)   |

2. omd_no2_weight.npy
Stores the latest learned OMD weight $x_t$. This file is automatically loaded on restart to ensure continuous learning.
---

## Future Work

- **Multi-Dimensional OMD:** Extend the model to a feature vector.
- **Advanced Prediction Models:** Implement OGD or true online regression.
- **Visualization Dashboard:** Real-time dashboard with Plotly or Dash.


---

## Citation / Data Use Disclaimer
Data used in this project is sourced from public environmental networks:

WAQI – World Air Quality Index (https://waqi.info/)

Sensoto Environmental Sensor Network (https://sensoto.io/en/)

Data usage must strictly adhere to the API terms and conditions of both providers.

