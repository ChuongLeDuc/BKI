import requests
import time
import json
import numpy as np
from datetime import datetime
from pymodbus.client import ModbusTcpClient
from tensorflow.keras.models import load_model
from joblib import load
import sseclient

# ---------------- THRESHOLD ----------------
threshold = 0.8

# ---------------- Load ANN model ----------------
try:
    model = load_model("ann_model.h5")
    print("[INFO] ANN model loaded successfully.")
except Exception as e:
    print(f"[ERROR] Failed to load ANN model: {type(e).__name__}: {e}")
    model = None

# ---------------- Load scaler ----------------
try:
    mean = load("scaler_X_mean.pkl")
    std = load("scaler_X_std.pkl")
    scaler_y = load("scaler_y.pkl")
    print("[INFO] Scaler X and Y loaded.")
except Exception as e:
    print(f"[WARNING] Could not load scaler values: {e}")
    mean = np.array([64.692, 11655.9065, 38.335])
    std = np.array([1.97217340e+01, 1.12813887e+04, 8.74448792e+00])
    scaler_y = None

# ---------------- Helper Functions ----------------
def readData(inputData, type):
    data = inputData.get('data')
    if type == "weather":
        return data['temperature'], data['irradiation'], data['humidity']
    if type == "power":
        return data['P_Inverter'], data['P_Solar'], data['P_Grid']

def ensure_modbus_connection(client, name):
    if not client.connect():
        print(f"[ERROR] Reconnecting to {name} Modbus failed.")
    else:
        print(f"[INFO] Reconnected to {name} Modbus.")

# ---------------- Modbus Clients ----------------
modbus_client = ModbusTcpClient('172.16.6.145', port=502)
M4M_client = ModbusTcpClient('172.16.6.238', port=502)

# ---------------- Main Loop ----------------
while True:
    try:
        today = datetime.now().strftime("%d-%m-%Y")
        base_url = "https://iot-weather-d6148-default-rtdb.asia-southeast1.firebasedatabase.app/Factory_1"
        weather_url = f"{base_url}/{today}/weather.json"
        power_url = f"{base_url}/{today}/power.json"
        trained_url = f"{base_url}/trained.json"
        predicted_url = f"{base_url}/{today}/predicted.json"

        print(f"[INFO] Listening to Firebase at {weather_url}")
        client = sseclient.SSEClient(weather_url)

        for event in client:
            if event.event in ['put', 'patch']:
                if datetime.now().strftime("%d-%m-%Y") != today:
                    break

                # --- Read Weather Data ---
                weather_data = json.loads(event.data)
                if len(weather_data.get('data')) > 10:
                    continue

                temperature, irradiation, humidity = readData(weather_data, "weather")

                # --- Predict ANN ---
                solar_predicted = None
                if model is not None:
                    try:
                        input_arr = np.array([[temperature, irradiation, humidity]])
                        input_scaled = (input_arr - mean) / std
                        pred_scaled = model.predict(input_scaled, verbose=0)[0][0]
                        if scaler_y:
                            solar_predicted = float(scaler_y.inverse_transform([[pred_scaled]])[0][0])
                        else:
                            solar_predicted = float(pred_scaled)  # fallback
                        print(f"[PREDICT] Solar_predicted: {solar_predicted:.2f} W")
                    except Exception as e_ann:
                        print(f"[WARNING] ANN prediction failed: {type(e_ann).__name__}: {e_ann}")

                # --- Read Real Power Data ---
                if not modbus_client.is_socket_open():
                    ensure_modbus_connection(modbus_client, "Inverter")
                if not M4M_client.is_socket_open():
                    ensure_modbus_connection(M4M_client, "M4M")

                Solar = modbus_client.read_holding_registers(address=40101, count=1)
                Inverter = modbus_client.read_holding_registers(address=40084, count=1)
                Grid = M4M_client.read_holding_registers(address=23322, count=1)

                if Solar.isError() or Inverter.isError() or Grid.isError():
                    print("[ERROR] Modbus read error.")
                    continue

                P_solar = Solar.registers[0] * 10
                P_inverter = Inverter.registers[0] * 10
                P_grid = Grid.registers[0]

                print(f"[DATA] P_solar: {P_solar}, P_inverter: {P_inverter}, P_grid: {P_grid}")

                # --- Prepare Payloads ---
                now = int(time.time() * 1000)
                trained_payload = {
                    "time": now,
                    "temperature": round(temperature, 2),
                    "humidity": round(humidity, 2),
                    "irradiation": round(irradiation, 2),
                    "P_Solar": round(P_solar, 2),
                }

                power_payload = {
                    "time": now,
                    "P_Inverter": round(P_inverter, 2),
                    "P_Grid": round(P_grid, 2),
                    "P_Solar": round(P_solar, 2),
                }

                predicted_payload = {
                    "time": now,
                    "Actual_Inverter": round(P_inverter, 2),
                    "Normal_Inverter": round(solar_predicted * threshold, 2) if solar_predicted else 0,
                    "Actual_Solar": round(P_solar, 2),
                    "Predicted_Solar": round(solar_predicted, 2) if solar_predicted else 0,
                }

                # --- POST to Firebase ---
                try:
                    requests.post(trained_url, json=trained_payload)
                    requests.post(power_url, json=power_payload)
                    requests.post(predicted_url, json=predicted_payload)
                    print("[POST] All data sent successfully.")
                except Exception as post_e:
                    print(f"[ERROR] Firebase POST failed: {post_e}")

        time.sleep(5)

    except Exception as e:
        print(f"[EXCEPTION] {type(e).__name__}: {e}")
        print("[INFO] Retrying in 10 seconds...")
        time.sleep(10)
        modbus_client.close()
        M4M_client.close()
        modbus_client.connect()
        M4M_client.connect()
