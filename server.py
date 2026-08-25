from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import warnings
import uvicorn
import requests
import threading

# Suppress sklearn warnings to keep terminal clean
warnings.filterwarnings('ignore')

# ==========================================
# 1. MACHINE LEARNING: Train on Startup
# ==========================================
print("Booting AI Core... Training Model...")
try:
    dataset = pd.read_csv("irrigation_data.csv")
    X = dataset[['Temperature', 'Humidity', 'Soil']]
    y = dataset['Pump_Status']
    
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    print("--> Model Trained Successfully!")
except Exception as e:
    print(f"ERROR loading CSV: {e}")
    print("Please ensure 'irrigation_data.csv' is in the same folder.")
    exit()

# ==========================================
# 2. FASTAPI DASHBOARD SERVER
# ==========================================
app = FastAPI(title="Smart Irrigation Server")

# Global variables to track the current state of the system
telemetry_state = {
    "temperature": 0.0,
    "humidity": 0.0,
    "soil": 1023, # Default to dry
    "raining": False,
    "pump_active": False,
}

control_state = {
    "mode": "AUTO",       # Starts in AUTO mode
    "pump_command": "PUMP_OFF"
}

# --- Dashboard UI Endpoints ---
@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/api/telemetry")
async def get_telemetry():
    return telemetry_state

@app.post("/api/mode")
async def update_mode(data: dict):
    control_state["mode"] = data.get("mode", "AUTO")
    print(f"Dashboard Mode Switched to: {control_state['mode']}")
    return {"status": "ok", "mode": control_state["mode"]}

@app.post("/api/pump")
async def manual_pump(data: dict):
    # Only allow manual dashboard clicks if in MANUAL mode
    if control_state["mode"] == "MANUAL":
        control_state["pump_command"] = data.get("command", "PUMP_OFF")
        telemetry_state["pump_active"] = (control_state["pump_command"] == "PUMP_ON")
        print(f"Manual Override: {control_state['pump_command']}")
    return {"status": "ok", "command": control_state["pump_command"]}

# --- ESP8266 Hardware Endpoint ---
class SensorPayload(BaseModel):
    temperature: float
    humidity: float
    soil: int
    raining: bool = False

@app.post("/api/esp/report")
async def esp_report(payload: SensorPayload):
    # 1. Update the live dashboard numbers
    telemetry_state["temperature"] = payload.temperature
    telemetry_state["humidity"] = payload.humidity
    telemetry_state["soil"] = payload.soil
    telemetry_state["raining"] = payload.raining

    # 2. THE AI DECISION ENGINE (Only runs if in AUTO mode)
    if control_state["mode"] == "AUTO":
        # Format the incoming live data for the ML model
        live_data = [[payload.temperature, payload.humidity, payload.soil]]
        
        # Ask the model what to do
        prediction = model.predict(live_data)
        
        if prediction[0] == 1:
            control_state["pump_command"] = "PUMP_ON"
            telemetry_state["pump_active"] = True
            print("🤖 AI Decision: Soil is dry. PUMP ON")
        else:
            control_state["pump_command"] = "PUMP_OFF"
            telemetry_state["pump_active"] = False
            print("🤖 AI Decision: Soil is fine. PUMP OFF")

    # --- GOOGLE SHEETS CLOUD LOGGING ---
    # Using your specific deployment URL
    GOOGLE_WEB_APP_URL = "https://script.google.com/macros/s/AKfycbwffBgl6IckeSp_IAHav_fV_ZHmcjIvOJRb83aENQoyjDxkGVbGeB0bDQUFcVqcgSw/exec"
    
    def log_to_sheets():
        sheet_data = {
            "Temperature": payload.temperature,
            "Humidity": payload.humidity,
            "Soil": payload.soil,
            "Rainy": 1 if payload.raining else 0,
            "Pump_Status": 1 if control_state["pump_command"] == "PUMP_ON" else 0
        }
        try:
            requests.post(GOOGLE_WEB_APP_URL, json=sheet_data)
        except Exception as e:
            print("Failed to log to Google Sheets:", e)

    # Run the logging in a background thread so the ESP8266 gets an instant response
    threading.Thread(target=log_to_sheets).start()

    # 3. Send the final command back to the ESP8266
    return {
        "mode": control_state["mode"],
        "command": control_state["pump_command"]
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)