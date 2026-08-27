from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import warnings
import uvicorn
import requests
import threading
import os
import time

warnings.filterwarnings("ignore")


# =========================================================
# CONFIGURATION
# =========================================================

DATASET_FILE = "irrigation_data.csv"

# Soil calibration
DRY_ADC = 900
WET_ADC = 300

# Soil target
SOIL_TARGET = 65

# ---------------------------------------------------------
# Rain sensor
#
# Your sensor is configured as:
#
# 0 = RAIN
# 1 = NO RAIN
#
# ESP8266 will send the correct boolean.
# ---------------------------------------------------------


# =========================================================
# TRAIN RANDOM FOREST
# =========================================================

print()
print("==============================================")
print("       EDGE-AI SMART IRRIGATION SERVER")
print("==============================================")

try:

    dataset = pd.read_csv(DATASET_FILE)

    required = [
        "Temperature",
        "Humidity",
        "Soil",
        "Pump_Status"
    ]

    missing = [
        c for c in required
        if c not in dataset.columns
    ]

    if missing:
        raise ValueError(
            f"Missing CSV columns: {missing}"
        )

    X = dataset[
        [
            "Temperature",
            "Humidity",
            "Soil"
        ]
    ]

    y = dataset["Pump_Status"]

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    model.fit(X, y)

    print("AI model trained successfully.")
    print(f"Training rows: {len(dataset)}")
    print(f"Classes: {sorted(y.unique().tolist())}")

except Exception as e:

    print("MODEL ERROR:")
    print(e)

    raise


# =========================================================
# FASTAPI
# =========================================================

app = FastAPI(
    title="Edge-AI Smart Irrigation"
)


# =========================================================
# GLOBAL STATE
# =========================================================

telemetry_state = {

    "temperature": 0.0,

    "humidity": 0.0,

    "soil": 1023,

    "soil_percent": 0,

    "raining": False,

    "rain_raw": 1,

    "pump_active": False,

    "pump_command": "PUMP_OFF",

    "mode": "AUTO",

    "ai_decision": "WAITING",

    "ai_reason": "Waiting for sensor data.",

    "recommended_action": "WAITING",

    "model_status": "Random Forest • Active",

    "last_update": 0,

    "esp_online": False
}


control_state = {

    "mode": "AUTO",

    "pump_command": "PUMP_OFF",

    "ai_decision": "WAITING",

    "ai_reason": "Waiting for telemetry."
}


# =========================================================
# SOIL ADC TO PERCENT
# =========================================================

def soil_adc_to_percent(adc):

    try:

        if DRY_ADC == WET_ADC:
            return 0

        percent = (
            (DRY_ADC - float(adc))
            / (DRY_ADC - WET_ADC)
        ) * 100

        percent = max(
            0,
            min(
                100,
                percent
            )
        )

        return round(percent)

    except Exception:

        return 0


# =========================================================
# SET PUMP STATE
# =========================================================

def set_pump_state(command):

    command = command.upper().strip()

    if command == "PUMP_ON":

        control_state["pump_command"] = "PUMP_ON"

        telemetry_state["pump_command"] = "PUMP_ON"

        telemetry_state["pump_active"] = True

    else:

        control_state["pump_command"] = "PUMP_OFF"

        telemetry_state["pump_command"] = "PUMP_OFF"

        telemetry_state["pump_active"] = False


# =========================================================
# NORMALIZE ML OUTPUT
# =========================================================

def ml_to_command(value):

    try:

        if hasattr(value, "item"):
            value = value.item()

    except Exception:
        pass

    value = str(value).strip().upper()

    if value in [
        "1",
        "TRUE",
        "YES",
        "ON",
        "PUMP_ON"
    ]:

        return "PUMP_ON"

    return "PUMP_OFF"


# =========================================================
# DASHBOARD
# =========================================================

@app.get("/", response_class=HTMLResponse)
async def dashboard():

    with open(
        "index.html",
        "r",
        encoding="utf-8"
    ) as f:

        return f.read()


# =========================================================
# TELEMETRY
# =========================================================

@app.get("/api/telemetry")
async def telemetry():

    return telemetry_state


# =========================================================
# MODE
# =========================================================

class ModePayload(BaseModel):

    mode: str


@app.post("/api/mode")
async def set_mode(data: ModePayload):

    mode = data.mode.upper().strip()

    if mode not in [
        "AUTO",
        "MANUAL"
    ]:

        return {
            "status": "error",
            "message": "Invalid mode."
        }


    control_state["mode"] = mode

    telemetry_state["mode"] = mode


    if mode == "AUTO":

        telemetry_state["model_status"] = (
            "Random Forest • Active"
        )

        control_state["ai_reason"] = (
            "AI automatic control enabled."
        )

        print()
        print("MODE -> AUTO")


    else:

        telemetry_state["model_status"] = (
            "Manual Override • Active"
        )

        control_state["ai_reason"] = (
            "Manual override enabled."
        )

        print()
        print("MODE -> MANUAL")


    return {

        "status": "ok",

        "mode": mode,

        "command":
            control_state["pump_command"],

        "pump_active":
            telemetry_state["pump_active"]
    }


# =========================================================
# MANUAL PUMP
# =========================================================

class PumpPayload(BaseModel):

    command: str


@app.post("/api/pump")
async def set_manual_pump(data: PumpPayload):

    command = data.command.upper().strip()


    if command not in [
        "PUMP_ON",
        "PUMP_OFF"
    ]:

        return {
            "status": "error",
            "message": "Invalid pump command."
        }


    if control_state["mode"] != "MANUAL":

        return {

            "status": "blocked",

            "message":
                "Switch to MANUAL mode first.",

            "mode":
                control_state["mode"]
        }


    set_pump_state(command)


    control_state["ai_decision"] = "MANUAL"


    if command == "PUMP_ON":

        control_state["ai_reason"] = (
            "Pump turned ON manually from dashboard."
        )

        telemetry_state["recommended_action"] = (
            "MANUAL WATERING"
        )

    else:

        control_state["ai_reason"] = (
            "Pump turned OFF manually from dashboard."
        )

        telemetry_state["recommended_action"] = (
            "PUMP OFF"
        )


    telemetry_state["ai_decision"] = "MANUAL"

    telemetry_state["ai_reason"] = (
        control_state["ai_reason"]
    )


    print(
        f"MANUAL COMMAND -> {command}"
    )


    return {

        "status": "ok",

        "mode": "MANUAL",

        "command": command,

        "pump_active":
            telemetry_state["pump_active"]
    }


# =========================================================
# FAST ESP COMMAND ENDPOINT
#
# ESP polls this every ~500 ms.
#
# This removes the old 3-second command delay.
# =========================================================

@app.get("/api/esp/command")
async def esp_command():

    return {

        "mode":
            control_state["mode"],

        "command":
            control_state["pump_command"],

        "pump_active":
            telemetry_state["pump_active"]
    }


# =========================================================
# ESP PAYLOAD
# =========================================================

class SensorPayload(BaseModel):

    temperature: float

    humidity: float

    soil: int

    raining: bool = False

    rain_raw: int | None = None


# =========================================================
# GOOGLE SHEETS
# =========================================================

GOOGLE_WEB_APP_URL = (
    "https://script.google.com/macros/s/"
    "AKfycbwffBgl6IckeSp_IAHav_fV_ZHmcjIvOJRb83aENQoyjDxkGVbGeB0bDQUFcVqcgSw"
    "/exec"
)


def log_to_sheets(
    temperature,
    humidity,
    soil,
    raining,
    pump_status
):

    data = {

        "Temperature":
            temperature,

        "Humidity":
            humidity,

        "Soil":
            soil,

        "Rainy":
            1 if raining else 0,

        "Pump_Status":
            1 if pump_status == "PUMP_ON" else 0
    }


    try:

        requests.post(
            GOOGLE_WEB_APP_URL,
            json=data,
            timeout=8
        )

    except Exception as e:

        print(
            "Google Sheets error:",
            e
        )


# =========================================================
# ESP REPORT
# =========================================================

@app.post("/api/esp/report")
async def esp_report(payload: SensorPayload):

    telemetry_state["temperature"] = (
        payload.temperature
    )

    telemetry_state["humidity"] = (
        payload.humidity
    )

    telemetry_state["soil"] = (
        payload.soil
    )

    telemetry_state["raining"] = (
        payload.raining
    )

    telemetry_state["rain_raw"] = (
        payload.rain_raw
        if payload.rain_raw is not None
        else (
            0 if payload.raining else 1
        )
    )

    telemetry_state["soil_percent"] = (
        soil_adc_to_percent(
            payload.soil
        )
    )

    telemetry_state["last_update"] = (
        time.time()
    )

    telemetry_state["esp_online"] = True


    soil_percent = (
        telemetry_state["soil_percent"]
    )


    print()
    print("----------------------------------------")
    print("ESP TELEMETRY")
    print(
        f"Temperature: {payload.temperature:.1f} C"
    )
    print(
        f"Humidity: {payload.humidity:.1f} %"
    )
    print(
        f"Soil ADC: {payload.soil}"
    )
    print(
        f"Soil %: {soil_percent}%"
    )
    print(
        f"Rain: {'YES' if payload.raining else 'NO'}"
    )
    print(
        f"Mode: {control_state['mode']}"
    )


    # =====================================================
    # MANUAL MODE
    # =====================================================

    if control_state["mode"] == "MANUAL":

        # AI DOES NOT TOUCH THE PUMP.

        command = (
            control_state["pump_command"]
        )

        telemetry_state["ai_decision"] = "MANUAL"

        telemetry_state["ai_reason"] = (
            "Manual override is controlling the pump."
        )

        telemetry_state["recommended_action"] = (
            "MANUAL WATERING"
            if command == "PUMP_ON"
            else "PUMP OFF"
        )

        set_pump_state(command)


    # =====================================================
    # AUTO MODE
    # =====================================================

    else:

        # -------------------------------------------------
        # RAIN PROTECTION
        # -------------------------------------------------

        if payload.raining:

            command = "PUMP_OFF"

            control_state["ai_decision"] = (
                "PUMP_OFF"
            )

            control_state["ai_reason"] = (
                "Rain detected. Irrigation stopped."
            )

            telemetry_state["recommended_action"] = (
                "STOP • RAIN DETECTED"
            )


        # -------------------------------------------------
        # SOIL TARGET
        # -------------------------------------------------

        elif soil_percent >= SOIL_TARGET:

            command = "PUMP_OFF"

            control_state["ai_decision"] = (
                "PUMP_OFF"
            )

            control_state["ai_reason"] = (
                "Soil moisture target reached."
            )

            telemetry_state["recommended_action"] = (
                "STOP • TARGET REACHED"
            )


        # -------------------------------------------------
        # RANDOM FOREST
        # -------------------------------------------------

        else:

            live_data = [[
                payload.temperature,
                payload.humidity,
                payload.soil
            ]]


            try:

                prediction =
                    model.predict(
                        live_data
                    )[0]

                ml_command =
                    ml_to_command(
                        prediction
                    )

            except Exception as e:

                print(
                    "ML prediction error:",
                    e
                )

                ml_command = "PUMP_OFF"


            if (
                ml_command == "PUMP_ON"
                and soil_percent < SOIL_TARGET
            ):

                command = "PUMP_ON"

                control_state["ai_decision"] = (
                    "PUMP_ON"
                )

                control_state["ai_reason"] = (
                    "Random Forest recommends irrigation "
                    "and soil moisture is below target."
                )

                telemetry_state["recommended_action"] = (
                    "START / CONTINUE WATERING"
                )

            else:

                command = "PUMP_OFF"

                control_state["ai_decision"] = (
                    "PUMP_OFF"
                )

                control_state["ai_reason"] = (
                    "Random Forest recommends no irrigation."
                )

                telemetry_state["recommended_action"] = (
                    "NO WATERING REQUIRED"
                )


        set_pump_state(command)

        telemetry_state["ai_decision"] = (
            control_state["ai_decision"]
        )

        telemetry_state["ai_reason"] = (
            control_state["ai_reason"]
        )


    telemetry_state["mode"] = (
        control_state["mode"]
    )


    # =====================================================
    # GOOGLE SHEETS
    # =====================================================

    threading.Thread(
        target=log_to_sheets,
        args=(
            payload.temperature,
            payload.humidity,
            payload.soil,
            payload.raining,
            control_state["pump_command"]
        ),
        daemon=True
    ).start()


    # =====================================================
    # RESPONSE TO ESP8266
    # =====================================================

    response = {

        "mode":
            control_state["mode"],

        "command":
            control_state["pump_command"],

        "pump_active":
            telemetry_state["pump_active"],

        "temperature":
            payload.temperature,

        "humidity":
            payload.humidity,

        "soil":
            payload.soil,

        "soil_percent":
            soil_percent,

        "raining":
            payload.raining,

        "rain_raw":
            telemetry_state["rain_raw"],

        "ai_decision":
            telemetry_state["ai_decision"],

        "ai_reason":
            telemetry_state["ai_reason"],

        "recommended_action":
            telemetry_state["recommended_action"],

        "model_status":
            telemetry_state["model_status"]
    }


    print(
        f"FINAL COMMAND: "
        f"{control_state['pump_command']}"
    )

    print("----------------------------------------")


    return response


# =========================================================
# HEALTH
# =========================================================

@app.get("/api/health")
async def health():

    return {

        "status": "online",

        "mode":
            control_state["mode"],

        "pump_active":
            telemetry_state["pump_active"],

        "esp_online":
            telemetry_state["esp_online"],

        "last_update":
            telemetry_state["last_update"]
    }


# =========================================================
# RENDER START
# =========================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            8000
        )
    )

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
