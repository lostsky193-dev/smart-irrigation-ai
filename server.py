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
# 1. CONFIGURATION
# =========================================================

DATASET_FILE = "irrigation_data.csv"

# ---------------------------------------------------------
# SOIL CALIBRATION
#
# These are DEFAULT values.
#
# Change these after testing your actual soil sensor.
#
# Example:
# DRY_ADC = 900
# WET_ADC = 300
#
# Assumption:
# HIGH ADC = DRY
# LOW ADC  = WET
# ---------------------------------------------------------

DRY_ADC = 900
WET_ADC = 300

# Target soil moisture percentage
SOIL_TARGET = 65

# Safety limit
SOIL_STOP_LEVEL = 65

# Do not continuously toggle the pump around the target.
# This gives a small hysteresis zone.
SOIL_RESTART_LEVEL = 58


# =========================================================
# 2. TRAIN MACHINE LEARNING MODEL
# =========================================================

print()
print("==============================================")
print("        SMART IRRIGATION AI SERVER")
print("==============================================")
print("Booting AI Core...")
print("Loading irrigation dataset...")

try:

    dataset = pd.read_csv(DATASET_FILE)

    required_columns = [
        "Temperature",
        "Humidity",
        "Soil",
        "Pump_Status"
    ]

    missing_columns = [
        col for col in required_columns
        if col not in dataset.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing columns in CSV: {missing_columns}"
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

    print("--> Model Trained Successfully!")
    print(f"--> Training samples: {len(dataset)}")
    print(f"--> Classes: {sorted(y.unique().tolist())}")

except Exception as e:

    print()
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print("ERROR LOADING / TRAINING MODEL")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(e)
    raise


# =========================================================
# 3. FASTAPI
# =========================================================

app = FastAPI(
    title="Edge-AI Smart Irrigation Server"
)


# =========================================================
# 4. SYSTEM STATE
# =========================================================

telemetry_state = {

    "temperature": 0.0,

    "humidity": 0.0,

    "soil": 1023,

    "soil_percent": 0,

    "raining": False,

    "rain_raw": 0,

    "pump_active": False,

    "pump_command": "PUMP_OFF",

    "mode": "AUTO",

    "ai_decision": "WAITING",

    "ai_reason": "Waiting for sensor data.",

    "recommended_action": "WAITING",

    "model_status": "Random Forest • Active",

    "last_update": 0
}


control_state = {

    # AUTO or MANUAL
    "mode": "AUTO",

    # Desired command
    "pump_command": "PUMP_OFF",

    # Last AI decision
    "ai_decision": "WAITING",

    "ai_reason": "Waiting for telemetry."
}


# =========================================================
# 5. PUMP STATE HELPER
# =========================================================

def set_pump_state(command: str):

    command = command.upper().strip()

    if command == "PUMP_ON":

        control_state["pump_command"] = "PUMP_ON"

        telemetry_state["pump_active"] = True

        telemetry_state["pump_command"] = "PUMP_ON"

    else:

        control_state["pump_command"] = "PUMP_OFF"

        telemetry_state["pump_active"] = False

        telemetry_state["pump_command"] = "PUMP_OFF"


# =========================================================
# 6. SOIL ADC -> PERCENTAGE
# =========================================================

def soil_adc_to_percent(adc_value: int) -> int:

    try:

        adc = float(adc_value)

        # Prevent division by zero
        if DRY_ADC == WET_ADC:
            return 0

        percent = (
            (DRY_ADC - adc)
            / (DRY_ADC - WET_ADC)
        ) * 100.0

        percent = max(
            0.0,
            min(100.0, percent)
        )

        return round(percent)

    except Exception:

        return 0


# =========================================================
# 7. NORMALIZE ML OUTPUT
# =========================================================

def ml_prediction_to_command(prediction):

    """
    Supports typical Random Forest outputs:

    1 / "1" / "ON" / "PUMP_ON" -> ON
    0 / "0" / "OFF" / "PUMP_OFF" -> OFF
    """

    value = prediction

    try:

        if hasattr(value, "item"):
            value = value.item()

    except Exception:
        pass

    text = str(value).strip().upper()

    if text in [
        "1",
        "ON",
        "PUMP_ON",
        "TRUE",
        "YES"
    ]:
        return "PUMP_ON"

    return "PUMP_OFF"


# =========================================================
# 8. DASHBOARD
# =========================================================

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():

    with open(
        "index.html",
        "r",
        encoding="utf-8"
    ) as f:

        return f.read()


# =========================================================
# 9. TELEMETRY API
# =========================================================

@app.get("/api/telemetry")
async def get_telemetry():

    return telemetry_state


# =========================================================
# 10. MODE MODEL
# =========================================================

class ModePayload(BaseModel):

    mode: str


@app.post("/api/mode")
async def update_mode(data: ModePayload):

    mode = data.mode.upper().strip()

    if mode not in ["AUTO", "MANUAL"]:

        return {
            "status": "error",
            "message": "Mode must be AUTO or MANUAL.",
            "mode": control_state["mode"]
        }


    control_state["mode"] = mode

    telemetry_state["mode"] = mode


    # -----------------------------------------------------
    # Switch to AUTO
    # -----------------------------------------------------

    if mode == "AUTO":

        control_state["ai_reason"] = (
            "AI control enabled."
        )

        telemetry_state["model_status"] = (
            "Random Forest • Active"
        )

        print()
        print("========================================")
        print("MODE: AUTO")
        print("AI CONTROL ENABLED")
        print("========================================")


    # -----------------------------------------------------
    # Switch to MANUAL
    # -----------------------------------------------------

    else:

        control_state["ai_reason"] = (
            "Manual override enabled."
        )

        telemetry_state["model_status"] = (
            "Manual Override • Active"
        )

        print()
        print("========================================")
        print("MODE: MANUAL")
        print("AI CONTROL DISABLED")
        print("========================================")


    telemetry_state["ai_decision"] = (
        "MANUAL"
        if mode == "MANUAL"
        else "WAITING"
    )


    return {

        "status": "ok",

        "mode": mode,

        "pump_command":
            control_state["pump_command"],

        "pump_active":
            telemetry_state["pump_active"]
    }


# =========================================================
# 11. MANUAL PUMP CONTROL
# =========================================================

class PumpPayload(BaseModel):

    command: str


@app.post("/api/pump")
async def manual_pump(data: PumpPayload):

    command = data.command.upper().strip()


    if command not in [
        "PUMP_ON",
        "PUMP_OFF"
    ]:

        return {

            "status": "error",

            "message":
                "Command must be PUMP_ON or PUMP_OFF.",

            "command":
                control_state["pump_command"]
        }


    # -----------------------------------------------------
    # ONLY allow dashboard pump control in MANUAL mode
    # -----------------------------------------------------

    if control_state["mode"] != "MANUAL":

        return {

            "status": "blocked",

            "message":
                "Switch dashboard to MANUAL mode first.",

            "mode":
                control_state["mode"],

            "command":
                control_state["pump_command"],

            "pump_active":
                telemetry_state["pump_active"]
        }


    # -----------------------------------------------------
    # Manual command accepted
    # -----------------------------------------------------

    set_pump_state(command)


    control_state["ai_decision"] = "MANUAL"


    if command == "PUMP_ON":

        control_state["ai_reason"] = (
            "Pump manually activated from dashboard."
        )

        telemetry_state["recommended_action"] = (
            "MANUAL WATERING"
        )

    else:

        control_state["ai_reason"] = (
            "Pump manually stopped from dashboard."
        )

        telemetry_state["recommended_action"] = (
            "PUMP OFF"
        )


    telemetry_state["ai_decision"] = "MANUAL"

    telemetry_state["ai_reason"] = (
        control_state["ai_reason"]
    )


    print()
    print("========================================")
    print("MANUAL COMMAND")
    print(f"COMMAND: {command}")
    print("========================================")


    return {

        "status": "ok",

        "mode": "MANUAL",

        "command": command,

        "pump_active":
            telemetry_state["pump_active"],

        "message":
            "Manual command accepted."
    }


# =========================================================
# 12. ESP PAYLOAD
# =========================================================

class SensorPayload(BaseModel):

    temperature: float

    humidity: float

    soil: int

    raining: bool = False

    # Optional raw rain sensor value
    rain_raw: int | None = None


# =========================================================
# 13. GOOGLE SHEETS LOGGING
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

    sheet_data = {

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
            json=sheet_data,
            timeout=8
        )

    except Exception as e:

        print(
            "Google Sheets logging failed:",
            e
        )


# =========================================================
# 14. ESP8266 REPORT ENDPOINT
# =========================================================

@app.post("/api/esp/report")
async def esp_report(payload: SensorPayload):

    # -----------------------------------------------------
    # Update raw telemetry
    # -----------------------------------------------------

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
            1 if payload.raining else 0
        )
    )

    telemetry_state["last_update"] = (
        time.time()
    )


    # -----------------------------------------------------
    # Calculate soil moisture percentage
    # -----------------------------------------------------

    soil_percent = soil_adc_to_percent(
        payload.soil
    )

    telemetry_state["soil_percent"] = (
        soil_percent
    )


    print()
    print("----------------------------------------")
    print("ESP8266 TELEMETRY")
    print("----------------------------------------")

    print(
        f"Temperature : {payload.temperature:.1f} °C"
    )

    print(
        f"Humidity    : {payload.humidity:.1f} %"
    )

    print(
        f"Soil ADC    : {payload.soil}"
    )

    print(
        f"Soil        : {soil_percent} %"
    )

    print(
        f"Rain        : "
        f"{'RAINING' if payload.raining else 'NO RAIN'}"
    )

    print(
        f"Mode        : "
        f"{control_state['mode']}"
    )


    # =====================================================
    # MANUAL MODE
    # =====================================================

    if control_state["mode"] == "MANUAL":

        # -------------------------------------------------
        # IMPORTANT:
        # AI DOES NOT CHANGE THE PUMP IN MANUAL MODE
        # -------------------------------------------------

        command = (
            control_state["pump_command"]
        )


        telemetry_state["ai_decision"] = (
            "MANUAL"
        )

        telemetry_state["ai_reason"] = (
            "Manual override is controlling the pump."
        )


        if command == "PUMP_ON":

            telemetry_state["recommended_action"] = (
                "MANUAL WATERING"
            )

        else:

            telemetry_state["recommended_action"] = (
                "PUMP OFF"
            )


        set_pump_state(command)


        print(
            f"MANUAL COMMAND: {command}"
        )


    # =====================================================
    # AUTO MODE
    # =====================================================

    else:

        # -------------------------------------------------
        # STEP 1: RAIN PROTECTION
        # -------------------------------------------------
        #
        # If raining, stop irrigation.
        #
        # This prevents watering during rainfall.
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
        # STEP 2: SOIL TARGET REACHED
        # -------------------------------------------------

        elif soil_percent >= SOIL_STOP_LEVEL:

            command = "PUMP_OFF"

            control_state["ai_decision"] = (
                "PUMP_OFF"
            )

            control_state["ai_reason"] = (
                "Soil moisture target reached."
            )

            telemetry_state["recommended_action"] = (
                "STOP WATERING • TARGET REACHED"
            )


        # -------------------------------------------------
        # STEP 3: RUN ML MODEL
        # -------------------------------------------------

        else:

            live_data = [[
                payload.temperature,
                payload.humidity,
                payload.soil
            ]]


            try:

                prediction = model.predict(
                    live_data
                )[0]


                ml_command = (
                    ml_prediction_to_command(
                        prediction
                    )
                )


            except Exception as e:

                print(
                    "ML prediction error:",
                    e
                )

                ml_command = "PUMP_OFF"


            # -------------------------------------------------
            # LOW SOIL = irrigation allowed
            # -------------------------------------------------

            if (
                ml_command == "PUMP_ON"
                and soil_percent < SOIL_TARGET
            ):

                command = "PUMP_ON"

                control_state["ai_decision"] = (
                    "PUMP_ON"
                )

                control_state["ai_reason"] = (
                    "ML model recommends irrigation "
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
                    "ML model recommends no irrigation."
                )

                telemetry_state["recommended_action"] = (
                    "NO WATERING REQUIRED"
                )


        # -------------------------------------------------
        # Update AUTO state
        # -------------------------------------------------

        set_pump_state(command)

        telemetry_state["ai_decision"] = (
            control_state["ai_decision"]
        )

        telemetry_state["ai_reason"] = (
            control_state["ai_reason"]
        )


        print(
            f"AI DECISION: {command}"
        )

        print(
            f"REASON: "
            f"{control_state['ai_reason']}"
        )


    # =====================================================
    # COMMON STATE
    # =====================================================

    telemetry_state["mode"] = (
        control_state["mode"]
    )

    telemetry_state["pump_command"] = (
        control_state["pump_command"]
    )

    telemetry_state["pump_active"] = (
        control_state["pump_command"]
        == "PUMP_ON"
    )


    # =====================================================
    # GOOGLE SHEETS
    # =====================================================

    thread = threading.Thread(

        target=log_to_sheets,

        args=(

            payload.temperature,

            payload.humidity,

            payload.soil,

            payload.raining,

            control_state["pump_command"]
        ),

        daemon=True
    )

    thread.start()


    # =====================================================
    # ESP8266 RESPONSE
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


    print("----------------------------------------")
    print(
        f"FINAL COMMAND: "
        f"{control_state['pump_command']}"
    )
    print("----------------------------------------")
    print()


    return response


# =========================================================
# 15. HEALTH CHECK
# =========================================================

@app.get("/api/health")
async def health():

    return {

        "status": "online",

        "mode":
            control_state["mode"],

        "pump_active":
            telemetry_state["pump_active"],

        "last_update":
            telemetry_state["last_update"],

        "model":
            "Random Forest"
    }


# =========================================================
# 16. START SERVER
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
