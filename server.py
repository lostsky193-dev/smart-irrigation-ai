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

# ---------------------------------------------------------
# SOIL CALIBRATION
#
# HIGH ADC = DRY
# LOW ADC  = WET
#
# Change these after testing your real sensor.
# ---------------------------------------------------------

DRY_ADC = 900
WET_ADC = 300

# Soil moisture target
SOIL_TARGET = 65


# =========================================================
# STARTUP
# =========================================================

print()
print("==================================================")
print("         EDGE-AI SMART IRRIGATION SERVER")
print("==================================================")
print("Booting AI core...")
print()


# =========================================================
# TRAIN RANDOM FOREST
# =========================================================

try:

    dataset = pd.read_csv(DATASET_FILE)

    required_columns = [
        "Temperature",
        "Humidity",
        "Soil",
        "Pump_Status"
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataset.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing required CSV columns: "
            + str(missing_columns)
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

    model.fit(
        X,
        y
    )


    print("AI model trained successfully.")
    print(
        f"Training samples: {len(dataset)}"
    )

    print(
        f"Model classes: {sorted(y.unique().tolist())}"
    )

    print()

except Exception as error:

    print("==================================================")
    print("AI MODEL ERROR")
    print("==================================================")
    print(error)
    print()

    raise


# =========================================================
# FASTAPI APP
# =========================================================

app = FastAPI(
    title="Edge-AI Smart Irrigation",
    version="1.0.0"
)


# =========================================================
# GLOBAL TELEMETRY STATE
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

    "ai_reason": "Waiting for ESP8266 telemetry.",

    "recommended_action": "WAITING",

    "model_status": "Random Forest • Active",

    "last_update": 0,

    "esp_online": False
}


# =========================================================
# CONTROL STATE
# =========================================================

control_state = {

    "mode": "AUTO",

    "pump_command": "PUMP_OFF",

    "ai_decision": "WAITING",

    "ai_reason": "Waiting for telemetry."
}


# =========================================================
# HELPER: SOIL ADC -> %
# =========================================================

def soil_adc_to_percent(adc_value: int) -> int:

    try:

        adc = float(adc_value)

        if DRY_ADC == WET_ADC:
            return 0

        percent = (
            (DRY_ADC - adc)
            /
            (DRY_ADC - WET_ADC)
        ) * 100.0

        percent = max(
            0.0,
            min(
                100.0,
                percent
            )
        )

        return round(percent)

    except Exception:

        return 0


# =========================================================
# HELPER: SET PUMP STATE
# =========================================================

def set_pump_state(command: str):

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
# HELPER: NORMALIZE ML OUTPUT
# =========================================================

def ml_to_command(value):

    try:

        if hasattr(value, "item"):
            value = value.item()

    except Exception:
        pass


    text = str(value).strip().upper()


    if text in [
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

@app.get(
    "/",
    response_class=HTMLResponse
)
async def serve_dashboard():

    with open(
        "index.html",
        "r",
        encoding="utf-8"
    ) as file:

        return file.read()


# =========================================================
# DASHBOARD TELEMETRY
# =========================================================

@app.get("/api/telemetry")
async def get_telemetry():

    return telemetry_state


# =========================================================
# MODE REQUEST
# =========================================================

class ModePayload(BaseModel):

    mode: str


@app.post("/api/mode")
async def update_mode(
    data: ModePayload
):

    mode = data.mode.upper().strip()


    if mode not in [
        "AUTO",
        "MANUAL"
    ]:

        return {

            "status": "error",

            "message":
                "Mode must be AUTO or MANUAL.",

            "mode":
                control_state["mode"]
        }


    control_state["mode"] = mode

    telemetry_state["mode"] = mode


    # -----------------------------------------------------
    # AUTO MODE
    # -----------------------------------------------------

    if mode == "AUTO":

        telemetry_state["model_status"] = (
            "Random Forest • Active"
        )

        control_state["ai_decision"] = (
            "WAITING"
        )

        control_state["ai_reason"] = (
            "AI automatic control enabled."
        )

        telemetry_state["ai_decision"] = (
            "WAITING"
        )

        telemetry_state["ai_reason"] = (
            "AI automatic control enabled."
        )

        print()
        print("------------------------------------------")
        print("MODE CHANGED -> AUTO")
        print("------------------------------------------")


    # -----------------------------------------------------
    # MANUAL MODE
    # -----------------------------------------------------

    else:

        telemetry_state["model_status"] = (
            "Manual Override • Active"
        )

        telemetry_state["ai_decision"] = (
            "MANUAL"
        )

        telemetry_state["ai_reason"] = (
            "Manual override enabled."
        )

        control_state["ai_decision"] = (
            "MANUAL"
        )

        control_state["ai_reason"] = (
            "Manual override enabled."
        )

        print()
        print("------------------------------------------")
        print("MODE CHANGED -> MANUAL")
        print("------------------------------------------")


    return {

        "status": "ok",

        "mode":
            mode,

        "command":
            control_state["pump_command"],

        "pump_active":
            telemetry_state["pump_active"]
    }


# =========================================================
# MANUAL PUMP CONTROL
# =========================================================

class PumpPayload(BaseModel):

    command: str


@app.post("/api/pump")
async def manual_pump(
    data: PumpPayload
):

    command = data.command.upper().strip()


    # -----------------------------------------------------
    # Validate command
    # -----------------------------------------------------

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
    # Manual mode required
    # -----------------------------------------------------

    if control_state["mode"] != "MANUAL":

        return {

            "status": "blocked",

            "message":
                "Switch to MANUAL mode first.",

            "mode":
                control_state["mode"],

            "command":
                control_state["pump_command"],

            "pump_active":
                telemetry_state["pump_active"]
        }


    # -----------------------------------------------------
    # Store manual command
    # -----------------------------------------------------

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


    telemetry_state["ai_decision"] = (
        "MANUAL"
    )

    telemetry_state["ai_reason"] = (
        control_state["ai_reason"]
    )


    print()
    print("------------------------------------------")
    print(
        f"MANUAL PUMP COMMAND -> {command}"
    )
    print("------------------------------------------")


    return {

        "status": "ok",

        "mode": "MANUAL",

        "command": command,

        "pump_active":
            telemetry_state["pump_active"],

        "message":
            "Command accepted."
    }


# =========================================================
# FAST ESP8266 COMMAND ENDPOINT
#
# ESP8266 checks this approximately every 500 ms.
#
# This makes dashboard control much faster.
# =========================================================

@app.get("/api/esp/command")
async def get_esp_command():

    return {

        "mode":
            control_state["mode"],

        "command":
            control_state["pump_command"],

        "pump_active":
            telemetry_state["pump_active"]
    }


# =========================================================
# ESP8266 SENSOR PAYLOAD
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
    "AKfycbwffbgl6IckeSp_IAHav_fV_ZHmcjIvOJRb83aENQoyjDxkGVbGeB0bDQUFcVqcgSw/"
    "exec"
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
            1 if pump_status == "PUMP_ON"
            else 0
    }


    try:

        requests.post(
            GOOGLE_WEB_APP_URL,
            json=sheet_data,
            timeout=8
        )

    except Exception as error:

        print(
            "Google Sheets logging error:",
            error
        )


# =========================================================
# ESP8266 REPORT ENDPOINT
# =========================================================

@app.post("/api/esp/report")
async def esp_report(
    payload: SensorPayload
):

    # -----------------------------------------------------
    # SAVE RAW SENSOR DATA
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


    if payload.rain_raw is not None:

        telemetry_state["rain_raw"] = (
            payload.rain_raw
        )

    else:

        telemetry_state["rain_raw"] = (
            0
            if payload.raining
            else 1
        )


    telemetry_state["last_update"] = (
        time.time()
    )

    telemetry_state["esp_online"] = True


    # -----------------------------------------------------
    # SOIL PERCENTAGE
    # -----------------------------------------------------

    soil_percent = soil_adc_to_percent(
        payload.soil
    )

    telemetry_state["soil_percent"] = (
        soil_percent
    )


    # -----------------------------------------------------
    # SERVER LOG
    # -----------------------------------------------------

    print()
    print("==========================================")
    print("ESP8266 TELEMETRY")
    print("==========================================")

    print(
        f"Temperature : {payload.temperature:.1f} C"
    )

    print(
        f"Humidity    : {payload.humidity:.1f} %"
    )

    print(
        f"Soil ADC    : {payload.soil}"
    )

    print(
        f"Soil %      : {soil_percent}%"
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
        #
        # AI DOES NOT MODIFY PUMP IN MANUAL MODE
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
        # PRIORITY 1: RAIN PROTECTION
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
        # PRIORITY 2: SOIL TARGET
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
        # PRIORITY 3: RANDOM FOREST
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

                ml_command = ml_to_command(
                    prediction
                )

            except Exception as error:

                print(
                    "ML prediction error:",
                    error
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


        # -------------------------------------------------
        # Apply AUTO decision
        # -------------------------------------------------

        set_pump_state(command)


        telemetry_state["ai_decision"] = (
            control_state["ai_decision"]
        )

        telemetry_state["ai_reason"] = (
            control_state["ai_reason"]
        )


        print(
            f"AI COMMAND: {command}"
        )

        print(
            f"AI REASON: "
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
    # GOOGLE SHEETS LOGGING
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

    print("==========================================")
    print()


    return response


# =========================================================
# HEALTH CHECK
# =========================================================

@app.get("/api/health")
async def health():

    return {

        "status": "online",

        "mode":
            control_state["mode"],

        "pump_active":
            telemetry_state["pump_active"],

        "pump_command":
            control_state["pump_command"],

        "esp_online":
            telemetry_state["esp_online"],

        "last_update":
            telemetry_state["last_update"],

        "model":
            "Random Forest"
    }


# =========================================================
# START SERVER
# =========================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            "8000"
        )
    )


    uvicorn.run(
        app,
        host="0.0.0.0",
        port=port
    )
