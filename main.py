import time
import pickle
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, Security, HTTPException, Depends
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.security.api_key import APIKeyHeader
from pydantic import BaseModel, Field
from dotenv import load_dotenv
from logger import logger
import os

# ----------------------------
# Load environment variables
# ----------------------------
load_dotenv()
API_KEY = os.getenv("API_KEY")
API_KEY_NAME = "X-API-Key"

# ----------------------------
# API Key Security Scheme
# ----------------------------
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

async def verify_api_key(api_key: str = Security(api_key_header)):
    if api_key is None:
        logger.warning({"event": "auth_failed", "reason": "No API key provided"})
        raise HTTPException(
            status_code=401,
            detail="API Key required. Pass it as 'X-API-Key' in headers."
        )
    if api_key != API_KEY:
        logger.warning({"event": "auth_failed", "reason": "Invalid API key", "key_used": api_key})
        raise HTTPException(
            status_code=403,
            detail="Invalid API Key. Access denied."
        )
    logger.info({"event": "auth_success"})
    return api_key

# ----------------------------
# Load model and scaler
# ----------------------------
try:
    with open("models/model.pkl", "rb") as f:
        model = pickle.load(f)
    with open("models/scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
    logger.info("Wine model and scaler loaded successfully")

except FileNotFoundError as e:
    logger.error(f"Model file not found: {e}")
    raise RuntimeError("Model files missing. Run train_model.py first.")

# 🍷 Wine config
CLASS_MAPPING = {0: "Barbera", 1: "Grigio", 2: "Nebbiolo"}

WINE_FEATURE_NAMES = [
    'alcohol', 'malic_acid', 'ash', 'alcalinity_of_ash',
    'magnesium', 'total_phenols', 'flavanoids',
    'nonflavanoid_phenols', 'proanthocyanins', 'color_intensity',
    'hue', 'od280/od315_of_diluted_wines', 'proline'
]

# ----------------------------
# FastAPI App
# ----------------------------
app = FastAPI(
    title="Wine Classification API",
    description="Secure Wine cultivar classification API with API Key authentication.",
    version="3.0.0",
)

# ----------------------------
# Exception Handlers
# ----------------------------
@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    logger.warning({
        "event"  : "validation_error",
        "path"   : str(request.url),
        "errors" : exc.errors(),
    })
    return JSONResponse(
        status_code=422,
        content={"error": "Invalid input", "details": exc.errors()},
    )

@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error({
        "event" : "unhandled_exception",
        "path"  : str(request.url),
        "error" : str(exc),
    })
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "details": str(exc)},
    )

# ----------------------------
# Middleware: Log every request & response
# ----------------------------
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.time()
    logger.info({
        "event" : "incoming_request",
        "method": request.method,
        "path"  : str(request.url),
    })
    response = await call_next(request)
    duration = round(time.time() - start_time, 4)
    logger.info({
        "event"            : "request_completed",
        "method"           : request.method,
        "path"             : str(request.url),
        "status_code"      : response.status_code,
        "duration_seconds" : duration,
    })
    return response

# ----------------------------
# Pydantic Schemas
# ----------------------------
class PredictRequest(BaseModel):
    alcohol               : float = Field(..., example=14.23, description="Alcohol (%)")
    malic_acid            : float = Field(..., example=1.71,  description="Malic acid (g/L)")
    ash                   : float = Field(..., example=2.43,  description="Ash (g/L)")
    alcalinity_of_ash     : float = Field(..., example=15.6,  description="Alcalinity of ash")
    magnesium             : float = Field(..., example=127,   description="Magnesium (mg/L)")
    total_phenols         : float = Field(..., example=2.80,  description="Total phenols")
    flavanoids            : float = Field(..., example=3.06,  description="Flavanoids")
    nonflavanoid_phenols  : float = Field(..., example=0.28,  description="Nonflavanoid phenols")
    proanthocyanins       : float = Field(..., example=2.29,  description="Proanthocyanins")
    color_intensity       : float = Field(..., example=5.64,  description="Color intensity")
    hue                   : float = Field(..., example=1.04,  description="Hue")
    od280_od315_of_diluted_wines: float = Field(..., example=3.92, description="OD280/OD315")
    proline               : float = Field(..., example=1065,  description="Proline (mg/L)")

class PredictResponse(BaseModel):
    predicted_class: int = Field(..., example=0)
    predicted_label: str = Field(..., example="Barbera")

# ----------------------------
# Endpoints
# ----------------------------

# ✅ Public endpoint — no auth needed
@app.get("/", tags=["Health"])
def root():
    logger.info({"event": "health_check"})
    return {"status": "ok", "message": "Wine Classification API is running."}


# 🔒 Secured endpoint — requires API key
@app.post(
    "/predict",
    response_model=PredictResponse,
    tags=["Prediction"],
    summary="Classify wine cultivar (🔒 Secured)",
    description="Requires X-API-Key header. Returns predicted wine cultivar.",
    dependencies=[Depends(verify_api_key)]   # ← Auth applied here
)
def predict(body: PredictRequest):
    logger.info({"event": "predict_request", "input": body.dict()})

    features_array = np.array([[
        body.alcohol, body.malic_acid, body.ash,
        body.alcalinity_of_ash, body.magnesium,
        body.total_phenols, body.flavanoids,
        body.nonflavanoid_phenols, body.proanthocyanins,
        body.color_intensity, body.hue,
        body.od280_od315_of_diluted_wines, body.proline,
    ]])

    features_df     = pd.DataFrame(features_array, columns=WINE_FEATURE_NAMES)
    features_scaled = scaler.transform(features_df)
    pred_class      = int(model.predict(features_scaled)[0])
    pred_label      = CLASS_MAPPING.get(pred_class, "unknown")

    logger.info({
        "event"          : "predict_response",
        "predicted_class": pred_class,
        "predicted_label": pred_label,
    })

    return PredictResponse(predicted_class=pred_class, predicted_label=pred_label)
