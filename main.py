import argparse
import asyncio
import time
import os
import json
import uuid
import logging
import pandas as pd
from contextlib import asynccontextmanager
from typing import Optional, Any

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.encoders import jsonable_encoder
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, field_validator
import uvicorn
from dotenv import load_dotenv

# V1 Advanced Security Imports
from slowapi import Limiter
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

# Load Environment Variables from .env
load_dotenv()

from src.model import PhishingModel
from src.feature_assembler import feature_assembler 
from src.model_logic import HybridDecisionEngine
from src.explainer import explain_decision
from src.scraper import fetch_html_async
from src.semantic_extractor import extract_semantic_features
from src.reputation import check_threat_intelligence
from src.cache_manager import get_cached_prediction, set_cached_prediction
from src.utils import log_prediction

# Pydantic schemas
class PredictRequest(BaseModel):
    url: str

    @field_validator("url")
    @classmethod
    def validate_url(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("URL is required.")
        if not normalized.startswith(("http://", "https://")):
            normalized = f"http://{normalized}"
        from urllib.parse import urlparse
        parsed = urlparse(normalized)
        if not parsed.hostname:
            raise ValueError("Invalid URL format.")
        return normalized


def success_response(data: Any, request_id: Optional[str] = None) -> dict:
    return {
        "status": "success",
        "data": data,
        "meta": {"request_id": request_id} if request_id else {}
    }


def error_response(code: str, message: str, details: Any = None, request_id: Optional[str] = None) -> dict:
    response = {
        "status": "error",
        "error": {
            "code": code,
            "message": message,
            "details": details
        },
        "meta": {"request_id": request_id} if request_id else {}
    }
    return response

# -----------------------------------------------------
# CORE HYBRID ENGINE (Intact from V0)
# -----------------------------------------------------
class HybridDecision:
    def __init__(self):
        self.ml_model = PhishingModel()
        try:
            self.ml_model.load()
        except:
            print("Warning: Model not found. Run 'python main.py train' first.")
        self.decision_engine = HybridDecisionEngine(self.ml_model)

    async def _deep_scan_background(self, url: str, fast_result: dict, features_dict: dict, features_df: pd.DataFrame):
        if get_cached_prediction(url) and get_cached_prediction(url).get("analysis_depth") == "deep_scan_cached":
            return
            
        semantic_features = {}
        reputation = {"blacklisted": False, "threat_intel_confidence": 0.0}
        
        html_task = asyncio.create_task(fetch_html_async(url))
        reputation_task = asyncio.create_task(check_threat_intelligence(url))
        
        results = await asyncio.gather(html_task, reputation_task, return_exceptions=True)
        html_content = results[0] if not isinstance(results[0], Exception) and results[0] else ""
        rep_result = results[1] if not isinstance(results[1], Exception) and results[1] else None
        
        if rep_result: reputation = rep_result
        if html_content: semantic_features = extract_semantic_features(html_content, url)
            
        heuristic_signal = 0.5
        decision_reason = "Hybrid ML Evaluation."
        if reputation["blacklisted"]:
            heuristic_signal = 1.0
            decision_reason += " | Threat Intel Blacklisted."
        elif semantic_features.get('brand_spoofing_flag') == 1:
            heuristic_signal = 0.95
            decision_reason += " | Semantic Label Brand Mismatch."
        elif semantic_features.get('suspicious_form_action') == 1:
            heuristic_signal = 0.90
            decision_reason += " | Malicious Password Action Array."

        dec_payload = self.decision_engine.evaluate(features_df, heuristic_signal)
        confidence = dec_payload["final_confidence"]

        if confidence > 0.75: verdict = "PHISHING (Red)"
        elif confidence >= 0.40: verdict = "SUSPICIOUS (Yellow)"
        else: verdict = "SAFE (Green)"
            
        cached_result = {
            "url": url,
            "final_verdict": verdict,
            "confidence_score": round(confidence, 4),
            "degraded_mode": False,
            "cache_hit": True,
            "processing_time_ms": 0.0,
            "analysis_depth": "deep_scan_cached",
            "decision_metadata": {
                "verdict_source": "Hybrid ML Engine",
                "ml_raw_score": round(dec_payload["ml_raw_score"], 4),
                "heuristic_signal": heuristic_signal,
                "top_contributing_features": fast_result["decision_metadata"]["top_contributing_features"]
            },
            "decision_reason": decision_reason,
            "raw_features": {"fused_vector": features_dict, "semantic": semantic_features, "reputation": reputation}
        }
        set_cached_prediction(url, cached_result)

    async def predict_with_fast_path(self, url: str, background_tasks: BackgroundTasks = None) -> dict:
        start_time = time.time()
        
        cached_result = get_cached_prediction(url)
        if cached_result is not None:
            cached_result["processing_time_ms"] = round((time.time() - start_time) * 1000, 2)
            log_prediction(url, cached_result["final_verdict"], cached_result["confidence_score"])
            return cached_result
            
        features_dict = feature_assembler.assemble(url)
        features_df = pd.DataFrame([features_dict])
        
        top_contribs = explain_decision(self.ml_model, features_dict)
        dec_payload = self.decision_engine.evaluate(features_df, heuristic_signal=None)
        confidence = dec_payload["final_confidence"]
        
        # Fast Path Hard Override Trigger
        decision_reason = "Lexical base prediction confident enough to skip network dependencies."
        if dec_payload.get("hard_override_reason"):
            decision_reason = dec_payload["hard_override_reason"]
            
        if confidence > 0.85 or confidence < 0.15:
            if confidence > 0.75: verdict = "PHISHING (Red)"
            elif confidence >= 0.40: verdict = "SUSPICIOUS (Yellow)"
            else: verdict = "SAFE (Green)"
                
            result = {
                "url": url,
                "final_verdict": verdict,
                "confidence_score": round(confidence, 4),
                "degraded_mode": False,
                "cache_hit": False,
                "analysis_depth": "lexical_fast_path",
                "decision_metadata": {
                    "verdict_source": "Hybrid ML Engine",
                    "ml_raw_score": round(dec_payload["ml_raw_score"], 4),
                    "heuristic_signal": dec_payload["heuristic_signal"],
                    "top_contributing_features": top_contribs
                },
                "decision_reason": decision_reason,
                "raw_features": {"fused_vector": features_dict},
                "processing_time_ms": round((time.time() - start_time) * 1000, 2)
            }
            if background_tasks is not None:
                background_tasks.add_task(self._deep_scan_background, url, result, features_dict, features_df)
            log_prediction(url, result["final_verdict"], confidence)
            return result
            
        html_task = asyncio.create_task(fetch_html_async(url))
        reputation_task = asyncio.create_task(check_threat_intelligence(url))
        
        done, pending = await asyncio.wait([html_task, reputation_task], timeout=0.250, return_when=asyncio.ALL_COMPLETED)
        for p in pending: p.cancel()
            
        html_content = ""
        reputation = {"blacklisted": False, "threat_intel_confidence": 0.0}
        degraded = False
        
        if html_task in done and not html_task.exception():
            content = html_task.result()
            html_content = content if content else ""
        else: degraded = True
            
        if reputation_task in done and not reputation_task.exception():
            rep = reputation_task.result()
            if rep: reputation = rep
        else: degraded = True

        semantic_features = {}
        if html_content:
            semantic_features = extract_semantic_features(html_content, url)
            
        heuristic_signal = 0.5
        decision_reason_deep = "Hybrid Soft-Voting Score."
        if reputation["blacklisted"]:
            heuristic_signal = 1.0
            decision_reason_deep += " | Threat Intel Blacklisted."
        elif semantic_features.get('brand_spoofing_flag') == 1:
            heuristic_signal = 0.95
            decision_reason_deep += " | Semantic Label Brand Mismatch."
        elif semantic_features.get('suspicious_form_action') == 1:
            heuristic_signal = 0.90
            decision_reason_deep += " | Malicious Password Action Array."
        
        if degraded:
            heuristic_signal = 0.5
            decision_reason_deep = "Degraded Mode: Target timed out. Pure ML Fallback inference."
            
        dec_payload_deep = self.decision_engine.evaluate(features_df, heuristic_signal)
        confidence = dec_payload_deep["final_confidence"]
        
        # Deep Scan Hard Override Trigger
        if dec_payload_deep.get("hard_override_reason"):
            decision_reason_deep = dec_payload_deep["hard_override_reason"]
            
        if confidence > 0.75: verdict = "PHISHING (Red)"
        elif confidence >= 0.40: verdict = "SUSPICIOUS (Yellow)"
        else: verdict = "SAFE (Green)"
            
        result = {
            "url": url,
            "final_verdict": verdict,
            "confidence_score": round(confidence, 4),
            "degraded_mode": degraded,
            "cache_hit": False,
            "processing_time_ms": round((time.time() - start_time) * 1000, 2),
            "analysis_depth": "deep_scan_live" if not degraded else "degraded_fast_path",
            "decision_metadata": {
                "verdict_source": "Hybrid ML Engine",
                "ml_raw_score": round(dec_payload_deep["ml_raw_score"], 4),
                "heuristic_signal": heuristic_signal,
                "top_contributing_features": top_contribs
            },
            "decision_reason": decision_reason_deep,
            "raw_features": {"fused_vector": features_dict, "semantic": semantic_features, "reputation": reputation}
        }
        
        set_cached_prediction(url, result)
        log_prediction(url, verdict, confidence)
        return result

hybrid_decision = HybridDecision()

# -----------------------------------------------------
# FASTAPI V1 DEPLOYMENT SETUP (UI, LIMITERS, HEALTH)
# -----------------------------------------------------
from starlette.middleware.base import BaseHTTPMiddleware
logger = logging.getLogger("phishing_api")
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s"
)

class ForceJsonContentTypeMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.method in ["POST", "PUT", "PATCH"] and request.url.path == "/predict":
            content_type = request.headers.get("content-type", "")
            if content_type == "application/x-www-form-urlencoded" or not content_type:
                # Dynamically rewrite the header so FastAPI's strict JSON parser 
                # engages instead of the Form parser, saving the user from 422 errors.
                headers = dict(request.scope["headers"])
                headers[b"content-type"] = b"application/json"
                request.scope["headers"] = [(k, v) for k, v in headers.items()]
        return await call_next(request)

class RequestContextMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        request_id = request.headers.get("x-request-id", str(uuid.uuid4()))
        request.state.request_id = request_id
        start = time.time()
        logger.info(json.dumps({
            "event": "request.start",
            "path": request.url.path,
            "method": request.method,
            "request_id": request_id
        }))
        response = await call_next(request)
        duration_ms = round((time.time() - start) * 1000, 2)
        response.headers["X-Request-ID"] = request_id
        logger.info(json.dumps({
            "event": "request.end",
            "path": request.url.path,
            "method": request.method,
            "status_code": response.status_code,
            "duration_ms": duration_ms,
            "request_id": request_id
        }))
        return response

from fastapi.middleware.cors import CORSMiddleware
limiter = Limiter(key_func=get_remote_address)
app = FastAPI(title="Real-Time Phishing Detection System V1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(ForceJsonContentTypeMiddleware)
app.add_middleware(RequestContextMiddleware)
app.state.limiter = limiter


@app.exception_handler(RateLimitExceeded)
async def rate_limit_exception_handler(request: Request, exc: RateLimitExceeded):
    return JSONResponse(
        status_code=429,
        content=error_response(
            code="RATE_LIMIT_EXCEEDED",
            message="Too many requests. Please retry later.",
            details={"limit": str(exc.detail)},
            request_id=getattr(request.state, "request_id", None)
        )
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """
    Graceful fallback: Detects if the user sent a valid JSON payload but 
    forgot the 'Content-Type: application/json' header. Intercepts the 422 
    error and processes the prediction immediately.
    """
    try:
        body = await request.body()
        if body.startswith(b"{") and body.endswith(b"}"):
            data = json.loads(body.decode('utf-8'))
            if "url" in data:
                bg_tasks = BackgroundTasks()
                payload = PredictRequest(url=data["url"])
                result = await hybrid_decision.predict_with_fast_path(payload.url, bg_tasks)
                return JSONResponse(
                    status_code=200,
                    content=success_response(result, getattr(request.state, "request_id", None)),
                    background=bg_tasks
                )
    except Exception:
        pass

    return JSONResponse(
        status_code=422,
        content=error_response(
            code="VALIDATION_ERROR",
            message="Invalid request payload.",
            details=jsonable_encoder(exc.errors()),
            request_id=getattr(request.state, "request_id", None)
        )
    )


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response(
            code="HTTP_ERROR",
            message=str(exc.detail),
            details=None,
            request_id=getattr(request.state, "request_id", None)
        )
    )


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.exception("unhandled_exception", extra={"request_id": getattr(request.state, "request_id", None)})
    return JSONResponse(
        status_code=500,
        content=error_response(
            code="INTERNAL_SERVER_ERROR",
            message="An unexpected server error occurred.",
            details=None,
            request_id=getattr(request.state, "request_id", None)
        )
    )

os.makedirs("static", exist_ok=True)
os.makedirs("templates", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/health")
async def health_check():
    """Liveness probe for Kubernetes & Docker orchestration."""
    payload = {"service_status": "healthy", "service": "Phishing-Detector-V1", "timestamp": time.time()}
    return success_response(payload)

@app.get("/", response_class=HTMLResponse)
@limiter.limit(os.getenv("RATE_LIMIT", "60/minute"))
async def serve_ui(request: Request):
    """Serves the standalone Glassmorphism Web UI."""
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
@limiter.limit(os.getenv("RATE_LIMIT", "60/minute"))
async def predict_endpoint(request: Request, payload: PredictRequest, background_tasks: BackgroundTasks):
    """JSON ML Inference Endpoint restricted by SlowAPI logic."""
    result = await hybrid_decision.predict_with_fast_path(payload.url, background_tasks)
    return success_response(result, getattr(request.state, "request_id", None))


# -----------------------------------------------------
# CLI EXECUTORS
# -----------------------------------------------------
def train_pipeline(data_path: str):
    """Train a phishing model and print core evaluation metrics."""
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    X = feature_assembler.assemble_batch(df['url'])
    y = df['label']
    
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    from src.model import PhishingModel
    model = PhishingModel()
    model.train(X_train, y_train)
    y_pred = model.predict(X_test)
    print("\n--- Test Set ---")
    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_test, y_pred, zero_division=0):.4f}")
    print(f"F1-Score: {f1_score(y_test, y_pred):.4f}")
    model.save('models/model.pkl')

def inference(url: str):
    import asyncio
    print(f"\nAnalyzing URL: {url}")
    result = asyncio.run(hybrid_decision.predict_with_fast_path(url, None))
    print(f"Verdict: {result['final_verdict'].upper()} | Confidence: {result['confidence_score']}")
    print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time Phishing Website Detection System (V1 Hybrid)")
    parser.add_argument("mode", choices=['train', 'serve', 'predict'], help="Mode to run the pipeline")
    parser.add_argument("--data", help="Path to training CSV (required for 'train')")
    parser.add_argument("--url", help="URL to predict (required for 'predict')")
    args = parser.parse_args()
    
    host = os.getenv("API_HOST", "0.0.0.0")
    port = int(os.getenv("API_PORT", 8000))
    
    if args.mode == 'train':
        train_pipeline(args.data)
    elif args.mode == 'serve':
        print(f"Starting FastAPI Hybrid ML Server on {host}:{port}")
        uvicorn.run(app, host=host, port=port)
    elif args.mode == 'predict':
        inference(args.url)
