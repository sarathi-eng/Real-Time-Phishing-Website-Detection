import argparse
import asyncio
import time
import pandas as pd
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel
import uvicorn

from src.model import PhishingModel
from src.feature_assembler import feature_assembler 
from src.model_logic import HybridDecisionEngine
from src.explainer import explain_decision

# Import Hybrid Modules
from src.scraper import fetch_html_async
from src.semantic_extractor import extract_semantic_features
from src.reputation import check_threat_intelligence
from src.cache_manager import get_cached_prediction, set_cached_prediction
from src.utils import log_prediction

class PredictRequest(BaseModel):
    url: str

class HybridDecision:
    def __init__(self):
        self.ml_model = PhishingModel()
        try:
            self.ml_model.load()
        except:
            print("Warning: Model not found. Run 'python main.py train' first.")
        self.decision_engine = HybridDecisionEngine(self.ml_model)

    async def _deep_scan_background(self, url: str, fast_result: dict, features_dict: dict, features_df: pd.DataFrame):
        """Background task to hydrate cache with Deep Scan results using Soft-Voting Hybrid ML logic."""
        if get_cached_prediction(url) and get_cached_prediction(url).get("analysis_depth") == "deep_scan_cached":
            return
            
        semantic_features = {}
        reputation = {"blacklisted": False, "threat_intel_confidence": 0.0}
        
        html_task = asyncio.create_task(fetch_html_async(url))
        reputation_task = asyncio.create_task(check_threat_intelligence(url))
        
        results = await asyncio.gather(html_task, reputation_task, return_exceptions=True)
        html_content = results[0] if not isinstance(results[0], Exception) and results[0] else ""
        rep_result = results[1] if not isinstance(results[1], Exception) and results[1] else None
        
        if rep_result:
            reputation = rep_result
            
        if html_content:
            semantic_features = extract_semantic_features(html_content, url)
            
        # Calculate Heuristic Signal
        heuristic_signal = 0.5 # Neutral baseline
        decision_reason = "Hybrid ML Evaluation."
        if reputation["blacklisted"]:
            heuristic_signal = 1.0
            decision_reason += " | Strong Semantic Threat Intel (Blacklisted)."
        elif semantic_features.get('brand_spoofing_flag') == 1:
            heuristic_signal = 0.95
            decision_reason += " | Semantic Brand Page Mismatch."
        elif semantic_features.get('suspicious_form_action') == 1:
            heuristic_signal = 0.90
            decision_reason += " | Malicious Password Form Action."

        # Pass pure ML score + Heuristic Signal to the Engine Soft-Voter
        dec_payload = self.decision_engine.evaluate(features_df, heuristic_signal)
        confidence = dec_payload["final_confidence"]

        if confidence > 0.75:
            verdict = "PHISHING (Red)"
        elif confidence >= 0.40:
            verdict = "SUSPICIOUS (Yellow)"
        else:
            verdict = "SAFE (Green)"
            
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
            "raw_features": {
                "fused_vector": features_dict,
                "semantic": semantic_features,
                "reputation": reputation
            }
        }
        set_cached_prediction(url, cached_result)

    async def predict_with_fast_path(self, url: str, background_tasks: BackgroundTasks = None) -> dict:
        start_time = time.time()
        
        # 0. SRE Caching Layer
        cached_result = get_cached_prediction(url)
        if cached_result is not None:
            cached_result["processing_time_ms"] = round((time.time() - start_time) * 1000, 2)
            log_prediction(url, cached_result["final_verdict"], cached_result["confidence_score"])
            return cached_result
            
        # 1. Feature Fusion Assembly
        features_dict = feature_assembler.assemble(url)
        features_df = pd.DataFrame([features_dict])
        
        # 2. XAI Explainability (Top weighted ML Features)
        top_contribs = explain_decision(self.ml_model, features_dict)
        
        # 3. Base ML Evaluation (Soft-Voting pure ML)
        dec_payload = self.decision_engine.evaluate(features_df, heuristic_signal=None)
        confidence = dec_payload["final_confidence"]
        
        # 4. Confidence-Based Early Exit 
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
                "decision_reason": "Lexical base prediction confident enough to skip network dependencies.",
                "raw_features": {"fused_vector": features_dict},
                "processing_time_ms": round((time.time() - start_time) * 1000, 2)
            }
            
            if background_tasks is not None:
                background_tasks.add_task(self._deep_scan_background, url, result, features_dict, features_df)
            
            log_prediction(url, result["final_verdict"], confidence)
            return result
            
        # 5. Grey Area -> Deep Scan with Micro-Timeouts
        html_task = asyncio.create_task(fetch_html_async(url))
        reputation_task = asyncio.create_task(check_threat_intelligence(url))
        
        done, pending = await asyncio.wait(
            [html_task, reputation_task], 
            timeout=0.250, 
            return_when=asyncio.ALL_COMPLETED
        )
        
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
            
        # Calculate Heuristic Signal dynamically
        heuristic_signal = 0.5 # Neutral baseline
        decision_reason = "Hybrid Soft-Voting Score."
        if reputation["blacklisted"]:
            heuristic_signal = 1.0
            decision_reason += " | Threat Intel Blacklisted."
        elif semantic_features.get('brand_spoofing_flag') == 1:
            heuristic_signal = 0.95
            decision_reason += " | Semantic Label Brand Mismatch."
        elif semantic_features.get('suspicious_form_action') == 1:
            heuristic_signal = 0.90
            decision_reason += " | Malicious Password Action Array."
        
        if degraded:
            heuristic_signal = 0.5 # Default to neutral fallback
            decision_reason = "Degraded Mode: Target timed out. Pure ML Fallback inference."
            
        dec_payload_deep = self.decision_engine.evaluate(features_df, heuristic_signal)
        confidence = dec_payload_deep["final_confidence"]
            
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
            "decision_reason": decision_reason,
            "raw_features": {
                "fused_vector": features_dict,
                "semantic": semantic_features,
                "reputation": reputation
            }
        }
        
        set_cached_prediction(url, result)
        log_prediction(url, verdict, confidence)
        return result

hybrid_decision = HybridDecision()
app = FastAPI(title="Real-Time Phishing Detection System")

@app.post("/predict")
async def predict_endpoint(request: PredictRequest, background_tasks: BackgroundTasks):
    if not request.url:
        raise HTTPException(status_code=400, detail="URL is required.")
    result = await hybrid_decision.predict_with_fast_path(request.url, background_tasks)
    return result

def train_pipeline(data_path: str):
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    if 'url' not in df.columns or 'label' not in df.columns:
        raise ValueError("CSV must contain 'url' and 'label' columns.")
    
    print("Extracting fused architectural features into analytical dimensions...")
    X = feature_assembler.assemble_batch(df['url'])
    y = df['label']
    
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    from src.model import PhishingModel
    model = PhishingModel()
    model.train(X_train, y_train)
    
    y_pred = model.predict(X_test)
    print("\n--- Test Set Metrics ---")
    print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
    print(f"Precision: {precision_score(y_test, y_pred):.4f}")
    print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
    print(f"F1-Score:  {f1_score(y_test, y_pred):.4f}")
    
    import numpy as np
    classifiers = model.model.calibrated_classifiers_
    importances = np.mean([clf.estimator.feature_importances_ for clf in classifiers], axis=0)
    
    print("\n--- Model Feature Importances (XAI) ---")
    for name, imp in sorted(zip(X.columns, importances), key=lambda x: x[1], reverse=True):
        print(f"{name}: {imp*100:.2f}%")
        
    print("------------------------\n")
    
    model.save('models/model.pkl')
    print("Training complete! Feature Vectors scaled and saved to models/model.pkl")

def inference(url: str):
    import asyncio
    print(f"\nAnalyzing URL: {url}")
    # Hybrid Decision uses None for background_tasks in CLI Mode
    result = asyncio.run(hybrid_decision.predict_with_fast_path(url, None))
    
    print("-" * 30)
    print(f"Verdict: {result['final_verdict'].upper()}")
    print(f"Confidence: {result['confidence_score']}")
    print(f"Engine Math: {result['decision_reason']} (ML Raw: {result['decision_metadata']['ml_raw_score']} | Heuristic: {result['decision_metadata']['heuristic_signal']})")
    print("--- XAI Top Math Weights ---")
    for tr in result['decision_metadata']['top_contributing_features']:
        print(tr)
    print(f"Depth: {result['analysis_depth']} | Hit: {result['cache_hit']} | Time: {result['processing_time_ms']}ms")
    print("-" * 30)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Real-Time Phishing Website Detection System (Hybrid)")
    parser.add_argument("mode", choices=['train', 'serve', 'predict'], help="Mode to run the pipeline")
    parser.add_argument("--data", help="Path to training CSV (required for 'train')")
    parser.add_argument("--url", help="URL to predict (required for 'predict')")
    parser.add_argument("--host", default="0.0.0.0", help="API Host (default 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="API Port (default 8000)")
    
    args = parser.parse_args()
    
    if args.mode == 'train':
        if not args.data:
            parser.error("--data is required in 'train' mode.")
        train_pipeline(args.data)
    elif args.mode == 'serve':
        print(f"Starting FastAPI Hybrid ML Server on {args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)
    elif args.mode == 'predict':
        if not args.url:
            parser.error("--url is required in 'predict' mode.")
        inference(args.url)
