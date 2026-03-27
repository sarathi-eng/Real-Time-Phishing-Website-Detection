import os
import logging
from datetime import datetime

# Ensure the logs directory exists as per the Clean Engineering project structure
os.makedirs("logs", exist_ok=True)

# Configure the logger to append to predictions.log
logging.basicConfig(
    filename="logs/predictions.log",
    level=logging.INFO,
    format="%(message)s"  # We handle the exact formatting in the log_prediction function
)

def log_prediction(url: str, prediction: str, confidence: float = None):
    """
    Logs the prediction result to the centralized predictions.log.
    Format: [TIMESTAMP] URL | PREDICTION | CONFIDENCE
    """
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    conf_str = f" | Confidence: {confidence:.2f}" if confidence is not None else ""
    log_message = f"[{timestamp}] URL: {url} | Prediction: {prediction}{conf_str}"
    
    # Write to predictions.log
    logging.info(log_message)
    
    # We don't necessarily need to print to stdout here, as the main pipeline handles user-facing output,
    # but the log file is fully managed here.
