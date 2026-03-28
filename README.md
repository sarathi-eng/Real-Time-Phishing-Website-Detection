# Real-Time Phishing Website Detection

![CI](https://github.com/sarathi-eng/Real-Time-Phishing-Website-Detection/actions/workflows/ci.yml/badge.svg)

Production-polished phishing detection service built with **FastAPI** and a **hybrid ML + rule-based engine**.

## Project Overview

This project classifies URLs as:
- `SAFE (Green)`
- `SUSPICIOUS (Yellow)`
- `PHISHING (Red)`

The system combines:
1. **Lexical/contextual ML features** (URL and domain patterns)
2. **Rule signals** (safelist + typosquatting overrides)
3. **Optional deep scan inputs** (semantic HTML checks + threat intel simulation)

This hybrid strategy keeps latency low while preserving high-risk override protections.

## Architecture

```
Client -> FastAPI /predict
       -> Feature Assembler (lexical + contextual + typo/safelist features)
       -> Hybrid Decision Engine
       -> [Optional async deep scan: HTML + threat intel]
       -> Standardized JSON response
       -> Structured logging + cache layer
```

### System Design Highlights

- **Fast path first**: low-latency lexical scoring for confident predictions
- **Deep scan fallback**: asynchronous semantic + reputation checks for uncertain cases
- **Decision governance**: hard overrides for high-confidence typosquatting and safelist protection
- **Caching strategy**: separate TTL tiers for deep vs degraded/fast-path outputs
- **Production API standards**: request validation, consistent response schema, request IDs

## Folder Structure

```
.
├── .github/workflows/ci.yml
├── data/
├── docs/
│   └── metrics.json
├── models/
├── scripts/
│   └── evaluate_model.py
├── src/
├── tests/
│   ├── integration/
│   └── unit/
├── main.py
├── requirements.txt
└── Dockerfile
```

## API Usage

### Health

```bash
curl -X GET http://localhost:8000/health
```

Example response:

```json
{
  "status": "success",
  "data": {
    "service_status": "healthy",
    "service": "Phishing-Detector-V1",
    "timestamp": 1710000000.0
  },
  "meta": {}
}
```

### Predict

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -H "X-Request-ID: demo-req-1" \
  -d '{"url": "https://example.com/login"}'
```

Example response:

```json
{
  "status": "success",
  "data": {
    "url": "https://example.com/login",
    "final_verdict": "SUSPICIOUS (Yellow)",
    "confidence_score": 0.53,
    "degraded_mode": false,
    "cache_hit": false,
    "processing_time_ms": 64.31,
    "analysis_depth": "deep_scan_live",
    "decision_metadata": {
      "verdict_source": "Hybrid ML Engine",
      "ml_raw_score": 0.47,
      "heuristic_signal": 0.7,
      "top_contributing_features": []
    },
    "decision_reason": "Hybrid Soft-Voting Score.",
    "raw_features": {}
  },
  "meta": {
    "request_id": "demo-req-1"
  }
}
```

Validation error response:

```json
{
  "status": "error",
  "error": {
    "code": "VALIDATION_ERROR",
    "message": "Invalid request payload.",
    "details": []
  },
  "meta": {
    "request_id": "..."
  }
}
```

## Metrics & Evaluation

Run evaluation:

```bash
python scripts/evaluate_model.py \
  --data data/dataset.csv \
  --test-size 0.2 \
  --random-state 42 \
  --cv-folds 5 \
  --output docs/metrics.json
```

Metrics are computed on a **held-out test set (no leakage)**:
- stratified split (`train_test_split(..., stratify=y)`)
- duplicate URL removal before splitting
- explicit train/test URL overlap check (`0` required)
- label-derived feature name guard in evaluation
- optional 5-fold stratified CV with mean/std reporting

Latest held-out metrics (`docs/metrics.json`):

- Accuracy: **1.0000**
- Precision: **1.0000**
- Recall: **1.0000**
- F1: **1.0000**

> Note: this dataset is small and highly separable, so perfect metrics can still occur even with leakage-safe evaluation.

## Testing

Run all tests:

```bash
python -m pytest -q
```

Current suite includes:
- Unit tests for feature assembly and decision logic
- Integration tests for FastAPI response contracts and validation behavior

## CI/CD

GitHub Actions workflow (`.github/workflows/ci.yml`) runs on every push and pull request:
- dependency install
- unit + integration tests
- metrics evaluation script execution

## Docker

Build and run:

```bash
docker build -t phishing-detector .
docker run -p 8000:8000 phishing-detector
```

## Suggested Commit Strategy (Logical History)

Use small, reviewable commits grouped by concern:

1. `test: add unit tests for feature assembly and decision logic`
2. `test: add FastAPI integration tests for response schema`
3. `feat(api): standardize success/error response contracts`
4. `feat(api): add stricter URL validation and request-id propagation`
5. `feat(obs): improve structured request logging`
6. `ci: add GitHub Actions workflow for tests and evaluation`
7. `chore: harden gitignore/dockerignore and dependency pins`
8. `docs: rewrite README with architecture, usage, and metrics`

## Development Commands

```bash
# Run API
python main.py serve

# Train model
python main.py train --data data/dataset.csv

# One-off prediction
python main.py predict --url "https://example.com"
```
