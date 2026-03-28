import argparse
import json
from pathlib import Path
import sys

import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.feature_assembler import feature_assembler
from src.model import PhishingModel


def evaluate(data_path: str, test_size: float = 0.2, random_state: int = 42) -> dict:
    dataset = pd.read_csv(data_path)
    if "url" not in dataset.columns or "label" not in dataset.columns:
        raise ValueError("Dataset must contain 'url' and 'label' columns.")

    features = feature_assembler.assemble_batch(dataset["url"].tolist())
    labels = dataset["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    model = PhishingModel()
    model.train(X_train, y_train)
    predictions = model.predict(X_test)

    metrics = {
        "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
        "precision": round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        "samples_total": int(len(dataset)),
        "samples_test": int(len(y_test)),
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate phishing model metrics.")
    parser.add_argument("--data", default="data/dataset.csv", help="Dataset CSV path")
    parser.add_argument(
        "--output",
        default="docs/metrics.json",
        help="Path to write computed metrics as JSON",
    )
    args = parser.parse_args()

    metrics = evaluate(args.data)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
