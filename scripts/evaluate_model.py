import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import StratifiedKFold, train_test_split

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.feature_assembler import feature_assembler
from src.model import PhishingModel


def _prepare_dataset(data_path: str) -> pd.DataFrame:
    dataset = pd.read_csv(data_path)
    if "url" not in dataset.columns or "label" not in dataset.columns:
        raise ValueError("Dataset must contain 'url' and 'label' columns.")
    dataset = dataset[["url", "label"]].copy()
    dataset["url"] = dataset["url"].astype(str).str.strip()
    dataset["label"] = dataset["label"].astype(int)
    dataset = dataset[(dataset["url"] != "") & (dataset["label"].isin([0, 1]))]

    conflicting = dataset.groupby("url")["label"].nunique()
    conflict_urls = conflicting[conflicting > 1]
    if not conflict_urls.empty:
        raise ValueError(
            f"Found identical URLs with conflicting labels, which is a leakage/bias source: {list(conflict_urls.index)[:5]}"
        )

    # Leakage guard: deduplicate exact URLs so the same sample cannot land in both train and test sets.
    dataset = dataset.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)
    if dataset["label"].nunique() < 2:
        raise ValueError("Evaluation requires at least two classes.")
    return dataset


def _prepare_external_dataset(data_path: str, train_urls: set[str], dataset_name: str) -> tuple[pd.DataFrame, dict]:
    external = _prepare_dataset(data_path)
    overlap_mask = external["url"].isin(train_urls)
    removed_overlap = int(overlap_mask.sum())
    external = external.loc[~overlap_mask].reset_index(drop=True)
    if external.empty:
        raise ValueError(
            f"External dataset '{dataset_name}' has no unseen URLs after removing overlaps with training set."
        )
    if external["label"].nunique() < 2:
        raise ValueError(f"External dataset '{dataset_name}' must contain both classes after overlap filtering.")
    return external, {
        "source_path": data_path,
        "samples_total_after_filter": int(len(external)),
        "removed_overlap_with_train": removed_overlap,
        "class_distribution": {str(k): int(v) for k, v in external["label"].value_counts().to_dict().items()},
    }


def _extract_features(urls: list[str]) -> pd.DataFrame:
    features = feature_assembler.assemble_batch(urls)
    forbidden_columns = {"label", "target", "ground_truth", "is_phishing", "y"}
    leaked_cols = [column for column in features.columns if column.lower() in forbidden_columns]
    if leaked_cols:
        raise ValueError(f"Feature leakage detected. Label-derived columns present: {leaked_cols}")
    return features


def _compute_metrics(y_true: pd.Series, y_pred: np.ndarray) -> dict:
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    return {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(precision_score(y_true, y_pred, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, y_pred, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "confusion_matrix": {
            "tn": int(cm[0][0]),
            "fp": int(cm[0][1]),
            "fn": int(cm[1][0]),
            "tp": int(cm[1][1]),
        },
    }


def _cross_validate(features: pd.DataFrame, labels: pd.Series, random_state: int, folds: int = 5) -> dict:
    folds = max(2, min(folds, int(labels.value_counts().min())))
    cv = StratifiedKFold(n_splits=folds, shuffle=True, random_state=random_state)
    scores = {"accuracy": [], "precision": [], "recall": [], "f1": []}

    for train_index, test_index in cv.split(features, labels):
        X_train, X_test = features.iloc[train_index], features.iloc[test_index]
        y_train, y_test = labels.iloc[train_index], labels.iloc[test_index]
        model = PhishingModel()
        model.train(X_train, y_train)
        pred = model.predict(X_test)
        fold_metrics = _compute_metrics(y_test, pred)
        for metric_name in scores:
            scores[metric_name].append(fold_metrics[metric_name])

    return {
        "folds": folds,
        "mean": {k: round(float(np.mean(v)), 4) for k, v in scores.items()},
        "std": {k: round(float(np.std(v)), 4) for k, v in scores.items()},
    }


def _evaluate_external_set(model: PhishingModel, dataset: pd.DataFrame) -> dict:
    external_features = _extract_features(dataset["url"].tolist())
    external_pred = model.predict(external_features)
    return _compute_metrics(dataset["label"], external_pred)


def evaluate(
    data_path: str,
    test_size: float = 0.2,
    random_state: int = 42,
    cross_validate: bool = True,
    cv_folds: int = 5,
    real_world_data_path: str | None = None,
    adversarial_data_path: str | None = None,
) -> dict:
    dataset = _prepare_dataset(data_path)
    features = _extract_features(dataset["url"].tolist())
    labels = dataset["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=test_size,
        random_state=random_state,
        stratify=labels,
    )

    # Mandatory no-overlap check using canonical URL strings.
    train_urls, test_urls = set(dataset.loc[X_train.index, "url"]), set(dataset.loc[X_test.index, "url"])
    overlap = train_urls.intersection(test_urls)
    if overlap:
        raise RuntimeError(f"Data leakage detected: {len(overlap)} URLs overlap between train and test splits.")

    model = PhishingModel()
    model.train(X_train, y_train)
    predictions = model.predict(X_test)

    prediction_distribution = pd.Series(predictions).value_counts().to_dict()
    if len(prediction_distribution) == 1:
        raise RuntimeError("Sanity check failed: model produced constant predictions on test set.")

    metrics = _compute_metrics(y_test, predictions)
    report = {
        "held_out_test_metrics": metrics,
        "sanity_checks": {
            "class_distribution_total": {str(k): int(v) for k, v in labels.value_counts().to_dict().items()},
            "class_distribution_train": {str(k): int(v) for k, v in y_train.value_counts().to_dict().items()},
            "class_distribution_test": {str(k): int(v) for k, v in y_test.value_counts().to_dict().items()},
            "prediction_distribution_test": {str(k): int(v) for k, v in prediction_distribution.items()},
            "train_test_overlap_urls": int(len(overlap)),
        },
        "split": {
            "strategy": "stratified_train_test_split",
            "test_size": test_size,
            "random_state": random_state,
            "samples_total": int(len(dataset)),
            "samples_train": int(len(y_train)),
            "samples_test": int(len(y_test)),
        },
        "leakage_guards": {
            "dropped_duplicate_urls": True,
            "conflicting_duplicate_labels_blocked": True,
            "label_derived_features_blocked": True,
        },
    }

    train_url_set = set(dataset.loc[X_train.index, "url"])
    challenge_sets: dict[str, dict] = {}
    challenge_metadata: dict[str, dict] = {}

    if real_world_data_path:
        real_world_dataset, meta = _prepare_external_dataset(
            data_path=real_world_data_path,
            train_urls=train_url_set,
            dataset_name="real_world",
        )
        challenge_sets["real_world_style_metrics"] = _evaluate_external_set(model, real_world_dataset)
        challenge_metadata["real_world"] = meta

    if adversarial_data_path:
        adversarial_dataset, meta = _prepare_external_dataset(
            data_path=adversarial_data_path,
            train_urls=train_url_set,
            dataset_name="adversarial",
        )
        challenge_sets["adversarial_metrics"] = _evaluate_external_set(model, adversarial_dataset)
        challenge_metadata["adversarial"] = meta

    if challenge_sets:
        report["external_challenge_metrics"] = challenge_sets
        report["external_challenge_metadata"] = challenge_metadata
        report["difficulty_gap_vs_held_out"] = {
            name: {
                "delta_accuracy": round(challenge["accuracy"] - metrics["accuracy"], 4),
                "delta_precision": round(challenge["precision"] - metrics["precision"], 4),
                "delta_recall": round(challenge["recall"] - metrics["recall"], 4),
                "delta_f1": round(challenge["f1"] - metrics["f1"], 4),
            }
            for name, challenge in challenge_sets.items()
        }

    if cross_validate:
        report["cross_validation"] = _cross_validate(
            features=features,
            labels=labels,
            random_state=random_state,
            folds=cv_folds,
        )

    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate phishing model metrics.")
    parser.add_argument("--data", default="data/dataset.csv", help="Dataset CSV path")
    parser.add_argument("--test-size", type=float, default=0.2, help="Held-out split ratio")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for reproducibility")
    parser.add_argument("--no-cross-validate", action="store_true", help="Disable 5-fold cross-validation")
    parser.add_argument("--cv-folds", type=int, default=5, help="Number of CV folds when enabled")
    parser.add_argument(
        "--real-world-data",
        default="data/real_world_eval.csv",
        help="CSV path for unseen real-world style evaluation URLs",
    )
    parser.add_argument(
        "--adversarial-data",
        default="data/adversarial_eval.csv",
        help="CSV path for adversarial evaluation URLs",
    )
    parser.add_argument(
        "--skip-external",
        action="store_true",
        help="Skip external real-world/adversarial evaluation layers",
    )
    parser.add_argument(
        "--output",
        default="docs/metrics.json",
        help="Path to write computed metrics as JSON",
    )
    args = parser.parse_args()

    metrics = evaluate(
        data_path=args.data,
        test_size=args.test_size,
        random_state=args.random_state,
        cross_validate=not args.no_cross_validate,
        cv_folds=args.cv_folds,
        real_world_data_path=None if args.skip_external else args.real_world_data,
        adversarial_data_path=None if args.skip_external else args.adversarial_data,
    )
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
