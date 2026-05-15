# machine_learning.py — QuantEdge MT5: Model Training, Validation & Inference
# RULES:
#   - NO future data in training (strict temporal split — never shuffle time series)
#   - Walk-forward validation ONLY — standard K-Fold is FORBIDDEN
#   - All models saved to D:\Trading\models\ with timestamp in filename

from datetime import datetime
from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from config import (
    ML_FORWARD_PERIODS,
    ML_MODEL_TYPE,
    ML_PRICE_THRESHOLD,
    ML_TEST_SIZE_RATIO,
    ML_WALK_FORWARD_SPLITS,
    MODEL_DIR,
)
from logger_config import setup_logger

logger = setup_logger(__name__)


# ---------------------------------------------------------------------------
# Label Engineering (Temporal — no lookahead)
# ---------------------------------------------------------------------------

def prepare_labels(
    df: pd.DataFrame,
    forward_periods: int = ML_FORWARD_PERIODS,
    threshold: float = ML_PRICE_THRESHOLD,
) -> pd.Series:
    """
    Create classification labels based on future price movement.

    Labels:
         1 = LONG  (future return > +threshold)
        -1 = SHORT (future return < -threshold)
         0 = FLAT  (return within ±threshold)

    IMPORTANT: Labels use FUTURE data — drop last `forward_periods` rows before
    training to avoid lookahead bias.

    Args:
        df:              Feature DataFrame with 'close' column.
        forward_periods: Bars to look ahead for labeling. Default: 5.
        threshold:       Minimum return to label as signal. Default: 0.001 (0.1%).

    Returns:
        Integer label Series aligned to df.index. Last `forward_periods` rows are NaN.
    """
    future_return = df["close"].shift(-forward_periods) / df["close"] - 1

    labels = pd.Series(0, index=df.index, dtype=int)
    labels[future_return >  threshold] =  1
    labels[future_return < -threshold] = -1
    labels[future_return.isna()]       = np.nan  # Drop boundary rows

    logger.info(
        f"Labels prepared: {(labels == 1).sum()} LONG, "
        f"{(labels == -1).sum()} SHORT, "
        f"{(labels == 0).sum()} FLAT"
    )
    return labels


# ---------------------------------------------------------------------------
# Model Factory
# ---------------------------------------------------------------------------

def _build_model(model_type: str) -> BaseEstimator:
    """Instantiate an untrained sklearn estimator by name."""
    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=300,
            max_depth=6,
            min_samples_leaf=20,
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        ),
        "gradient_boost": GradientBoostingClassifier(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            subsample=0.8,
            random_state=42,
        ),
        "svm": SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            probability=True,
            random_state=42,
        ),
    }
    if model_type not in models:
        raise ValueError(
            f"Unknown model_type '{model_type}'. "
            f"Choose from: {list(models.keys())}"
        )
    return models[model_type]


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = ML_MODEL_TYPE,
    scale: bool = True,
) -> Tuple[BaseEstimator, StandardScaler]:
    """
    Train a classification model on feature matrix X and labels y.

    CRITICAL: X and y must be chronologically ordered. DO NOT shuffle.

    Args:
        X:          Feature DataFrame (from build_feature_matrix, labels removed).
        y:          Label Series from prepare_labels().
        model_type: One of "random_forest", "gradient_boost", "svm".
        scale:      Whether to standardize features. Required for SVM.

    Returns:
        Tuple of (fitted_model, fitted_scaler). Scaler is identity if scale=False.
    """
    # Drop rows where label is NaN (boundary rows from future lookahead)
    valid_mask = y.notna()
    X_clean = X[valid_mask]
    y_clean = y[valid_mask].astype(int)

    if len(X_clean) < 100:
        raise ValueError(f"Insufficient training data: {len(X_clean)} rows (need ≥ 100).")

    # Strict temporal split — NO SHUFFLE
    split_idx = int(len(X_clean) * (1 - ML_TEST_SIZE_RATIO))
    X_train, X_test = X_clean.iloc[:split_idx], X_clean.iloc[split_idx:]
    y_train, y_test = y_clean.iloc[:split_idx], y_clean.iloc[split_idx:]

    logger.info(
        f"Training {model_type} | Train: {len(X_train)} | Test: {len(X_test)} | "
        f"Features: {X_train.shape[1]}"
    )

    # Scaling
    scaler = StandardScaler()
    if scale:
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)
    else:
        X_train_s = X_train.values
        X_test_s  = X_test.values

    model = _build_model(model_type)
    model.fit(X_train_s, y_train)

    # Evaluation on hold-out test set
    y_pred = model.predict(X_test_s)
    acc    = accuracy_score(y_test, y_pred)
    f1     = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    prec   = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec    = recall_score(y_test, y_pred, average="weighted", zero_division=0)

    logger.info(
        f"Model trained | Acc: {acc:.3f} | F1: {f1:.3f} | "
        f"Precision: {prec:.3f} | Recall: {rec:.3f}"
    )
    logger.debug(f"Classification report:\n{classification_report(y_test, y_pred, zero_division=0)}")

    return model, scaler


# ---------------------------------------------------------------------------
# Walk-Forward Validation
# ---------------------------------------------------------------------------

def walk_forward_validate(
    X: pd.DataFrame,
    y: pd.Series,
    model_type: str = ML_MODEL_TYPE,
    n_splits: int = ML_WALK_FORWARD_SPLITS,
) -> dict:
    """
    Validate model using TimeSeriesSplit (walk-forward).

    IMPORTANT: This is the ONLY permitted cross-validation strategy.
    Standard K-Fold with shuffle is FORBIDDEN for financial time series.

    Args:
        X:          Feature DataFrame (chronological order).
        y:          Label Series from prepare_labels().
        model_type: Model type string.
        n_splits:   Number of temporal folds. Default: 5.

    Returns:
        Dict with per-fold and aggregate metrics:
        {
            "folds": [{"fold": int, "train_size": int, "test_size": int,
                       "accuracy": float, "f1": float}, ...],
            "mean_accuracy": float,
            "mean_f1": float,
            "std_accuracy": float,
        }
    """
    valid_mask = y.notna()
    X_clean = X[valid_mask]
    y_clean = y[valid_mask].astype(int)

    tscv   = TimeSeriesSplit(n_splits=n_splits)
    scaler = StandardScaler()
    folds  = []

    logger.info(f"Walk-forward validation | Model: {model_type} | Folds: {n_splits}")

    for fold_num, (train_idx, test_idx) in enumerate(tscv.split(X_clean), start=1):
        X_tr, X_te = X_clean.iloc[train_idx], X_clean.iloc[test_idx]
        y_tr, y_te = y_clean.iloc[train_idx], y_clean.iloc[test_idx]

        X_tr_s = scaler.fit_transform(X_tr)
        X_te_s = scaler.transform(X_te)

        model = _build_model(model_type)
        model.fit(X_tr_s, y_tr)

        y_pred = model.predict(X_te_s)
        acc    = accuracy_score(y_te, y_pred)
        f1     = f1_score(y_te, y_pred, average="weighted", zero_division=0)

        folds.append({
            "fold":       fold_num,
            "train_size": len(X_tr),
            "test_size":  len(X_te),
            "accuracy":   round(acc, 4),
            "f1":         round(f1, 4),
        })
        logger.info(f"  Fold {fold_num}: Acc={acc:.3f} | F1={f1:.3f} | Train={len(X_tr)} | Test={len(X_te)}")

    accuracies = [f["accuracy"] for f in folds]
    f1s        = [f["f1"]       for f in folds]

    result = {
        "folds":         folds,
        "mean_accuracy": round(float(np.mean(accuracies)), 4),
        "std_accuracy":  round(float(np.std(accuracies)),  4),
        "mean_f1":       round(float(np.mean(f1s)),        4),
    }
    logger.info(
        f"Walk-forward complete | Mean Acc: {result['mean_accuracy']:.3f} ± "
        f"{result['std_accuracy']:.3f} | Mean F1: {result['mean_f1']:.3f}"
    )
    return result


# ---------------------------------------------------------------------------
# Model Persistence
# ---------------------------------------------------------------------------

def save_model(
    model: BaseEstimator,
    scaler: StandardScaler,
    model_type: str,
    symbol: str,
    model_dir: Path = MODEL_DIR,
) -> Path:
    """
    Serialize model and scaler to disk with a timestamped filename.

    Args:
        model:      Trained sklearn estimator.
        scaler:     Fitted StandardScaler.
        model_type: Model type string (for filename).
        symbol:     Symbol the model was trained on (for filename).
        model_dir:  Save directory. Default: D:\\Trading\\models\\.

    Returns:
        Path to the saved .joblib file.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename  = model_dir / f"{symbol}_{model_type}_{timestamp}.joblib"

    payload = {"model": model, "scaler": scaler, "model_type": model_type, "symbol": symbol}
    joblib.dump(payload, filename, compress=3)
    logger.info(f"Model saved: {filename}")
    return filename


def load_model(path: str | Path) -> Tuple[BaseEstimator, StandardScaler]:
    """
    Load a previously saved model and scaler from disk.

    Args:
        path: Path to .joblib file.

    Returns:
        Tuple of (model, scaler).

    Raises:
        FileNotFoundError: If path does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    payload = joblib.load(path)
    logger.info(
        f"Model loaded: {path.name} | "
        f"Type: {payload['model_type']} | Symbol: {payload['symbol']}"
    )
    return payload["model"], payload["scaler"]


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def predict(
    model: BaseEstimator,
    scaler: StandardScaler,
    features: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run inference on a feature matrix.

    Args:
        model:    Fitted sklearn estimator.
        scaler:   Fitted StandardScaler (from same training run).
        features: Feature DataFrame (same columns as training data).

    Returns:
        Tuple of (predictions, probabilities):
            predictions:  np.ndarray of int labels (1, -1, or 0).
            probabilities: np.ndarray of shape (n_samples, n_classes).
    """
    if features.empty:
        logger.warning("predict() received empty feature DataFrame.")
        return np.array([]), np.array([])

    X_scaled = scaler.transform(features)

    predictions   = model.predict(X_scaled)
    probabilities = (
        model.predict_proba(X_scaled)
        if hasattr(model, "predict_proba")
        else np.zeros((len(predictions), 3))
    )

    latest_pred = predictions[-1]
    latest_prob = probabilities[-1].max() if len(probabilities) > 0 else 0.0

    logger.debug(
        f"Inference complete | Latest prediction: {latest_pred} | "
        f"Max probability: {latest_prob:.3f}"
    )
    return predictions, probabilities


def get_latest_signal_confidence(
    model: BaseEstimator,
    scaler: StandardScaler,
    features: pd.DataFrame,
) -> Tuple[int, float]:
    """
    Get prediction and confidence for the latest bar only.

    Args:
        model:    Fitted estimator.
        scaler:   Fitted scaler.
        features: Feature DataFrame (last row used).

    Returns:
        Tuple of (direction: int, confidence: float).
        direction in {1, -1, 0}. confidence in [0.0, 1.0].
    """
    preds, probs = predict(model, scaler, features.iloc[[-1]])

    if len(preds) == 0:
        return 0, 0.0

    direction  = int(preds[-1])
    confidence = float(probs[-1].max()) if probs.size > 0 else 0.0

    return direction, confidence
