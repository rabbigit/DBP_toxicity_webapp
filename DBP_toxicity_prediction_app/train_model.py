"""Train, evaluate, and save the LightGBM toxicity regression model."""

from datetime import datetime, timezone
from pathlib import Path
import platform

import joblib
import lightgbm
import numpy as np
import pandas as pd
import sklearn
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score, root_mean_squared_error

from preprocessing import (
    CELL_MAPPING,
    ENDPOINT_MAPPING,
    MORGAN_BITS,
    MORGAN_RADIUS,
    canonicalize_smiles,
)


PROJECT_DIR = Path(__file__).resolve().parent
TRAINING_PATH = PROJECT_DIR / "data" / "training data 872026.csv"
TEST_PATH = PROJECT_DIR / "data" / "Test data 872026.csv"
MODEL_PATH = PROJECT_DIR / "model" / "toxicity_model.joblib"

NON_FEATURE_COLUMNS = ["Canonical SMILES", "Value"]
CATEGORICAL_FEATURES = ["Endpoint", "Cell"]


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    training = pd.read_csv(TRAINING_PATH)
    test = pd.read_csv(TEST_PATH)
    if training.columns.tolist() != test.columns.tolist():
        raise ValueError("Training and test columns do not match in exact order.")
    return training, test


def validate_data(training: pd.DataFrame, test: pd.DataFrame) -> list[str]:
    required = set(NON_FEATURE_COLUMNS + CATEGORICAL_FEATURES)
    missing = sorted(required - set(training.columns))
    if missing:
        raise ValueError("Required columns are missing: " + ", ".join(missing))

    feature_names = [
        column for column in training.columns if column not in NON_FEATURE_COLUMNS
    ]
    for name, frame in [("training", training), ("test", test)]:
        numeric = frame[feature_names + ["Value"]].apply(pd.to_numeric, errors="coerce")
        if numeric.isna().any().any():
            bad = numeric.columns[numeric.isna().any()].tolist()
            raise ValueError(f"{name} data contain missing/non-numeric values in: {bad}")
        if not np.isfinite(numeric.to_numpy(dtype=float)).all():
            raise ValueError(f"{name} data contain infinite numeric values.")
    return feature_names


def train_and_save() -> dict:
    training, test = load_data()
    feature_names = validate_data(training, test)
    descriptor_names = [
        name for name in feature_names if name not in CATEGORICAL_FEATURES
    ]

    x_train = training[feature_names]
    y_train = training["Value"].to_numpy()
    x_test = test[feature_names]
    y_test = test["Value"].to_numpy()

    model = LGBMRegressor(
        random_state=42,
        verbosity=-1,
        n_jobs=1,
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    metrics = {
        "r2": float(r2_score(y_test, predictions)),
        "rmse": float(root_mean_squared_error(y_test, predictions)),
        "training_rows": int(len(training)),
        "test_rows": int(len(test)),
    }

    descriptor_ranges = {
        name: {
            "min": float(training[name].min()),
            "max": float(training[name].max()),
        }
        for name in descriptor_names
    }

    ad_reference_smiles = list(
        dict.fromkeys(
            canonicalize_smiles(smiles)[0]
            for smiles in training["Canonical SMILES"].astype(str)
        )
    )

    bundle = {
        "model": model,
        "feature_names": feature_names,
        "descriptor_names": descriptor_names,
        "descriptor_ranges": descriptor_ranges,
        "ad_reference_smiles": ad_reference_smiles,
        "ad_method": {
            "fingerprint": "Morgan",
            "radius": MORGAN_RADIUS,
            "n_bits": MORGAN_BITS,
            "similarity": "Tanimoto",
            "reported_value": "maximum similarity to unique training compounds",
        },
        "endpoint_mapping": ENDPOINT_MAPPING,
        "cell_mapping": CELL_MAPPING,
        "metrics": metrics,
        "target_name": "Value",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "versions": {
            "python": platform.python_version(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "lightgbm": lightgbm.__version__,
        },
    }

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, MODEL_PATH)
    print(f"Test R2: {metrics['r2']:.4f}")
    print(f"Test RMSE: {metrics['rmse']:.4f}")
    print(f"Saved model bundle to: {MODEL_PATH}")
    return bundle


if __name__ == "__main__":
    train_and_save()
