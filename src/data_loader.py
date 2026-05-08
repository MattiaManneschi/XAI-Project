"""Download, cache and preprocess the Heart Failure Clinical Records dataset."""

import os
import ssl
import certifi
import numpy as np
import pandas as pd
from dataclasses import dataclass
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer

from config import DATA_DIR, RANDOM_STATE, TEST_SIZE

# macOS Python (python.org installer) ships without system CA certificates.
# Point the SSL stack to certifi's bundle before any HTTPS call is made.
os.environ.setdefault("SSL_CERT_FILE", certifi.where())
os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
ssl._create_default_https_context = lambda: ssl.create_default_context(cafile=certifi.where())


@dataclass
class PreparedData:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: np.ndarray
    y_test: np.ndarray
    X_train_scaled: pd.DataFrame    # StandardScaled (for RuleFit)
    X_test_scaled: pd.DataFrame
    X_train_disc: np.ndarray        # one-hot binary bins (for BayesianRuleList)
    X_test_disc: np.ndarray
    feature_names: list[str]
    disc_feature_names: list[str]   # decoded names for each discretised column
    scale_pos_weight: float         # class imbalance weight for XGBoost


def load_dataset() -> tuple[pd.DataFrame, pd.Series]:
    """Return (X, y) from UCI or from local cache."""
    cache_X = DATA_DIR / "X.csv"
    cache_y = DATA_DIR / "y.csv"

    if cache_X.exists() and cache_y.exists():
        print("Loading dataset from local cache...")
        X = pd.read_csv(cache_X)
        y = pd.read_csv(cache_y).iloc[:, 0]
    else:
        print("Downloading dataset from UCI repository...")
        DATA_DIR.mkdir(exist_ok=True)
        ds = fetch_ucirepo(id=519)
        assert ds.data is not None, "fetch_ucirepo returned no data"
        X = ds.data.features.copy()
        y = ds.data.targets.squeeze().rename("death_event").copy()
        X.to_csv(cache_X, index=False)
        y.to_csv(cache_y, index=False)
        print("Dataset saved to cache.")

    print(f"Samples: {X.shape[0]}  |  Features: {X.shape[1]}")
    print(f"Class distribution:\n{y.value_counts().to_string()}")
    print(f"Positive rate: {y.mean():.1%}\n")
    return X, y


def prepare(X: pd.DataFrame, y: pd.Series) -> PreparedData:
    """Stratified split + scaling + one-hot discretisation."""
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X.columns)
    X_test_scaled  = pd.DataFrame(scaler.transform(X_test),      columns=X.columns)

    # BayesianRuleList requires strictly binary (0/1) features.
    # Binary features produce 1 bin; continuous features produce 4 bins.
    # Actual column count depends on the data (typically ~33, not 48).
    discretiser = KBinsDiscretizer(n_bins=4, encode="onehot-dense", strategy="quantile")
    X_train_disc = np.asarray(discretiser.fit_transform(X_train.values)).astype(int)
    X_test_disc  = np.asarray(discretiser.transform(X_test.values)).astype(int)

    disc_feature_names: list[str] = []
    for i, feat in enumerate(X.columns):
        edges = discretiser.bin_edges_[i]
        for j in range(len(edges) - 1):
            disc_feature_names.append(f"{feat} ∈ ({edges[j]:.2f}, {edges[j+1]:.2f}]")

    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())

    print(f"Train: {len(X_train)} samples  |  Test: {len(X_test)} samples")
    print(f"Train positive rate: {y_train.mean():.1%}")
    print(f"Test  positive rate: {y_test.mean():.1%}\n")

    return PreparedData(
        X_train=X_train.reset_index(drop=True),
        X_test=X_test.reset_index(drop=True),
        y_train=y_train.values,
        y_test=y_test.values,
        X_train_scaled=X_train_scaled,
        X_test_scaled=X_test_scaled,
        X_train_disc=X_train_disc,
        X_test_disc=X_test_disc,
        feature_names=list(X.columns),
        disc_feature_names=disc_feature_names,
        scale_pos_weight=scale_pos_weight,
    )
