"""Grid search on validation set (maximises ROC-AUC)."""

import itertools
import warnings
from typing import Callable, Any

from sklearn.metrics import roc_auc_score


def _val_roc_auc(model, X_val, y_val) -> float:
    try:
        y_prob = model.predict_proba(X_val)[:, 1]
        return float(roc_auc_score(y_val, y_prob))
    except Exception:
        return 0.0


def grid_search_val(
    builder: Callable[..., Any],
    param_grid: dict[str, list],
    X_train, y_train,
    X_val, y_val,
    name: str = "",
) -> tuple[dict, float, list[tuple[dict, float]]]:
    """Try every combination in param_grid.

    Returns (best_params, best_val_roc_auc, all_scores) where all_scores is a
    list of (params_dict, val_roc_auc) for every combination that succeeded.
    """
    keys = list(param_grid.keys())
    combos = list(itertools.product(*param_grid.values()))
    best_score, best_params = -1.0, {}
    all_scores: list[tuple[dict, float]] = []

    print(f"  Tuning {name} ({len(combos)} combinations)...")
    for combo in combos:
        params = dict(zip(keys, combo))
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = builder(**params)
                m.fit(X_train, y_train)
            score = _val_roc_auc(m, X_val, y_val)
            all_scores.append((params, score))
            if score > best_score:
                best_score = score
                best_params = params
        except Exception:
            pass

    print(f"    → best: {best_params}  val ROC-AUC={best_score:.4f}")
    return best_params, best_score, all_scores
