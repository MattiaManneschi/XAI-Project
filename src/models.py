"""Factory functions for all classifiers: rule-based and ensemble."""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from imodels import (
    RuleFitClassifier,
    GreedyRuleListClassifier,
    BayesianRuleListClassifier,
    FIGSClassifier,
    SkopeRulesClassifier,
)

from config import (
    RANDOM_STATE,
    RULEFIT_MAX_RULES,
    GRL_MAX_DEPTH,
    BRL_MAX_ITER, BRL_MAX_CARDINALITY,
    FIGS_MAX_RULES,
    SKOPE_N_ESTIMATORS, SKOPE_PRECISION_MIN, SKOPE_RECALL_MIN,
    RIPPER_K,
    RF_N_ESTIMATORS,
    GB_N_ESTIMATORS, GB_LEARNING_RATE, GB_MAX_DEPTH,
    XGB_N_ESTIMATORS, XGB_LEARNING_RATE, XGB_MAX_DEPTH,
)


# ── Rule-based ────────────────────────────────────────────────────────────────

def build_rulefit(**params) -> RuleFitClassifier:
    cfg = {"max_rules": RULEFIT_MAX_RULES, "random_state": RANDOM_STATE}
    cfg.update(params)
    return RuleFitClassifier(**cfg)


def build_greedy_rule_list(**params) -> GreedyRuleListClassifier:
    cfg = {"max_depth": GRL_MAX_DEPTH}
    cfg.update(params)
    return GreedyRuleListClassifier(**cfg)


def build_bayesian_rule_list(**params) -> BayesianRuleListClassifier:
    cfg = {
        "max_iter": BRL_MAX_ITER,
        "listlengthprior": 3,
        "listwidthprior": 1,
        "maxcardinality": BRL_MAX_CARDINALITY,
        "random_state": RANDOM_STATE,
    }
    cfg.update(params)
    return BayesianRuleListClassifier(**cfg)


def build_figs(**params) -> FIGSClassifier:
    cfg = {"max_rules": FIGS_MAX_RULES}
    cfg.update(params)
    return FIGSClassifier(**cfg)


class _RIPPERWrapper(BaseEstimator, ClassifierMixin):
    """sklearn-compatible wrapper: fixes pos_class=1 and handles DataFrame/ndarray.
    RIPPER object is created inside fit() so sklearn clone() works correctly."""

    def __init__(self, k: int = 2, random_state: int | None = None) -> None:
        self.k = k
        self.random_state = random_state

    def _as_df(self, X) -> pd.DataFrame:
        if isinstance(X, pd.DataFrame):
            return X
        return pd.DataFrame(X, columns=getattr(self, "_feature_names_", None))

    def fit(self, X, y):
        import wittgenstein as lw
        if isinstance(X, pd.DataFrame):
            self._feature_names_ = list(X.columns)
        self._ripper_ = lw.RIPPER(k=self.k, random_state=self.random_state)
        self._ripper_.fit(self._as_df(X), y, pos_class=1)
        return self

    def predict(self, X) -> np.ndarray:
        return np.asarray(self._ripper_.predict(self._as_df(X))).astype(int)

    def predict_proba(self, X) -> np.ndarray:
        return self._ripper_.predict_proba(self._as_df(X))

    @property
    def ruleset_(self):
        return self._ripper_.ruleset_


def build_ripper() -> _RIPPERWrapper:
    return _RIPPERWrapper(k=RIPPER_K, random_state=RANDOM_STATE)


def build_skope_rules(**params) -> SkopeRulesClassifier:
    cfg = {
        "n_estimators": SKOPE_N_ESTIMATORS,
        "precision_min": SKOPE_PRECISION_MIN,
        "recall_min": SKOPE_RECALL_MIN,
        "random_state": RANDOM_STATE,
    }
    cfg.update(params)
    return SkopeRulesClassifier(**cfg)


# ── Ensemble ──────────────────────────────────────────────────────────────────

def build_random_forest(**params) -> RandomForestClassifier:
    cfg = {
        "n_estimators": RF_N_ESTIMATORS,
        "min_samples_leaf": 2,
        "class_weight": "balanced",
        "random_state": RANDOM_STATE,
        "n_jobs": -1,
    }
    cfg.update(params)
    return RandomForestClassifier(**cfg)


def build_gradient_boosting(**params) -> GradientBoostingClassifier:
    cfg = {
        "n_estimators": GB_N_ESTIMATORS,
        "learning_rate": GB_LEARNING_RATE,
        "max_depth": GB_MAX_DEPTH,
        "subsample": 0.8,
        "random_state": RANDOM_STATE,
    }
    cfg.update(params)
    return GradientBoostingClassifier(**cfg)


def build_xgboost(scale_pos_weight: float, **params) -> xgb.XGBClassifier:
    cfg = {
        "n_estimators": XGB_N_ESTIMATORS,
        "learning_rate": XGB_LEARNING_RATE,
        "max_depth": XGB_MAX_DEPTH,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": scale_pos_weight,
        "eval_metric": "logloss",
        "verbosity": 0,
        "random_state": RANDOM_STATE,
    }
    cfg.update(params)
    return xgb.XGBClassifier(**cfg)
