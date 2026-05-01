"""Factory functions for all classifiers: rule-based and ensemble."""

import xgboost as xgb
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
    RF_N_ESTIMATORS,
    GB_N_ESTIMATORS, GB_LEARNING_RATE, GB_MAX_DEPTH,
    XGB_N_ESTIMATORS, XGB_LEARNING_RATE, XGB_MAX_DEPTH,
)


# ── Rule-based ────────────────────────────────────────────────────────────────

def build_rulefit() -> RuleFitClassifier:
    return RuleFitClassifier(max_rules=RULEFIT_MAX_RULES, random_state=RANDOM_STATE)


def build_greedy_rule_list() -> GreedyRuleListClassifier:
    return GreedyRuleListClassifier(max_depth=GRL_MAX_DEPTH)


def build_bayesian_rule_list() -> BayesianRuleListClassifier:
    return BayesianRuleListClassifier(
        max_iter=BRL_MAX_ITER,
        listlengthprior=3,
        listwidthprior=1,
        maxcardinality=BRL_MAX_CARDINALITY,
        random_state=RANDOM_STATE,
    )


def build_figs() -> FIGSClassifier:
    return FIGSClassifier(max_rules=FIGS_MAX_RULES)


def build_skope_rules() -> SkopeRulesClassifier:
    return SkopeRulesClassifier(
        n_estimators=SKOPE_N_ESTIMATORS,
        precision_min=SKOPE_PRECISION_MIN,
        recall_min=SKOPE_RECALL_MIN,
        random_state=RANDOM_STATE,
    )


# ── Ensemble ──────────────────────────────────────────────────────────────────

def build_random_forest() -> RandomForestClassifier:
    return RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )


def build_gradient_boosting() -> GradientBoostingClassifier:
    return GradientBoostingClassifier(
        n_estimators=GB_N_ESTIMATORS,
        learning_rate=GB_LEARNING_RATE,
        max_depth=GB_MAX_DEPTH,
        subsample=0.8,
        random_state=RANDOM_STATE,
    )


def build_xgboost(scale_pos_weight: float) -> xgb.XGBClassifier:
    return xgb.XGBClassifier(
        n_estimators=XGB_N_ESTIMATORS,
        learning_rate=XGB_LEARNING_RATE,
        max_depth=XGB_MAX_DEPTH,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        verbosity=0,
        random_state=RANDOM_STATE,
    )
