"""Metrics computation, result collection and all plot generation."""

import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend for script use
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
    roc_curve, auc as sk_auc, classification_report,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score

from config import RANDOM_STATE, CV_FOLDS, RESULTS_DIR, FILES_DIR, PLOTS_DIR

RESULTS_DIR.mkdir(exist_ok=True)
FILES_DIR.mkdir(exist_ok=True)
PLOTS_DIR.mkdir(exist_ok=True)
plt.rcParams["figure.dpi"] = 150
sns.set_theme(style="whitegrid", palette="muted")

RULE_BASED_NAMES = {"RuleFit", "GreedyRuleList", "BayesianRuleList", "FIGS", "SkopeRules", "RIPPER"}


# ── helpers ──────────────────────────────────────────────────────────────────

def fit_and_score(model, X_train, y_train, X_test, y_test, name: str) -> dict:
    """Fit model and return a metrics dict."""
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    try:
        y_prob = model.predict_proba(X_test)[:, 1]
        roc = roc_auc_score(y_test, y_prob)
    except Exception:
        y_prob = None
        roc = float("nan")

    print(f"\n{'─'*50}")
    print(f"  {name}")
    print(f"{'─'*50}")
    print(classification_report(y_test, y_pred, target_names=["Survived", "Deceased"]))

    return {
        "name":      name,
        "model":     model,
        "y_pred":    y_pred,
        "y_prob":    y_prob,
        "Accuracy":  round(float(accuracy_score(y_test, y_pred)), 4),
        "Precision": round(float(precision_score(y_test, y_pred, zero_division=0)), 4),
        "Recall":    round(float(recall_score(y_test, y_pred, zero_division=0)), 4),
        "F1":        round(float(f1_score(y_test, y_pred, zero_division=0)), 4),
        "ROC-AUC":   round(float(roc), 4),
    }


# ── summary table ─────────────────────────────────────────────────────────────

def summary_table(results: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame([
        {k: v for k, v in r.items() if k not in ("model", "y_pred", "y_prob")}
        for r in results
    ]).set_index("name")
    df["Type"] = df.index.map(lambda n: "Rule-based" if n in RULE_BASED_NAMES else "Ensemble")
    df = df.sort_values("ROC-AUC", ascending=False)
    df.to_csv(FILES_DIR / "metrics_summary.csv")
    print("\n=== Metrics Summary ===")
    print(df.to_string())
    return df


# ── plots ─────────────────────────────────────────────────────────────────────

def plot_eda(X: pd.DataFrame, y: pd.Series, binary_features: list[str]) -> None:
    continuous = [c for c in X.columns if c not in binary_features]
    df = X.copy()
    df["DEATH_EVENT"] = y.values

    # Continuous feature distributions
    fig, axes = plt.subplots(4, 2, figsize=(8, 11))
    axes = axes.flatten()
    for i, feat in enumerate(continuous):
        for label, color in zip([0, 1], ["steelblue", "tomato"]):
            axes[i].hist(
                df.loc[df["DEATH_EVENT"] == label, feat],
                bins=20, alpha=0.6, color=color,
                label="Survived" if label == 0 else "Deceased",
            )
        axes[i].set_title(feat)
        axes[i].legend(fontsize=8)
    for j in range(len(continuous), len(axes)):
        axes[j].set_visible(False)
    plt.suptitle("Distribuzione feature continue per classe", fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "eda_continuous.png")
    plt.close()

    # Correlation heatmap
    fig, ax = plt.subplots(figsize=(8, 8))
    corr = df.corr(numeric_only=True)
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                cmap="RdBu_r", center=0, ax=ax, linewidths=0.5, square=True)
    ax.set_title("Matrice di correlazione", fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "eda_correlation.png")
    plt.close()

    print("EDA plots saved to results/")


def plot_metrics_comparison(df: pd.DataFrame) -> None:
    metrics = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    clean = df[metrics].dropna()

    # Rule-based first, then ensemble (separator line depends on this order)
    rule_rows = clean[[n in RULE_BASED_NAMES for n in clean.index]]
    ens_rows  = clean[[n not in RULE_BASED_NAMES for n in clean.index]]
    plot_df = pd.concat([rule_rows, ens_rows])
    n_rule = len(rule_rows)

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(plot_df))
    width = 0.15
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2"]
    for i, metric in enumerate(metrics):
        ax.bar(x + i * width, plot_df[metric], width, label=metric,
               color=colors[i], alpha=0.85)

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(plot_df.index, rotation=30, ha="right")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("Score")
    ax.set_title("Confronto metriche — tutti i modelli", fontweight="bold")
    ax.legend(loc="lower right")

    if n_rule > 0 and n_rule < len(plot_df):
        ax.axvline(x=n_rule - 0.5, color="gray", linestyle="--", linewidth=1.5, alpha=0.7)
        ax.text(n_rule / 2 - 0.5, 1.04, "Rule-based", ha="center", color="gray")
        ax.text(n_rule + (len(plot_df) - n_rule) / 2 - 0.5, 1.04, "Ensemble", ha="center", color="gray")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "metrics_comparison.png")
    plt.close()
    print("Saved: results/metrics_comparison.png")


def plot_roc_curves(results: list[dict], y_test: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(8, 8))
    colors = ["#4C72B0", "#55A868", "#DD8452", "#C44E52", "#8172B2",
              "#937860", "#DA8BC3", "#8C8C8C"]
    linestyles = ["-", "--", "-.", ":", "-", "--", "-.", ":"]

    for r, color, ls in zip(results, colors, linestyles):
        if r["y_prob"] is not None:
            fpr, tpr, _ = roc_curve(y_test, r["y_prob"])
            area = sk_auc(fpr, tpr)
            ax.plot(fpr, tpr, label=f'{r["name"]} (AUC={area:.3f})',
                    color=color, linestyle=ls, linewidth=2)

    ax.plot([0, 1], [0, 1], "k--", linewidth=1, alpha=0.5)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Curve ROC — confronto modelli", fontweight="bold")
    ax.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "roc_curves.png")
    plt.close()
    print("Saved: results/roc_curves.png")


def plot_confusion_matrices(results: list[dict], y_test: np.ndarray) -> None:
    n = len(results)
    cols = 2
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.5, rows * 3))
    axes = axes.flatten()

    for i, r in enumerate(results):
        cm = confusion_matrix(y_test, r["y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=axes[i],
                    xticklabels=["Survived", "Deceased"],
                    yticklabels=["Survived", "Deceased"],
                    cbar=False)
        axes[i].set_title(r["name"], fontweight="bold", fontsize=10)
        axes[i].set_xlabel("Predicted")
        axes[i].set_ylabel("Actual")

    for j in range(n, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Matrici di Confusione — Test Set", fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "confusion_matrices.png")
    plt.close()
    print("Saved: results/confusion_matrices.png")


def plot_rulefit_rules(rulefit_model) -> None:
    rules_df = rulefit_model._get_rules(exclude_zero_coef=True)
    rules_df = rules_df[rules_df["coef"].abs() > 0.01].copy()
    top_pos = rules_df[rules_df["coef"] > 0].nlargest(8, "coef")
    top_neg = rules_df[rules_df["coef"] < 0].nsmallest(8, "coef")

    def shorten(s, n=60):
        return s[:n] + "..." if len(s) > n else s

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 10))
    ax1.barh(range(len(top_pos)), top_pos["coef"], color="tomato")
    ax1.set_yticks(range(len(top_pos)))
    ax1.set_yticklabels([shorten(r) for r in top_pos["rule"]], fontsize=9)
    ax1.set_title("RuleFit — Regole pro DECESSO", fontweight="bold", fontsize=11)
    ax1.set_xlabel("Coefficiente")

    ax2.barh(range(len(top_neg)), top_neg["coef"], color="steelblue")
    ax2.set_yticks(range(len(top_neg)))
    ax2.set_yticklabels([shorten(r) for r in top_neg["rule"]], fontsize=9)
    ax2.set_title("RuleFit — Regole pro SOPRAVVIVENZA", fontweight="bold", fontsize=11)
    ax2.set_xlabel("Coefficiente")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "rulefit_rules.png")
    plt.close()
    print("Saved: results/rulefit_rules.png")


def plot_feature_importance(rf_model, feature_names: list[str]) -> None:
    fi = pd.Series(rf_model.feature_importances_, index=feature_names).sort_values()
    fig, ax = plt.subplots(figsize=(7, 9))
    fi.plot(kind="barh", ax=ax, color="steelblue", alpha=0.8)
    ax.set_title("Random Forest — Feature Importance (MDI)", fontweight="bold")
    ax.set_xlabel("Importance")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "feature_importance_rf.png")
    plt.close()
    print("Saved: results/feature_importance_rf.png")


def _proba_roc_auc_scorer(estimator, X, y):
    """Custom scorer that calls predict_proba directly, bypassing sklearn's
    is_classifier() check (needed for GreedyRuleListClassifier)."""
    y_prob = estimator.predict_proba(X)[:, 1]
    return roc_auc_score(y, y_prob)


def run_cross_validation(cv_entries: list[dict], y_train: np.ndarray) -> None:
    """cv_entries: list of {'name': str, 'model': ..., 'X': ndarray}"""
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = {}

    print(f"\n=== {CV_FOLDS}-Fold Stratified Cross-Validation (ROC-AUC) ===")
    for entry in cv_entries:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            scores = cross_val_score(
                entry["model"], entry["X"], y_train,
                cv=cv, scoring=_proba_roc_auc_scorer, n_jobs=-1,
            )
        cv_scores[entry["name"]] = scores
        print(f"  {entry['name']:22s}  {scores.mean():.4f} ± {scores.std():.4f}")

    # Boxplot — use matplotlib directly so patch_artist + bp["boxes"] work correctly
    names = list(cv_scores.keys())
    colors = ["#4C72B0" if n in RULE_BASED_NAMES else "#DD8452" for n in names]
    data_to_plot = [cv_scores[n] for n in names]

    fig, ax = plt.subplots(figsize=(9, 9))
    bp = ax.boxplot(data_to_plot, patch_artist=True)
    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels(names)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_ylabel("ROC-AUC")
    ax.set_title(
        f"Cross-Validation ({CV_FOLDS}-fold) — ROC-AUC\n(blu=rule-based, arancio=ensemble)",
        fontweight="bold",
    )
    n_rule = sum(1 for n in names if n in RULE_BASED_NAMES)
    if n_rule > 0 and n_rule < len(names):
        ax.axvline(x=n_rule + 0.5, color="gray", linestyle="--", alpha=0.6)
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "cross_validation.png")
    plt.close()
    print("Saved: results/cross_validation.png")


def plot_hyperparam_analysis(all_scores: dict[str, list[tuple[dict, float]]]) -> None:
    """Two figures: rule-based models and ensemble models.

    all_scores maps model_name → [(params_dict, val_roc_auc), ...]
    For multi-parameter grids the marginal best is shown: for each value of a
    parameter, take the maximum val AUC across all combinations that include it.
    """
    def _marginal(scores, param):
        """Return {param_value: max_val_auc} marginalising over other params."""
        best = {}
        for params, score in scores:
            v = params.get(param)
            if v is None:
                continue
            v_key = str(v)
            if v_key not in best or score > best[v_key]:
                best[v_key] = score
        return best

    def _barplot(ax, scores, param, title, best_params, color):
        marg = _marginal(scores, param)
        if not marg:
            ax.set_visible(False)
            return
        keys = list(marg.keys())
        vals = [marg[k] for k in keys]
        best_val = str(best_params.get(param, ""))
        bar_colors = [color if k == best_val else "#cccccc" for k in keys]
        ax.bar(keys, vals, color=bar_colors, edgecolor="white", linewidth=0.8)
        ax.set_ylim(max(0, min(vals) - 0.05), min(1.0, max(vals) + 0.06))
        ax.set_title(title, fontsize=9, fontweight="bold")
        ax.set_xlabel(param, fontsize=8)
        ax.set_ylabel("Val ROC-AUC", fontsize=8)
        ax.tick_params(labelsize=8)
        for bar, v in zip(ax.patches, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{v:.3f}", ha="center", va="bottom", fontsize=7)

    # ── Figure 1: rule-based ──────────────────────────────────────────────────
    rb_layout = [
        ("GreedyRuleList",   "max_depth",      "#4C72B0"),
        ("BayesianRuleList", "maxcardinality",  "#4C72B0"),
        ("SkopeRules",       "max_depth",       "#4C72B0"),
        ("SkopeRules",       "precision_min",   "#4C72B0"),
        ("SkopeRules",       "recall_min",      "#4C72B0"),
    ]

    # Load best params for highlight (passed as part of all_scores metadata? no — derive)
    best_map = {name: dict(max(scores, key=lambda x: x[1])[0])
                for name, scores in all_scores.items() if scores}

    fig, axes = plt.subplots(2, 3, figsize=(12, 7))
    axes = axes.flatten()
    for idx, (model, param, color) in enumerate(rb_layout):
        scores = all_scores.get(model, [])
        _barplot(axes[idx], scores, param, f"{model}\n{param}", best_map.get(model, {}), color)
    axes[5].set_visible(False)
    plt.suptitle("Analisi iperparametri — Modelli Rule-Based\n(barre evidenziate = valore ottimale)",
                 fontweight="bold", fontsize=11)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "hyperparam_rule_based.png")
    plt.close()
    print("Saved: results/hyperparam_rule_based.png")

    # ── Figure 2: ensemble ────────────────────────────────────────────────────
    ens_layout = [
        ("RandomForest",     "n_estimators",    "#DD8452"),
        ("RandomForest",     "max_depth",       "#DD8452"),
        ("RandomForest",     "min_samples_leaf","#DD8452"),
        ("GradientBoosting", "n_estimators",    "#DD8452"),
        ("GradientBoosting", "learning_rate",   "#DD8452"),
        ("GradientBoosting", "max_depth",       "#DD8452"),
        ("XGBoost",          "n_estimators",    "#DD8452"),
        ("XGBoost",          "learning_rate",   "#DD8452"),
        ("XGBoost",          "max_depth",       "#DD8452"),
    ]
    fig, axes = plt.subplots(3, 3, figsize=(12, 10))
    axes = axes.flatten()
    for idx, (model, param, color) in enumerate(ens_layout):
        scores = all_scores.get(model, [])
        _barplot(axes[idx], scores, param, f"{model}\n{param}", best_map.get(model, {}), color)
    plt.suptitle("Analisi iperparametri — Modelli Ensemble\n(barre evidenziate = valore ottimale)",
                 fontweight="bold", fontsize=11)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "hyperparam_ensemble.png")
    plt.close()
    print("Saved: results/hyperparam_ensemble.png")
