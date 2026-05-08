"""Entry point: runs the full XAI pipeline end-to-end."""

import json
import re
import warnings
warnings.filterwarnings("ignore")

from data_loader import load_dataset, prepare
from models import (
    build_rulefit, build_greedy_rule_list, build_bayesian_rule_list,
    build_figs, build_skope_rules,
    build_random_forest, build_gradient_boosting, build_xgboost,
)
from evaluate import (
    fit_and_score, summary_table,
    plot_eda, plot_metrics_comparison, plot_roc_curves,
    plot_confusion_matrices, plot_rulefit_rules,
    plot_feature_importance, run_cross_validation,
)
from config import BINARY_FEATURES, RESULTS_DIR
from report import generate_report


def main():
    # ── 1. Data ───────────────────────────────────────────────────────────────
    X, y = load_dataset()
    data = prepare(X, y)

    # ── 2. EDA plots ──────────────────────────────────────────────────────────
    plot_eda(X, y, BINARY_FEATURES)

    # ── 3. Rule-based models ──────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  RULE-BASED MODELS")
    print("="*60)

    rulefit = build_rulefit()
    r_rulefit = fit_and_score(
        rulefit,
        data.X_train_scaled, data.y_train,
        data.X_test_scaled,  data.y_test,
        "RuleFit",
    )

    grl = build_greedy_rule_list()
    r_grl = fit_and_score(
        grl,
        data.X_train, data.y_train,
        data.X_test,  data.y_test,
        "GreedyRuleList",
    )
    # Save and print decoded GRL rules
    grl_rows = []
    print("\nGreedyRuleList rules:")
    for i, rule in enumerate(grl.rules_[:-1]):
        feat = data.feature_names[rule["index_col"]]
        op   = "<=" if rule["flip"] else ">"
        cond = f"{feat} {op} {rule['cutoff']:.3f}"
        risk = f"{rule['val_right']:.1%}"
        print(f"  Rule {i+1}: IF {cond}  →  {risk} risk  ({rule['num_pts_right']} pts)")
        grl_rows.append([cond, risk, str(rule["num_pts_right"])])
    default = grl.rules_[-1]
    print(f"  Default: {default['val']:.1%} risk  ({default['num_pts']} pts)")
    grl_rows.append(["DEFAULT (nessuna regola attivata)", f"{default['val']:.1%}", str(default["num_pts"])])
    with open(RESULTS_DIR / "grl_rules.json", "w") as f:
        json.dump(grl_rows, f)

    brl = build_bayesian_rule_list()
    r_brl = fit_and_score(
        brl,
        data.X_train_disc, data.y_train,
        data.X_test_disc,  data.y_test,
        "BayesianRuleList",
    )
    # Save and print decoded BRL rules
    brl_rows = []
    print("\nBayesianRuleList rules (decoded):")
    for line in str(brl).splitlines():
        m = re.search(r"X_(\d+)", line)
        if m:
            decoded = data.disc_feature_names[int(m.group(1))]
            line = line.replace(m.group(0), decoded)
        print(" ", line)
        # Extract structured data for report table
        m_if = re.search(
            r"((?:ELSE )?IF|ELSE)\s*(.*?)(?:\s*THEN probability of class 1:\s*([\d.]+)%\s*\(([\d.]+)%-([\d.]+)%\))?$",
            line.strip())
        m_prob = re.search(r"probability of class 1:\s*([\d.]+)%\s*\(([\d.]+)%-([\d.]+)%\)", line)
        if m_prob:
            prob, lo, hi = m_prob.group(1), m_prob.group(2), m_prob.group(3)
            if "ELSE" in line and "ELSE IF" not in line:
                cond = "DEFAULT (nessuna regola attivata)"
            else:
                cond_m = re.search(r"IF\s+(.+?)\s+THEN", line)
                cond = cond_m.group(1).replace(" > 0.5", "").strip() if cond_m else line.strip()
            brl_rows.append([cond, f"{prob}%", f"{lo}%–{hi}%"])
    with open(RESULTS_DIR / "brl_rules.json", "w") as f:
        json.dump(brl_rows, f)

    figs = build_figs()
    r_figs = fit_and_score(
        figs,
        data.X_train, data.y_train,
        data.X_test,  data.y_test,
        "FIGS",
    )
    print("\nFIGS structure:")
    print(figs)

    skope = build_skope_rules()
    r_skope = fit_and_score(
        skope,
        data.X_train, data.y_train,
        data.X_test,  data.y_test,
        "SkopeRules",
    )
    if hasattr(skope, "rules_") and skope.rules_:
        print(f"Rules learned: {len(skope.rules_)}")
        for i, item in enumerate(skope.rules_[:10]):
            print(f"  Rule {i+1}: {item}")

    # ── 4. Ensemble models ────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("  ENSEMBLE MODELS")
    print("="*60)

    rf  = build_random_forest()
    gb  = build_gradient_boosting()
    xgb = build_xgboost(data.scale_pos_weight)

    r_rf  = fit_and_score(rf,  data.X_train.values, data.y_train, data.X_test.values, data.y_test, "RandomForest")
    r_gb  = fit_and_score(gb,  data.X_train.values, data.y_train, data.X_test.values, data.y_test, "GradientBoosting")
    r_xgb = fit_and_score(xgb, data.X_train.values, data.y_train, data.X_test.values, data.y_test, "XGBoost")

    # ── 5. Evaluation & plots ─────────────────────────────────────────────────
    all_results = [r_rulefit, r_grl, r_brl, r_figs, r_skope, r_rf, r_gb, r_xgb]

    summary = summary_table(all_results)

    plot_metrics_comparison(summary)
    plot_roc_curves(all_results, data.y_test)
    plot_confusion_matrices(all_results, data.y_test)
    plot_rulefit_rules(rulefit)
    plot_feature_importance(rf, data.feature_names)

    # ── 6. Cross-validation ───────────────────────────────────────────────────
    cv_entries = [
        {"name": "RuleFit",          "model": build_rulefit(),          "X": data.X_train_scaled.values},
        {"name": "GreedyRuleList",   "model": build_greedy_rule_list(), "X": data.X_train},
        {"name": "FIGS",             "model": build_figs(),             "X": data.X_train},
        {"name": "RandomForest",     "model": build_random_forest(),    "X": data.X_train.values},
        {"name": "GradientBoosting", "model": build_gradient_boosting(),"X": data.X_train.values},
        {"name": "XGBoost",          "model": build_xgboost(data.scale_pos_weight), "X": data.X_train.values},
    ]
    run_cross_validation(cv_entries, data.y_train)

    generate_report()
    print("\n✓ Pipeline completed. All outputs saved in results/")


if __name__ == "__main__":
    main()
