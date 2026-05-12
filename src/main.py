"""Entry point: runs the full XAI pipeline end-to-end."""

import json
import re
import warnings
import pandas as pd
warnings.filterwarnings("ignore")

from data_loader import load_dataset, prepare
from models import (
    build_rulefit, build_greedy_rule_list, build_bayesian_rule_list,
    build_figs, build_skope_rules, build_ripper,
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
    _rf_df = rulefit._get_rules(exclude_zero_coef=True)
    _rf_df = _rf_df[_rf_df["coef"] > 0.01].nlargest(8, "coef")
    with open(RESULTS_DIR / "rulefit_top_rules.json", "w") as f:
        json.dump([[str(r), f"{c:.3f}"] for r, c in zip(_rf_df["rule"], _rf_df["coef"])], f)

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
        val  = rule["val_right"]
        risk = f"{val:.1%}"
        pred = "Decesso" if val > 0.5 else "Sopravvissuto"
        print(f"  Rule {i+1}: IF {cond}  →  {risk} ({pred})  ({rule['num_pts_right']} pts)")
        grl_rows.append([cond, risk, pred, str(rule["num_pts_right"])])
    default = grl.rules_[-1]
    val_def  = default["val"]
    pred_def = "Decesso" if val_def > 0.5 else "Sopravvissuto"
    print(f"  Default: {val_def:.1%} ({pred_def})  ({default['num_pts']} pts)")
    grl_rows.append(["DEFAULT", f"{val_def:.1%}", pred_def, str(default["num_pts"])])
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
    with open(RESULTS_DIR / "figs_structure.txt", "w") as f:
        f.write(str(figs))

    skope = build_skope_rules()
    r_skope = fit_and_score(
        skope,
        data.X_train, data.y_train,
        data.X_test,  data.y_test,
        "SkopeRules",
    )
    skope_rows = []
    if hasattr(skope, "rules_") and skope.rules_:
        print(f"Rules learned: {len(skope.rules_)}")
        for i, item in enumerate(skope.rules_):
            print(f"  Rule {i+1}: {item}")
            try:
                rule_str = item.rule if hasattr(item, "rule") else str(item)
                args = getattr(item, "args_", None)
                prec = f"{args[0]:.3f}" if args and args[0] is not None else "—"
                rec  = f"{args[1]:.3f}" if args and len(args) > 1 else "—"
            except Exception:
                rule_str, prec, rec = str(item), "—", "—"
            skope_rows.append([rule_str, prec, rec])
    with open(RESULTS_DIR / "skope_rules.json", "w") as f:
        json.dump(skope_rows, f)

    ripper = build_ripper()
    r_ripper = fit_and_score(
        ripper,
        data.X_train, data.y_train,
        data.X_test,  data.y_test,
        "RIPPER",
    )
    # Decode RIPPER rules with sequential sample coverage
    ripper_rows = []
    print("\nRIPPER rules:")
    X_rem = data.X_train.copy()
    for i, rule in enumerate(ripper.ruleset_.rules):
        parts = []
        mask = pd.Series(True, index=X_rem.index)
        for cond in rule.conds:
            val = cond.val
            col = X_rem[cond.feature]
            if isinstance(val, str):
                if val.startswith('<'):
                    mask &= col <= float(val[1:])
                    parts.append(f"{cond.feature} <= {val[1:]}")
                elif val.startswith('>'):
                    mask &= col > float(val[1:])
                    parts.append(f"{cond.feature} > {val[1:]}")
                elif ' - ' in val:
                    lo, hi = val.split(' - ', 1)
                    mask &= (col > float(lo)) & (col <= float(hi))
                    parts.append(f"{cond.feature} ∈ ({lo}, {hi}]")
                else:
                    mask &= col == val
                    parts.append(f"{cond.feature} = {val}")
            else:
                mask &= col == val
                parts.append(f"{cond.feature} = {val}")
        antecedent = " AND ".join(parts)
        n_pts = int(mask.sum())
        print(f"  Rule {i+1}: IF {antecedent}  →  Decesso  ({n_pts} pts)")
        ripper_rows.append([antecedent, str(n_pts)])
        X_rem = X_rem[~mask]
    n_default = len(X_rem)
    print(f"  Default: Sopravvissuto  ({n_default} pts)")
    ripper_rows.append(["DEFAULT", str(n_default)])
    with open(RESULTS_DIR / "ripper_rules.json", "w") as f:
        json.dump(ripper_rows, f)

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
    all_results = [r_rulefit, r_grl, r_brl, r_figs, r_skope, r_ripper, r_rf, r_gb, r_xgb]

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
        {"name": "RIPPER",           "model": build_ripper(),           "X": data.X_train},
        {"name": "FIGS",             "model": build_figs(),             "X": data.X_train},
        {"name": "RandomForest",     "model": build_random_forest(),    "X": data.X_train.values},
        {"name": "GradientBoosting", "model": build_gradient_boosting(),"X": data.X_train.values},
        {"name": "XGBoost",          "model": build_xgboost(data.scale_pos_weight), "X": data.X_train.values},
    ]
    run_cross_validation(cv_entries, data.y_train)

    # ── 7. Ablation: RF e GRL senza feature 'time' ───────────────────────────
    print("\n" + "="*60)
    print("  ABLATION: RF e GRL senza feature 'time'")
    print("="*60)

    X_nt = X.drop(columns=["time"])
    data_nt = prepare(X_nt, y)

    rf_nt  = build_random_forest()
    grl_nt = build_greedy_rule_list()

    r_rf_nt  = fit_and_score(rf_nt,  data_nt.X_train.values, data_nt.y_train,
                             data_nt.X_test.values, data_nt.y_test, "RF_no_time")
    r_grl_nt = fit_and_score(grl_nt, data_nt.X_train,        data_nt.y_train,
                             data_nt.X_test,        data_nt.y_test, "GRL_no_time")

    # Save ablation metrics
    abl_keys = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    with open(RESULTS_DIR / "ablation_no_time.csv", "w") as f:
        f.write("name," + ",".join(abl_keys) + "\n")
        for r in [r_rf, r_grl, r_rf_nt, r_grl_nt]:
            f.write(r["name"] + "," + ",".join(str(r[k]) for k in abl_keys) + "\n")

    # Save GRL no-time rules (same format as grl_rules.json: 4-element rows)
    grl_nt_rows = []
    print("\nGRL (no time) rules:")
    for i, rule in enumerate(grl_nt.rules_[:-1]):
        feat = data_nt.feature_names[rule["index_col"]]
        op   = "<=" if rule["flip"] else ">"
        cond = f"{feat} {op} {rule['cutoff']:.3f}"
        val  = rule["val_right"]
        risk = f"{val:.1%}"
        pred = "Decesso" if val > 0.5 else "Sopravvissuto"
        print(f"  Rule {i+1}: IF {cond}  →  {risk} ({pred})  ({rule['num_pts_right']} pts)")
        grl_nt_rows.append([cond, risk, pred, str(rule["num_pts_right"])])
    default_nt  = grl_nt.rules_[-1]
    val_nt_def  = default_nt["val"]
    pred_nt_def = "Decesso" if val_nt_def > 0.5 else "Sopravvissuto"
    print(f"  Default: {val_nt_def:.1%} ({pred_nt_def})  ({default_nt['num_pts']} pts)")
    grl_nt_rows.append(["DEFAULT", f"{val_nt_def:.1%}", pred_nt_def, str(default_nt["num_pts"])])
    with open(RESULTS_DIR / "grl_no_time_rules.json", "w") as f:
        json.dump(grl_nt_rows, f)

    generate_report()
    print("\n✓ Pipeline completed. All outputs saved in results/")


if __name__ == "__main__":
    main()
