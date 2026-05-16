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
    plot_hyperparam_analysis,
)
from config import BINARY_FEATURES, RESULTS_DIR, FILES_DIR
from tune import grid_search_val
from report_latex import generate_report_latex


def _eval_rule_mask(rule_str: str, X: pd.DataFrame) -> pd.Series:
    """Parse 'feat op val and ...' rule string → boolean mask over X.index."""
    mask = pd.Series(True, index=X.index)
    for cond in rule_str.split(" and "):
        cond = cond.strip()
        for op in ("<=", ">=", ">", "<"):
            if op in cond:
                feat, val = cond.split(op, 1)
                method = {"<=": "le", ">=": "ge", ">": "gt", "<": "lt"}[op]
                mask &= getattr(X[feat.strip()], method)(float(val.strip()))
                break
    return mask


def main():
    # ── 1. Data ───────────────────────────────────────────────────────────────
    X, y = load_dataset()
    data = prepare(X, y)

    # ── 2. EDA plots ──────────────────────────────────────────────────────────
    plot_eda(X, y, BINARY_FEATURES)

    # ── 3. Hyperparameter tuning on validation set ────────────────────────────
    print("\n" + "="*60)
    print("  HYPERPARAMETER TUNING (validation set)")
    print("="*60)

    grl_params, _, grl_scores = grid_search_val(
        lambda **kw: build_greedy_rule_list(**kw),
        {"max_depth": [1, 2, 3, 4, 6]},
        data.X_train, data.y_train, data.X_val, data.y_val,
        name="GreedyRuleList",
    )

    # BRL: use reduced max_iter during search to keep runtime manageable
    brl_params, _, brl_scores = grid_search_val(
        lambda **kw: build_bayesian_rule_list(max_iter=3000, **kw),
        {"maxcardinality": [1, 2, 3]},
        data.X_train_disc, data.y_train, data.X_val_disc, data.y_val,
        name="BayesianRuleList",
    )

    skope_params, _, skope_scores = grid_search_val(
        lambda **kw: build_skope_rules(**kw),
        {"max_depth": [2, 3, 4], "precision_min": [0.4, 0.5], "recall_min": [0.2, 0.3]},
        data.X_train, data.y_train, data.X_val, data.y_val,
        name="SkopeRules",
    )

    rf_params, _, rf_scores = grid_search_val(
        lambda **kw: build_random_forest(**kw),
        {"n_estimators": [100, 200], "max_depth": [None, 5, 10], "min_samples_leaf": [1, 2, 5]},
        data.X_train.values, data.y_train, data.X_val.values, data.y_val,
        name="RandomForest",
    )

    gb_params, _, gb_scores = grid_search_val(
        lambda **kw: build_gradient_boosting(**kw),
        {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3, 4, 5]},
        data.X_train.values, data.y_train, data.X_val.values, data.y_val,
        name="GradientBoosting",
    )

    xgb_params, _, xgb_scores = grid_search_val(
        lambda **kw: build_xgboost(data.scale_pos_weight, **kw),
        {"n_estimators": [100, 200], "learning_rate": [0.05, 0.1], "max_depth": [3, 4, 5]},
        data.X_train.values, data.y_train, data.X_val.values, data.y_val,
        name="XGBoost",
    )

    all_hyperparam_scores = {
        "GreedyRuleList":    grl_scores,
        "BayesianRuleList":  brl_scores,
        "SkopeRules":        skope_scores,
        "RandomForest":      rf_scores,
        "GradientBoosting":  gb_scores,
        "XGBoost":           xgb_scores,
    }
    # Serialise (params dict keys may contain None → convert to str for JSON)
    serialisable = {
        model: [({str(k): str(v) for k, v in p.items()}, s) for p, s in scores]
        for model, scores in all_hyperparam_scores.items()
    }
    with open(FILES_DIR / "hyperparam_scores.json", "w") as f:
        json.dump(serialisable, f, indent=2)

    best_params_all = {
        "GreedyRuleList":    grl_params,
        "BayesianRuleList":  brl_params,
        "SkopeRules":        skope_params,
        "RandomForest":      rf_params,
        "GradientBoosting":  gb_params,
        "XGBoost":           xgb_params,
    }
    with open(FILES_DIR / "best_params.json", "w") as f:
        json.dump(best_params_all, f, indent=2)
    print("Best params saved → results/best_params.json")

    # ── 4. Rule-based models (with best params) ───────────────────────────────
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
    with open(FILES_DIR / "rulefit_top_rules.json", "w") as f:
        json.dump([[str(r), f"{c:.3f}"] for r, c in zip(_rf_df["rule"], _rf_df["coef"])], f)

    grl = build_greedy_rule_list(**grl_params)
    r_grl = fit_and_score(
        grl,
        data.X_train, data.y_train,
        data.X_test,  data.y_test,
        "GreedyRuleList",
    )
    # Save and print decoded GRL rules
    grl_rows = []
    _n_pos = int(data.y_train.sum())
    _n_neg = len(data.y_train) - _n_pos
    print("\nGreedyRuleList rules:")
    for i, rule in enumerate(grl.rules_[:-1]):
        feat = data.feature_names[rule["index_col"]]
        op   = "<=" if rule["flip"] else ">"
        cond = f"{feat} {op} {rule['cutoff']:.3f}"
        val  = rule["val_right"]
        n_pts = rule["num_pts_right"]
        pred = "Decesso" if val > 0.5 else "Sopravvissuto"
        if pred == "Decesso":
            prec = val
            rec  = (n_pts * val) / _n_pos if _n_pos else 0.0
        else:
            prec = 1.0 - val
            rec  = (n_pts * (1.0 - val)) / _n_neg if _n_neg else 0.0
        print(f"  Rule {i+1}: IF {cond}  →  {val:.1%} ({pred})  ({n_pts} pts)")
        grl_rows.append([cond, f"{prec:.3f}", f"{rec:.3f}", pred, str(n_pts)])
    default = grl.rules_[-1]
    val_def  = default["val"]
    n_def    = default["num_pts"]
    pred_def = "Decesso" if val_def > 0.5 else "Sopravvissuto"
    if pred_def == "Decesso":
        prec_def = val_def
        rec_def  = (n_def * val_def) / _n_pos if _n_pos else 0.0
    else:
        prec_def = 1.0 - val_def
        rec_def  = (n_def * (1.0 - val_def)) / _n_neg if _n_neg else 0.0
    print(f"  Default: {val_def:.1%} ({pred_def})  ({n_def} pts)")
    grl_rows.append(["DEFAULT", f"{prec_def:.3f}", f"{rec_def:.3f}", pred_def, str(n_def)])
    with open(FILES_DIR / "grl_rules.json", "w") as f:
        json.dump(grl_rows, f)

    # Final BRL uses full max_iter; maxcardinality from tuning
    brl = build_bayesian_rule_list(**brl_params)
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
        m_prob = re.search(r"probability of class 1:\s*([\d.]+)%\s*\(([\d.]+)%-([\d.]+)%\)", line)
        if m_prob:
            prob, lo, hi = m_prob.group(1), m_prob.group(2), m_prob.group(3)
            if "ELSE" in line and "ELSE IF" not in line:
                cond = "DEFAULT (nessuna regola attivata)"
            else:
                cond_m = re.search(r"IF\s+(.+?)\s+THEN", line)
                cond = cond_m.group(1).replace(" > 0.5", "").strip() if cond_m else line.strip()
            brl_rows.append([cond, f"{prob}%", f"{lo}%–{hi}%"])
    with open(FILES_DIR / "brl_rules.json", "w") as f:
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
    with open(FILES_DIR / "figs_structure.txt", "w") as f:
        f.write(str(figs))

    skope = build_skope_rules(**skope_params)
    r_skope = fit_and_score(
        skope,
        data.X_train, data.y_train,
        data.X_test,  data.y_test,
        "SkopeRules",
    )
    skope_rows = []
    _sk_y = pd.Series(data.y_train, index=data.X_train.index)
    _sk_n_pos = int(data.y_train.sum())
    if hasattr(skope, "rules_") and skope.rules_:
        print(f"Rules learned: {len(skope.rules_)}")
        for i, item in enumerate(skope.rules_):
            print(f"  Rule {i+1}: {item}")
            try:
                rule_str = item.rule if hasattr(item, "rule") else str(item)
                mask = _eval_rule_mask(rule_str, data.X_train)
                n_cov = int(mask.sum())
                tp    = int(_sk_y[mask].sum())
                prec  = f"{tp/n_cov:.3f}" if n_cov > 0 else "---"
                rec   = f"{tp/_sk_n_pos:.3f}" if _sk_n_pos > 0 else "---"
            except Exception:
                rule_str, prec, rec = str(item), "---", "---"
            skope_rows.append([rule_str, prec, rec])
    with open(FILES_DIR / "skope_rules.json", "w") as f:
        json.dump(skope_rows, f)

    ripper = build_ripper()
    r_ripper = fit_and_score(
        ripper,
        data.X_train, data.y_train,
        data.X_test,  data.y_test,
        "RIPPER",
    )
    # Decode RIPPER rules with sequential sample coverage + precision/recall
    ripper_rows = []
    _rip_n_pos = int(data.y_train.sum())
    _rip_n_neg = len(data.y_train) - _rip_n_pos
    _y_train_s = pd.Series(data.y_train, index=data.X_train.index)
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
        tp = int(_y_train_s.loc[X_rem.index][mask].sum())
        prec = tp / n_pts if n_pts else 0.0
        rec  = tp / _rip_n_pos if _rip_n_pos else 0.0
        print(f"  Rule {i+1}: IF {antecedent}  →  Decesso  prec={prec:.3f} rec={rec:.3f}  ({n_pts} pts)")
        ripper_rows.append([antecedent, f"{prec:.3f}", f"{rec:.3f}", str(n_pts)])
        X_rem = X_rem[~mask]
    n_default = len(X_rem)
    y_rem = _y_train_s.loc[X_rem.index]
    tn_def = int((y_rem == 0).sum())
    prec_def = tn_def / n_default if n_default else 0.0
    rec_def  = tn_def / _rip_n_neg if _rip_n_neg else 0.0
    print(f"  Default: Sopravvissuto  prec={prec_def:.3f} rec={rec_def:.3f}  ({n_default} pts)")
    ripper_rows.append(["DEFAULT", f"{prec_def:.3f}", f"{rec_def:.3f}", str(n_default)])
    with open(FILES_DIR / "ripper_rules.json", "w") as f:
        json.dump(ripper_rows, f)

    # ── 5. Ensemble models (with best params) ─────────────────────────────────
    print("\n" + "="*60)
    print("  ENSEMBLE MODELS")
    print("="*60)

    rf  = build_random_forest(**rf_params)
    gb  = build_gradient_boosting(**gb_params)
    xgb = build_xgboost(data.scale_pos_weight, **xgb_params)

    r_rf  = fit_and_score(rf,  data.X_train.values, data.y_train, data.X_test.values, data.y_test, "RandomForest")
    r_gb  = fit_and_score(gb,  data.X_train.values, data.y_train, data.X_test.values, data.y_test, "GradientBoosting")
    r_xgb = fit_and_score(xgb, data.X_train.values, data.y_train, data.X_test.values, data.y_test, "XGBoost")

    # ── 6. Evaluation & plots ─────────────────────────────────────────────────
    all_results = [r_rulefit, r_grl, r_brl, r_figs, r_skope, r_ripper, r_rf, r_gb, r_xgb]

    summary = summary_table(all_results)

    plot_metrics_comparison(summary)
    plot_roc_curves(all_results, data.y_test)
    plot_confusion_matrices(all_results, data.y_test)
    plot_rulefit_rules(rulefit)
    plot_feature_importance(rf, data.feature_names)
    plot_hyperparam_analysis(all_hyperparam_scores)

    # ── 7. Cross-validation (on trainval to maximise data) ────────────────────
    # RuleFit needs scaled features — scale X_trainval with the same scaler
    from sklearn.preprocessing import StandardScaler as _SS
    _scaler_tv = _SS()
    X_trainval_scaled = _scaler_tv.fit_transform(data.X_trainval.values)

    cv_entries = [
        {"name": "RuleFit",          "model": build_rulefit(),                                              "X": X_trainval_scaled},
        {"name": "GreedyRuleList",   "model": build_greedy_rule_list(**grl_params),                        "X": data.X_trainval},
        {"name": "RIPPER",           "model": build_ripper(),                                               "X": data.X_trainval},
        {"name": "FIGS",             "model": build_figs(),                                                 "X": data.X_trainval},
        {"name": "RandomForest",     "model": build_random_forest(**rf_params),                             "X": data.X_trainval.values},
        {"name": "GradientBoosting", "model": build_gradient_boosting(**gb_params),                        "X": data.X_trainval.values},
        {"name": "XGBoost",          "model": build_xgboost(data.scale_pos_weight, **xgb_params),          "X": data.X_trainval.values},
    ]
    run_cross_validation(cv_entries, data.y_trainval)

    # ── 8. Ablation: RF e GRL senza feature 'time' ───────────────────────────
    print("\n" + "="*60)
    print("  ABLATION: RF e GRL senza feature 'time'")
    print("="*60)

    X_nt = X.drop(columns=["time"])
    data_nt = prepare(X_nt, y)

    # Re-tune on no-time data (validation set of data_nt)
    print("\n  [ablation] re-tuning on no-time data...")
    rf_nt_params, _, _ = grid_search_val(
        lambda **kw: build_random_forest(**kw),
        {"n_estimators": [100, 200], "max_depth": [None, 5, 10], "min_samples_leaf": [1, 2, 5]},
        data_nt.X_train.values, data_nt.y_train, data_nt.X_val.values, data_nt.y_val,
        name="RF_no_time",
    )
    grl_nt_params, _, _ = grid_search_val(
        lambda **kw: build_greedy_rule_list(**kw),
        {"max_depth": [1, 2, 3, 4, 6]},
        data_nt.X_train, data_nt.y_train, data_nt.X_val, data_nt.y_val,
        name="GRL_no_time",
    )

    rf_nt  = build_random_forest(**rf_nt_params)
    grl_nt = build_greedy_rule_list(**grl_nt_params)

    r_rf_nt  = fit_and_score(rf_nt,  data_nt.X_train.values, data_nt.y_train,
                             data_nt.X_test.values, data_nt.y_test, "RF_no_time")
    r_grl_nt = fit_and_score(grl_nt, data_nt.X_train,        data_nt.y_train,
                             data_nt.X_test,        data_nt.y_test, "GRL_no_time")

    # Save ablation metrics
    abl_keys = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    with open(FILES_DIR / "ablation_no_time.csv", "w") as f:
        f.write("name," + ",".join(abl_keys) + "\n")
        for r in [r_rf, r_grl, r_rf_nt, r_grl_nt]:
            f.write(r["name"] + "," + ",".join(str(r[k]) for k in abl_keys) + "\n")

    # Save GRL no-time rules (same format as grl_rules.json: 5-element rows)
    grl_nt_rows = []
    _nt_n_pos = int(data_nt.y_train.sum())
    _nt_n_neg = len(data_nt.y_train) - _nt_n_pos
    print("\nGRL (no time) rules:")
    for i, rule in enumerate(grl_nt.rules_[:-1]):
        feat = data_nt.feature_names[rule["index_col"]]
        op   = "<=" if rule["flip"] else ">"
        cond = f"{feat} {op} {rule['cutoff']:.3f}"
        val  = rule["val_right"]
        n_pts = rule["num_pts_right"]
        pred = "Decesso" if val > 0.5 else "Sopravvissuto"
        if pred == "Decesso":
            prec_nt = val
            rec_nt  = (n_pts * val) / _nt_n_pos if _nt_n_pos else 0.0
        else:
            prec_nt = 1.0 - val
            rec_nt  = (n_pts * (1.0 - val)) / _nt_n_neg if _nt_n_neg else 0.0
        print(f"  Rule {i+1}: IF {cond}  →  {val:.1%} ({pred})  ({n_pts} pts)")
        grl_nt_rows.append([cond, f"{prec_nt:.3f}", f"{rec_nt:.3f}", pred, str(n_pts)])
    default_nt  = grl_nt.rules_[-1]
    val_nt_def  = default_nt["val"]
    n_nt_def    = default_nt["num_pts"]
    pred_nt_def = "Decesso" if val_nt_def > 0.5 else "Sopravvissuto"
    if pred_nt_def == "Decesso":
        prec_nt_def = val_nt_def
        rec_nt_def  = (n_nt_def * val_nt_def) / _nt_n_pos if _nt_n_pos else 0.0
    else:
        prec_nt_def = 1.0 - val_nt_def
        rec_nt_def  = (n_nt_def * (1.0 - val_nt_def)) / _nt_n_neg if _nt_n_neg else 0.0
    print(f"  Default: {val_nt_def:.1%} ({pred_nt_def})  ({n_nt_def} pts)")
    grl_nt_rows.append(["DEFAULT", f"{prec_nt_def:.3f}", f"{rec_nt_def:.3f}", pred_nt_def, str(n_nt_def)])
    with open(FILES_DIR / "grl_no_time_rules.json", "w") as f:
        json.dump(grl_nt_rows, f)

    generate_report_latex()
    print("\n✓ Pipeline completed. All outputs saved in results/ and Report/")


if __name__ == "__main__":
    main()
