"""Generate a detailed PDF report from the pipeline artefacts in results/."""

import textwrap
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.backends.backend_pdf import PdfPages

from config import RESULTS_DIR, PLOTS_DIR, DATA_DIR

REPORT_PATH = RESULTS_DIR / "report.pdf"

TITLE_COLOR  = "#1a1a2e"
ACCENT_COLOR = "#4C72B0"
LIGHT_GRAY   = "#f5f5f5"
A4           = (8.27, 11.69)
L_MARGIN     = 0.08   # figure-coordinate left margin

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})


# ── low-level helpers ─────────────────────────────────────────────────────────

def _new_fig() -> Figure:
    return plt.figure(figsize=A4)


def _add_header(fig: Figure, section: str, title: str) -> None:
    """Section label + title + horizontal rule, all in figure coords."""
    fig.text(L_MARGIN, 0.955, section.upper(),
             fontsize=7, color="gray", fontweight="bold", va="top")
    fig.text(L_MARGIN, 0.930, title,
             fontsize=15, color=TITLE_COLOR, fontweight="bold", va="top")
    fig.add_artist(Line2D([L_MARGIN, 0.92], [0.910, 0.910],
                          transform=fig.transFigure,
                          color=ACCENT_COLOR, linewidth=1.2))


def _embed_image(ax: Axes, path, title: str = "") -> None:
    if not path.exists():
        ax.text(0.5, 0.5, f"[immagine non trovata:\n{path.name}]",
                ha="center", va="center", color="gray", fontsize=9)
    else:
        ax.imshow(mpimg.imread(str(path)))
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=10, pad=4)


def _render_table(ax: Axes, df: pd.DataFrame, col_widths=None) -> None:
    ax.axis("off")
    tbl = ax.table(cellText=df.values.tolist(), colLabels=df.columns.tolist(),
                   cellLoc="center", loc="center", colWidths=col_widths)
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8.5)
    tbl.scale(1, 1.6)
    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor(ACCENT_COLOR)
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor(LIGHT_GRAY)
        cell.set_edgecolor("white")


def _write_paragraphs(fig: Figure, paragraphs: list[tuple[str, str]],
                      start_y: float = 0.88) -> float:
    """
    Place (heading, body) pairs on the figure using fig.text() — never clipped.
    Returns the y position after the last paragraph.
    """
    y = start_y
    for heading, body in paragraphs:
        if heading:
            fig.text(L_MARGIN, y, heading, fontsize=11, fontweight="bold",
                     color=TITLE_COLOR, va="top")
            y -= 0.038
        for line in textwrap.wrap(body, width=100):
            fig.text(L_MARGIN, y, line, fontsize=9.5, color="#333333", va="top")
            y -= 0.024
        y -= 0.016   # gap between paragraphs
    return y


def _chapter_divider(pdf: PdfPages, number: str, title: str, subtitle: str) -> None:
    fig = _new_fig()
    fig.patch.set_facecolor(TITLE_COLOR)
    fig.text(0.5, 0.58, f"Capitolo {number}", fontsize=13,
             color="#7788bb", ha="center", fontweight="bold")
    fig.text(0.5, 0.51, title, fontsize=28, color="white",
             ha="center", fontweight="bold")
    fig.add_artist(Line2D([0.20, 0.80], [0.48, 0.48],
                          transform=fig.transFigure,
                          color=ACCENT_COLOR, linewidth=1))
    fig.text(0.5, 0.44, subtitle, fontsize=12,
             color="#aabbcc", ha="center")
    pdf.savefig(fig)
    plt.close(fig)


# ── cover ─────────────────────────────────────────────────────────────────────

def _page_cover(pdf: PdfPages) -> None:
    fig = _new_fig()
    fig.patch.set_facecolor(TITLE_COLOR)
    fig.text(0.5, 0.62, "XAI Project", fontsize=36, color="white",
             ha="center", fontweight="bold")
    fig.text(0.5, 0.55, "Heart Failure Clinical Records", fontsize=18,
             color="#aaaacc", ha="center")
    fig.text(0.5, 0.47,
             "Confronto tra metodi interpretabili basati su regole\n"
             "e metodi non interpretabili basati su ensemble di alberi",
             fontsize=12, color="#ccccdd", ha="center", linespacing=1.8)
    fig.add_artist(Line2D([0.20, 0.80], [0.44, 0.44],
                          transform=fig.transFigure,
                          color=ACCENT_COLOR, linewidth=1))
    fig.text(0.5, 0.38, "Dataset: UCI Heart Failure Clinical Records (id=519)",
             fontsize=9, color="#aaaacc", ha="center")
    fig.text(0.5, 0.35, "Librerie: imodels · scikit-learn · xgboost",
             fontsize=9, color="#aaaacc", ha="center")
    pdf.savefig(fig)
    plt.close(fig)


# ── capitolo 1: introduzione ──────────────────────────────────────────────────

def _page_intro(pdf: PdfPages) -> None:
    fig = _new_fig()
    _add_header(fig, "1.1 · Introduzione", "Contesto e motivazione")
    _write_paragraphs(fig, [
        ("Il problema",
         "L'insufficienza cardiaca è una condizione clinica grave in cui il cuore non riesce a "
         "pompare sangue in modo sufficiente. Identificare i fattori di rischio associati al "
         "decesso è fondamentale per supportare le decisioni cliniche e migliorare la prognosi "
         "dei pazienti."),
        ("Obiettivo",
         "Il progetto confronta due famiglie di modelli di machine learning per la predizione "
         "della mortalità: modelli interpretabili basati su regole (rule-based) e modelli non "
         "interpretabili basati su ensemble di alberi. L'obiettivo è valutare il trade-off tra "
         "performance predittiva e interpretabilità in un contesto clinico ad alta criticità."),
        ("Explainable AI in ambito clinico",
         "In medicina, un modello non spiegabile — per quanto accurato — non è direttamente "
         "utilizzabile come strumento di supporto decisionale: il medico deve poter comprendere "
         "e validare le motivazioni di ogni predizione. I modelli intrinsecamente interpretabili "
         "soddisfano questo requisito senza richiedere metodi post-hoc aggiuntivi (es. SHAP, LIME), "
         "che possono introdurre approssimazioni e spiegazioni fuorvianti."),
        ("Struttura del report",
         "Il Capitolo 1 descrive il dataset e l'analisi esplorativa. Il Capitolo 2 illustra i "
         "modelli utilizzati. Il Capitolo 3 presenta i risultati, il confronto delle performance "
         "e le conclusioni."),
    ])
    pdf.savefig(fig)
    plt.close(fig)


def _page_dataset(pdf: PdfPages) -> None:
    cache_X = DATA_DIR / "X.csv"
    cache_y = DATA_DIR / "y.csv"

    fig = _new_fig()
    _add_header(fig, "1.2 · Introduzione", "Il Dataset")

    if cache_X.exists() and cache_y.exists():
        X = pd.read_csv(cache_X)
        y = pd.read_csv(cache_y).iloc[:, 0]
        stats = (f"Campioni: {len(X)}   |   Feature: {X.shape[1]}   |   "
                 f"Decessi: {int(y.sum())} ({y.mean():.1%})   |   "
                 f"Sopravvissuti: {int((y == 0).sum())} ({(y == 0).mean():.1%})")
    else:
        stats = "Dati non disponibili — eseguire prima main.py"

    _write_paragraphs(fig, [
        ("Descrizione",
         "Il dataset Heart Failure Clinical Records (UCI, id=519) raccoglie cartelle cliniche "
         "di 299 pazienti con insufficienza cardiaca seguiti in follow-up. La variabile target "
         "DEATH_EVENT indica se il paziente è deceduto durante il periodo di osservazione."),
        ("", stats),
        ("Gestione delle feature: binarie vs continue",
         "Le 12 feature si dividono in 5 binarie (anaemia, diabetes, high_blood_pressure, sex, "
         "smoking — valori 0/1) e 7 continue (age, creatinine_phosphokinase, ejection_fraction, "
         "platelets, serum_creatinine, serum_sodium, time). I modelli ensemble e la maggior parte "
         "dei rule-based le ricevono direttamente. RuleFit richiede standardizzazione (StandardScaler), "
         "mentre BayesianRuleList richiede feature esclusivamente binarie: le variabili continue "
         "vengono quindi discretizzate in bin tramite KBinsDiscretizer con one-hot encoding prima "
         "del fitting."),
    ], start_y=0.88)

    # feature table
    ax = fig.add_axes((0.05, 0.04, 0.90, 0.50))
    _render_table(ax, pd.DataFrame({
        "Feature": [
            "age", "anaemia", "creatinine_phosphokinase", "diabetes",
            "ejection_fraction", "high_blood_pressure", "platelets",
            "serum_creatinine", "serum_sodium", "sex", "smoking", "time",
        ],
        "Tipo": [
            "continua", "binaria", "continua", "binaria",
            "continua", "binaria", "continua",
            "continua", "continua", "binaria", "binaria", "continua",
        ],
        "Descrizione": [
            "Età del paziente (anni)",
            "Presenza di anemia (0/1)",
            "Livello CPK nel sangue (mcg/L)",
            "Presenza di diabete (0/1)",
            "Percentuale di sangue pompata dal ventricolo sinistro (%)",
            "Presenza di ipertensione (0/1)",
            "Conta piastrinica (kiloplatelets/mL)",
            "Livello di creatinina serica (mg/dL)",
            "Livello di sodio sierico (mEq/L)",
            "Sesso biologico (0=F, 1=M)",
            "Fumatore (0/1)",
            "Durata del follow-up (giorni)",
        ],
    }), col_widths=[0.22, 0.12, 0.66])

    pdf.savefig(fig)
    plt.close(fig)


def _page_eda(pdf: PdfPages) -> None:
    for img_file, section, title in [
        ("eda_continuous.png",  "1.3 · Introduzione", "EDA — Distribuzioni per classe"),
        ("eda_correlation.png", "1.4 · Introduzione", "EDA — Matrice di correlazione"),
    ]:
        fig = _new_fig()
        _add_header(fig, section, title)
        ax = fig.add_axes((0.04, 0.04, 0.92, 0.84))
        _embed_image(ax, PLOTS_DIR / img_file)
        pdf.savefig(fig)
        plt.close(fig)


# ── capitolo 2: modelli ───────────────────────────────────────────────────────

def _page_models_rules(pdf: PdfPages) -> None:
    fig = _new_fig()
    _add_header(fig, "2.1 · Modelli", "Modelli Interpretabili Basati su Regole")
    _write_paragraphs(fig, [
        ("",
         "I modelli basati su regole producono predizioni nella forma di liste di condizioni "
         "if-then direttamente leggibili. Ogni regola può essere validata da un esperto di "
         "dominio senza strumenti aggiuntivi."),
        ("RuleFit",
         "Estrae regole da un ensemble di alberi, poi allena una regressione lineare penalizzata "
         "(Lasso) sui valori delle regole. Il coefficiente di ogni regola indica il suo peso "
         "nella predizione finale. Richiede feature standardizzate."),
        ("GreedyRuleList",
         "Costruisce una lista ordinata di regole if-then in modo greedy, massimizzando ad ogni "
         "passo la riduzione dell'impurità. La lista si legge dall'alto verso il basso: si "
         "applica la prima regola che si attiva sul campione."),
        ("BayesianRuleList (BRL)",
         "Apprende una lista di regole tramite inferenza Bayesiana (MCMC). Richiede feature "
         "binarie: le variabili continue vengono discretizzate in bin (one-hot encoding) prima "
         "del fitting. Produce liste compatte e altamente leggibili."),
        ("FIGS — Fast Interpretable Greedy-Tree Sums",
         "Il modello è una somma di piccoli alberi decisionali, ciascuno appreso in modo greedy "
         "sui residui del passo precedente. Combina interpretabilità strutturale con una "
         "capacità espressiva maggiore rispetto a un singolo albero."),
        ("SkopeRules",
         "Genera regole ad alta precisione e recall tramite sub-campionamenti ripetuti di "
         "ensemble di alberi, poi de-duplica le regole simili. Ogni regola è corredata da "
         "statistiche di precisione, recall e support sul training set."),
    ])
    pdf.savefig(fig)
    plt.close(fig)


def _page_models_ensemble(pdf: PdfPages) -> None:
    fig = _new_fig()
    _add_header(fig, "2.2 · Modelli", "Modelli Ensemble Non Interpretabili")
    _write_paragraphs(fig, [
        ("",
         "I modelli ensemble combinano centinaia di alberi decisionali per ottenere performance "
         "superiori, ma il processo decisionale risulta opaco (black box). Sono usati come "
         "baseline di riferimento per valutare il gap di performance rispetto ai modelli "
         "interpretabili."),
        ("Random Forest",
         "Allena N alberi decisionali su sotto-campioni casuali del dataset (bagging) e su "
         "sottoinsiemi casuali delle feature (feature bagging). La predizione finale è la "
         "media delle probabilità dei singoli alberi. Riduce la varianza rispetto al singolo "
         "albero. Configurato con class_weight='balanced' per gestire lo sbilanciamento."),
        ("Gradient Boosting (sklearn)",
         "Allena alberi sequenzialmente: ogni albero corregge gli errori del precedente "
         "minimizzando il gradiente della loss (boosting). Riduce il bias rispetto al "
         "bagging. Parametri: 200 stimatori, learning rate 0.05, profondità massima 4, "
         "subsample 0.8."),
        ("XGBoost",
         "Implementazione ottimizzata del gradient boosting con regolarizzazione L1/L2, "
         "gestione nativa dei valori mancanti e parallelismo. Parametro scale_pos_weight "
         "impostato automaticamente per bilanciare le classi in base alla distribuzione "
         "del training set."),
        ("Preprocessing",
         "I modelli ensemble non richiedono standardizzazione delle feature. Le variabili "
         "continue e binarie sono passate direttamente come array NumPy. Solo RuleFit "
         "utilizza feature standardizzate (StandardScaler); BRL utilizza feature "
         "discretizzate in bin binari (KBinsDiscretizer + one-hot encoding)."),
    ])
    pdf.savefig(fig)
    plt.close(fig)


# ── capitolo 3: risultati ─────────────────────────────────────────────────────

def _page_metrics(pdf: PdfPages) -> None:
    csv_path = RESULTS_DIR / "metrics_summary.csv"
    if not csv_path.exists():
        return

    df = pd.read_csv(csv_path).rename(columns={"name": "Modello"})
    df["ROC-AUC"] = df["ROC-AUC"].round(4)

    fig = _new_fig()
    _add_header(fig, "3.1 · Risultati", "Metriche Comparative")

    ax_tbl = fig.add_axes((0.04, 0.52, 0.92, 0.36))
    cols = ["Modello", "Type", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    _render_table(ax_tbl, df[[c for c in cols if c in df.columns]].fillna("—"))

    ax_img = fig.add_axes((0.04, 0.04, 0.92, 0.46))
    _embed_image(ax_img, PLOTS_DIR / "metrics_comparison.png")

    pdf.savefig(fig)
    plt.close(fig)


def _page_roc(pdf: PdfPages) -> None:
    fig = _new_fig()
    _add_header(fig, "3.2 · Risultati", "Curve ROC")
    ax = fig.add_axes((0.04, 0.04, 0.92, 0.84))
    _embed_image(ax, PLOTS_DIR / "roc_curves.png")
    pdf.savefig(fig)
    plt.close(fig)


def _page_cm(pdf: PdfPages) -> None:
    fig = _new_fig()
    _add_header(fig, "3.3 · Risultati", "Matrici di Confusione")
    ax = fig.add_axes((0.04, 0.04, 0.92, 0.84))
    _embed_image(ax, PLOTS_DIR / "confusion_matrices.png")
    pdf.savefig(fig)
    plt.close(fig)


def _page_interpretability(pdf: PdfPages) -> None:
    for img_file, section, title in [
        ("rulefit_rules.png",        "3.4 · Risultati", "Interpretabilità — Regole RuleFit"),
        ("feature_importance_rf.png","3.4 · Risultati", "Interpretabilità — Feature Importance RF"),
        ("cross_validation.png",     "3.5 · Risultati", "Cross-Validation 10-fold"),
    ]:
        fig = _new_fig()
        _add_header(fig, section, title)
        ax = fig.add_axes((0.04, 0.04, 0.92, 0.84))
        _embed_image(ax, PLOTS_DIR / img_file)
        pdf.savefig(fig)
        plt.close(fig)


def _page_conclusions(pdf: PdfPages) -> None:
    csv_path = RESULTS_DIR / "metrics_summary.csv"

    fig = _new_fig()
    _add_header(fig, "3.6 · Risultati", "Conclusioni")

    y = _write_paragraphs(fig, [
        ("Trade-off interpretabilità vs performance",
         "I modelli ensemble raggiungono generalmente performance superiori in termini di "
         "ROC-AUC rispetto ai modelli basati su regole. Tuttavia il divario non è sempre "
         "significativo, e i modelli interpretabili risultano competitivi su dataset di "
         "dimensioni ridotte come questo (299 campioni)."),
        ("Rilevanza clinica delle regole",
         "Le feature più ricorrenti nelle regole apprese — serum_creatinine, ejection_fraction "
         "e time — sono coerenti con la letteratura clinica sull'insufficienza cardiaca. "
         "Questo costituisce una forma di validazione qualitativa del modello."),
        ("Vantaggio dei modelli interpretabili",
         "In contesti ad alta criticità come la medicina clinica, la possibilità di ispezionare "
         "esplicitamente le regole è fondamentale. I modelli intrinsecamente interpretabili "
         "soddisfano i requisiti di trasparenza senza richiedere metodi post-hoc, riducendo "
         "il rischio di spiegazioni fuorvianti."),
        ("Nota sul dataset",
         "Con soli 299 campioni e un moderato sbilanciamento delle classi (~32% positivi), "
         "i risultati sul test set sono soggetti a varianza. La cross-validation a 10 fold "
         "fornisce una stima più robusta delle performance reali."),
    ])

    if csv_path.exists():
        df = pd.read_csv(csv_path)
        best_rule = df[df["Type"] == "Rule-based"].sort_values("ROC-AUC", ascending=False).iloc[0]
        best_ens  = df[df["Type"] == "Ensemble"].sort_values("ROC-AUC", ascending=False).iloc[0]

        fig.text(L_MARGIN, y - 0.01, "Migliori modelli per categoria:",
                 fontsize=10, fontweight="bold", color=TITLE_COLOR, va="top")
        y -= 0.055

        summary_df = pd.DataFrame([
            ["Rule-based", best_rule["name"], f"{best_rule['ROC-AUC']:.4f}", f"{best_rule['F1']:.4f}"],
            ["Ensemble",   best_ens["name"],  f"{best_ens['ROC-AUC']:.4f}",  f"{best_ens['F1']:.4f}"],
        ], columns=["Categoria", "Modello", "ROC-AUC", "F1"])

        table_h = 0.12
        table_y = max(0.04, y - table_h)
        ax_tbl = fig.add_axes((L_MARGIN, table_y, 1 - 2 * L_MARGIN, table_h))
        _render_table(ax_tbl, summary_df)

    pdf.savefig(fig)
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def generate_report() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    with PdfPages(REPORT_PATH) as pdf:
        _page_cover(pdf)

        _chapter_divider(pdf, "1", "Introduzione",
                         "Contesto del problema, dataset e analisi esplorativa")
        _page_intro(pdf)
        _page_dataset(pdf)
        _page_eda(pdf)

        _chapter_divider(pdf, "2", "Modelli",
                         "Approcci interpretabili basati su regole ed ensemble non interpretabili")
        _page_models_rules(pdf)
        _page_models_ensemble(pdf)

        _chapter_divider(pdf, "3", "Risultati",
                         "Valutazione delle performance e confronto tra approcci")
        _page_metrics(pdf)
        _page_roc(pdf)
        _page_cm(pdf)
        _page_interpretability(pdf)
        _page_conclusions(pdf)

        pdf.infodict().update({
            "Title":   "XAI Project — Heart Failure Clinical Records",
            "Subject": "Confronto modelli interpretabili vs ensemble",
        })

    print(f"Report saved to {REPORT_PATH}")


if __name__ == "__main__":
    generate_report()
