"""Generate a detailed PDF report from the pipeline artefacts in results/."""

import json
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

REPORT_PATH = RESULTS_DIR / "Manneschi_XAI.pdf"

TITLE_COLOR  = "#1a1a2e"
ACCENT_COLOR = "#4C72B0"
LIGHT_GRAY   = "#f5f5f5"
A4           = (8.27, 11.69)
L_MARGIN     = 0.08

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})


# ── low-level helpers ─────────────────────────────────────────────────────────

def _new_fig() -> Figure:
    return plt.figure(figsize=A4)


def _add_header(fig: Figure, section: str, title: str) -> None:
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
    for (r, _), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor(ACCENT_COLOR)
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor(LIGHT_GRAY)
        cell.set_edgecolor("white")


def _write_paragraphs(fig: Figure, paragraphs: list[tuple[str, str]],
                      start_y: float = 0.88) -> float:
    """Place (heading, body) pairs using fig.text(). Returns final y."""
    y = start_y
    for heading, body in paragraphs:
        if heading:
            fig.text(L_MARGIN, y, heading, fontsize=11, fontweight="bold",
                     color=TITLE_COLOR, va="top")
            y -= 0.038
        for line in textwrap.wrap(body, width=100):
            fig.text(L_MARGIN, y, line, fontsize=9.5, color="#333333", va="top")
            y -= 0.024
        y -= 0.016
    return y


def _add_description(fig: Figure, text: str, y_start: float = 0.22) -> None:
    """Description text below the image area."""
    y = y_start
    for line in textwrap.wrap(text, width=100):
        fig.text(L_MARGIN, y, line, fontsize=9.5, color="#555555", va="top")
        y -= 0.024


# ── title page + table of contents ───────────────────────────────────────────

def _page_toc(pdf: PdfPages) -> None:
    fig = _new_fig()

    # Title block
    fig.text(0.5, 0.88, "XAI Project", fontsize=30, color=TITLE_COLOR,
             ha="center", fontweight="bold", va="top")
    fig.text(0.5, 0.82, "Heart Failure Clinical Records", fontsize=16,
             color=ACCENT_COLOR, ha="center", va="top")
    fig.text(0.5, 0.77,
             "Confronto tra modelli interpretabili basati su regole\n"
             "e modelli ensemble non interpretabili",
             fontsize=10, color="#555555", ha="center", va="top", linespacing=1.7)
    fig.add_artist(Line2D([L_MARGIN, 1 - L_MARGIN], [0.73, 0.73],
                          transform=fig.transFigure, color=ACCENT_COLOR, linewidth=1.2))

    # TOC heading
    fig.text(L_MARGIN, 0.695, "Indice", fontsize=13, color=TITLE_COLOR,
             fontweight="bold", va="top")

    toc: list[tuple[str, str, int | None]] = [
        ("1.", "Introduzione del problema e dataset", None),
        ("1.1", "Contesto e motivazione", 2),
        ("1.2", "Il Dataset", 3),
        ("1.3", "EDA — Distribuzioni per classe", 4),
        ("1.4", "EDA — Matrice di correlazione", 5),
        ("2.", "Architettura e Implementazione", None),
        ("2.1", "Struttura del Progetto", 6),
        ("2.2", "Pipeline di Elaborazione dei Dati", 7),
        ("2.3", "Modelli Interpretabili Basati su Regole", 9),
        ("2.4", "Modelli Ensemble Non Interpretabili", 10),
        ("2.5", "Configurazione e Iperparametri", 11),
        ("3.", "Risultati", None),
        ("3.1", "Metriche Comparative", 12),
        ("3.2", "Curve ROC", 13),
        ("3.3", "Matrici di Confusione", 14),
        ("3.4", "Regole Complete: tutti i modelli interpretabili", 15),
        ("3.5", "Cross-Validation 10-fold", 17),
        ("3.6", "RF e GRL senza feature time", 18),
        ("3.7", "Interpretabilità — Regole RuleFit e Feature Importance", 19),
        ("4.", "Conclusioni", None),
        ("4.1", "Confronto, analisi clinica e limiti", 21),
    ]

    y = 0.645
    for num, title, page in toc:
        is_chapter = page is None
        indent = L_MARGIN if is_chapter else L_MARGIN + 0.04
        size = 11 if is_chapter else 9.5
        weight = "bold" if is_chapter else "normal"
        color = TITLE_COLOR if is_chapter else "#333333"
        if is_chapter:
            y -= 0.006
        fig.text(indent, y, f"{num}  {title}", fontsize=size,
                 fontweight=weight, color=color, va="top")
        if page is not None:
            fig.text(1 - L_MARGIN, y, str(page), fontsize=9.5,
                     color="#888888", va="top", ha="right")
        y -= 0.032 if is_chapter else 0.025

    pdf.savefig(fig)
    plt.close(fig)


# ── capitolo 1: introduzione ──────────────────────────────────────────────────

def _page_intro(pdf: PdfPages) -> None:
    fig = _new_fig()
    _add_header(fig, "1.1 · Introduzione del problema e dataset", "Contesto e motivazione")
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
         "Il Capitolo 1 introduce il problema clinico, descrive il dataset e presenta l'analisi "
         "esplorativa. Il Capitolo 2 illustra l'architettura del progetto, la pipeline di "
         "elaborazione dei dati, i modelli utilizzati e la loro configurazione. Il Capitolo 3 "
         "presenta i risultati e il confronto delle performance. Il Capitolo 4 raccoglie le "
         "conclusioni, l'analisi clinica dei risultati e i limiti del lavoro."),
    ])
    pdf.savefig(fig)
    plt.close(fig)


def _page_dataset(pdf: PdfPages) -> None:
    cache_X = DATA_DIR / "X.csv"
    cache_y = DATA_DIR / "y.csv"

    fig = _new_fig()
    _add_header(fig, "1.2 · Introduzione del problema e dataset", "Il Dataset")

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
    pages = [
        (
            "eda_continuous.png", "1.3 · Introduzione del problema e dataset", "EDA — Distribuzioni per classe",
            "Ogni istogramma sovrappone la distribuzione di una feature per sopravvissuti (blu) e "
            "deceduti (rosso): più le due curve sono separate, maggiore è il potere predittivo di "
            "quella variabile. time è il discriminatore più potente — un follow-up breve corrisponde "
            "quasi sempre a decesso, segno di rapido deterioramento clinico. serum_creatinine alta "
            "indica compromissione renale, spesso associata a esiti peggiori; ejection_fraction bassa "
            "segnala che il ventricolo sinistro pompa una quota insufficiente di sangue. platelets e "
            "creatinine_phosphokinase mostrano distribuzioni largamente sovrapposte: contributo "
            "predittivo individuale marginale.",
        ),
        (
            "eda_correlation.png", "1.4 · Introduzione del problema e dataset", "EDA — Matrice di correlazione",
            "Ogni cella mostra la correlazione di Pearson tra due variabili: vale +1 se quando "
            "una sale anche l'altra sale sempre, -1 se quando una sale l'altra scende sempre, 0 "
            "se non c'è alcuna relazione lineare tra le due. Valori vicini a ±0.5 o oltre "
            "indicano una relazione abbastanza chiara; vicini a 0 indicano variabili "
            "sostanzialmente indipendenti. time ha la correlazione più forte con DEATH_EVENT "
            "(-0.53): pazienti con follow-up lungo tendono a sopravvivere. serum_creatinine "
            "(+0.29) ed ejection_fraction (-0.27) sono i segnali biochimici più informativi. "
            "Le feature binarie mostrano correlazioni deboli, sotto ±0.15.",
        ),
    ]
    for img_file, section, title, description in pages:
        fig = _new_fig()
        _add_header(fig, section, title)
        ax = fig.add_axes((0.04, 0.24, 0.92, 0.65))
        _embed_image(ax, PLOTS_DIR / img_file)
        _add_description(fig, description)
        pdf.savefig(fig)
        plt.close(fig)


# ── capitolo 2: architettura e implementazione ────────────────────────────────

def _page_project_structure(pdf: PdfPages) -> None:
    fig = _new_fig()
    _add_header(fig, "2.1 · Architettura e Implementazione", "Struttura del Progetto")

    _write_paragraphs(fig, [
        ("Organizzazione del codice",
         "Il progetto è organizzato in una cartella src/ con sei moduli distinti, ciascuno "
         "con una responsabilità precisa. Questa separazione permette di modificare una "
         "componente senza toccare le altre e rende il codice più leggibile e testabile."),
    ], start_y=0.88)

    ax = fig.add_axes((0.05, 0.52, 0.90, 0.30))
    _render_table(ax, pd.DataFrame({
        "Modulo": [
            "config.py", "data_loader.py", "models.py",
            "evaluate.py", "main.py", "report.py",
        ],
        "Responsabilità": [
            "Centralizza iperparametri, path e costanti — unico punto di modifica",
            "Download UCI, cache CSV, split stratificato, tre varianti di preprocessing",
            "Factory function per ciascuno degli 8 modelli — nessuna logica di training",
            "Calcolo metriche, generazione di tutti i grafici, cross-validation",
            "Orchestrazione dell'intera pipeline end-to-end, in ordine fisso",
            "Lettura degli artefatti salvati e generazione del PDF report",
        ],
    }), col_widths=[0.20, 0.80])

    _write_paragraphs(fig, [
        ("Output della pipeline",
         "Ogni esecuzione di main.py produce artefatti salvati su disco: i dati grezzi "
         "vengono cachati in data/ (X.csv, y.csv) per evitare di riscaricarli ogni volta. "
         "Le metriche vengono salvate in results/metrics_summary.csv. I grafici vengono "
         "salvati in results/plots/ come PNG ad alta risoluzione (150 DPI). Il report "
         "finale viene scritto in results/report.pdf. Questa separazione tra esecuzione "
         "e report permette di rigenerare il PDF senza rieseguire l'intera pipeline."),
        ("Dipendenze principali",
         "scikit-learn gestisce preprocessing, metriche e cross-validation. imodels "
         "fornisce le implementazioni dei modelli rule-based (RuleFit, GreedyRuleList, "
         "BayesianRuleList, FIGS, SkopeRules). xgboost implementa XGBoost. "
         "ucimlrepo scarica il dataset direttamente dal repository UCI. matplotlib e "
         "seaborn gestiscono la generazione di tutti i grafici e del report PDF."),
    ], start_y=0.48)

    pdf.savefig(fig)
    plt.close(fig)


def _page_data_pipeline(pdf: PdfPages) -> None:
    # ── Page 1: acquisizione, split, tre varianti ─────────────────────────────
    fig = _new_fig()
    _add_header(fig, "2.2 · Architettura e Implementazione", "Pipeline di Elaborazione dei Dati")
    _write_paragraphs(fig, [
        ("Split stratificato",
         "Il dataset viene diviso in training (80%, 239 campioni) e test (20%, 60 campioni) "
         "con train_test_split(stratify=y): la stratificazione garantisce che entrambe le parti "
         "mantengano la stessa proporzione di deceduti (~32%) presente nel dataset originale. "
         "Senza stratificazione, con un dataset così piccolo, uno split casuale potrebbe "
         "produrre un test set con proporzioni molto diverse, rendendo le metriche poco "
         "affidabili. Il seed fisso (RANDOM_STATE=42) garantisce la riproducibilità."),
        ("Tre varianti di preprocessing",
         "I modelli richiedono rappresentazioni diverse delle stesse feature, quindi il "
         "preprocessing produce tre versioni del training e test set in parallelo. "
         "La versione raw è il DataFrame originale con le 12 feature così come sono: "
         "usata da GreedyRuleList, FIGS, SkopeRules e dai modelli ensemble (come array "
         "NumPy). La versione scalata applica StandardScaler, che porta ogni feature "
         "continua a media 0 e varianza 1: necessaria per RuleFit, che usa la regressione "
         "Lasso internamente e quindi è sensibile alla scala delle variabili. La versione "
         "discretizzata usa KBinsDiscretizer(n_bins=4, encode='onehot-dense', "
         "strategy='quantile'): ogni feature continua viene suddivisa in 4 intervalli di "
         "uguale frequenza, ciascuno codificato come colonna binaria. Il risultato è un "
         "array di circa 33 colonne binarie (le feature continue producono 4 bin ciascuna, "
         "le 5 binarie ne producono 1), obbligatorio per "
         "BayesianRuleList che per design accetta solo valori strettamente 0/1."),
    ])
    pdf.savefig(fig)
    plt.close(fig)

    # ── Page 2: PreparedData, sbilanciamento ──────────────────────────────────
    fig2 = _new_fig()
    _add_header(fig2, "2.2 · Architettura e Implementazione", "Pipeline di Elaborazione dei Dati (continua)")
    _write_paragraphs(fig2, [
        ("PreparedData — struttura dati condivisa",
         "Il risultato del preprocessing è incapsulato in un dataclass PreparedData che "
         "raccoglie tutte le versioni in un unico oggetto: X_train/X_test (raw), "
         "X_train_scaled/X_test_scaled, X_train_disc/X_test_disc, y_train/y_test, "
         "la lista dei nomi delle feature e scale_pos_weight per XGBoost. main.py accede "
         "ai campi tramite data.X_train_scaled, data.y_test ecc., senza dover gestire "
         "decine di variabili separate. Lo scaler e il discretizzatore sono fittati solo "
         "sul training set e applicati al test set con transform(), evitando data leakage."),
        ("Sbilanciamento delle classi",
         "Il dataset contiene 203 sopravvissuti e 96 deceduti (~32% positivi). Questo "
         "sbilanciamento è gestito in modo diverso per ciascun modello: Random Forest usa "
         "class_weight='balanced' che pesa ogni campione inversamente alla frequenza della "
         "sua classe. XGBoost usa scale_pos_weight calcolato automaticamente come rapporto "
         "tra negativi e positivi nel training set. Gli altri modelli non hanno meccanismi "
         "nativi e trattano le classi con peso uguale."),
    ])
    pdf.savefig(fig2)
    plt.close(fig2)


def _page_models_rules(pdf: PdfPages) -> None:
    fig = _new_fig()
    _add_header(fig, "2.3 · Architettura e Implementazione", "Modelli Interpretabili Basati su Regole")
    _write_paragraphs(fig, [
        ("",
         "I modelli basati su regole producono predizioni if-then direttamente leggibili da un "
         "esperto di dominio, senza richiedere strumenti post-hoc aggiuntivi."),
        ("RuleFit",
         "Estrae regole da un ensemble di alberi e le pondera con regressione Lasso. Il "
         "coefficiente di ogni regola indica il suo contributo alla predizione; richiede feature standardizzate."),
        ("GreedyRuleList",
         "Costruisce una lista ordinata di regole if-then in modo greedy. Si applica la prima "
         "regola soddisfatta; se nessuna si attiva, vale la regola di default in fondo alla lista."),
        ("BayesianRuleList (BRL)",
         "Apprende una lista di regole tramite MCMC su feature binarie. Le variabili continue "
         "vengono discretizzate in bin prima del fitting; produce stime probabilistiche con IC al 95%."),
        ("FIGS — Fast Interpretable Greedy-Tree Sums",
         "Somma di piccoli alberi decisionali, ciascuno appreso greedy sui residui del precedente. "
         "Struttura additiva trasparente con capacità espressiva maggiore di un singolo albero."),
        ("SkopeRules",
         "Genera regole ad alta precisione tramite sub-campionamenti ripetuti di ensemble di alberi, "
         "poi de-duplica quelle simili. Ogni regola è corredata da precisione, recall e support."),
        ("RIPPER — Repeated Incremental Pruning to Produce Error Reduction",
         "Apprende regole congiuntive (AND tra più condizioni simultanee) in modo incrementale, "
         "potando ogni regola per ridurre l'errore. Il classificatore scorre le regole in ordine: "
         "la prima che si attiva predice Decesso; se nessuna si attiva, la predizione è "
         "Sopravvissuto (DEFAULT). A differenza di GRL, può combinare più feature per regola."),
    ])
    pdf.savefig(fig)
    plt.close(fig)


def _page_models_ensemble(pdf: PdfPages) -> None:
    fig = _new_fig()
    _add_header(fig, "2.4 · Architettura e Implementazione", "Modelli Ensemble Non Interpretabili")
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
    ])
    pdf.savefig(fig)
    plt.close(fig)


def _page_config(pdf: PdfPages) -> None:
    fig = _new_fig()
    _add_header(fig, "2.5 · Architettura e Implementazione", "Configurazione e Iperparametri")

    intro = ("Tutti i parametri del progetto sono centralizzati in config.py, importato da "
             "ogni modulo. Questo garantisce che una singola modifica si propaghi all'intera "
             "pipeline senza rischio di disallineamenti. Di seguito la motivazione di ogni scelta.")
    y = 0.88
    for line in textwrap.wrap(intro, width=100):
        fig.text(L_MARGIN, y, line, fontsize=9.5, color="#333333", va="top")
        y -= 0.024
    y -= 0.010

    def _section_table(title, rows, top_y, height):
        fig.text(L_MARGIN, top_y + height + 0.018, title,
                 fontsize=10, fontweight="bold", color=TITLE_COLOR, va="bottom")
        ax = fig.add_axes((L_MARGIN, top_y, 1 - 2 * L_MARGIN, height))
        df_t = pd.DataFrame(rows, columns=["Parametro", "Valore", "Motivazione"])
        _render_table(ax, df_t, col_widths=[0.28, 0.10, 0.62])

    # ── Parametri generali ───────────────────────────────────────────────────
    gen_rows = [
        ["RANDOM_STATE",    "42",   "Seed fisso per riproducibilità di split e modelli stocastici"],
        ["TEST_SIZE",       "0.2",  "20% test (60 campioni); bilancia valutazione e training"],
        ["CV_FOLDS",        "10",   "10-fold CV: stima robusta su dataset piccoli"],
        ["BINARY_FEATURES", "(5)",  "Feature binarie escluse da StandardScaler"],
    ]
    gen_h = 0.105
    gen_top = y - 0.045 - gen_h
    _section_table("Parametri generali", gen_rows, gen_top, gen_h)
    y = gen_top - 0.048

    # ── Modelli rule-based ───────────────────────────────────────────────────
    rule_rows = [
        ["RULEFIT_MAX_RULES",   "30",     "Limita le regole candidate; troppe degenerano in black-box"],
        ["GRL_MAX_DEPTH",       "6",      "Profondità massima della lista greedy (default libreria)"],
        ["BRL_MAX_ITER",        "20 000", "Iterazioni MCMC sufficienti per la convergenza"],
        ["BRL_MAX_CARDINALITY", "2",      "Max 2 condizioni per regola: mantiene leggibilità"],
        ["FIGS_MAX_RULES",      "10",     "Alberi nella somma FIGS; oltre 10 perde interpretabilità"],
        ["SKOPE_N_ESTIMATORS",  "100",    "Pool ampio di alberi base per diversità delle regole"],
        ["SKOPE_PRECISION_MIN", "0.5",    "Soglia minima di precisione per accettare una regola"],
        ["SKOPE_RECALL_MIN",    "0.4",    "Copertura minima dei positivi nel training per regola"],
    ]
    rule_h = 0.200
    rule_top = y - 0.045 - rule_h
    _section_table("Modelli rule-based", rule_rows, rule_top, rule_h)
    y = rule_top - 0.048

    # ── Modelli ensemble ─────────────────────────────────────────────────────
    ens_rows = [
        ["RF_N_ESTIMATORS",   "200",  "Abbastanza alberi da stabilizzare le stime RF"],
        ["GB_N_ESTIMATORS",   "200",  "Iterazioni di boosting sufficienti senza overfitting"],
        ["GB_LEARNING_RATE",  "0.05", "Passi piccoli nel gradiente: riduce il rischio di overfitting"],
        ["GB_MAX_DEPTH",      "4",    "Alberi profondi limitati; il boosting compensa il bias"],
        ["XGB_N_ESTIMATORS",  "200",  "Allineato a GB per confronto diretto"],
        ["XGB_LEARNING_RATE", "0.05", "Come GB; XGBoost aggiunge anche regolarizzazione L1/L2"],
        ["XGB_MAX_DEPTH",     "4",    "Come GB: limita i learner deboli nel boosting sequenziale"],
    ]
    ens_h = 0.178
    ens_top = max(0.03, y - 0.045 - ens_h)
    _section_table("Modelli ensemble", ens_rows, ens_top, ens_h)

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

    ax_tbl = fig.add_axes((0.04, 0.61, 0.92, 0.26))
    cols = ["Modello", "Type", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    _render_table(ax_tbl, df[[c for c in cols if c in df.columns]].fillna("—"))

    ax_img = fig.add_axes((0.04, 0.27, 0.92, 0.31))
    _embed_image(ax_img, PLOTS_DIR / "metrics_comparison.png")

    _add_description(fig,
        "Ogni gruppo di barre rappresenta un modello; i cinque colori corrispondono alle cinque "
        "metriche. Accuracy e F1 riassumono la qualità complessiva, ma su classi sbilanciate "
        "possono essere fuorvianti. Precision misura quante predizioni 'deceduto' sono corrette; "
        "Recall quante morti reali vengono identificate — in clinica un Recall basso (morte non "
        "rilevata) è più pericoloso di una Precision bassa (falso allarme). ROC-AUC è la metrica "
        "più robusta: misura la capacità di separare le classi indipendentemente dalla soglia. "
        "GreedyRuleList e RIPPER ottengono F1=0.667 e Accuracy=0.833 identici — rispettivamente "
        "con regole a condizione singola e regole multi-condizione AND. Tuttavia RIPPER ha "
        "ROC-AUC=0.750, il più basso di tutti i modelli: buona classificazione a soglia fissa "
        "non implica buona capacità discriminativa generale. SkopeRules ottiene Precision=1.0 "
        "con Recall=0.263: identifica pochi deceduti ma senza mai sbagliare.",
        y_start=0.25,
    )
    pdf.savefig(fig)
    plt.close(fig)


def _page_roc(pdf: PdfPages) -> None:
    fig = _new_fig()
    _add_header(fig, "3.2 · Risultati", "Curve ROC")
    ax = fig.add_axes((0.04, 0.24, 0.92, 0.65))
    _embed_image(ax, PLOTS_DIR / "roc_curves.png")
    _add_description(fig,
        "La curva ROC traccia il TPR (Recall) contro il FPR al variare della soglia; l'AUC "
        "riassume la performance in un unico numero indipendente dalla soglia (0.50 = casuale, "
        "1.0 = perfetto). Random Forest raggiunge AUC 0.901, il più alto. SkopeRules e "
        "GreedyRuleList seguono (~0.89), risultato notevole per modelli intrinsecamente "
        "interpretabili. RIPPER ha AUC 0.750 — il più basso di tutti, inferiore anche a "
        "FIGS (0.767): nonostante F1 e Accuracy identici a GRL, il ranking probabilistico "
        "è meno calibrato."
    )
    pdf.savefig(fig)
    plt.close(fig)


def _page_cm(pdf: PdfPages) -> None:
    fig = _new_fig()
    _add_header(fig, "3.3 · Risultati", "Matrici di Confusione")
    ax = fig.add_axes((0.04, 0.24, 0.92, 0.65))
    _embed_image(ax, PLOTS_DIR / "confusion_matrices.png")
    _add_description(fig,
        "Ogni cella indica: riga = classe reale, colonna = classe predetta. La diagonale "
        "principale (celle più scure) raccoglie le predizioni corrette; gli elementi fuori "
        "diagonale sono gli errori. In clinica i due tipi di errore hanno costo asimmetrico: "
        "un falso negativo (morte non rilevata) è più grave di un falso positivo (falso allarme). "
        "BayesianRuleList massimizza il Recall sui deceduti — pochi falsi negativi — a scapito "
        "di più falsi positivi. Gli ensemble (RandomForest, XGBoost) bilanciano meglio i due "
        "errori. GreedyRuleList e RIPPER producono matrici identiche (10 TP, 1 FP, 9 FN, 40 TN): "
        "due architetture molto diverse — regola singola vs AND multi-condizione — convergono "
        "alla stessa frontiera decisionale. SkopeRules ha Precision=1.0 ma Recall=0.263: "
        "identifica solo 5 deceduti su 19, senza mai produrre falsi positivi."
    )
    pdf.savefig(fig)
    plt.close(fig)


def _page_crossval(pdf: PdfPages) -> None:
    fig = _new_fig()
    _add_header(fig, "3.5 · Risultati", "Cross-Validation 10-fold")
    ax = fig.add_axes((0.04, 0.24, 0.92, 0.65))
    _embed_image(ax, PLOTS_DIR / "cross_validation.png")
    _add_description(fig,
        "Il boxplot mostra la distribuzione del ROC-AUC su 10 fold stratificati: mediana, "
        "IQR (bordi del box) e range (baffi). La stratificazione garantisce che ogni fold "
        "mantenga la stessa proporzione di deceduti (~32%) del dataset completo, rendendo "
        "la stima più affidabile. Random Forest ha la mediana più alta e IQR contenuto: "
        "performance stabile e consistente tra i fold. RuleFit è il miglior rule-based in CV. "
        "FIGS mostra alta varianza — instabile su dataset piccoli. GreedyRuleList ottiene "
        "0.843 ± 0.096 tramite un custom scorer che aggira un'incompatibilità con l'API sklearn.")
    pdf.savefig(fig)
    plt.close(fig)


def _page_rules_complete(pdf: PdfPages) -> None:
    grl_path = RESULTS_DIR / "grl_rules.json"
    brl_path = RESULTS_DIR / "brl_rules.json"
    rip_path = RESULTS_DIR / "ripper_rules.json"
    if not grl_path.exists() or not brl_path.exists():
        return
    with open(grl_path) as f:
        grl_rows = json.load(f)
    with open(brl_path) as f:
        brl_rows = json.load(f)
    rip_rows = json.load(open(rip_path)) if rip_path.exists() else []

    fig = _new_fig()
    _add_header(fig, "3.4 · Risultati", "Regole Complete: GRL, BRL e RIPPER")

    row_h = 0.022   # compact row height to fit three tables on one page

    # ── GRL ──────────────────────────────────────────────────────────────────
    y = _write_paragraphs(fig, [
        ("GreedyRuleList",
         "Lista IF/THEN sequenziale: la prima regola soddisfatta determina la classe, "
         "che non è necessariamente Decesso — ogni regola predice la classe di maggioranza "
         "del proprio segmento. % deceduti > 50% → Predizione = Decesso; altrimenti Sopravvissuto."),
    ], start_y=0.88)
    grl_h = row_h * (len(grl_rows) + 1)
    ax_grl = fig.add_axes((L_MARGIN, y - 0.01 - grl_h, 1 - 2 * L_MARGIN, grl_h))
    df_grl = pd.DataFrame(grl_rows, columns=["Condizione", "% deceduti", "Predizione", "Campioni"])
    _render_table(ax_grl, df_grl, col_widths=[0.47, 0.15, 0.21, 0.17])

    # ── BRL ──────────────────────────────────────────────────────────────────
    y2 = _write_paragraphs(fig, [
        ("BayesianRuleList",
         "Lista IF/ELSE con MCMC su feature discretizzate (4 bin, quantile). "
         "Ogni antecedente può avere fino a 3 condizioni congiuntive. "
         "P(decesso) = probabilità posteriore con IC al 95%."),
    ], start_y=y - 0.01 - grl_h - 0.01)
    brl_h = row_h * (len(brl_rows) + 1)
    ax_brl = fig.add_axes((L_MARGIN, y2 - 0.01 - brl_h, 1 - 2 * L_MARGIN, brl_h))
    df_brl = pd.DataFrame(brl_rows, columns=["Condizione", "P(decesso)", "IC 95%"])
    _render_table(ax_brl, df_brl, col_widths=[0.58, 0.20, 0.22])

    # ── RIPPER ───────────────────────────────────────────────────────────────
    if rip_rows:
        y3 = _write_paragraphs(fig, [
            ("RIPPER",
             "Ogni regola ha un antecedente con più condizioni AND (tutte devono essere "
             "verificate simultaneamente). Se tutte le condizioni di una regola sono "
             "soddisfatte → Decesso; nessuna regola soddisfatta → Sopravvissuto (DEFAULT)."),
        ], start_y=y2 - 0.01 - brl_h - 0.01)
        rip_h = row_h * (len(rip_rows) + 1)
        ax_rip = fig.add_axes((L_MARGIN, y3 - 0.01 - rip_h, 1 - 2 * L_MARGIN, rip_h))
        rip_data = [
            [r[0], "Sopravvissuto" if r[0] == "DEFAULT" else "Decesso", r[1]]
            for r in rip_rows
        ]
        df_rip = pd.DataFrame(rip_data, columns=["Regola", "Predizione", "Campioni"])
        _render_table(ax_rip, df_rip, col_widths=[0.60, 0.22, 0.18])

    pdf.savefig(fig)
    plt.close(fig)


def _page_rules_complete2(pdf: PdfPages) -> None:
    """Continuation of 3.5: RuleFit top rules, SkopeRules, FIGS."""
    rf_path    = RESULTS_DIR / "rulefit_top_rules.json"
    skope_path = RESULTS_DIR / "skope_rules.json"
    figs_path  = RESULTS_DIR / "figs_structure.txt"
    if not rf_path.exists():
        return

    with open(rf_path) as f:
        rf_rows = json.load(f)
    skope_rows = json.load(open(skope_path)) if skope_path.exists() else []
    figs_text  = figs_path.read_text() if figs_path.exists() else ""

    fig = _new_fig()
    _add_header(fig, "3.4 · Risultati (continua)", "Regole Complete: RuleFit, SkopeRules e FIGS")

    row_h = 0.022

    # ── RuleFit ───────────────────────────────────────────────────────────────
    y = _write_paragraphs(fig, [
        ("RuleFit — regole con coefficiente positivo (pro Decesso)",
         "Ogni regola è un percorso estratto da un ensemble di alberi; il coefficiente Lasso "
         "indica il peso nella predizione finale. Valori più alti → maggiore contributo al "
         "rischio di decesso. Le regole possono combinare più feature (AND implicito sul percorso)."),
    ], start_y=0.88)
    rf_h = row_h * (len(rf_rows) + 1)
    ax_rf = fig.add_axes((L_MARGIN, y - 0.01 - rf_h, 1 - 2 * L_MARGIN, rf_h))
    df_rf = pd.DataFrame(rf_rows, columns=["Regola", "Coefficiente"])
    _render_table(ax_rf, df_rf, col_widths=[0.82, 0.18])

    # ── SkopeRules ────────────────────────────────────────────────────────────
    y2 = _write_paragraphs(fig, [
        ("SkopeRules",
         "Regole ad alta precisione estratte tramite sub-campionamenti di ensemble di alberi, "
         "poi de-duplicate. Ogni regola è corredata da Precisione e Recall sul training set."),
    ], start_y=y - 0.01 - rf_h - 0.012)
    if skope_rows:
        sk_h = row_h * (len(skope_rows) + 1)
        ax_sk = fig.add_axes((L_MARGIN, y2 - 0.01 - sk_h, 1 - 2 * L_MARGIN, sk_h))
        df_sk = pd.DataFrame(skope_rows, columns=["Regola", "Precisione", "Recall"])
        _render_table(ax_sk, df_sk, col_widths=[0.64, 0.18, 0.18])
        y2 = y2 - 0.01 - sk_h
    else:
        fig.text(L_MARGIN, y2 - 0.01, "Nessuna regola appresa con le soglie impostate.",
                 fontsize=9, color="#888888", va="top")
        y2 -= 0.04

    # ── FIGS ──────────────────────────────────────────────────────────────────
    y_figs = _write_paragraphs(fig, [
        ("FIGS — Fast Interpretable Greedy-Tree Sums",
         "FIGS è una somma di alberi: ogni albero assegna al campione un valore numerico "
         "(la sua contribuzione), e se la somma supera 0.5 la predizione è Decesso. Sono "
         "stati appresi due alberi: Tree 0 usa time (≤ 60.5 → 0.68) e serum_creatinine, "
         "Tree 1 usa ejection_fraction (≤ 27.5 → +0.31)."),
    ], start_y=y2 - 0.012)
    if figs_text:
        for i, line in enumerate(figs_text.splitlines()[:12]):
            fig.text(L_MARGIN + 0.02, y_figs - i * 0.012, line,
                     fontsize=7.5, color="#444444", va="top",
                     fontfamily="monospace")

    pdf.savefig(fig)
    plt.close(fig)


def _page_rulefit_interp(pdf: PdfPages) -> None:
    """Section 3.7: RuleFit coefficient bar chart."""
    fig = _new_fig()
    _add_header(fig, "3.7 · Risultati", "Interpretabilità — Regole RuleFit")
    ax = fig.add_axes((0.04, 0.24, 0.92, 0.65))
    _embed_image(ax, PLOTS_DIR / "rulefit_rules.png")
    _add_description(fig,
        "Coefficienti Lasso delle regole estratte da RuleFit: positivo (barre rosse) → la regola "
        "aumenta il rischio di decesso; negativo (blu) → lo riduce. Le regole pro-decesso con "
        "peso più alto coinvolgono time basso e serum_creatinine alta. Le regole pro-sopravvivenza "
        "implicano ejection_fraction più alta e follow-up lungo. La penalizzazione Lasso azzera "
        "i coefficienti ridondanti, mantenendo solo le regole con effetto reale sulla predizione."
    )
    pdf.savefig(fig)
    plt.close(fig)


def _page_interpretability(pdf: PdfPages) -> None:
    """Section 3.8: RF feature importance with cross-model interpretability synthesis."""
    fig = _new_fig()
    _add_header(fig, "3.7 · Risultati (continua)", "Interpretabilità — Feature Importance RF")

    ax = fig.add_axes((0.04, 0.24, 0.92, 0.65))
    _embed_image(ax, PLOTS_DIR / "feature_importance_rf.png")

    _add_description(fig,
        "L'importanza MDI misura di quanto ogni feature riduce l'impurità media nei 200 alberi "
        "del Random Forest: time domina nettamente, seguita da serum_creatinine ed "
        "ejection_fraction. Questa gerarchia è confermata in modo indipendente da RuleFit "
        "(sezione 3.4): le regole con coefficiente Lasso più alto coinvolgono time ≤ 60.5 e "
        "serum_creatinine alta, quelle con coefficiente negativo implicano follow-up lungo ed "
        "ejection_fraction alta. Due metodi radicalmente diversi — importance da ensemble di "
        "200 alberi vs regressione Lasso su regole estratte — convergono sugli stessi tre "
        "segnali fisiopatologici, coerenti con la letteratura sull'insufficienza cardiaca."
    )

    pdf.savefig(fig)
    plt.close(fig)


def _page_ablation(pdf: PdfPages) -> None:
    abl_path     = RESULTS_DIR / "ablation_no_time.csv"
    grl_nt_path  = RESULTS_DIR / "grl_no_time_rules.json"
    if not abl_path.exists() or not grl_nt_path.exists():
        return

    df_abl = pd.read_csv(abl_path, index_col=0)
    with open(grl_nt_path) as f:
        grl_nt_rows = json.load(f)

    fig = _new_fig()
    _add_header(fig, "3.6 · Risultati", "RF e GRL senza feature time")

    y = _write_paragraphs(fig, [
        ("Motivazione",
         "La feature time (durata del follow-up) è la più predittiva del dataset, come mostrano "
         "la feature importance del Random Forest e le regole di GreedyRuleList. In contesti "
         "reali di triage questa informazione non è sempre disponibile al momento della "
         "prima valutazione. Questo esperimento valuta RF e GRL privandoli di time, forzandoli "
         "a imparare dai segnali biochimici e demografici rimanenti (serum_creatinine, "
         "ejection_fraction, age, ecc.)."),
    ], start_y=0.88)

    # ── Comparison table ──────────────────────────────────────────────────────
    labels = {
        "RandomForest":  "RF (con time)",
        "RF_no_time":    "RF (senza time)",
        "GreedyRuleList":"GRL (con time)",
        "GRL_no_time":   "GRL (senza time)",
    }
    rows = []
    for name in ["RandomForest", "RF_no_time", "GreedyRuleList", "GRL_no_time"]:
        if name in df_abl.index:
            r = df_abl.loc[name]
            rows.append([labels[name],
                         f"{r['Accuracy']:.4f}", f"{r['F1']:.4f}", f"{r['ROC-AUC']:.4f}"])

    tbl_h = 0.026 * (len(rows) + 1)
    y_tbl = y - 0.01
    ax_tbl = fig.add_axes((L_MARGIN, y_tbl - tbl_h, 1 - 2 * L_MARGIN, tbl_h))
    df_cmp = pd.DataFrame(rows, columns=["Modello", "Accuracy", "F1", "ROC-AUC"])
    _render_table(ax_tbl, df_cmp, col_widths=[0.40, 0.20, 0.20, 0.20])

    # ── GRL no-time rules ─────────────────────────────────────────────────────
    grl_sec_y = y_tbl - tbl_h - 0.01
    y2 = _write_paragraphs(fig, [
        ("Regole GRL senza time",
         "Con la rimozione di time, GreedyRuleList impara a discriminare dai segnali "
         "biochimici. serum_creatinine ed ejection_fraction diventano le feature dominanti, "
         "coerentemente con la feature importance del Random Forest. Ogni regola classifica "
         "i campioni in base alla Predizione indicata (Decesso se % deceduti > 50%)."),
    ], start_y=grl_sec_y)

    grl_h = 0.026 * (len(grl_nt_rows) + 1)
    top_grl = y2 - 0.01
    ax_grl = fig.add_axes((L_MARGIN, top_grl - grl_h, 1 - 2 * L_MARGIN, grl_h))
    df_grl = pd.DataFrame(grl_nt_rows, columns=["Condizione", "% deceduti", "Predizione", "Campioni"])
    _render_table(ax_grl, df_grl, col_widths=[0.47, 0.15, 0.21, 0.17])

    pdf.savefig(fig)
    plt.close(fig)


def _page_conclusions(pdf: PdfPages) -> None:
    csv_path = RESULTS_DIR / "metrics_summary.csv"

    # ── Page 1 ────────────────────────────────────────────────────────────────
    fig = _new_fig()
    _add_header(fig, "4 · Conclusioni", "Conclusioni")

    y = _write_paragraphs(fig, [
        ("Confronto tra approcci",
         "I modelli ensemble raggiungono ROC-AUC superiore (Random Forest 0.901, XGBoost 0.851) "
         "ma il divario con i rule-based non è netto: GreedyRuleList (0.885) e RuleFit (0.883) "
         "li superano entrambi. Su soli 299 campioni la semplicità strutturale dei modelli a "
         "regole riduce il rischio di overfitting, risultando competitiva. GreedyRuleList offre "
         "il miglior F1 rule-based (0.667) con una lista ordinata di condizioni leggibile da un "
         "clinico. RuleFit assegna coefficienti espliciti a ogni regola, rendendo visibile il "
         "peso relativo di ciascuna. BayesianRuleList privilegia il recall (0.842): sovrastima "
         "i decessi, il che in medicina è spesso preferibile a sottostimarli. FIGS ha AUC più "
         "basso (0.767) ma struttura additiva particolarmente trasparente. SkopeRules raggiunge "
         "Precision=1.0 con Recall=0.263: non produce mai falsi positivi, ma identifica solo "
         "5 deceduti su 19 — utile quando ogni allarme deve corrispondere a un caso critico. "
         "RIPPER è l'unico rule-list a produrre regole congiuntive (AND tra più condizioni): "
         "ottiene F1=0.667 identico a GRL, ma con ROC-AUC=0.750 — il più basso di tutti. "
         "Il vantaggio strutturale (regole multi-feature) non si traduce in migliore capacità "
         "discriminativa su questo dataset."),
        ("Precision vs Recall in ambito clinico",
         "Il trade-off tra precisione e recall non è neutro in contesti medici. Alta recall "
         "minimizza i falsi negativi (morti non rilevate), che in clinica sono più pericolosi "
         "dei falsi positivi (falsi allarmi). Alta precisione garantisce invece che ogni "
         "segnalazione sia affidabile, preferibile quando si pianificano interventi invasivi. "
         "La scelta del modello dipende quindi dal contesto d'uso, e la trasparenza dei modelli "
         "rule-based è la condizione necessaria per fare questa scelta in modo consapevole."),
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

    # ── Page 2 ────────────────────────────────────────────────────────────────
    fig2 = _new_fig()
    _add_header(fig2, "4 · Conclusioni", "Conclusioni (continua)")

    _write_paragraphs(fig2, [
        ("Validazione qualitativa delle regole",
         "Le feature più ricorrenti — time (follow-up), serum_creatinine ed ejection_fraction "
         "— sono coerenti con la letteratura clinica sull'insufficienza cardiaca: follow-up "
         "breve segnala una fase acuta critica, creatinina alta indica insufficienza renale "
         "spesso letale in questo contesto, ejection fraction bassa riflette ridotta capacità "
         "di pompa. Il fatto che i modelli apprendano autonomamente queste relazioni costituisce "
         "una validazione qualitativa: catturano pattern fisiopatologici reali, non solo "
         "correlazioni statistiche."),
        ("Stabilità in cross-validation",
         "Random Forest (0.918 ± 0.044) e XGBoost (0.910 ± 0.049) sono i modelli più stabili. "
         "RuleFit raggiunge 0.884 ± 0.058 — il miglior risultato tra i rule-based in CV e "
         "competitivo con gli ensemble. GreedyRuleList ottiene 0.843 ± 0.096, con varianza "
         "più alta ma attesa su fold di circa 24 campioni. FIGS ha la varianza maggiore "
         "(0.847 ± 0.110), indice di instabilità strutturale su dataset piccoli."),
        ("Ablazione della feature time",
         "Rimuovendo time da RF e GRL, il ROC-AUC scende da 0.901 a 0.802 per RF e da 0.885 "
         "a 0.772 per GRL: un costo di circa 0.10, atteso dato il peso dominante di time. "
         "Senza time i modelli si affidano a serum_creatinine ed ejection_fraction — feature "
         "biochimiche disponibili alla prima valutazione, prima del follow-up — e rimangono "
         "utilizzabili in contesti di triage, anche se con discriminazione ridotta."),
        ("Limiti e sviluppi futuri",
         "Con 60 campioni nel test set (19 deceduti) ogni metrica è soggetta ad alta varianza: "
         "un solo paziente classificato diversamente sposta F1 di oltre 0.05. I risultati sono "
         "quindi indicativi e richiedono validazione su coorti esterne prima di qualsiasi "
         "applicazione clinica. Ulteriori sviluppi potrebbero includere la calibrazione delle "
         "probabilità predette e l'integrazione in un sistema di supporto decisionale con "
         "supervisione medica — contesto in cui la trasparenza dei modelli rule-based resta "
         "il requisito imprescindibile."),
    ])

    pdf.savefig(fig2)
    plt.close(fig2)


# ── PDF post-processing (page numbers + clickable TOC links) ─────────────────

def _postprocess_pdf(path) -> None:
    try:
        import fitz
    except ImportError:
        print("PyMuPDF not available — skipping PDF postprocessing")
        return
    import os

    doc = fitz.open(str(path))

    # 1. Page numbers at the bottom center of every page
    for i in range(doc.page_count):
        page = doc[i]
        r = page.rect
        page.insert_textbox(
            fitz.Rect(0, r.height - 28, r.width, r.height - 6),
            str(i + 1),
            fontsize=9,
            color=(0.55, 0.55, 0.55),
            align=1,  # center
        )

    # 2. Clickable link annotations on TOC page (index 0)
    # Each tuple: (section_num, target_page_1indexed, is_chapter)
    # Chapters link to their first page even though no page number is displayed.
    toc_links = [
        ("1.",  2,  True),
        ("1.1", 2,  False), ("1.2", 3,  False), ("1.3", 4,  False), ("1.4", 5,  False),
        ("2.",  6,  True),
        ("2.1", 6,  False), ("2.2", 7,  False), ("2.3", 9,  False),
        ("2.4", 10, False), ("2.5", 11, False),
        ("3.",  12, True),
        ("3.1", 12, False), ("3.2", 13, False), ("3.3", 14, False),
        ("3.4", 15, False),
        ("3.5", 17, False), ("3.6", 18, False),
        ("3.7", 19, False),
        ("4.",  21, True),
        ("4.1", 21, False),
    ]

    toc_page = doc[0]
    pw, ph = toc_page.rect.width, toc_page.rect.height

    y = 0.645
    for _, page_num, is_chapter in toc_links:
        if is_chapter:
            y -= 0.006  # pre-spacing before chapter heading

        # Link rect: full width between margins, row height ~16 pt
        top_pdf = (1.0 - y) * ph
        left_pdf = L_MARGIN * pw
        right_pdf = (1.0 - L_MARGIN) * pw
        link_rect = fitz.Rect(left_pdf, top_pdf - 5, right_pdf, top_pdf + 12)
        toc_page.insert_link({
            "kind": fitz.LINK_GOTO,
            "from": link_rect,
            "page": page_num - 1,  # fitz uses 0-based page indices
        })

        y -= 0.032 if is_chapter else 0.025  # post-spacing

    tmp = str(path) + ".tmp"
    doc.save(tmp)
    doc.close()
    os.replace(tmp, str(path))
    print("PDF postprocessed: page numbers and TOC links added.")


# ── main ──────────────────────────────────────────────────────────────────────

def generate_report() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)

    with PdfPages(REPORT_PATH) as pdf:
        _page_toc(pdf)
        _page_intro(pdf)
        _page_dataset(pdf)
        _page_eda(pdf)
        _page_project_structure(pdf)
        _page_data_pipeline(pdf)
        _page_models_rules(pdf)
        _page_models_ensemble(pdf)
        _page_config(pdf)
        _page_metrics(pdf)
        _page_roc(pdf)
        _page_cm(pdf)
        _page_rules_complete(pdf)
        _page_rules_complete2(pdf)
        _page_crossval(pdf)
        _page_ablation(pdf)
        _page_rulefit_interp(pdf)
        _page_interpretability(pdf)
        _page_conclusions(pdf)

        pdf.infodict().update({
            "Title":   "XAI Project — Heart Failure Clinical Records",
            "Subject": "Confronto modelli interpretabili vs ensemble",
        })

    print(f"Report saved to {REPORT_PATH}")
    _postprocess_pdf(REPORT_PATH)


if __name__ == "__main__":
    generate_report()
