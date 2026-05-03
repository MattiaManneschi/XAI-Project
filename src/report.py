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
    for (r, c), cell in tbl.get_celld().items():
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
        ("3.4", "Interpretabilità — Regole RuleFit", 15),
        ("3.5", "Interpretabilità — Feature Importance RF", 16),
        ("3.6", "Cross-Validation 10-fold", 17),
        ("4.", "Conclusioni", None),
        ("4.1", "Confronto, analisi clinica e limiti", 18),
    ]

    y = 0.645
    for num, title, page in toc:
        is_chapter = page is None
        indent = L_MARGIN if is_chapter else L_MARGIN + 0.04
        size = 11 if is_chapter else 9.5
        weight = "bold" if is_chapter else "normal"
        color = TITLE_COLOR if is_chapter else "#333333"
        if is_chapter:
            y -= 0.010
        fig.text(indent, y, f"{num}  {title}", fontsize=size,
                 fontweight=weight, color=color, va="top")
        if page is not None:
            fig.text(1 - L_MARGIN, y, str(page), fontsize=9.5,
                     color="#888888", va="top", ha="right")
        y -= 0.042 if is_chapter else 0.033

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
         "array di 48 colonne binarie (12 feature x 4 bin), obbligatorio per "
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

    ax_tbl = fig.add_axes((0.04, 0.57, 0.92, 0.30))
    cols = ["Modello", "Type", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
    _render_table(ax_tbl, df[[c for c in cols if c in df.columns]].fillna("—"))

    ax_img = fig.add_axes((0.04, 0.19, 0.92, 0.36))
    _embed_image(ax_img, PLOTS_DIR / "metrics_comparison.png")

    _add_description(fig,
        "Ogni gruppo di barre rappresenta un modello; i cinque colori corrispondono alle cinque "
        "metriche. Accuracy e F1 riassumono la qualità complessiva, ma su classi sbilanciate "
        "possono essere fuorvianti. Precision misura quante predizioni 'deceduto' sono corrette; "
        "Recall quante morti reali vengono identificate — in clinica un Recall basso (morte non "
        "rilevata) è più pericoloso di una Precision bassa (falso allarme). ROC-AUC è la metrica "
        "più robusta: misura la capacità di separare le classi indipendentemente dalla soglia. "
        "GreedyRuleList è il miglior rule-based per F1 e Accuracy. SkopeRules ottiene ROC-AUC "
        "0.858 con Precision=1.0 e Recall=0.263: identifica pochi deceduti ma senza mai sbagliare.",
        y_start=0.17,
    )
    pdf.savefig(fig)
    plt.close(fig)


def _page_roc(pdf: PdfPages) -> None:
    fig = _new_fig()
    _add_header(fig, "3.2 · Risultati", "Curve ROC")
    ax = fig.add_axes((0.04, 0.24, 0.92, 0.65))
    _embed_image(ax, PLOTS_DIR / "roc_curves.png")
    _add_description(fig,
        "La curva ROC traccia il True Positive Rate (Recall, asse y) contro il False Positive "
        "Rate (1 - Specificità, asse x) al variare della soglia di classificazione. Un modello "
        "perfetto punta all'angolo in alto a sinistra (TPR=1, FPR=0); uno casuale segue la "
        "diagonale tratteggiata (AUC=0.50). L'area sotto la curva (AUC) riassume la performance "
        "in un unico numero indipendente dalla soglia: valori sopra 0.80 sono considerati buoni "
        "in ambito clinico. Random Forest raggiunge AUC 0.901, il più alto. SkopeRules e "
        "GreedyRuleList seguono (~0.89), risultato notevole per modelli intrinsecamente "
        "interpretabili. FIGS è il più debole (~0.77), penalizzato dalla struttura rigida."
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
        "errori. SkopeRules ha Precision=1.0 ma Recall=0.263: identifica solo 5 deceduti su 19, "
        "senza mai produrre falsi positivi — comportamento by design delle sue soglie di accettazione."
    )
    pdf.savefig(fig)
    plt.close(fig)


def _page_interpretability(pdf: PdfPages) -> None:
    pages = [
        (
            "rulefit_rules.png", "3.4 · Risultati", "Interpretabilità — Regole RuleFit",
            "RuleFit assegna a ogni regola estratta un coefficiente tramite regressione Lasso: "
            "coefficiente positivo (barre rosse, grafico superiore) significa che quella condizione "
            "aumenta la probabilità di decesso; negativo (blu, inferiore) la diminuisce. Le regole "
            "più influenti nel predire il decesso coinvolgono time basso e serum_creatinine alta, "
            "confermando i pattern dell'EDA. Le regole pro-sopravvivenza implicano ejection_fraction "
            "più alta e follow-up lungo. La penalizzazione Lasso azzera i coefficienti delle regole "
            "ridondanti, mantenendo solo quelle con effetto reale sulla predizione.",
        ),
        (
            "feature_importance_rf.png", "3.5 · Risultati", "Interpretabilità — Feature Importance RF",
            "L'importanza MDI (Mean Decrease in Impurity) misura di quanto ogni feature riduce "
            "l'impurità media nei nodi degli alberi del Random Forest: valori più alti indicano "
            "feature più utilizzate e discriminative. time domina nettamente, seguita da "
            "serum_creatinine ed ejection_fraction — le stesse tre variabili che emergono dalle "
            "regole di RuleFit e GreedyRuleList: una validazione incrociata tra approcci molto "
            "diversi. Le feature binarie (anaemia, diabetes, sex, smoking) hanno importanza "
            "marginale, coerente con le basse correlazioni dell'EDA.",
        ),
        (
            "cross_validation.png", "3.6 · Risultati", "Cross-Validation 10-fold",
            "Il boxplot mostra la distribuzione del ROC-AUC su 10 fold stratificati: mediana, "
            "IQR (bordi del box) e range (baffi). La stratificazione garantisce che ogni fold "
            "mantenga la stessa proporzione di deceduti (~32%) del dataset completo, rendendo "
            "la stima più affidabile. Random Forest ha la mediana più alta e IQR contenuto: "
            "performance stabile e consistente tra i fold. RuleFit è il miglior rule-based in CV. "
            "FIGS mostra alta varianza — instabile su dataset piccoli. GreedyRuleList ottiene "
            "0.843 ± 0.096 tramite un custom scorer che aggira un'incompatibilità con l'API sklearn.",
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
         "5 deceduti su 19 — utile quando ogni allarme deve corrispondere a un caso realmente critico."),
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
        _page_interpretability(pdf)
        _page_conclusions(pdf)

        pdf.infodict().update({
            "Title":   "XAI Project — Heart Failure Clinical Records",
            "Subject": "Confronto modelli interpretabili vs ensemble",
        })

    print(f"Report saved to {REPORT_PATH}")


if __name__ == "__main__":
    generate_report()
