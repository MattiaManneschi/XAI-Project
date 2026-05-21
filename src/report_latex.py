"""Generate a LaTeX report from pipeline artefacts and compile to PDF."""

import json
import shutil
import subprocess
from pathlib import Path
import pandas as pd

from config import RESULTS_DIR, FILES_DIR, PLOTS_DIR, DATA_DIR

_ROOT       = Path(__file__).parent.parent
REPORT_DIR  = _ROOT / "Report"
FIGURES_DIR = REPORT_DIR / "figures"
TEX_PATH    = REPORT_DIR / "Manneschi_XAI.tex"

_IMAGES = [
    "eda_continuous.png", "eda_correlation.png",
    "metrics_comparison.png", "roc_curves.png",
    "confusion_matrices.png", "rulefit_rules.png",
    "feature_importance_rf.png", "cross_validation.png",
    "hyperparam_rule_based.png", "hyperparam_ensemble.png",
    "pdp_ice_1d.png", "pdp_2d.png",
]


# ── LaTeX helpers ─────────────────────────────────────────────────────────────

def _e(s) -> str:
    """Escape dynamic data for LaTeX text mode."""
    s = str(s) if s is not None else r"\textit{null}"
    s = s.replace("\\", "XBSX")
    s = s.replace("{",  "XOBX")
    s = s.replace("}",  "XCBX")
    s = s.replace("$",  r"\$")
    s = s.replace("&",  r"\&")
    s = s.replace("#",  r"\#")
    s = s.replace("^",  r"\^{}")
    s = s.replace("_",  r"\_")
    s = s.replace("~",  r"\textasciitilde{}")
    s = s.replace("%",  r"\%")
    s = s.replace("≤",  r"$\leq$")
    s = s.replace("≥",  r"$\geq$")
    s = s.replace("∈",  r"$\in$")
    s = s.replace("→",  r"$\rightarrow$")
    s = s.replace("±",  r"$\pm$")
    s = s.replace("—",  "---")
    s = s.replace("–",  "--")
    s = s.replace("XBSX", r"\textbackslash{}")
    s = s.replace("XOBX", r"\{")
    s = s.replace("XCBX", r"\}")
    return s


class _Raw(str):
    """Pre-formatted LaTeX cell — bypasses _e() in _tbl."""


_LINEWIDTH_PT = 471.0   # text width for a4paper with left=2.2cm, right=2.2cm
_TABCOLSEP_PT = 6.0    # LaTeX default \tabcolsep


def _tbl(columns, rows, fracs, long=False) -> str:
    """Booktabs table with blue header; use long=True for longtable."""
    n = len(fracs)
    overhead = 2 * n * _TABCOLSEP_PT / _LINEWIDTH_PT
    width_factor = min(0.965, 1.0 - overhead - 0.01)
    total = sum(fracs)
    f = [x / total * width_factor for x in fracs]
    col_spec = "".join(
        f">{{\\raggedright\\arraybackslash}}p{{{v:.4f}\\linewidth}}" for v in f
    )
    hdr = " & ".join(
        f"\\cellcolor{{tblhdr}}\\color{{white}}\\bfseries {_e(c)}" for c in columns
    )
    body = []
    for i, row in enumerate(rows):
        cells = " & ".join(c if isinstance(c, _Raw) else _e(c) for c in row)
        if i % 2 == 1 and not long:
            body.append(f"\\rowcolor{{ltrow}}{cells} \\\\")
        else:
            body.append(f"{cells} \\\\")

    if long:
        return "\n".join(
            [
                "{\\small",
                f"\\begin{{longtable}}{{{col_spec}}}",
                "\\toprule",
                f"{hdr} \\\\",
                "\\midrule",
                "\\endfirsthead",
                "\\toprule",
                f"{hdr} \\\\",
                "\\midrule",
                "\\endhead",
                "\\midrule",
                "\\endlastfoot",
            ]
            + body
            + ["\\end{longtable}", "}"]
        )
    else:
        return "\n".join(
            [
                "\\begin{center}",
                f"{{\\small\\begin{{tabular}}{{{col_spec}}}",
                "\\toprule",
                f"{hdr} \\\\",
                "\\midrule",
            ]
            + body
            + ["\\bottomrule", "\\end{tabular}}}", "\\end{center}"]
        )


def _img(name: str, width: str = "\\textwidth") -> str:
    return (
        "\\begin{figure}[H]\n"
        "  \\centering\n"
        f"  \\includegraphics[width={width}]{{figures/{name}}}\n"
        "\\end{figure}"
    )


def _load(filename):
    p = FILES_DIR / filename
    if not p.exists():
        return None
    if p.suffix == ".json":
        return json.loads(p.read_text())
    if p.suffix == ".csv":
        return pd.read_csv(p)
    return p.read_text()


def _metric(df, name, col, default="?"):
    if df is None:
        return default
    row = df[df["name"] == name] if "name" in df.columns else df[df.index == name]
    if row.empty:
        return default
    return f"{float(row.iloc[0][col]):.3f}"


# ── Preamble ──────────────────────────────────────────────────────────────────

def _preamble() -> str:
    return r"""\documentclass[a4paper,11pt]{article}
\usepackage[a4paper, top=2.5cm, bottom=2.5cm, left=2.2cm, right=2.2cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage[utf8]{inputenc}
\usepackage{lmodern}
\usepackage{microtype}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage[table]{xcolor}
\usepackage{colortbl}
\usepackage{titlesec}
\usepackage{fancyhdr}
\usepackage[hidelinks,
            pdftitle={XAI Project - Heart Failure Clinical Records},
            pdfauthor={Mattia Manneschi}]{hyperref}
\usepackage{caption}
\usepackage{float}
\usepackage{parskip}
\usepackage{array}
\usepackage{verbatim}

\definecolor{titlecol}{HTML}{1a1a2e}
\definecolor{accentcol}{HTML}{4C72B0}
\definecolor{tblhdr}{HTML}{4C72B0}
\definecolor{ltrow}{HTML}{f5f5f5}

\titleformat{\section}
  {\Large\bfseries\color{titlecol}}{\thesection}{0.8em}{}
  [{\color{accentcol}\hrule height 0.8pt\vspace{3pt}}]
\titleformat{\subsection}
  {\large\bfseries\color{titlecol}}{\thesubsection}{0.8em}{}
\titlespacing*{\section}{0pt}{18pt}{8pt}
\titlespacing*{\subsection}{0pt}{12pt}{4pt}

\pagestyle{fancy}
\fancyhf{}
\renewcommand{\headrulewidth}{0pt}
\fancyhead[L]{\small\color{gray}XAI Project --- Heart Failure Clinical Records}
\fancyhead[R]{\small\color{gray}\thepage}

\setlength{\parskip}{5pt}
\setlength{\parindent}{0pt}
\captionsetup{justification=raggedright,singlelinecheck=false}
"""


# ── Title page ────────────────────────────────────────────────────────────────

def _titlepage() -> str:
    return r"""
\begin{titlepage}
  \centering
  \vspace*{3cm}
  {\Huge\bfseries\color{titlecol} XAI Project\par}
  \vspace{0.8cm}
  {\LARGE\color{accentcol} Heart Failure Clinical Records\par}
  \vspace{0.6cm}
  {\large\color{gray}
    Confronto tra modelli interpretabili basati su regole\\[4pt]
    e modelli ensemble non interpretabili\par}
  \vspace{0.5cm}
  {\color{accentcol}\rule{\textwidth}{0.8pt}}
  \vspace{1.2cm}
  {\large Mattia Manneschi\par}
  \vspace{0.3cm}
  {\normalsize Laboratorio di Explainable AI\par}
  \vspace*{\fill}
\end{titlepage}
"""


# ── Chapter 1: Introduzione ───────────────────────────────────────────────────

def _ch1(X_rows: int, y_pos: int) -> str:
    stats = (
        f"Campioni: {X_rows} $\\;|\\;$ Decessi: {y_pos} "
        f"({y_pos/X_rows*100:.1f}\\%) $\\;|\\;$ Sopravvissuti: {X_rows-y_pos} "
        f"({(X_rows-y_pos)/X_rows*100:.1f}\\%)"
        if X_rows else "Dati non disponibili."
    )
    def _fn(name: str) -> _Raw:
        """Feature name: escape underscores and allow line-break at each one."""
        return _Raw(name.replace("_", r"\_\allowbreak{}"))

    dataset_tbl = _tbl(
        ["Feature", "Tipo", "Descrizione"],
        [
            [_fn("age"),                        "continua", "Eta' del paziente (anni)"],
            [_fn("anaemia"),                     "binaria",  "Presenza di anemia (0/1)"],
            [_fn("creatinine_phosphokinase"),    "continua", "Livello CPK nel sangue (mcg/L)"],
            [_fn("diabetes"),                    "binaria",  "Presenza di diabete (0/1)"],
            [_fn("ejection_fraction"),           "continua", "Percentuale di sangue pompata dal ventricolo sinistro (%)"],
            [_fn("high_blood_pressure"),         "binaria",  "Presenza di ipertensione (0/1)"],
            [_fn("platelets"),                   "continua", "Conta piastrinica (kiloplatelets/mL)"],
            [_fn("serum_creatinine"),            "continua", "Livello di creatinina serica (mg/dL)"],
            [_fn("serum_sodium"),                "continua", "Livello di sodio sierico (mEq/L)"],
            [_fn("sex"),                         "binaria",  "Sesso biologico (0=F, 1=M)"],
            [_fn("smoking"),                     "binaria",  "Fumatore (0/1)"],
            [_fn("time"),                        "continua", "Durata del follow-up (giorni)"],
        ],
        [0.26, 0.12, 0.62],
    )
    return rf"""
\section{{Introduzione del problema e dataset}}

\subsection{{Contesto e motivazione}}

\textbf{{Il problema.}}
L'insufficienza cardiaca è una condizione clinica grave in cui il cuore non riesce a pompare
sangue in modo sufficiente. Identificare i fattori di rischio associati al decesso è fondamentale
per supportare le decisioni cliniche e migliorare la prognosi dei pazienti.

\textbf{{Obiettivo.}}
Il progetto confronta due famiglie di modelli di machine learning per la predizione della
mortalità: modelli interpretabili basati su regole (rule-based) e modelli non interpretabili
basati su ensemble di alberi. L'obiettivo è valutare il trade-off tra performance predittiva e
interpretabilità in un contesto clinico ad alta criticità.

\textbf{{Explainable AI in ambito clinico.}}
In medicina, un modello non spiegabile --- per quanto accurato --- non è direttamente utilizzabile
come strumento di supporto decisionale: il medico deve poter comprendere e validare le motivazioni
di ogni predizione. I modelli intrinsecamente interpretabili soddisfano questo requisito senza
richiedere metodi post-hoc aggiuntivi (es.\ SHAP, LIME), che possono introdurre approssimazioni
e spiegazioni fuorvianti.

\textbf{{Struttura del report.}}
Il Capitolo~1 introduce il problema clinico, descrive il dataset e presenta l'analisi esplorativa.
Il Capitolo~2 descrive la metodologia: pipeline di elaborazione dei dati, modelli utilizzati e
selezione degli iperparametri. Il Capitolo~3 presenta i risultati: metriche comparative, curve
ROC, matrici di confusione, regole apprese, cross-validation, esperimento di ablazione della
feature \texttt{{time}}, analisi delle feature (importance globale) e Partial Dependence Plots
con ICE. Il Capitolo~4 raccoglie le conclusioni e le considerazioni cliniche.

\subsection{{Il Dataset}}

Il dataset \emph{{Heart Failure Clinical Records}} (UCI, id=519) raccoglie cartelle cliniche di
{X_rows} pazienti con insufficienza cardiaca seguiti in follow-up. La variabile target
\texttt{{DEATH\_EVENT}} indica se il paziente è deceduto durante il periodo di osservazione.

\medskip
\noindent {stats}

\medskip
\textbf{{Gestione delle feature: binarie vs continue.}}
Le 12 feature si dividono in 5 binarie (\texttt{{anaemia}}, \texttt{{diabetes}},
\texttt{{high\_blood\_pressure}}, \texttt{{sex}}, \texttt{{smoking}}) e 7 continue.
I modelli ensemble e la maggior parte dei rule-based le ricevono direttamente.
RuleFit richiede standardizzazione (StandardScaler), mentre BayesianRuleList richiede feature
esclusivamente binarie: le variabili continue vengono discretizzate in bin tramite
KBinsDiscretizer (one-hot, 4 bin quantili) prima del fitting.

\medskip
{dataset_tbl}

\clearpage
\subsection{{EDA --- Distribuzioni per classe}}

{_img("eda_continuous.png", "0.88\\textwidth")}

Ogni istogramma sovrappone la distribuzione di una feature per sopravvissuti (blu) e deceduti
(rosso). \texttt{{time}} è il discriminatore più potente: un follow-up breve corrisponde quasi
sempre a decesso. \texttt{{serum\_creatinine}} alta indica compromissione renale; \texttt{{ejection\_fraction}}
bassa segnala ridotta capacità di pompa. \texttt{{platelets}} e \texttt{{creatinine\_phosphokinase}}
mostrano distribuzioni largamente sovrapposte.

\clearpage
\subsection{{EDA --- Matrice di correlazione}}

{_img("eda_correlation.png")}

\texttt{{time}} ha la correlazione più forte con DEATH\_EVENT ($-$0.53): pazienti con follow-up
lungo tendono a sopravvivere. \texttt{{serum\_creatinine}} (+0.29) ed
\texttt{{ejection\_fraction}} ($-$0.27) sono i segnali biochimici più informativi.
Le feature binarie mostrano correlazioni deboli ($< \pm 0.15$).
"""


# ── Chapter 2: Architettura ───────────────────────────────────────────────────

def _ch2(best: dict) -> str:
    modules_tbl = _tbl(
        ["Modulo", "Responsabilita'"],
        [
            ["config.py",         "Centralizza iperparametri, path e costanti"],
            ["data\\_loader.py",  "Download UCI, cache CSV, split stratificato 60/20/20, tre varianti di preprocessing"],
            ["models.py",         "Factory function per ciascuno dei 9 modelli; accetta **params per override"],
            ["evaluate.py",       "Calcolo metriche, grafici, cross-validation e grid search sul validation set (ROC-AUC)"],
            ["main.py",           "Orchestrazione dell'intera pipeline end-to-end"],
            ["report\\_latex.py", "Legge gli artefatti salvati e genera il report LaTeX/PDF"],
        ],
        [0.22, 0.78],
    )

    def _bval(model, param, fallback="---"):
        model_params = best.get(model, {})
        if param not in model_params:
            return fallback
        v = model_params[param]
        return "None" if v is None else str(v)

    grid_rows = [
        ["GreedyRuleList",   "max\\_depth",         "1, 2, 3, 4, 6",        _bval("GreedyRuleList",   "max_depth")],
        ["GreedyRuleList",   "criterion",            "gini, entropy",        _bval("GreedyRuleList",   "criterion")],
        ["GreedyRuleList",   "class\\_weight",       "None, balanced",       _bval("GreedyRuleList",   "class_weight")],
        ["BayesianRuleList", "maxcardinality",       "2, 3",                 _bval("BayesianRuleList", "maxcardinality")],
        ["BayesianRuleList", "listlengthprior",      "2, 3, 4",              _bval("BayesianRuleList", "listlengthprior")],
        ["BayesianRuleList", "minsupport",           "0.05, 0.1, 0.2",       _bval("BayesianRuleList", "minsupport")],
        ["SkopeRules",       "max\\_depth",          "2, 3, 4",              _bval("SkopeRules",       "max_depth")],
        ["SkopeRules",       "precision\\_min",      "0.5, 0.6",        _bval("SkopeRules",       "precision_min")],
        ["SkopeRules",       "recall\\_min",         "0.3, 0.4",        _bval("SkopeRules",       "recall_min")],
        ["RandomForest",     "n\\_estimators",       "100, 200",        _bval("RandomForest",     "n_estimators")],
        ["RandomForest",     "max\\_depth",          "None, 5, 10",     _bval("RandomForest",     "max_depth")],
        ["RandomForest",     "min\\_samples\\_leaf", "1, 2, 5",         _bval("RandomForest",     "min_samples_leaf")],
        ["GradientBoosting", "n\\_estimators",       "100, 200",        _bval("GradientBoosting", "n_estimators")],
        ["GradientBoosting", "learning\\_rate",      "0.05, 0.1",       _bval("GradientBoosting", "learning_rate")],
        ["GradientBoosting", "max\\_depth",          "3, 4, 5",         _bval("GradientBoosting", "max_depth")],
        ["XGBoost",          "n\\_estimators",       "100, 200",        _bval("XGBoost",          "n_estimators")],
        ["XGBoost",          "learning\\_rate",      "0.05, 0.1",       _bval("XGBoost",          "learning_rate")],
        ["XGBoost",          "max\\_depth",          "3, 4, 5",         _bval("XGBoost",          "max_depth")],
    ]
    grid_tbl = _tbl(
        ["Modello", "Parametro", "Valori esplorati", "Ottimo (val)"],
        grid_rows,
        [0.22, 0.22, 0.30, 0.26],
    )

    fixed_tbl = _tbl(
        ["Parametro", "Valore", "Motivazione"],
        [
            ["RANDOM\\_STATE",     "42",      "Seed fisso per riproducibilita'"],
            ["TEST\\_SIZE",        "0.20",    "20\\% test ($\\approx$60 campioni), mantenuto separato durante il tuning"],
            ["VAL\\_SIZE",         "0.20",    "20\\% validation ($\\approx$60 campioni) per la scelta degli iperparametri"],
            ["CV\\_FOLDS",         "10",      "10-fold CV su trainval ($\\approx$240 campioni): stima robusta della stabilita'"],
            ["BRL\\_MAX\\_ITER",   "20\\,000","Iterazioni MCMC per la convergenza (ridotto a 3\\,000 durante il tuning)"],
            ["RULEFIT\\_MAX\\_RULES","30",    "Limita le regole candidate; troppe degenerano in black-box"],
            ["FIGS\\_MAX\\_RULES", "10",      "Alberi nella somma FIGS; oltre 10 si perde interpretabilita'"],
        ],
        [0.26, 0.10, 0.64],
    )

    brl_card = _bval("BayesianRuleList", "maxcardinality", "?")
    grl_depth = _bval("GreedyRuleList",   "max_depth",       "?")
    grl_crit  = _bval("GreedyRuleList",   "criterion",       "?")
    grl_cw    = _bval("GreedyRuleList",   "class_weight",    "?")
    brl_llp   = _bval("BayesianRuleList", "listlengthprior", "?")
    brl_ms    = _bval("BayesianRuleList", "minsupport",      "?")

    return rf"""
\section{{Metodologia}}

\subsection{{Struttura del Progetto}}

Il progetto è organizzato in una cartella \texttt{{src/}} con sei moduli distinti, ciascuno
con una responsabilità precisa. Questa separazione permette di modificare una componente senza
toccare le altre e rende il codice più leggibile e testabile.

\medskip
{modules_tbl}

\medskip
\textbf{{Output della pipeline.}}
Ogni esecuzione di \texttt{{main.py}} produce artefatti su disco: i dati grezzi vengono cachati
in \texttt{{data/}}; le metriche e le regole vengono salvate in \texttt{{results/files/}}; i
grafici in \texttt{{results/plots/}}; il report finale in \texttt{{Report/}}.

\textbf{{Dipendenze principali.}}
\texttt{{scikit-learn}} gestisce preprocessing, metriche e cross-validation.
\texttt{{imodels}} fornisce le implementazioni dei modelli rule-based (RuleFit, GreedyRuleList,
BayesianRuleList, FIGS, SkopeRules). \texttt{{wittgenstein}} implementa RIPPER.
\texttt{{xgboost}} implementa XGBoost. \texttt{{ucimlrepo}} scarica il dataset direttamente dal
repository UCI.

\subsection{{Pipeline di Elaborazione dei Dati}}

\textbf{{Split stratificato 60/20/20.}}
Il dataset viene diviso in tre parti con doppio \texttt{{train\_test\_split}} stratificato:
training ($\approx$179 campioni, 60\%), validation ($\approx$60 campioni, 20\%) e test
($\approx$60 campioni, 20\%).
La stratificazione garantisce che ogni partizione mantenga la stessa proporzione di deceduti
($\sim$32\%) presente nel dataset originale.
Il validation set viene usato esclusivamente per la selezione degli iperparametri;
il test set è mantenuto completamente separato e usato una sola volta per la valutazione finale.

\textbf{{Tre varianti di preprocessing.}}
I modelli richiedono rappresentazioni diverse: (1)~\emph{{raw}} --- il DataFrame originale con le
12 feature, usato da GreedyRuleList, FIGS, SkopeRules, RIPPER e dagli ensemble;
(2)~\emph{{scalata}} --- StandardScaler porta le feature continue a media 0 e varianza 1,
necessaria per RuleFit che usa Lasso internamente;
(3)~\emph{{discretizzata}} --- KBinsDiscretizer (4 bin quantili, one-hot) produce $\approx$33
colonne binarie, obbligatoria per BayesianRuleList.

\textbf{{PreparedData --- struttura dati condivisa.}}
Il preprocessing è incapsulato in un dataclass \texttt{{PreparedData}} con i campi:
\texttt{{X\_train/X\_val/X\_test}} (raw), le varianti scalate e discretizzate per ciascuna
partizione, \texttt{{X\_trainval/y\_trainval}} (train+val, usato per la cross-validation),
\texttt{{y\_train/y\_val/y\_test}} e \texttt{{scale\_pos\_weight}} per XGBoost.
Scaler e discretizzatore sono fittati solo sul training set e applicati con
\texttt{{transform()}} a validation e test, evitando data leakage.

\textbf{{Sbilanciamento delle classi.}}
Il dataset contiene 203 sopravvissuti e 96 deceduti ($\sim$32\% positivi).
Random Forest usa \texttt{{class\_weight='balanced'}}; XGBoost usa \texttt{{scale\_pos\_weight}}
calcolato come rapporto negativi/positivi nel training set.

\subsection{{Modelli Interpretabili Basati su Regole}}

I modelli basati su regole producono predizioni if-then direttamente leggibili da un esperto
di dominio, senza richiedere strumenti post-hoc aggiuntivi.

\textbf{{RuleFit.}}
Estrae regole da un ensemble di alberi e le pondera con regressione Lasso. Il coefficiente
di ogni regola indica il suo contributo alla predizione; richiede feature standardizzate.

\textbf{{GreedyRuleList (GRL).}}
Costruisce una lista ordinata di regole if-then in modo greedy. Ogni regola è un decision
stump (una condizione su una feature); il parametro \texttt{{max\_depth}} controlla la lunghezza
della lista. Si applica la prima regola soddisfatta; se nessuna si attiva, vale la regola di
default. Importante: GRL produce sempre regole a condizione singola per design
della libreria \texttt{{imodels}}.

\textbf{{BayesianRuleList (BRL).}}
Apprende una lista di regole tramite MCMC su feature binarie. Le variabili continue vengono
discretizzate in bin prima del fitting; produce stime probabilistiche con IC al 95\%.
Il parametro \texttt{{maxcardinality}} controlla il numero massimo di condizioni per regola;
dal tuning sul validation set è risultato ottimale il valore {brl_card}.

\textbf{{FIGS --- Fast Interpretable Greedy-Tree Sums.}}
Somma di piccoli alberi decisionali, ciascuno appreso greedy sui residui del precedente.
La predizione è la somma dei valori ``Val'' raggiunti traversando ogni albero; per i
classificatori viene poi applicata una softmax. Struttura additiva trasparente con capacità
espressiva maggiore di un singolo albero.

\textbf{{SkopeRules.}}
Genera regole ad alta precisione tramite sub-campionamenti ripetuti di ensemble di alberi,
poi de-duplica quelle simili. La profondità degli alberi base (\texttt{{max\_depth}}) controlla
il numero di condizioni AND per regola; ogni regola è corredata da precisione e recall.

\textbf{{RIPPER --- Repeated Incremental Pruning to Produce Error Reduction.}}
Apprende regole congiuntive (AND tra più condizioni simultanee) in modo incrementale, potando
ogni regola per ridurre l'errore. La prima regola soddisfatta predice Decesso; se nessuna si
attiva, la predizione è Sopravvissuto (DEFAULT). A differenza di GRL, può combinare più
feature in un singolo antecedente.

\subsection{{Modelli Ensemble Non Interpretabili}}

I modelli ensemble combinano centinaia di alberi decisionali per ottenere performance
superiori, ma il processo decisionale risulta opaco (black box). Sono usati come baseline
di riferimento per valutare il gap di performance rispetto ai modelli interpretabili.

\textbf{{Random Forest.}}
Allena $N$ alberi su sotto-campioni casuali del dataset (bagging) e su sottoinsiemi casuali
delle feature (feature bagging). La predizione finale è la media delle probabilità dei
singoli alberi. Configurato con \texttt{{class\_weight='balanced'}}.

\textbf{{Gradient Boosting (sklearn).}}
Allena alberi sequenzialmente: ogni albero corregge gli errori del precedente minimizzando
il gradiente della loss. Riduce il bias rispetto al bagging.

\textbf{{XGBoost.}}
Implementazione ottimizzata del gradient boosting con regolarizzazione L1/L2, gestione
nativa dei valori mancanti e parallelismo. Usa \texttt{{scale\_pos\_weight}} per bilanciare
le classi.

\subsection{{Selezione degli Iperparametri}}

Il dataset (299 campioni) è suddiviso in tre parti stratificate: train 60\%, validation 20\%,
test 20\% ($\approx$179/60/60 campioni). Gli iperparametri di GreedyRuleList, BayesianRuleList,
SkopeRules e dei tre ensemble vengono scelti massimizzando il ROC-AUC sul validation set
tramite grid search esaustiva. RuleFit, FIGS e RIPPER usano i parametri di default della
libreria. La valutazione finale avviene sul test set, mantenuto separato durante tutto il
processo di selezione.

\begin{{minipage}}{{\textwidth}}
\noindent\textbf{{Grid di ricerca --- migliori valori sul validation set}}
\medskip

{grid_tbl}
\end{{minipage}}

\vspace{{1.2em}}
\begin{{minipage}}{{\textwidth}}
\noindent\textbf{{Parametri fissi (non ottimizzati)}}
\medskip

{fixed_tbl}
\end{{minipage}}

\vspace{{1.8em}}
\textbf{{Sensitivity agli iperparametri.}}
Per ciascun modello ottimizzato è riportato il ROC-AUC sul validation set al variare di ogni
iperparametro (marginalizzando sugli altri: per ogni valore del parametro si considera il
massimo AUC tra tutte le combinazioni che lo includono). Le barre evidenziate corrispondono
al valore ottimale selezionato.

{_img("hyperparam_rule_based.png")}

\textbf{{Modelli rule-based.}}
Per GreedyRuleList \texttt{{max\_depth={grl_depth}}} è il punto di equilibrio: liste troppo
corte non catturano abbastanza pattern; liste lunghe non portano benefici su questo dataset.
Il criterio di split ottimale è \texttt{{{grl_crit}}}; \texttt{{class\_weight={grl_cw}}}
indica che il modello gestisce lo sbilanciamento delle classi tramite la struttura delle regole
stesse, senza ponderazione esplicita.
Per BayesianRuleList la cardinalità ottimale è {brl_card}: un valore più alto permette
al MCMC di esplorare congiunzioni più complesse durante la ricerca, sebbene in pratica
converga su regole a condizione singola (come discusso in sezione~3.4).
\texttt{{listlengthprior={brl_llp}}} favorisce liste più lunghe, adatte alla variabilità del
dataset; \texttt{{minsupport={brl_ms}}} richiede che ogni regola copra almeno il
{int(float(brl_ms)*100) if brl_ms != '?' else '?'}\% dei campioni di training, riducendo
regole troppo specifiche.
Per SkopeRules, \texttt{{max\_depth=2}} genera regole bi-condizionali ad alta precisione;
profondità maggiori producono overfitting. \texttt{{precision\_min=0.6}} e
\texttt{{recall\_min=0.3}} bilanciano qualità e copertura delle regole selezionate, limitando
la proliferazione di regole ridondanti.

\medskip
{_img("hyperparam_ensemble.png")}

\textbf{{Modelli ensemble.}}
Tutti e tre i modelli mostrano scarsa sensibilità al numero di stimatori nell'intervallo
[100, 200]: la performance è già stabile a 100. Per Random Forest \texttt{{max\_depth=None}}
(alberi completi) domina; \texttt{{min\_samples\_leaf=5}} riduce la varianza su dataset piccoli.
Gradient Boosting e XGBoost preferiscono alberi poco profondi (max\_depth=3--4), coerente con
la dimensione ridotta del dataset.
"""


# ── Chapter 3: Risultati ──────────────────────────────────────────────────────

def _ch3(
    metrics_df,
    grl_rows, brl_rows, rip_rows,
    rf_rows, skope_rows, figs_text,
    abl_df, grl_nt_rows, brl_nt_rows,
) -> str:

    # ── 3.1 Metrics table ─────────────────────────────────────────────────────
    if metrics_df is not None:
        m = metrics_df.copy()
        if "name" in m.columns:
            m = m.set_index("name")
        met_rows = []
        for name, row in m.iterrows():
            met_rows.append([
                str(name),
                str(row.get("Type", "---")),
                f"{float(row['Accuracy']):.4f}",
                f"{float(row['Precision']):.4f}",
                f"{float(row['Recall']):.4f}",
                f"{float(row['F1']):.4f}",
                f"{float(row['ROC-AUC']):.4f}",
            ])
    else:
        met_rows = [["---"] * 7]
    metrics_tbl = _tbl(
        ["Modello", "Tipo", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"],
        met_rows,
        [0.24, 0.14, 0.10, 0.10, 0.10, 0.10, 0.12],
    )

    # ── 3.4 Rules tables ──────────────────────────────────────────────────────
    # grl_rows: [cond, prec, rec, pred, n_pts]
    grl_tbl = _tbl(
        ["Condizione", "Precisione", "Recall", "Predizione", "Campioni"],
        grl_rows or [["---"] * 5],
        [0.38, 0.12, 0.12, 0.22, 0.16],
    )
    brl_tbl = _tbl(
        ["Condizione", "P(decesso)", "IC 95\\%"],
        brl_rows or [["---", "---", "---"]],
        [0.58, 0.20, 0.22],
    )
    # rip_rows: [antecedent, prec, rec, n_pts] — predizione derivata
    if rip_rows:
        rip_data = [
            [r[0], "Sopravvissuto" if r[0] == "DEFAULT" else "Decesso",
             r[1], r[2], r[3]]
            for r in rip_rows
        ]
    else:
        rip_data = [["---"] * 5]
    rip_tbl = _tbl(
        ["Regola", "Predizione", "Precisione", "Recall", "Campioni"],
        rip_data,
        [0.38, 0.18, 0.12, 0.12, 0.20],
    )
    rf_rules_tbl = _tbl(
        ["Regola", "Coefficiente"],
        rf_rows or [["---", "---"]],
        [0.82, 0.18],
    )

    # SkopeRules: show top 15 by precision, note total
    _SK_MAX = 15
    sk_sorted = sorted(skope_rows, key=lambda x: float(x[1]), reverse=True) if skope_rows else []
    sk_total = len(sk_sorted)
    sk_display = sk_sorted[:_SK_MAX]
    sk_note = (
        f"(mostrate le prime {_SK_MAX} su {sk_total} regole apprese, ordinate per precisione decrescente)"
        if sk_total > _SK_MAX else ""
    )
    sk_tbl = _tbl(
        ["Regola", "Precisione", "Recall"],
        sk_display or [["Nessuna regola appresa.", "---", "---"]],
        [0.64, 0.18, 0.18],
    )
    sk_note_line = f"\\smallskip\\noindent\\textit{{{_e(sk_note)}}}\n" if sk_note else ""

    # FIGS verbatim block (max 30 lines)
    figs_verb = ""
    if figs_text:
        lines = figs_text.splitlines()[:30]
        inner = "\n".join(lines)
        figs_verb = f"\\begin{{small}}\n\\begin{{verbatim}}\n{inner}\n\\end{{verbatim}}\n\\end{{small}}"

    # ── 3.6 Ablation ──────────────────────────────────────────────────────────
    if abl_df is not None:
        abl_index = abl_df.set_index("name") if "name" in abl_df.columns else abl_df
        abl_rows_tbl = []
        for key in ["RandomForest", "RF_no_time", "GreedyRuleList", "GRL_no_time",
                    "BayesianRuleList", "BRL_no_time"]:
            lbl = key.replace("_", "\\_")
            if key in abl_index.index:
                r = abl_index.loc[key]
                abl_rows_tbl.append([
                    lbl,
                    f"{float(r['Accuracy']):.4f}",
                    f"{float(r['F1']):.4f}",
                    f"{float(r['ROC-AUC']):.4f}",
                ])
    else:
        abl_rows_tbl = [["---"] * 4]
    abl_tbl = _tbl(
        ["Modello", "Accuracy", "F1", "ROC-AUC"],
        abl_rows_tbl,
        [0.40, 0.20, 0.20, 0.20],
    )
    grl_nt_tbl = _tbl(
        ["Condizione", "Precisione", "Recall", "Predizione", "Campioni"],
        grl_nt_rows or [["---"] * 5],
        [0.38, 0.12, 0.12, 0.22, 0.16],
    )
    brl_nt_tbl = _tbl(
        ["Condizione", "P(decesso)", "IC 95\\%"],
        brl_nt_rows or [["---", "---", "---"]],
        [0.58, 0.20, 0.22],
    )

    return rf"""
\clearpage
\section{{Risultati}}

\subsection{{Metriche Comparative}}

{metrics_tbl}

\medskip
{_img("metrics_comparison.png")}

Accuracy e F1 riassumono la qualità complessiva, ma su classi sbilanciate possono essere
fuorvianti. Precision misura quante predizioni ``deceduto'' sono corrette; Recall quante
morti reali vengono identificate --- in clinica un Recall basso è più pericoloso di una
Precision bassa. ROC-AUC è la metrica più robusta: misura la capacità di separare le classi
indipendentemente dalla soglia.
GreedyRuleList raggiunge buon F1 e Accuracy con regole a condizione singola.
SkopeRules ottiene Precision=1.0 con Recall molto basso: identifica pochi deceduti ma senza
mai sbagliare. RIPPER produce regole congiuntive (AND) ma ha il ROC-AUC più basso tra tutti i
modelli, dimostrando che la complessità strutturale non implica migliore capacità discriminativa.

\subsection{{Curve ROC}}

{_img("roc_curves.png")}

La curva ROC traccia il TPR (Recall) contro il FPR al variare della soglia; l'AUC riassume
la performance in un unico numero indipendente dalla soglia (0.50\,=\,casuale, 1.0\,=\,perfetto).
Random Forest raggiunge l'AUC più alto tra tutti i modelli. SkopeRules è il miglior rule-based
in AUC (0.860), competitivo con i tre ensemble. RIPPER ha l'AUC più basso e anche F1 e Accuracy
ridotti: la maggiore espressività strutturale delle sue regole congiuntive non si traduce in
migliore capacità discriminativa su questo dataset.

\clearpage
\subsection{{Matrici di Confusione}}

{_img("confusion_matrices.png", "0.85\\textwidth")}

Ogni cella indica: riga = classe reale, colonna = classe predetta. La diagonale raccoglie le
predizioni corrette; gli elementi fuori diagonale sono gli errori. In clinica i due tipi di
errore hanno costo asimmetrico: un falso negativo (morte non rilevata) è più grave di un falso
positivo (falso allarme). BayesianRuleList massimizza il Recall sui deceduti --- pochi falsi
negativi --- a scapito di più falsi positivi. Gli ensemble bilanciano meglio i due errori.
SkopeRules ha Precision=1.0 ma Recall molto basso: identifica pochi deceduti, senza mai
produrre falsi positivi.

\clearpage
\subsection{{Regole Complete: tutti i modelli interpretabili}}

\textbf{{GreedyRuleList.}}
Lista IF/THEN sequenziale: la prima regola soddisfatta determina la classe, che non è
necessariamente Decesso --- ogni regola predice la classe di maggioranza del proprio segmento
($>$50\% deceduti $\Rightarrow$ Decesso). GRL è strutturalmente limitato a una condizione per
regola; il parametro \texttt{{max\_depth}} controlla la lunghezza della lista.

\medskip
{grl_tbl}

\medskip
\textbf{{BayesianRuleList.}}
Lista IF/ELSE con MCMC su feature discretizzate. Ogni antecedente ha al massimo
\texttt{{maxcardinality}} condizioni congiuntive; il valore ottimale ({{brl_card}}) è stato
scelto sul validation set. P(decesso) è la probabilità posteriore con IC al 95\%.
Nonostante \texttt{{maxcardinality={{brl_card}}}} permetta antecedenti multi-feature, il MCMC
converge su regole a condizione singola. Come confermato dall'esperimento in sezione~3.6 ---
dove BRL viene riaddestrato senza \texttt{{time}} --- la preferenza per antecedenti unari è
una proprietà intrinseca del MCMC su questo dataset: anche in assenza della feature dominante,
le congiunzioni rimangono ridondanti dal punto di vista probabilistico.

\medskip
{brl_tbl}

\medskip
\textbf{{RIPPER.}}
Ogni regola ha un antecedente con più condizioni AND (tutte devono essere verificate
simultaneamente). Se tutte le condizioni di una regola sono soddisfatte $\Rightarrow$ Decesso;
nessuna regola soddisfatta $\Rightarrow$ Sopravvissuto (DEFAULT).

\medskip
{rip_tbl}

\clearpage
\textbf{{RuleFit --- regole con coefficiente positivo (pro Decesso).}}
Ogni regola è un percorso estratto da un ensemble di alberi; il coefficiente Lasso indica
il peso nella predizione finale. Valori più alti indicano maggiore contributo al rischio di
decesso. Le regole possono combinare più feature (AND implicito sul percorso).

\medskip
{rf_rules_tbl}

\medskip
\textbf{{SkopeRules.}}
Regole ad alta precisione estratte tramite sub-campionamenti di ensemble di alberi, poi
de-duplicate. La profondità degli alberi base (\texttt{{max\_depth}} ottimizzato sul validation
set) controlla il numero di condizioni per regola. Ogni regola è corredata da Precisione e
Recall sul training set.

\medskip
{sk_tbl}
{sk_note_line}
\textbf{{FIGS --- Fast Interpretable Greedy-Tree Sums.}}
FIGS è una somma di alberi: ogni albero produce una coppia di valori (uno per classe), le
coppie vengono sommate elemento per elemento e una softmax sui due valori risultanti
fornisce le probabilità finali; la classe con probabilità maggiore è la predizione. Di seguito
la struttura completa degli alberi appresi:

\medskip
{figs_verb}

\clearpage
\subsection{{Cross-Validation 10-fold}}

{_img("cross_validation.png")}

Il boxplot mostra la distribuzione del ROC-AUC su 10 fold stratificati del set trainval
($\approx$240 campioni): mediana, IQR (bordi del box) e range (baffi). La stratificazione
garantisce che ogni fold mantenga la stessa proporzione di deceduti ($\sim$32\%). I modelli
usano gli iperparametri ottimizzati sul validation set. Random Forest ha la mediana più alta e
IQR contenuto: performance stabile e consistente. RuleFit è il miglior rule-based in CV.
FIGS mostra alta varianza --- instabile su dataset piccoli. GreedyRuleList usa un custom scorer
che aggira un'incompatibilità con l'API sklearn.

\clearpage
\subsection{{RF, GRL e BRL senza feature \texttt{{time}}}}

\textbf{{Motivazione.}}
La feature \texttt{{time}} (durata del follow-up) è la più predittiva del dataset. In contesti
reali di triage questa informazione non è sempre disponibile al momento della prima valutazione.
Questo esperimento valuta RF, GRL e BRL privandoli di \texttt{{time}}, forzandoli ad appoggiarsi
ai segnali biochimici e demografici rimanenti. Tutti e tre i modelli vengono ri-tunati sul
validation set (senza \texttt{{time}}) per garantire un confronto equo. Per BRL l'esperimento
verifica inoltre se, in assenza della feature dominante, il MCMC produce antecedenti
multi-condizionali.

\medskip
{abl_tbl}

\medskip
Rimuovendo \texttt{{time}}, il ROC-AUC scende per tutti i modelli. RF e GRL subiscono un calo
di circa 0.10--0.13 ma rimangono utilizzabili, appoggiandosi a \texttt{{serum\_creatinine}} ed
\texttt{{ejection\_fraction}} --- feature biochimiche disponibili alla prima valutazione.
BRL soffre maggiormente (ROC-AUC scende a 0.65, accuracy al 32\%): senza \texttt{{time}}, il
modello tende a classificare la maggior parte dei campioni come deceduti, riflettendo la
difficoltà del MCMC nel bilanciare le soglie senza il segnale dominante.

\medskip
\textbf{{Regole GRL senza \texttt{{time}}.}}
Con la rimozione di \texttt{{time}}, GreedyRuleList apprende dai segnali biochimici:
\texttt{{serum\_creatinine}} ed \texttt{{ejection\_fraction}} diventano le feature dominanti.

\medskip
{grl_nt_tbl}

\medskip
\textbf{{Regole BRL senza \texttt{{time}}.}}
Rimuovendo \texttt{{time}}, BayesianRuleList continua a produrre antecedenti a condizione
singola: \texttt{{ejection\_fraction}} e \texttt{{serum\_creatinine}} vengono usate in regole
sequenziali separate, non congiunte. La preferenza per antecedenti unari è quindi una
proprietà intrinseca del MCMC su questo dataset, indipendente dalla dominanza di \texttt{{time}}.

\medskip
{brl_nt_tbl}

\clearpage
\subsection{{Analisi delle Feature}}

\textbf{{Regole RuleFit --- coefficienti Lasso.}}

{_img("rulefit_rules.png")}

Coefficienti Lasso delle regole estratte da RuleFit: positivo (barre rosse) $\Rightarrow$ la
regola aumenta il rischio di decesso; negativo (blu) $\Rightarrow$ lo riduce. Le regole
pro-decesso con peso più alto coinvolgono \texttt{{time}} basso e \texttt{{serum\_creatinine}}
alta. Le regole pro-sopravvivenza implicano \texttt{{ejection\_fraction}} più alta e follow-up
lungo. La penalizzazione Lasso azzera i coefficienti ridondanti, mantenendo solo le regole con
effetto reale sulla predizione.

\textbf{{Feature Importance --- Random Forest (MDI).}}

{_img("feature_importance_rf.png")}

L'importanza MDI misura di quanto ogni feature riduce l'impurità media negli alberi del Random
Forest (n\_estimators=100): \texttt{{time}} domina nettamente, seguita da \texttt{{serum\_creatinine}} ed
\texttt{{ejection\_fraction}}. Questa gerarchia è confermata in modo indipendente da RuleFit
(sezione~3.4): le regole con coefficiente Lasso più alto coinvolgono \texttt{{time}} basso e
\texttt{{serum\_creatinine}} alta. Due metodi radicalmente diversi --- importance da ensemble di
alberi vs regressione Lasso su regole estratte --- convergono sugli stessi tre segnali
fisiopatologici, coerenti con la letteratura sull'insufficienza cardiaca.

\subsection{{Partial Dependence Plots e ICE}}

I Partial Dependence Plot (PDP) e le Individual Conditional Expectation (ICE) sono strumenti
post-hoc per visualizzare l'effetto marginale di una feature sulla predizione, mantenendo le
altre feature ai loro valori osservati (marginalizzazione).

\textbf{{Motivazione: perché sui modelli ensemble.}}
PDP e ICE sono strumenti nati per spiegare modelli black-box: la loro utilità è massima sui
modelli ensemble (RF, GradientBoosting, XGBoost) la cui logica interna non è direttamente
ispezionabile. Per i modelli rule-based (RuleFit, GRL, BRL, FIGS, SkopeRules, RIPPER) le regole
apprese (o i coefficienti Lasso nel caso di RuleFit) sono già la spiegazione completa del
comportamento del modello, quindi un PDP sarebbe ridondante.
L'analisi che segue si concentra dunque sui tre ensemble, verificando se modelli con bias
induttivi diversi (bagging vs boosting) apprendono pattern simili.

\textbf{{PDP 1D + ICE --- tre feature principali.}}
Ogni cella mostra la curva PDP media (linea colorata) sovrapposta alle curve ICE individuali
(grigio, 50 campioni). Le tre feature analizzate --- \texttt{{time}},
\texttt{{serum\_creatinine}} ed \texttt{{ejection\_fraction}} --- sono le più informative
secondo la feature importance MDI. I tre modelli confrontati sono Random Forest (arancio),
Gradient Boosting (blu) e XGBoost (verde).

{_img("pdp_ice_1d.png")}

\textbf{{Lettura dei risultati.}}
Tutti e tre gli ensemble apprendono pattern coerenti: \texttt{{time}} ha relazione
fortemente decrescente con soglia critica intorno a 70--80 giorni; \texttt{{serum\_creatinine}}
ha effetto crescente (alta creatinina = alto rischio renale); \texttt{{ejection\_fraction}}
ha effetto decrescente. La convergenza su questi pattern --- nonostante RF usi bagging e
GB/XGBoost usino boosting --- è una validazione qualitativa robusta: i tre modelli, partendo
da bias induttivi diversi, identificano le stesse relazioni fisiopatologiche.

\textbf{{Differenze qualitative tra i modelli.}}
RF produce curve più lisce e monotone, conseguenza del bagging che media molte predizioni
indipendenti. Gradient Boosting e XGBoost, costruendo alberi sequenzialmente sui residui,
mostrano transizioni più nette e talvolta non-monotonie locali (es.\ piccoli rimbalzi su
\texttt{{time}} attorno a 150--180 giorni), riflettendo la maggiore capacità del boosting di
modellare pattern fine. Le curve ICE rivelano che per \texttt{{time}} la maggior parte dei
campioni segue il trend medio (eterogeneità contenuta), mentre per \texttt{{serum\_creatinine}}
alcune ICE divergono in particolare nei modelli boosting --- segnale di interazioni con altre
feature catturate dal modello.

\bigskip
\textbf{{PDP 2D --- interazione \texttt{{time}} $\times$ \texttt{{serum\_creatinine}}.}}
Il PDP bidimensionale mostra la probabilità di decesso media in funzione congiunta di
\texttt{{time}} e \texttt{{serum\_creatinine}}, marginalizzando sulle restanti feature.
Non esiste un analogo ICE per il caso 2D (richiederebbe una superficie per ogni campione).

\textbf{{Motivazione della scelta della coppia.}}
La coppia \texttt{{time}} $\times$ \texttt{{serum\_creatinine}} è stata selezionata in base
a due criteri: (i)~sono le due feature più informative secondo la feature importance MDI,
con \texttt{{serum\_creatinine}} che ha la correlazione più alta con \texttt{{DEATH\_EVENT}}
tra le feature biochimiche (+0.29, contro $-$0.27 di \texttt{{ejection\_fraction}});
(ii)~compaiono congiuntamente nella maggior parte delle SkopeRules apprese
(es.\ \texttt{{time <= 73.5 AND serum\_creatinine > 0.85}}) e nelle regole RuleFit con
coefficiente Lasso più alto --- segnale che il modello stesso le considera interagenti, e
quindi la coppia più candidata a esibire una vera interazione non-additiva.

{_img("pdp_2d.png")}

\textbf{{Lettura dei risultati.}}
L'angolo in alto a sinistra (follow-up breve e creatinina alta) corrisponde al massimo rischio
in tutti e tre i modelli (P(decesso) $\approx$ 0.6--0.88). L'angolo in basso a destra
(follow-up lungo e creatinina bassa) corrisponde alla minima probabilità di decesso
(P(decesso) $\approx$ 0.14--0.22). La regione di transizione è governata principalmente da
\texttt{{time}}: la soglia critica appare intorno a 70--80 giorni in modo abbastanza
indipendente dal valore di \texttt{{serum\_creatinine}}, coerente con la dominanza di
\texttt{{time}} osservata nella feature importance.

\textbf{{Confronto tra modelli.}}
Random Forest mostra curve di livello smooth e prevalentemente verticali: l'effetto dominante
è \texttt{{time}}, con \texttt{{serum\_creatinine}} che modula gradualmente il rischio
soprattutto a follow-up brevi. Gradient Boosting evidenzia un pattern più complesso, con
"isole" di rischio elevato (P $\approx$ 0.88) e regioni di basso rischio anche per follow-up
intermedi: il boosting cattura una vera non-additività tra le due feature. XGBoost si colloca
tra i due, con transizioni nette ma più frammentate rispetto a RF.
La convergenza qualitativa dei tre modelli (gradiente diagonale dall'angolo alto-sinistro a
quello basso-destro) conferma che la struttura del rischio è solida; le differenze locali
riflettono il bias induttivo di ciascun algoritmo: bagging $\rightarrow$ smoothing, boosting
$\rightarrow$ pattern fine.

"""


# ── Chapter 4: Conclusioni ────────────────────────────────────────────────────

def _ch4(metrics_df, best: dict) -> str:
    def m(name, col):
        return _metric(metrics_df, name, col)

    brl_card = str(best.get("BayesianRuleList", {}).get("maxcardinality", "?"))
    rf_auc  = m("RandomForest",     "ROC-AUC")
    xgb_auc = m("XGBoost",          "ROC-AUC")
    grl_auc = m("GreedyRuleList",   "ROC-AUC")
    rf_auc_cv  = m("RandomForest",  "ROC-AUC")  # from metrics table
    grl_f1  = m("GreedyRuleList",   "F1")
    brl_f1  = m("BayesianRuleList", "F1")
    brl_rec = m("BayesianRuleList", "Recall")

    # Best models summary
    best_rule_row = best_ens_row = None
    if metrics_df is not None:
        df = metrics_df.copy()
        if "name" not in df.columns and df.index.name == "name":
            df = df.reset_index()
        rb = df[df["Type"] == "Rule-based"].sort_values("ROC-AUC", ascending=False)
        en = df[df["Type"] == "Ensemble"].sort_values("ROC-AUC", ascending=False)
        if not rb.empty:
            r = rb.iloc[0]
            best_rule_row = [r["name"], str(r["Type"]),
                             f"{float(r['ROC-AUC']):.4f}", f"{float(r['F1']):.4f}"]
        if not en.empty:
            r = en.iloc[0]
            best_ens_row = [r["name"], str(r["Type"]),
                            f"{float(r['ROC-AUC']):.4f}", f"{float(r['F1']):.4f}"]

    best_rows = []
    if best_rule_row: best_rows.append(best_rule_row)
    if best_ens_row:  best_rows.append(best_ens_row)
    best_tbl = _tbl(
        ["Modello", "Categoria", "ROC-AUC", "F1"],
        best_rows or [["---"] * 4],
        [0.34, 0.22, 0.22, 0.22],
    )

    return rf"""
\section{{Conclusioni}}

\textbf{{Confronto tra approcci.}}
I modelli ensemble raggiungono ROC-AUC superiore (Random Forest {rf_auc}, XGBoost {xgb_auc})
ma il divario con i rule-based non è netto: GreedyRuleList ({grl_auc}) è competitivo su un
dataset di soli 299 campioni, dove la semplicità strutturale riduce il rischio di overfitting.
BayesianRuleList ottiene il miglior F1 tra i rule-based ({brl_f1}) e produce stime
probabilistiche con IC al 95\%, utili per comunicare l'incertezza al clinico.
GreedyRuleList offre il secondo F1 rule-based ({grl_f1}) con una lista ordinata di condizioni
particolarmente leggibile da un clinico. RuleFit assegna coefficienti espliciti a ogni regola,
rendendo visibile il peso relativo di ciascuna. FIGS ha struttura additiva
particolarmente trasparente. SkopeRules raggiunge Precision=1.0 con Recall molto basso: non
produce mai falsi positivi, ma identifica solo una piccola frazione dei deceduti --- utile quando
ogni allarme deve corrispondere a un caso critico. RIPPER apprende regole congiuntive (AND tra più condizioni) in modo incrementale con potatura:
nonostante la maggiore espressività strutturale rispetto a GRL, ottiene il ROC-AUC più basso
di tutti i modelli. Il vantaggio strutturale non si traduce in migliore capacità discriminativa
su questo dataset.

\textbf{{Tuning degli iperparametri.}}
La selezione sul validation set ha mostrato che per BayesianRuleList la cardinalità ottimale è
{brl_card} (sebbene in pratica il MCMC converga su regole a condizione singola, come discusso
in sezione~3.4), mentre per
SkopeRules \texttt{{max\_depth=2}} con soglie \texttt{{precision\_min=0.6}} e
\texttt{{recall\_min=0.3}} produce regole di qualità senza eccessiva proliferazione. Per gli
ensemble, i valori ottimali tendono a configurazioni più semplici (meno stimatori, alberi poco
profondi), coerente con la dimensione ridotta del dataset.

\textbf{{Precision vs Recall in ambito clinico.}}
Il trade-off tra precisione e recall non è neutro in contesti medici. Alta recall minimizza i
falsi negativi (morti non rilevate), che in clinica sono più pericolosi dei falsi positivi
(falsi allarmi). Alta precisione garantisce invece che ogni segnalazione sia affidabile,
preferibile quando si pianificano interventi invasivi. La scelta del modello dipende quindi
dal contesto d'uso, e la trasparenza dei modelli rule-based è la condizione necessaria per
fare questa scelta in modo consapevole.

\textbf{{Validazione qualitativa delle regole.}}
Le feature più ricorrenti --- \texttt{{time}} (follow-up), \texttt{{serum\_creatinine}} ed
\texttt{{ejection\_fraction}} --- sono coerenti con la letteratura clinica sull'insufficienza
cardiaca: follow-up breve segnala una fase acuta critica, creatinina alta indica insufficienza
renale, ejection fraction bassa riflette ridotta capacità di pompa. Il fatto che i modelli
apprendano autonomamente queste relazioni costituisce una validazione qualitativa.

\textbf{{Ablazione della feature \texttt{{time}}.}}
Come mostrato nella sezione~3.6, rimuovendo \texttt{{time}} il ROC-AUC scende per tutti e tre
i modelli, confermando il suo peso dominante. RF e GRL rimangono utilizzabili in triage;
BRL degrada significativamente, evidenziando una maggiore dipendenza dal segnale di follow-up.

\textbf{{Sintesi finale.}}
L'analisi delle feature (sezione~3.7) e i Partial Dependence Plots (sezione~3.8) hanno
mostrato che metodi radicalmente diversi --- feature importance MDI, coefficienti Lasso su
regole e PDP/ICE sugli ensemble --- convergono sugli stessi tre segnali clinici dominanti:
\texttt{{time}}, \texttt{{serum\_creatinine}} ed \texttt{{ejection\_fraction}}. Questa
convergenza, unita alla coerenza con la letteratura sull'insufficienza cardiaca, è il
risultato più solido del progetto.
"""


# ── Main ──────────────────────────────────────────────────────────────────────

def generate_report_latex() -> None:
    # Create directories
    REPORT_DIR.mkdir(exist_ok=True)
    FIGURES_DIR.mkdir(exist_ok=True)

    # Copy figures
    for img in _IMAGES:
        src = PLOTS_DIR / img
        if src.exists():
            shutil.copy2(src, FIGURES_DIR / img)

    # Load data
    best         = _load("best_params.json") or {}
    metrics_df   = _load("metrics_summary.csv")
    grl_rows     = _load("grl_rules.json") or []
    brl_rows     = _load("brl_rules.json") or []
    rip_rows     = _load("ripper_rules.json") or []
    rf_rows      = _load("rulefit_top_rules.json") or []
    skope_rows   = _load("skope_rules.json") or []
    figs_text    = _load("figs_structure.txt") or ""
    abl_df       = _load("ablation_no_time.csv")
    grl_nt_rows  = _load("grl_no_time_rules.json") or []
    brl_nt_rows  = _load("brl_no_time_rules.json") or []

    # Dataset stats
    X_rows, y_pos = 0, 0
    cx = DATA_DIR / "X.csv"
    cy = DATA_DIR / "y.csv"
    if cx.exists() and cy.exists():
        X_rows = len(pd.read_csv(cx))
        y_pos  = int(pd.read_csv(cy).iloc[:, 0].sum())

    # BRL cardinality for inline text
    brl_card = str(best.get("BayesianRuleList", {}).get("maxcardinality", "?"))

    # Assemble LaTeX source
    ch3 = _ch3(metrics_df, grl_rows, brl_rows, rip_rows,
               rf_rows, skope_rows, figs_text, abl_df, grl_nt_rows, brl_nt_rows)
    # Patch {brl_card} in ch3 that was left as placeholder
    ch3 = ch3.replace("{brl_card}", brl_card)

    doc = "\n".join([
        _preamble(),
        r"\begin{document}",
        _titlepage(),
        r"\tableofcontents",
        r"\clearpage",
        _ch1(X_rows, y_pos),
        _ch2(best),
        ch3,
        _ch4(metrics_df, best),
        r"\end{document}",
    ])

    TEX_PATH.write_text(doc, encoding="utf-8")
    print(f"LaTeX source written to {TEX_PATH}")

    # Compile with pdflatex (2 passes for TOC)
    try:
        result = subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-output-directory", str(REPORT_DIR),
             str(TEX_PATH)],
            capture_output=True, text=True, encoding="latin-1", errors="replace",
            cwd=str(REPORT_DIR), timeout=120,
        )
        if result.returncode != 0:
            print("pdflatex pass 1 had warnings (continuing).")

        # Second pass for TOC page numbers
        subprocess.run(
            ["pdflatex", "-interaction=nonstopmode", "-output-directory", str(REPORT_DIR),
             str(TEX_PATH)],
            capture_output=True, text=True, encoding="latin-1", errors="replace",
            cwd=str(REPORT_DIR), timeout=120,
        )
        pdf_out = REPORT_DIR / "Manneschi_XAI.pdf"
        if pdf_out.exists():
            print(f"PDF compiled: {pdf_out}")
        else:
            print("pdflatex ran but PDF not found; check Report/Manneschi_XAI.log")
    except FileNotFoundError:
        print("pdflatex not found — LaTeX source saved, compile manually.")
    except Exception as e:
        print(f"Compilation error: {e}")


if __name__ == "__main__":
    generate_report_latex()
