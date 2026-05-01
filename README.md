# XAI Project — Heart Failure Clinical Records

Laboratorio di 3 CFU su Explainable Artificial Intelligence.

Confronto tra modelli **interpretabili basati su regole** (RuleFit, GreedyRuleList, BayesianRuleList, FIGS, SkopeRules) e modelli **non interpretabili basati su ensemble** (Random Forest, Gradient Boosting, XGBoost) sul dataset [Heart Failure Clinical Records](https://archive.ics.uci.edu/dataset/519/heart+failure+clinical+records) (UCI, 299 pazienti).

## Struttura

```
src/
├── config.py        # iperparametri e percorsi
├── data_loader.py   # download, cache, split e preprocessing
├── models.py        # factory dei modelli rule-based ed ensemble
├── evaluate.py      # metriche, plot e cross-validation
└── main.py          # pipeline end-to-end
```

## Utilizzo

```bash
pip install -r requirements.txt
python3 src/main.py
```

I grafici e la tabella delle metriche vengono salvati in `results/`.
