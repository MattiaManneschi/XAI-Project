from pathlib import Path

RANDOM_STATE = 42
TEST_SIZE = 0.2    # fraction held out as test set
VAL_SIZE  = 0.2    # fraction held out as validation set (for hyperparameter tuning)
CV_FOLDS = 10

_ROOT = Path(__file__).parent.parent
DATA_DIR = _ROOT / "data"
RESULTS_DIR = _ROOT / "results"
FILES_DIR   = RESULTS_DIR / "files"   # JSON / CSV artefacts
PLOTS_DIR   = RESULTS_DIR / "plots"

# Rule-based hyperparams
RULEFIT_MAX_RULES = 30
GRL_MAX_DEPTH = 6
BRL_MAX_ITER = 20_000
BRL_MAX_CARDINALITY = 3
FIGS_MAX_RULES = 10
SKOPE_N_ESTIMATORS = 100
SKOPE_PRECISION_MIN = 0.5
SKOPE_RECALL_MIN = 0.4
RIPPER_K = 2

# Ensemble hyperparams
RF_N_ESTIMATORS = 200
GB_N_ESTIMATORS = 200
GB_LEARNING_RATE = 0.05
GB_MAX_DEPTH = 4
XGB_N_ESTIMATORS = 200
XGB_LEARNING_RATE = 0.05
XGB_MAX_DEPTH = 4

BINARY_FEATURES = ["anaemia", "diabetes", "high_blood_pressure", "sex", "smoking"]
