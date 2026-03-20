DATASET = "M4"
FREQUENCY = "Monthly"
N_SERIES = 150
HORIZON = 18
SEASON_LENGTH = 12
RANDOM_SEED = 42

LAGS = list(range(1, 25))
CALENDAR_FEATURES = ["month"]
VAL_FRAC = 0.15

CATBOOST_PARAMS_MIMO = {
    "iterations": 1000,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "loss_function": "MultiRMSE",
    "random_seed": RANDOM_SEED,
    "verbose": 0,
    "early_stopping_rounds": 50,
}

CATBOOST_PARAMS_1STEP = {
    "iterations": 1000,
    "learning_rate": 0.05,
    "depth": 6,
    "l2_leaf_reg": 3.0,
    "loss_function": "RMSE",
    "random_seed": RANDOM_SEED,
    "verbose": 0,
    "early_stopping_rounds": 50,
    "task_type": "GPU",
    "devices": "0",
}

PATCHTST_PARAMS = {
    "input_size": 48,
    "h": HORIZON,
    "max_steps": 500,
    "learning_rate": 1e-3,
    "batch_size": 32,
    "patch_len": 12,
    "stride": 12,
    "hidden_size": 64,
    "n_heads": 4,
    "encoder_layers": 2,
    "random_seed": RANDOM_SEED,
    "early_stop_patience_steps": 10,
    "val_check_steps": 50,
    "scaler_type": "standard",
}
