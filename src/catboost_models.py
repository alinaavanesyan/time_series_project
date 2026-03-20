import numpy as np
import pandas as pd
from tqdm import tqdm
from catboost import CatBoostRegressor

from src.features import (
    create_lag_features, create_rolling_features,
    create_calendar_features, temporal_train_val_split,
)

def build_mimo_dataset(train_df, lags, horizon):
    X_rows, Y_rows = [], []
    max_lag = max(lags)
    for uid, grp in train_df.groupby("unique_id"):
        grp = grp.sort_values("ds").reset_index(drop=True)
        values, ds_values = grp["y"].values, grp["ds"].values
        for end_idx in range(max_lag, len(values) - horizon + 1):
            history = values[:end_idx]
            future = values[end_idx:end_idx + horizon]
            feat = create_lag_features(history, lags)
            feat.update(create_rolling_features(history))
            feat.update(create_calendar_features(ds_values[end_idx]))
            feat["unique_id"] = uid
            X_rows.append(feat)
            Y_rows.append({f"y_h{h+1}": future[h] for h in range(horizon)})
    return pd.DataFrame(X_rows), pd.DataFrame(Y_rows)


def build_mimo_test_features(train_df, lags):
    X_rows = []
    for uid, grp in train_df.groupby("unique_id"):
        grp = grp.sort_values("ds").reset_index(drop=True)
        values, ds_values = grp["y"].values, grp["ds"].values
        feat = create_lag_features(values, lags)
        feat.update(create_rolling_features(values))
        feat.update(create_calendar_features(ds_values[-1] + 1))
        feat["unique_id"] = uid
        X_rows.append(feat)
    return pd.DataFrame(X_rows)


def build_recursive_dataset(train_df, lags):
    X_rows, y_list = [], []
    max_lag = max(lags)
    for uid, grp in train_df.groupby("unique_id"):
        grp = grp.sort_values("ds").reset_index(drop=True)
        values, ds_values = grp["y"].values, grp["ds"].values
        for t in range(max_lag, len(values)):
            history = values[:t]
            feat = create_lag_features(history, lags)
            feat.update(create_rolling_features(history))
            feat.update(create_calendar_features(ds_values[t]))
            feat["unique_id"] = uid
            X_rows.append(feat)
            y_list.append(values[t])
    return pd.DataFrame(X_rows), pd.Series(y_list, name="y")


def build_direct_dataset(train_df, step, lags):
    X_rows, y_list = [], []
    max_lag = max(lags)
    for uid, grp in train_df.groupby("unique_id"):
        grp = grp.sort_values("ds").reset_index(drop=True)
        values, ds_values = grp["y"].values, grp["ds"].values
        for end_idx in range(max_lag, len(values) - step + 1):
            history = values[:end_idx]
            target = values[end_idx + step - 1]
            feat = create_lag_features(history, lags)
            feat.update(create_rolling_features(history))
            feat.update(create_calendar_features(ds_values[end_idx]))
            feat["unique_id"] = uid
            X_rows.append(feat)
            y_list.append(target)
    return pd.DataFrame(X_rows), pd.Series(y_list, name="y")


def train_catboost_mimo(train_df, lags, horizon, params, val_frac=0.15):
    X, Y = build_mimo_dataset(train_df, lags, horizon)
    feature_cols = [c for c in X.columns if c != "unique_id"]
    X_train, X_val, Y_train, Y_val = temporal_train_val_split(X, Y, val_frac=val_frac)
    model = CatBoostRegressor(**params)
    model.fit(X_train, Y_train, eval_set=(X_val, Y_val),
              cat_features=["month"] if "month" in X_train.columns else [])
    return model, feature_cols


def predict_catboost_mimo(model, train_df, lags, horizon, feature_cols):
    X_test = build_mimo_test_features(train_df, lags)
    uids = X_test["unique_id"].values
    preds = model.predict(X_test[feature_cols])
    if preds.ndim == 1:
        preds = preds.reshape(-1, horizon)
    rows = []
    for i, uid in enumerate(uids):
        for h in range(horizon):
            rows.append({"unique_id": uid, "h": h + 1, "prediction": preds[i, h]})
    return pd.DataFrame(rows)


def train_catboost_recursive(train_df, lags, params, val_frac=0.15):
    X, y = build_recursive_dataset(train_df, lags)
    X_train, X_val, y_train, y_val = temporal_train_val_split(X, y, val_frac=val_frac)
    model = CatBoostRegressor(**params)
    model.fit(X_train, y_train, eval_set=(X_val, y_val),
              cat_features=["month"] if "month" in X_train.columns else [])
    return model


def predict_catboost_recursive(model, train_df, lags, horizon):
    rows = []
    feature_cols = None
    for uid, grp in train_df.groupby("unique_id"):
        grp = grp.sort_values("ds").reset_index(drop=True)
        values = grp["y"].values.tolist()
        ds_last = int(grp["ds"].values[-1])
        for h in range(horizon):
            arr = np.array(values, dtype=float)
            feat = create_lag_features(arr, lags)
            feat.update(create_rolling_features(arr))
            feat.update(create_calendar_features(ds_last + h + 1))
            feat["unique_id"] = uid
            feat_df = pd.DataFrame([feat])
            if feature_cols is None:
                feature_cols = [c for c in feat_df.columns if c != "unique_id"]
            pred = model.predict(feat_df[feature_cols])[0]
            values.append(pred)
            rows.append({"unique_id": uid, "h": h + 1, "prediction": pred})
    return pd.DataFrame(rows)


def train_catboost_direct(train_df, lags, horizon, params, val_frac=0.15):
    models = []
    for step in tqdm(range(1, horizon + 1), desc="Direct models"):
        X, y = build_direct_dataset(train_df, step, lags)
        X_train, X_val, y_train, y_val = temporal_train_val_split(X, y, val_frac=val_frac)
        model = CatBoostRegressor(**params)
        model.fit(X_train, y_train, eval_set=(X_val, y_val),
                  cat_features=["month"] if "month" in X_train.columns else [])
        models.append(model)
    return models


def predict_catboost_direct(models, train_df, lags):
    X_test = build_mimo_test_features(train_df, lags)
    feat_cols = [c for c in X_test.columns if c != "unique_id"]
    uids = X_test["unique_id"].values
    rows = []
    for h, model in enumerate(models):
        preds = model.predict(X_test[feat_cols])
        for i, uid in enumerate(uids):
            rows.append({"unique_id": uid, "h": h + 1, "prediction": preds[i]})
    return pd.DataFrame(rows)
