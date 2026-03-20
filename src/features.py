import numpy as np
import pandas as pd


def create_lag_features(series, lags):
    features = {}
    n = len(series)
    for lag in lags:
        idx = n - lag
        features[f"lag_{lag}"] = series[idx] if idx >= 0 else np.nan
    return features


def create_rolling_features(series):
    features = {}
    for window in [3, 6, 12]:
        if len(series) >= window:
            features[f"rolling_mean_{window}"] = np.mean(series[-window:])
            features[f"rolling_std_{window}"] = np.std(series[-window:])
        else:
            features[f"rolling_mean_{window}"] = np.nan
            features[f"rolling_std_{window}"] = np.nan
    return features


def create_calendar_features(ds_value, season_length=12):
    return {"month": int(ds_value % season_length)}


def temporal_train_val_split(X, Y, uid_col="unique_id", val_frac=0.15):
    train_idx, val_idx = [], []

    for uid, grp in X.groupby(uid_col):
        idx = grp.index.tolist()
        split_point = int(len(idx) * (1 - val_frac))
        train_idx.extend(idx[:split_point])
        val_idx.extend(idx[split_point:])

    feat_cols = [c for c in X.columns if c != uid_col]

    X_train = X.loc[train_idx, feat_cols]
    X_val = X.loc[val_idx, feat_cols]

    if isinstance(Y, pd.DataFrame):
        Y_train, Y_val = Y.loc[train_idx], Y.loc[val_idx]
    else:
        Y_train, Y_val = Y.loc[train_idx], Y.loc[val_idx]

    return X_train, X_val, Y_train, Y_val
