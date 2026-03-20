import numpy as np
import pandas as pd


def smape(y_true, y_pred):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    denom = (np.abs(y_true) + np.abs(y_pred)) / 2
    mask = denom > 0
    if mask.sum() == 0:
        return 0.0
    return np.mean(np.abs(y_true[mask] - y_pred[mask]) / denom[mask]) * 100


def mase(y_true, y_pred, y_train, season_length=12):
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)
    y_train = np.array(y_train, dtype=float)
    if len(y_train) <= season_length:
        return np.nan
    naive_errors = np.abs(y_train[season_length:] - y_train[:-season_length])
    scale = np.mean(naive_errors)
    if scale == 0 or np.isnan(scale):
        return np.nan
    return np.mean(np.abs(y_true - y_pred)) / scale


def compute_per_step_metrics(test_df, preds_df, train_series, season_length=12):
    merged = test_df.merge(preds_df, on=["unique_id", "h"], how="inner")
    results = []
    for h in sorted(merged["h"].unique()):
        step_data = merged[merged["h"] == h]
        smape_vals, mase_vals = [], []
        for _, row in step_data.iterrows():
            uid = row["unique_id"]
            y_true = np.array([row["y"]])
            y_pred = np.array([row["prediction"]])
            smape_vals.append(smape(y_true, y_pred))
            if uid in train_series:
                m = mase(y_true, y_pred, train_series[uid], season_length)
                if not np.isnan(m):
                    mase_vals.append(m)
        results.append({
            "h": int(h),
            "sMAPE": float(np.mean(smape_vals)) if smape_vals else np.nan,
            "MASE": float(np.mean(mase_vals)) if mase_vals else np.nan,
        })
    return pd.DataFrame(results)


def compute_aggregated_metrics(test_df, preds_df, train_series, season_length=12):
    merged = test_df.merge(preds_df, on=["unique_id", "h"], how="inner")
    smape_vals, mase_vals = [], []
    for uid, grp in merged.groupby("unique_id"):
        y_true = grp["y"].values
        y_pred = grp["prediction"].values
        smape_vals.append(smape(y_true, y_pred))
        if uid in train_series:
            m = mase(y_true, y_pred, train_series[uid], season_length)
            if not np.isnan(m):
                mase_vals.append(m)
    return {
        "sMAPE": float(np.mean(smape_vals)) if smape_vals else np.nan,
        "MASE": float(np.mean(mase_vals)) if mase_vals else np.nan,
    }
