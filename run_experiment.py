import warnings
warnings.filterwarnings("ignore")

import json
import pandas as pd

from config import *
from src.data import load_m4_monthly, prepare_m4_data
from src.baselines import fit_baselines, baseline_to_long
from src.metrics import compute_per_step_metrics, compute_aggregated_metrics
from src.catboost_models import (
    train_catboost_mimo, predict_catboost_mimo,
    train_catboost_recursive, predict_catboost_recursive,
    train_catboost_direct, predict_catboost_direct,
)
from src.patchtst_model import train_and_predict_patchtst


def main():
    print("Загрузка данных M4 Monthly...")
    train_full, meta = load_m4_monthly()
    train_df, test_df, sampled_ids = prepare_m4_data(
        train_full, n_series=N_SERIES, seed=RANDOM_SEED, horizon=HORIZON
    )

    train_series = {
        uid: grp["y"].values
        for uid, grp in train_df.groupby("unique_id")
    }

    all_per_step = {}
    all_agg = {}

    print("Обучение бейзлайнов...")
    baseline_forecasts = fit_baselines(train_df, HORIZON, SEASON_LENGTH)

    for name in ["Naive", "SeasonalNaive", "AutoTheta", "AutoETS"]:
        preds = baseline_to_long(baseline_forecasts, name)
        all_per_step[name] = compute_per_step_metrics(test_df, preds, train_series)
        all_agg[name] = compute_aggregated_metrics(test_df, preds, train_series)
        print(f"{name}: sMAPE={all_agg[name]['sMAPE']:.4f}, MASE={all_agg[name]['MASE']:.4f}")

    print("CatBoost MIMO...")
    cb_mimo, feat_cols = train_catboost_mimo(
        train_df, LAGS, HORIZON, CATBOOST_PARAMS_MIMO, VAL_FRAC
    )
    mimo_preds = predict_catboost_mimo(cb_mimo, train_df, LAGS, HORIZON, feat_cols)
    all_per_step["CatBoost_MIMO"] = compute_per_step_metrics(test_df, mimo_preds, train_series)
    all_agg["CatBoost_MIMO"] = compute_aggregated_metrics(test_df, mimo_preds, train_series)
    print(f"  sMAPE={all_agg['CatBoost_MIMO']['sMAPE']:.4f}, MASE={all_agg['CatBoost_MIMO']['MASE']:.4f}")

    print("CatBoost Recursive...")
    cb_rec = train_catboost_recursive(train_df, LAGS, CATBOOST_PARAMS_1STEP, VAL_FRAC)
    rec_preds = predict_catboost_recursive(cb_rec, train_df, LAGS, HORIZON)
    all_per_step["CatBoost_Recursive"] = compute_per_step_metrics(test_df, rec_preds, train_series)
    all_agg["CatBoost_Recursive"] = compute_aggregated_metrics(test_df, rec_preds, train_series)
    print(f"  sMAPE={all_agg['CatBoost_Recursive']['sMAPE']:.4f}, MASE={all_agg['CatBoost_Recursive']['MASE']:.4f}")

    print("CatBoost Direct...")
    direct_models = train_catboost_direct(train_df, LAGS, HORIZON, CATBOOST_PARAMS_1STEP, VAL_FRAC)
    direct_preds = predict_catboost_direct(direct_models, train_df, LAGS)
    all_per_step["CatBoost_Direct"] = compute_per_step_metrics(test_df, direct_preds, train_series)
    all_agg["CatBoost_Direct"] = compute_aggregated_metrics(test_df, direct_preds, train_series)
    print(f"  sMAPE={all_agg['CatBoost_Direct']['sMAPE']:.4f}, MASE={all_agg['CatBoost_Direct']['MASE']:.4f}")

    print("PatchTST MIMO...")
    ptst_preds = train_and_predict_patchtst(train_df, PATCHTST_PARAMS, HORIZON)
    all_per_step["PatchTST_MIMO"] = compute_per_step_metrics(test_df, ptst_preds, train_series)
    all_agg["PatchTST_MIMO"] = compute_aggregated_metrics(test_df, ptst_preds, train_series)
    print(f"  sMAPE={all_agg['PatchTST_MIMO']['sMAPE']:.4f}, MASE={all_agg['PatchTST_MIMO']['MASE']:.4f}")

    print("ИТОГОВАЯ ТАБЛИЦА:")
    summary = pd.DataFrame([
        {"Модель": name, **metrics}
        for name, metrics in all_agg.items()
    ]).sort_values("sMAPE").reset_index(drop=True)
    print(summary.to_string(index=False))

    summary.to_csv("results/summary.csv", index=False)

    per_step_all = {}
    for name, df in all_per_step.items():
        per_step_all[name] = df.to_dict(orient="records")
    with open("results/per_step_metrics.json", "w") as f:
        json.dump(per_step_all, f, indent=2, ensure_ascii=False)

    print("Результаты сохранены в results/")


if __name__ == "__main__":
    main()
