import pandas as pd
from statsforecast import StatsForecast
from statsforecast.models import Naive, SeasonalNaive, AutoTheta, AutoETS


def fit_baselines(train_df, horizon, season_length=12):
    models = [
        Naive(),
        SeasonalNaive(season_length=season_length),
        AutoTheta(season_length=season_length),
        AutoETS(season_length=season_length),
    ]
    sf = StatsForecast(models=models, freq=1, n_jobs=-1)
    forecasts = sf.forecast(df=train_df, h=horizon)
    return forecasts.reset_index()


def baseline_to_long(forecasts, model_name):
    df = forecasts[["unique_id", model_name]].copy()
    df["h"] = df.groupby("unique_id").cumcount() + 1
    df = df.rename(columns={model_name: "prediction"})
    return df[["unique_id", "h", "prediction"]]
