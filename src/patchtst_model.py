import pandas as pd
from neuralforecast import NeuralForecast
from neuralforecast.models import PatchTST


def train_and_predict_patchtst(train_df, params, horizon):
    model = PatchTST(**params)
    nf = NeuralForecast(models=[model], freq=1)
    nf.fit(df=train_df, val_size=horizon)
    raw = nf.predict()

    if isinstance(raw.index, pd.MultiIndex):
        raw = raw.reset_index()

    raw = raw.rename(columns={"PatchTST": "prediction"})
    raw["h"] = raw.groupby("unique_id").cumcount() + 1
    return raw[["unique_id", "h", "prediction"]].copy()
