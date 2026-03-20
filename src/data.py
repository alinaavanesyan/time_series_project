import numpy as np
import pandas as pd
from datasetsforecast.m4 import M4


def load_m4_monthly(directory="data"):
    data = M4.load(directory=directory, group="Monthly")
    train_full = data[0]
    meta = data[2]
    return train_full, meta


def prepare_m4_data(train_full, n_series=150, seed=42, horizon=18):
    all_ids = train_full["unique_id"].unique()
    rng = np.random.RandomState(seed)
    sampled_ids = rng.choice(all_ids, size=min(n_series, len(all_ids)), replace=False)

    df = train_full[train_full["unique_id"].isin(sampled_ids)].copy()
    df = df.sort_values(["unique_id", "ds"]).reset_index(drop=True)

    train_parts, test_parts = [], []

    for uid, grp in df.groupby("unique_id"):
        grp = grp.sort_values("ds").reset_index(drop=True)
        if len(grp) <= horizon:
            continue

        train_part = grp.iloc[:-horizon].copy()
        test_part = grp.iloc[-horizon:].copy()
        test_part["h"] = range(1, len(test_part) + 1)

        train_parts.append(train_part)
        test_parts.append(test_part)

    train_df = pd.concat(train_parts, ignore_index=True)
    test_df = pd.concat(test_parts, ignore_index=True)

    train_df["ds"] = train_df["ds"].astype(int)
    test_df["ds"] = test_df["ds"].astype(int)

    return train_df, test_df, sampled_ids
