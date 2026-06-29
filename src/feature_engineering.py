import pandas as pd

import numpy as np

def engineer_features(df):

    df["ANNUITY_INCOME_RATIO"] = (
    df["AMT_ANNUITY"] /
    df["AMT_INCOME_TOTAL"]
    )

    df["CREDIT_ANNUITY_RATIO"] = (
    df["AMT_CREDIT"] /
    df["AMT_ANNUITY"]
    )

    df["INCOME_PER_PERSON"] = (
    df["AMT_INCOME_TOTAL"] /
    df["CNT_FAM_MEMBERS"]
    ) 
    df["AGE"] = -df["DAYS_BIRTH"]/365
    df["YEARS_EMPLOYED"] = -df["DAYS_EMPLOYED"]/365

    df["EMPLOYED_PERCENT"] = (
    df["YEARS_EMPLOYED"] /
    df["AGE"]
    )
    df["EXT_MEAN"] = df[
    ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    ].mean(axis=1)
    df["EXT_MAX"] = df[
    ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    ].max(axis=1)
    df["EXT_MIN"] = df[
    ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    ].min(axis=1)
    return df