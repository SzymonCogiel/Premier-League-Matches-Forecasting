# Data processing model nodes

import numpy as np
import pandas as pd
from datetime import datetime
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler


def rm_nan(df: pd.DataFrame) -> pd.DataFrame:
    df = df.iloc[:-1, :]
    df = df.dropna(axis=1)
    return df

    
# Function that converts dates to the same format
def guess_date(string):
    """Converts string to date.

    Args:
        string: single date.
    Returns:
        Date in the correct format.
    """
    for fmt in ["%Y/%m/%d", "%d-%m-%Y", "%Y%m%d", "%d/%m/%Y", "%Y-%m-%d"]:
        try:
            return datetime.strptime(string, fmt).date()
        except ValueError:
            continue
    raise ValueError(string)


# date cleaning function
def date_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    """Converts date (string) to datetime.

    Args:
        df: dataframe containing dates.
    Returns:
        Dataframe with correct date.
    """
    df = df[df['Date'].notna()]
    df_date_data = pd.Series(df['Date']).reset_index()
    df_date_data = df_date_data.drop(columns="index")

    # adding up to year 20 to keep the date complete
    for key in range(len(df_date_data)):
        if "/" == str(df_date_data.loc[key, "Date"]).strip()[-3]:
            df_date_data.loc[key] = df_date_data.loc[key, "Date"][:-2] + "20" + df_date_data.loc[key, "Date"][-2:]

    for key in range(len(df_date_data)):
        df_date_data.loc[key, "Date"] = guess_date(str(df_date_data.loc[key, "Date"]))

    df = df.drop("Date", axis=1)
    df = pd.concat([df, df_date_data], axis=1)
    df = df.sort_values(by="Date").reset_index(drop=True)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def recive_points(df_res: pd.Series) -> pd.Series:
    scale_mapper = {'H': 3,
                    'D': 1,
                    'A': 0
                    }
    df_res = df_res.replace(scale_mapper)
    return df_res
    
    
def rm_dup_ref(df: pd.Series) -> pd.Series:
    df = df.replace({'M Atkinson': 'Mn Atkinson'}, regex=True)
    df = df.replace({'L Mason': 'l Mason'}, regex=True)
    df = df.replace({'St Bennett': 'S Bennett'}, regex=True)
    return df


def ref_encoder(df: pd.DataFrame) -> pd.DataFrame:
    ohe = OneHotEncoder()
    oe_results = ohe.fit_transform(df[["Referee"]])
    ref_df = pd.DataFrame(oe_results.toarray(), columns=ohe.categories_)
    return ref_df
    
    
def team_encoder(df_teams: pd.DataFrame) -> pd.DataFrame:
    encoder = LabelEncoder()
    df_teams = df_teams.apply(encoder.fit_transform)
    return df_teams
    
    
def join_team(df : pd.DataFrame) -> pd.DataFrame:
    df_teams = team_encoder(df[['HomeTeam', 'AwayTeam']])
    df = df.drop(columns=["HomeTeam", "AwayTeam"])
    df = df.join(df_teams)
    return df
    
    
def date_scalling(df: pd.DataFrame) -> pd.DataFrame:
    df['date_delta'] = (df['Date'] - df['Date'].min()) / np.timedelta64(1, 'D')
    return df


def scale_numer(df: pd.DataFrame) -> pd.DataFrame:
    scaler = StandardScaler()
    scaled = scaler.fit_transform(df)
#     normalizer = MinMaxScaler()
#     model_nor = normalizer.fit(df)
#     scaled = model_nor.transform(df)
    return scaled


def preporcessing(df: pd.DataFrame) -> pd.DataFrame:
    """Converts categorical data to numerical.

    Args:
        df: Data containing features and target.
    Returns:
        preporcesed data.
    """
    df = rm_nan(df)
    df = date_preprocessing(df)
    df["HTR"] = recive_points(df["HTR"])
    df["FTR"] = recive_points(df["FTR"])
    df = df.drop(['Div'], axis=1)
    df["Referee"] = rm_dup_ref(df["Referee"])
    df = df.join(ref_encoder(df))
    df = df.drop(columns="Referee")
    df = join_team(df)
    df = date_scalling(df)
    return df

def numerical_prep(df: pd.DataFrame) -> pd.DataFrame:
    """Scales numeric data.

    Args:
        df: Data containing features and target.
    Returns:
        preporcesed data.
    """
    df_numer = df.iloc[:, 18:39]
    df.iloc[:, 18:39] = scale_numer(df_numer)
    return df



