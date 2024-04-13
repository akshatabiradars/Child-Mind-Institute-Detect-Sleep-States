import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import pyarrow
import fastparquet
import os
import pickle

from pandas.api.types import is_string_dtype
from pandas.api.types import is_numeric_dtype

import streamlit as st


def line_break(n=3):
    """ print n line break in the app"""
    for i in range(n):
        st.write('\n')


def check_df(df):
    """
    check that df has required columns with correct types:
    series_id - str
    step - int or float, numeric
    timestamp - str
    anglez - float, numeric
    enmo - float, numeric
    pd.DataFrame -> bool/list[str]
    """
    list_errors = []
    columns = df.columns
    needed_columns = ["series_id", "step", "timestamp", "anglez", "enmo"]

    # check that df is not empty
    if df.shape[0] == 0:
        return ["Dataframe is empty"]

    # check that df has all the required columns
    for i, col in enumerate(needed_columns):
        if col not in columns:
            list_errors.append(f"column {col} not found")
        else:
            # check that columns have correct type
            if col == needed_columns[0]:  # series_id
                #if not isinstance(df[col][0], str):
                if not is_string_dtype(df[col]):
                    list_errors.append(f"column {col} has type {type(df[col][0])}, expected str")
            if (col == needed_columns[1]) or (col == needed_columns[3]) or (col == needed_columns[4]):  # step, anglez or enmo
                #if not (isinstance(df[col][0], int) or isinstance(df[col][0], float)):
                if not is_numeric_dtype(df[col]):
                    list_errors.append(f"column {col} has type {df[col][0].dtype}, expected int or float")
    
    
    if list_errors == []:
        return True
    else:
        return list_errors


def get_series(df, series_id):
    """
    return portion of data corresponding to series_id
    DataFrame * str -> DataFrame
    """
    return df[df["series_id"] == series_id]


def get_moment(hour):
    """
    return the moment of day
    night if 0 < hour < 6
    morning if 6 < hour < 12
    afternoon if 12 < hour < 18
    evening if 18 < hour < 24
    int -> str
    """
    if (0 < hour) and (hour < 6):
        return "night"
    elif (6 < hour) and (hour < 12):
        return "morning"
    elif (12 < hour) and (hour < 18):
        return "afternoon"
    else:
        return "evening"


def preprocess_col(df, col_name, rolling_val=1000):
    """
    return dataframe with same cols as df + rolling_mean, high_to_mean and centered for col_name
    col rolling_mean : mean value of col_name for window = rolling_val, by series_id
    high_to_mean column : values of rolling_mean or mean of rolling_mean by series_id if value > mean
    center : values of high_to_mean centered abs(on mean + std) / 2 by seies_id
    df: DataFrame with columns series_id, col_name
    DataFrame * str * int -> DataFrame
    """

    df_result = pd.DataFrame(columns=df.columns)
    list_ids = df["series_id"].unique()

    for id in list_ids:

        # get data frame for each series_id
        df_tmp = get_series(df, id)

        # compute rolling_mean column
        df_tmp[f"{col_name}_rolling_mean"] =\
            df_tmp[col_name].rolling(window=rolling_val, center=True).mean().fillna(method="bfill").fillna(method="ffill")

        # compute high_to_mean column
        mean_value = df_tmp[f"{col_name}_rolling_mean"].mean()
        df_tmp[f"{col_name}_high_to_mean"] = df_tmp[f"{col_name}_rolling_mean"].apply(lambda x : mean_value if x > mean_value else x)

        # compute centered_column
        center_value = abs(df_tmp[f"{col_name}_high_to_mean"].mean() + df_tmp[f"{col_name}_high_to_mean"].std()) / 2
        df_tmp[f"{col_name}_centered"] = df_tmp[f"{col_name}_high_to_mean"].apply(lambda x : x - center_value)
        df_result = pd.concat([df_result, df_tmp])
    df_result = df_result.drop(columns=[f"{col_name}_rolling_mean", f"{col_name}_high_to_mean"])
    return df_result


def get_features(df):
    """
    Return DataFrame with more features from enmo and anglez:
    enmo_centered, datetime, hour, month, moment
    anglez_abs, anglez_diff, enmo_diff, anglez_rolling_mean, enmo_rolling_mean,
    enmo_x_anglez, enmo_x_anlez_abs, weekday, is_weekend
    DataFrame -> DataFrame
    """
    # mean number of rows by series
    size_series = df.shape[0] // df["series_id"].nunique()

    periods = 12
    # 12 * 5 secondes -> 1 minute

    # column enmo centered
    df_result = df.copy()

    # timestamp to datetime
    df_result["datetime"] = pd.to_datetime(df["timestamp"])

    # month
    df_result["month"] = df_result["datetime"].apply(lambda x : x.month)
    
    # hour
    df_result["hour"] = df_result["datetime"].apply(lambda x : x.hour)

    # moment of day
    df_result["moment"] = df_result["hour"].apply(lambda x : get_moment(x))
    
    # abs anglez
    df_result["anglez_abs"] = df["anglez"].apply(lambda x : abs(x))
    
    # diff between anglez_abs and n (periods) previous anglez
    # we need bfill because first values can't be computed
    df_result["anglez_diff"] = df_result.groupby("series_id")["anglez_abs"].diff(periods=periods).fillna(method="bfill")

    # diff between enmo and n (periods) previous enmo
    # we need bfill because first values can't be computed
    df_result["enmo_diff"] = df_result.groupby("series_id")["enmo"].diff(periods=periods).fillna(method="bfill")

    # rolling mean anglez abs
    # we need bfill and ffill because we have missing values at the begining and the end (center=True)
    df_result["anglez_rolling_mean"] = df_result["anglez_abs"].rolling(periods, center=True).mean().fillna(method="bfill").fillna(method="ffill")

    # rolling mean enmo
    # we need bfill and ffill because we have missing values at the begining and the end (center=True)
    df_result["enmo_rolling_mean"] = df["enmo"].rolling(periods, center=True).mean().fillna(method="bfill").fillna(method="ffill")
    
    
    # enmo * anglez
    df_result["enmo_x_anglez"] = df.apply(lambda x : x["enmo"] * x["anglez"], axis=1)
    
    # enmo * anglez_abs
    df_result["enmo_x_anglez_abs"] = df_result.apply(lambda x : x["enmo"] * x["anglez_abs"], axis=1)
    
    # is weekend
    df_result["weekday"] = df_result["datetime"].apply(lambda x : x.weekday())
    # Timestamp.weekday(): Monday == 0 … Sunday == 6.
    df_result["is_weekend"] = df_result["weekday"].apply(lambda x: 1 if x >= 5 else 0)

    return df_result

def feature_engineering(df):
    """
    Add features to dataframe df from columns enmo, anglez and timestamp
    """
    nb_series = df["series_id"].nunique()

    # rolling val: average number of rows / 10
    rolling_val =  int(df.shape[0] / nb_series / 10)

    # column enmo_centered
    df_result = preprocess_col(df, "enmo", rolling_val= rolling_val)

    # other features
    df_result = get_features(df_result)

    df_result = df_result.drop(columns=["datetime", "month", "weekday"])

    return df_result


def get_predictions_probas(df, pipeline):
    """
    Apply pipeline on df to get predictions and probas
    """
    cols_to_train = ['anglez', 'enmo', 'enmo_centered', 'hour', 'moment',\
                 'anglez_abs', 'anglez_diff', 'enmo_diff', 'anglez_rolling_mean', 'enmo_rolling_mean',\
                 'enmo_x_anglez', 'enmo_x_anglez_abs', 'is_weekend']
    # get predictions
    y_pred = pipeline.predict(df[cols_to_train])
    # get probas
    probas = pipeline.predict_proba(df)
    y_probas = [max(proba) for proba in probas]

    return y_pred, y_probas

def smooth_results(df, y_pred, smooth_val):
    """
    return a new array of predictions calculated with rolling mean for each series
    df : DataFrame with columns series_id
    y_pred : predictions for df (0 or )
    smooth_val : window of rolling
    DataFrame * array/Series * int -> DataFrame
    """
    list_ids = df["series_id"].unique()
    y_pred = pd.DataFrame(y_pred, columns=["pred"])
    y_result = pd.DataFrame(columns=y_pred.columns)

    for id in list_ids:
        # select series in df and y_pred
        df_tmp = get_series(df, id)
        start_id = df_tmp.index[0]
        end_id = df_tmp.index[-1]
        y_tmp = y_pred.iloc[start_id : end_id+1]
        
        y_tmp["pred"] =\
            y_tmp["pred"].rolling(window=smooth_val, center=True).mean().fillna(method="bfill").fillna(method="ffill")
        y_tmp["pred"] = y_tmp["pred"].apply(lambda x : 1 if x >= 0.5 else 0)
        
        y_result = pd.concat([y_result, y_tmp])


    return np.array(y_result["pred"]).astype("int")


def get_events(df, y_pred, y_probas):
    """
    Add column event, pred and score to df and return new df
    event = 0, 1 or np.nan
    
    df : DataFrame with columns series_id
    y_pred : DataFram with column pred of 0 and 1
    y_probas : dataFrame with probabilities from model
    
    DataFrame -> DataFrame
    """
    # add column pred and score to df
    y_pred = pd.DataFrame(y_pred, columns=["pred"])
    y_probas = pd.DataFrame(y_probas, columns=["score"])
    df = pd.concat([df, y_pred, y_probas], axis=1)
    
    df_result = pd.DataFrame(columns=df.columns)
    list_ids = df["series_id"].unique()

    for id in list_ids:
        
        # get data frame for each series_id
        df_tmp = get_series(df, id)

        # create column diff
        df_tmp["pred_diff"] = df_tmp["pred"].diff().fillna(method="bfill")

        # use diff to determine event
        # when diff < 0 ie diff == -1 value went from 1 to 0 -> onset
        # when diff > 0 ie diff == 1 value went from 0 to 1 -> wakeup
        # 0 -> no changes
        df_tmp["event"] = df_tmp["pred_diff"].apply(lambda x : "onset" if x < 0 else ("wakeup" if x > 0 else np.nan))
        
        df_result = pd.concat([df_result, df_tmp])
        
    return df_result



def get_submission(df):
    """
    Returns a dataFrame ready for submission
    df : DataFrame with columns series_id, step, event, score
    """

    # remove nan values (no event)
    df = df.dropna()

    # keep only necessary columns
    df = df[["series_id", "step", "event", "score"]]

    # reset index
    df = df.reset_index(drop = True)

    # add first column: row_id
    row_id = df.index.values
    df.insert(0, 'row_id', row_id)

    return df



def keep_periods(df, min_period):
    """
    Take a dataframe ready for submission and return same data frame minus periods of sleep < min_period
    Makes sure that sleep periods begin and end (start with onset, end with wakeup) for each series
    """
    
    df_result = pd.DataFrame(columns=df.columns)
    list_ids = df["series_id"].unique()

    for id in list_ids:

        # get data frame for each series_id
        df_tmp = get_series(df, id)

        # get steps for onset and wakeup as lists
        pred_onsets = df_tmp[df_tmp["event"] == "onset"]["step"].to_list()
        pred_wakeups = df_tmp[df_tmp["event"] == "wakeup"]["step"].to_list()

        # check that all sleep periods start with onset and end wit wakeup
        # compare steps
        if min(pred_wakeups) < min(pred_onsets):     # first step of pred_wakeups smaller than first step of pred_onsets
            pred_wakeups = pred_wakeups[1:]          # don't keep first element of pred_wakeups
            print("delete wakeup")
        if max(pred_onsets) > max(pred_wakeups):     # last onset bigger than last wakeup
            pred_onsets = pred_onsets[:-1]           # don't keep last element of pred_onsets
            print("delete onset")


        # keep only sleep periods > min_period
        pred_onsets_2 = []
        pred_wakeups_2 = []
        for onset, wakeup in zip(pred_onsets, pred_wakeups):
            # we compare onset and wakeup couples
            if wakeup - onset >= min_period:
                pred_onsets_2.append(onset)
                pred_wakeups_2.append(wakeup)

        # keep only activity periods > min_period
        steps_to_keep = [pred_onsets_2[0]]               # keep first onset
        # we compare wakeup and onset couples
        for i, wakeup in enumerate(pred_wakeups_2[:-1]): # last wakeup can't be compared to any onset
            if pred_onsets_2[i+1] - wakeup >= min_period:
                steps_to_keep.append(wakeup)             # add couples of wakeup/onset
                steps_to_keep.append(pred_onsets_2[i+1])
        steps_to_keep.append(pred_wakeups_2[-1])        # add last wakeup event
        

        # select events en df_tmp
        df_tmp = df_tmp[df_tmp["step"].isin(steps_to_keep)]
        
        df_result = pd.concat([df_result, df_tmp])

        
    # reset index
    df_result = df_result.reset_index(drop = True)

    # reset row_id
    row_id = df_result.index.values
    df_result["row_id"] = row_id

    return df_result


def build_submission(df, y_pred, y_probas):
    """
    Build a submission dataframe for kaggle competition
    """
    # smooth predictions
    y_smooth = smooth_results(df, y_pred, 1000)
    # get events corresponding to predictions
    df_events = get_events(df, y_smooth, y_probas)
    # create submission dataFrame
    df_submission = get_submission(df_events)
    # keep_periods > 1h
    df_submission = keep_periods(df_submission, 12*60*1)

    return df_submission

def get_random_id(data):
    """
    Randomly selects a series in data and return the corresponding dataframe
    DataFrame -> DataFrame
    """
    list_id = data["series_id"].unique()
    id = np.random.choice(list_id, size=1)[0]
    return id

def plot_enmo(data, id):
    """
    plot column enmo by column step
    data: pd.DataFrame
    """

    data_series = get_series(data, id)

    fig, ax = plt.subplots(figsize=(20, 4))

    ax.plot(data_series["step"], data_series["enmo"], linewidth=0.5)
    ax.set_xlabel("step")
    ax.set_ylabel("enmo")

    fig.suptitle("Evolution of enmo")

    return fig

def plot_prediction(df, df_submission, id):
    """
    Plot column enmo by column step and predictions from df_submission
    id: id of series to plot
    """
    df_viz = get_series(df, id)
    df_sub_viz = get_series(df_submission, id)
    onset_viz = df_sub_viz[df_sub_viz["event"] == "onset"]
    wakeup_viz = df_sub_viz[df_sub_viz["event"] == "wakeup"]

    fig, ax = plt.subplots(figsize=(20, 4))

    # plot enmo
    ax.plot(df_viz["step"], df_viz["enmo"], linewidth=0.5)


    # plot events onset
    ax.vlines(x=onset_viz["step"].values, ymin=0, ymax=4, colors = 'm', label = "Predicted onset", linestyles="dashed")
    # plot events wakeup
    ax.vlines(x=wakeup_viz["step"].values, ymin=0, ymax=4, colors = 'r', label = "Predicted wakeup", linestyles="dashed")

    ax.set_xlabel("step")
    ax.set_ylabel("enmo")

    fig.suptitle("Predictions")
    plt.legend()

    return fig


def read_file(filename, file):
    """
    Read a parquet or csv file and return the result
    str -> pd.DataFrame
    """
    file_name, file_extension = os.path.splitext(filename)
    if file_extension == ".csv":
        return pd.read_csv(file)
    if file_extension == ".parquet":
        return pd.read_parquet(file)
    else:
        return "Incorrect file extension"
