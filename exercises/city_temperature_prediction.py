import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

TEMP = "Temp"

MAX_TEMP = 56.7
MIN_TEMP = -89.2
MAX_DAY = 31
MAX_MONTH = 12

pio.templates.default = "simple_white"


def get_valid_df(filename):
    """
    Returns a valid DataFrame from the csv while removing
    rows with invalid features
    Parameters
    ----------
    filename: str
        Path to temperature dataset

    Returns
    -------
    DataFrame that includes only valid rows
    """
    df = pd.read_csv(filename)
    df.date = pd.to_datetime(df.Date, errors='coerce')
    return df[
        (1 <= df.Month) & (df.Month <= MAX_MONTH) & (1 <= df.Day) & (
                df.Day <= MAX_DAY) & (df.Temp >= MIN_TEMP) & (
                df.Temp <= MAX_TEMP)].dropna()


def load_data(filename: str) -> pd.DataFrame:
    """
    Load city daily temperature dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to temperature dataset

    Returns
    -------
    Design matrix and response vector (Temp)
    """
    df = get_valid_df(filename)
    df["DayOfYear"] = pd.DatetimeIndex(df.Date).dayofyear
    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    data = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    raise NotImplementedError()

    # Question 3 - Exploring differences between countries
    raise NotImplementedError()

    # Question 4 - Fitting model for different values of `k`
    raise NotImplementedError()

    # Question 5 - Evaluating fitted model on different countries
    raise NotImplementedError()
