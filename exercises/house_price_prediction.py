import os.path

from IMLearn.utils import split_train_test
from IMLearn.learners.regressors import LinearRegression

from typing import NoReturn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio

pio.templates.default = "simple_white"

ZIPCODE_SCORE = "zipcode_score"
ZIPCODE = 'zipcode'
PRICE = "price"
LAST_RENOV_AGE = "last_renov_age"
RENOVATED = "renovated"
HOUSE_AGE = "house_age"

MIN_SQUARE_FOOTAGE = 120
FILTERED_COLS = ["id", "lat", "long", "date", "yr_built", "yr_renovated",
                 "zipcode"]


def get_valid_df(filename):
    """
    Returns a valid DataFrame from the csv while removing
    rows with invalid features
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    DataFrame that includes only valid rows
    """
    df = pd.read_csv(filename)
    df.date = pd.to_datetime(df.date, errors='coerce')
    return df[(df.id > 0) & (df.price > 0) & (df.bedrooms > 0) & (
            df.yr_built > 0) & ((df.yr_renovated == 0) | (
            df.yr_renovated >= df.yr_built)) &
              (df.sqft_living >= MIN_SQUARE_FOOTAGE)].dropna()


def load_data(filename: str):
    """
    Load house prices dataset and preprocess data.
    Parameters
    ----------
    filename: str
        Path to house prices dataset

    Returns
    -------
    Design matrix and response vector (prices) - either as a single
    DataFrame or a Tuple[DataFrame, Series]
    """
    df = get_valid_df(filename)
    df = add_age_features(df)
    df = add_zipcode_score(df)
    df = df.drop(columns=FILTERED_COLS)

    return df.drop(columns=[PRICE]), df.price


def add_zipcode_score(df: pd.DataFrame):
    """
        Adds zipcode score (mean price) feature to DataFrame
        Parameters
        ----------
        df: pd.DataFrame
            DataFrame to work on

        Returns
        -------
        df: pd.DataFrame
            DataFrame with added zipcode score (mean price in zipcode)
        """
    zipcode_mean_price = df.groupby(ZIPCODE).mean()[[PRICE]].rename(
        columns={PRICE: ZIPCODE_SCORE})
    zipcode_mean_price.reset_index(inplace=True)
    zipcode_mean_price = zipcode_mean_price[[ZIPCODE, ZIPCODE_SCORE]]
    df = pd.merge(df, zipcode_mean_price, on=ZIPCODE, how='left')
    return df


def add_age_features(df: pd.DataFrame):
    """
    Adds house age and renovation features
    Parameters
    ----------
    df: pd.DataFrame
        DataFrame to work on

    Returns
    -------
    df: pd.DataFrame
        DataFrame with added features of house age and renovation age
    """
    df[HOUSE_AGE] = pd.DatetimeIndex(df.date).year - df.yr_built
    df[RENOVATED] = np.where(df.yr_renovated > 0, 1, 0)
    df[LAST_RENOV_AGE] = np.where(df.yr_renovated > 0, pd.DatetimeIndex(
        df.date).year - df.yr_renovated, df[HOUSE_AGE])
    return df


def feature_evaluation(X: pd.DataFrame, y: pd.Series,
                       output_path: str = ".") -> NoReturn:
    """
    Create scatter plot between each feature and the response.
        - Plot title specifies feature name
        - Plot title specifies Pearson Correlation between feature and response
        - Plot saved under given folder with file name including feature name
    Parameters
    ----------
    X : DataFrame of shape (n_samples, n_features)
        Design matrix of regression problem

    y : array-like of shape (n_samples, )
        Response vector to evaluate against

    output_path: str (default ".")
        Path to folder in which plots are saved
    """
    for feature in X:
        feature_data = X[feature]
        cov = np.cov(feature_data, y)
        pearson_corr = cov[0][1] / (
                np.sqrt(cov[0][0]) * np.sqrt(cov[1][1]))
        go.Figure([go.Scatter(x=feature_data, y=y,
                              mode='markers')],
                  layout=go.Layout(
                      title=f"$\\text{{price as a function of {feature} - Corr =}} {pearson_corr}$",
                      xaxis_title=f"$\\text{{{feature}}}$",
                      yaxis_title=r"$\text{price}$",
                      height=500)).write_image(
            os.path.join(output_path, f"{feature}.png"))


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of housing prices dataset
    X, y = load_data("../datasets/house_prices.csv")

    # Question 2 - Feature evaluation with respect to response
    feature_evaluation(X, y)

    # Question 3 - Split samples into training- and testing sets.
    train_X, train_y, test_X, test_y = split_train_test(X, y)

    # Question 4 - Fit model over increasing percentages of the overall training data
    # For every percentage p in 10%, 11%, ..., 100%, repeat the following 10 times:
    #   1) Sample p% of the overall training data
    #   2) Fit linear model (including intercept) over sampled set
    #   3) Test fitted model over test set
    #   4) Store average and variance of loss over test set
    # Then plot average loss as function of training size with error ribbon of size (mean-2*std, mean+2*std)
    raise NotImplementedError()
