import IMLearn.learners.regressors.linear_regression
from IMLearn.learners.regressors import PolynomialFitting
from IMLearn.utils import split_train_test

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.io as pio

COUNTRY = "Country"
MONTH = "Month"
YEAR = "Year"

pio.templates.default = "simple_white"
pio.renderers.default = "browser"

DAY_OF_YEAR = "DayOfYear"
TEMP = "Temp"

MAX_TEMP = 56.7
MIN_TEMP = -60
MAX_DAY = 31
MAX_MONTH = 12


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
    df.Date = pd.to_datetime(df.Date, errors='coerce')
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
    df[DAY_OF_YEAR] = pd.DatetimeIndex(df.Date).dayofyear
    return df


if __name__ == '__main__':
    np.random.seed(0)
    # Question 1 - Load and preprocessing of city temperature dataset
    dataset = load_data("../datasets/City_Temperature.csv")

    # Question 2 - Exploring data for specific country
    israel_data = dataset[dataset.Country == "Israel"].copy()
    israel_data.reset_index(drop=True, inplace=True)
    israel_data.Year = israel_data.Year.astype(str)

    # Temp as a function of day of year
    fig = px.scatter(israel_data, x=DAY_OF_YEAR, y=TEMP, color=YEAR,
                     title=r"Average Temp (Israel) As A Function Of "
                           r"Day Of Year",
                     width=1000, height=500)
    fig.show()
    israel_data.Year = israel_data.Year.astype(int)

    # Std as a function of month
    israel_monthly = israel_data[[MONTH, TEMP]].groupby(MONTH)
    israel_monthly = israel_monthly.agg(np.std)

    fig = px.bar(israel_monthly, y=TEMP,
                 labels={TEMP: "STD of Average Temp"},
                 title=r"STD of Average Temp (Israel) on Different Months",
                 width=600)
    fig.show()

    # Question 3 - Exploring differences between countries
    grouped_data = dataset[[MONTH, COUNTRY, TEMP]].groupby([MONTH, COUNTRY])
    grouped_data = grouped_data.agg(mean=(TEMP, np.mean), std=(TEMP, np.std))
    grouped_data = grouped_data.reset_index(level=[COUNTRY])

    fig = px.line(grouped_data, color=COUNTRY, y="mean", error_y="std",
                  labels={"mean": "Average Monthly Temp"},
                  title="Monthly Average Temp As A Function of Month",
                  width=800)
    fig.show()

    # Question 4 - Fitting model for different values of `k`
    train_X, train_y, test_X, test_y = split_train_test(
        israel_data[DAY_OF_YEAR].to_frame(), israel_data.Temp)
    test_errors = []
    for k in range(1, 11):
        poly = PolynomialFitting(k)
        poly.fit(train_X.to_numpy(), train_y.to_numpy())
        loss = np.round(poly.loss(test_X.to_numpy(), test_y.to_numpy()), 2)
        test_errors.append(loss)
        print(f"k = {k} - Loss = {loss}")

    poly_loss = pd.DataFrame({"k": range(1, 11), "loss": test_errors})
    fig = px.bar(poly_loss, x="k", y="loss",
                 title=r"(4) Loss On Fitting On Different Polynomial Degree",
                 width=600)
    fig.show()

    # Question 5 - Evaluating fitted model on different countries
    # Fit over polynom of degree 6 which has the minimal loss
    poly_fit = PolynomialFitting(5)
    poly_fit.fit(israel_data[DAY_OF_YEAR].to_numpy(),
                 israel_data.Temp.to_numpy())
    # Get all countries except Israel
    countries = dataset.Country.unique().tolist()
    countries.remove("Israel")
    country_losses = []
    # Calculate loss for each country
    for country in countries:
        country_data = dataset[dataset.Country == country]
        country_losses.append(
            poly_fit.loss(country_data[DAY_OF_YEAR].to_numpy(),
                          country_data.Temp.to_numpy()))

    country_loss_df = pd.DataFrame(
        {"Country": countries, "loss": country_losses})
    fig = px.bar(country_loss_df, x="Country", y="loss",
                 title=r"(5) Loss Of Model On Different Countries",
                 width=600)
    fig.show()
