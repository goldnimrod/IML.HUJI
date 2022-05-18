from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn import datasets
from IMLearn.metrics import mean_square_error
from IMLearn.utils import split_train_test
from IMLearn.model_selection import cross_validate
from IMLearn.learners.regressors import PolynomialFitting, LinearRegression, \
    RidgeRegression
from sklearn.linear_model import Lasso

from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def select_polynomial_degree(n_samples: int = 100, noise: float = 5):
    """
    Simulate data from a polynomial model and use cross-validation to select the best fitting degree

    Parameters
    ----------
    n_samples: int, default=100
        Number of samples to generate

    noise: float, default = 5
        Noise level to simulate in responses
    """

    # Question 1 - Generate dataset for model f(x)=(x+3)(x+2)(x+1)(x-1)(x-2) + eps for eps Gaussian noise
    # and split into training- and testing portions
    def response(x):
        return (x + 3) * (x + 2) * (x + 1) * (x - 1) * (x - 2)

    X = np.linspace(-1.2, 2, n_samples)
    y_noiseless = response(X)
    y = y_noiseless + np.random.normal(scale=noise, size=n_samples)

    train_X, train_y, test_X, test_y = split_train_test(pd.DataFrame(X),
                                                        pd.Series(y),
                                                        train_proportion=2 / 3.0)
    train_X = train_X.to_numpy().flatten()
    train_y = train_y.to_numpy().flatten()
    test_X = test_X.to_numpy().flatten()
    test_y = test_y.to_numpy().flatten()

    fig = go.Figure()
    fig.add_traces(
        [go.Scatter(x=X, y=y_noiseless, mode="markers", name="Real Points",
                    marker=dict(color="black", opacity=.7)),
         go.Scatter(x=train_X, y=train_y, mode="markers", name="Train",
                    marker=dict(color="red", opacity=.7)),
         go.Scatter(x=test_X, y=test_y, mode="markers", name="Test",
                    marker=dict(color="blue", opacity=.7))])
    fig.update_layout(
        title=f"$\\text{{Splitted Sample Points, noise}} = {noise} \\text{{, m}} = {n_samples}$",
        width=800).show()

    # Question 2 - Perform CV for polynomial fitting with degrees 0,1,...,10
    train_errors = []
    validation_errors = []
    for k in range(11):
        train_error, val_error = cross_validate(PolynomialFitting(k), train_X,
                                                train_y,
                                                mean_square_error)
        train_errors.append(train_error)
        validation_errors.append(val_error)
    fig = go.Figure()
    fig.add_traces(
        [go.Scatter(x=np.arange(11), y=train_errors, mode="markers+lines",
                    name="Train Error",
                    marker=dict(color="red", opacity=.7)),
         go.Scatter(x=np.arange(11), y=validation_errors, mode="markers+lines",
                    name="Validation Error",
                    marker=dict(color="blue", opacity=.7))])
    fig.update_layout(
        title=f"$\\text{{Cross Validate Errors as Function of Polynom Degree, noise}} = {noise} \\text{{, m}} = {n_samples}$",
        xaxis=dict(title="Polynom degree"),
        yaxis=dict(title="Error", type="log"),
        width=800).show()

    # Question 3 - Using best value of k, fit a k-degree polynomial model and report test error
    opt_k = np.array(validation_errors).argmin()
    model = PolynomialFitting(opt_k).fit(test_X, test_y)
    print(f"Optimal k: {opt_k}")
    print(f"Test Error: {np.around(model.loss(test_X, test_y), 2)}")


def select_regularization_parameter(n_samples: int = 50,
                                    n_evaluations: int = 500):
    """
    Using sklearn's diabetes dataset use cross-validation to select the best fitting regularization parameter
    values for Ridge and Lasso regressions

    Parameters
    ----------
    n_samples: int, default=50
        Number of samples to generate

    n_evaluations: int, default = 500
        Number of regularization parameter values to evaluate for each of the algorithms
    """
    # Question 6 - Load diabetes dataset and split into training and testing portions
    X, y = datasets.load_diabetes(return_X_y=True,
                                  as_frame=True)
    train_X, train_y, test_X, test_y = split_train_test(X, y,
                                                        n_samples / float(
                                                            len(y)))

    train_X = train_X.to_numpy()
    train_y = train_y.to_numpy()
    test_X = test_X.to_numpy()
    test_y = test_y.to_numpy()

    # Question 7 - Perform CV for different values of the regularization parameter for Ridge and Lasso regressions
    models = {
        "Ridge": (RidgeRegression, np.linspace(0.001, 0.5, n_evaluations)),
        "Lasso": (Lasso, np.linspace(0, 1, n_evaluations))}
    best_lambdas = {"Ridge": 0, "Lasso": 0}
    for name, (model, space) in models.items():
        train_errors = []
        validation_errors = []
        for lam in space:
            train_error, val_error = cross_validate(model(lam),
                                                    train_X,
                                                    train_y,
                                                    mean_square_error)
            train_errors.append(train_error)
            validation_errors.append(val_error)
        fig = go.Figure()
        fig.add_traces(
            [go.Scatter(x=space, y=train_errors,
                        mode="markers+lines",
                        name="Train Error",
                        marker=dict(color="red", opacity=.7)),
             go.Scatter(x=space, y=validation_errors,
                        mode="markers+lines",
                        name="Validation Error",
                        marker=dict(color="blue", opacity=.7))])
        fig.update_layout(
            title=f"$\\text{{Error as a Function of }} \lambda \\text{{ in model {name}}}$",
            xaxis=dict(title="$\lambda$"),
            yaxis=dict(title="Error", type="log"),
            width=800).show()
        best_lambdas[name] = space[np.array(validation_errors).argmin()]
        print(
            f"Best lambda on validation error for {name} is: {best_lambdas[name]}")

    # Question 8 - Compare best Ridge model, best Lasso model and Least Squares model
    models = {"Ridge": RidgeRegression(best_lambdas["Ridge"]),
              "Lasso": Lasso(best_lambdas["Lasso"]),
              "Least Squares": LinearRegression()}
    for name, model in models.items():
        models[name].fit(train_X, train_y)
        print(
            f"Test Error for Model {name} is: {mean_square_error(test_y, model.predict(test_X))}")


if __name__ == '__main__':
    np.random.seed(0)
    select_polynomial_degree()
    select_polynomial_degree(noise=0)
    select_polynomial_degree(n_samples=1500, noise=10)
    select_regularization_parameter()
