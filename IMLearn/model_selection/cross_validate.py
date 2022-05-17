from __future__ import annotations
from copy import deepcopy
from typing import Tuple, Callable
import numpy as np
from IMLearn import BaseEstimator


def cross_validate(estimator: BaseEstimator, X: np.ndarray, y: np.ndarray,
                   scoring: Callable[[np.ndarray, np.ndarray, ...], float],
                   cv: int = 5) -> Tuple[float, float]:
    """
    Evaluate metric by cross-validation for given estimator

    Parameters
    ----------
    estimator: BaseEstimator
        Initialized estimator to use for fitting the data

    X: ndarray of shape (n_samples, n_features)
       Input data to fit

    y: ndarray of shape (n_samples, )
       Responses of input data to fit to

    scoring: Callable[[np.ndarray, np.ndarray, ...], float]
        Callable to use for evaluating the performance of the cross-validated model.
        When called, the scoring function receives the true- and predicted values for each sample
        and potentially additional arguments. The function returns the score for given input.

    cv: int
        Specify the number of folds.

    Returns
    -------
    train_score: float
        Average train score over folds

    validation_score: float
        Average validation score over folds
    """
    folds = np.array_split(np.arange(y.shape[0]), cv)
    train_scores = []
    validation_scores = []

    for k in range(cv):
        k_estimator = estimator.fit(X[~folds[k]], y[~folds[k]])
        train_scores.append(
            scoring(y[~folds[k]], k_estimator.predict(X[~folds[k]])))
        validation_scores.append(
            scoring(y[folds[k]], k_estimator.predict(X[folds[k]])))

    return np.array(train_scores).mean(), np.array(validation_scores).mean()
