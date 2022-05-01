from __future__ import annotations
from typing import Tuple, NoReturn
from ...base import BaseEstimator
import numpy as np
from ...metrics import misclassification_error
from itertools import product


class DecisionStump(BaseEstimator):
    """
    A decision stump classifier for {-1,1} labels according to the CART algorithm

    Attributes
    ----------
    self.threshold_ : float
        The threshold by which the data is split

    self.j_ : int
        The index of the feature by which to split the data

    self.sign_: int
        The label to predict for samples where the value of the j'th feature is about the threshold
    """

    def __init__(self) -> DecisionStump:
        """
        Instantiate a Decision stump classifier
        """
        super().__init__()
        self.threshold_, self.j_, self.sign_ = None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a decision stump to the given data

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        min_error = 1
        # TODO: maybe change according to answer in forum
        # ran on all sign combinations instead of determining the majority
        for feature_index, sign in product(range(X.shape[1]),
                                           np.unique(np.sign(y))):
            threshold, error = self._find_threshold(X[:, feature_index], y,
                                                    sign)
            if error <= min_error:
                min_error = error
                self.threshold_ = threshold
                self.sign_ = sign
                self.j_ = feature_index

    def _predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict responses for given samples using fitted estimator

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to predict responses for

        Returns
        -------
        responses : ndarray of shape (n_samples, )
            Predicted responses of given samples

        Notes
        -----
        Feature values strictly below threshold are predicted as `-sign` whereas values which equal
        to or above the threshold are predicted as `sign`
        """
        return np.where(X[:, self.j_] < self.threshold_, -self.sign_,
                        self.sign_)

    def _find_threshold(self, values: np.ndarray, labels: np.ndarray,
                        sign: int) -> Tuple[float, float]:
        """
        Given a feature vector and labels, find a threshold by which to perform a split
        The threshold is found according to the value minimizing the misclassification
        error along this feature

        Parameters
        ----------
        values: ndarray of shape (n_samples,)
            A feature vector to find a splitting threshold for

        labels: ndarray of shape (n_samples,)
            The labels to compare against

        sign: int
            Predicted label assigned to values equal to or above threshold

        Returns
        -------
        thr: float
            Threshold by which to perform split

        thr_err: float between 0 and 1
            Misclassificaiton error of returned threshold

        Notes
        -----
        For every tested threshold, values strictly below threshold are predicted as `-sign` whereas values
        which equal to or above the threshold are predicted as `sign`
        """
        sorted_values = values[values.argsort()]
        sorted_labels = labels[values.argsort()]
        error_count = np.sum(np.abs(sorted_labels[
                                        np.not_equal(np.sign(sorted_labels),
                                                     np.ones(
                                                         sorted_values.shape[
                                                             0]) * sign)]))

        def calc_thr_value_error(i):
            """
            Calculates the misclassificaiton error of the threshold with
            The value in index i

            Parameters
            ----------
            i: int
                The index of the value in the sorted_values array

            Returns
            -------
            thr_err: float between 0 and 1
                Misclassificaiton error of the threshold

            """
            # TODO: maybe add according to answer in forum
            # sign = np.argmax(np.histogram(sorted_labels[i:]))
            # threshold_labels = np.where(np.arange(sorted_values.shape[0]) < i,
            #                             -sign, sign)
            # return np.sum(np.abs(sorted_labels[
            #                          np.not_equal(np.sign(sorted_labels),
            #                                       np.sign(threshold_labels))]))
            nonlocal error_count
            if i == 0:
                return error_count
            if np.sign(sorted_labels[i - 1]) == -sign:
                error_count -= np.abs(sorted_labels[i - 1])
            else:
                error_count += np.abs(sorted_labels[i - 1])
            return error_count

        errors = np.vectorize(calc_thr_value_error)(
            np.arange(sorted_values.shape[0]))
        min_error_index = np.argmin(errors)
        return sorted_values[min_error_index], errors[min_error_index]

    def _loss(self, X: np.ndarray, y: np.ndarray) -> float:
        """
        Evaluate performance under misclassification loss function

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Test samples

        y : ndarray of shape (n_samples, )
            True labels of test samples

        Returns
        -------
        loss : float
            Performance under missclassification loss function
        """
        return misclassification_error(np.sign(y), np.sign(self.predict(X)))
