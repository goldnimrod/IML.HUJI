from typing import NoReturn
from ...base import BaseEstimator
import numpy as np


class GaussianNaiveBayes(BaseEstimator):
    """
    Gaussian Naive-Bayes classifier
    """

    def __init__(self):
        """
        Instantiate a Gaussian Naive Bayes classifier

        Attributes
        ----------
        self.classes_ : np.ndarray of shape (n_classes,)
            The different labels classes. To be set in `GaussianNaiveBayes.fit`

        self.mu_ : np.ndarray of shape (n_classes,n_features)
            The estimated features means for each class. To be set in `GaussianNaiveBayes.fit`

        self.vars_ : np.ndarray of shape (n_classes, n_features)
            The estimated features variances for each class. To be set in `GaussianNaiveBayes.fit`

        self.pi_: np.ndarray of shape (n_classes)
            The estimated class probabilities. To be set in `GaussianNaiveBayes.fit`
        """
        super().__init__()
        self.classes_, self.mu_, self.vars_, self.pi_ = None, None, None, None

    def _fit(self, X: np.ndarray, y: np.ndarray) -> NoReturn:
        """
        fits a gaussian naive bayes model

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            Input data to fit an estimator for

        y : ndarray of shape (n_samples, )
            Responses of input data to fit to
        """
        self.classes_ = np.unique(y)
        self.pi_ = np.vectorize(
            lambda k: np.count_nonzero(y == k) / y.shape[0])(self.classes_)
        self.mu_ = np.array(
            [np.sum(X[np.where(y == k)], axis=0) / np.count_nonzero(
                y == k) for k in self.classes_])
        mu_yi = np.array([self.mu_[yi] for yi in y])
        self.vars_ = np.array(
            [np.sum(
                [np.diag(np.outer(row, row)) for row in
                 (X - mu_yi)[np.where(y == k)]],
                axis=0) / np.count_nonzero(y == k)
             for k in self.classes_])

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
        """

        def calc_predict(x: np.ndarray, k: int):
            ak = np.diag(self.vars_[k]) @ self.mu_[k]
            bk = np.log(self.pi_[k]) - 0.5 * self.mu_[k] @ ak
            return ak.T @ x + bk

        def predict_x(x: np.ndarray):
            class_predicts = np.vectorize(lambda k: calc_predict(x, k))(
                self.classes_)
            return self.classes_[np.argmax(class_predicts)]

        return np.apply_along_axis(predict_x, 1, X)

    def likelihood(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate the likelihood of a given data over the estimated model

        Parameters
        ----------
        X : np.ndarray of shape (n_samples, n_features)
            Input data to calculate its likelihood over the different classes.

        Returns
        -------
        likelihoods : np.ndarray of shape (n_samples, n_classes)
            The likelihood for each sample under each of the classes

        """
        if not self.fitted_:
            raise ValueError(
                "Estimator must first be fitted before calling `likelihood` function")

        def calc_pdf(x, k):
            cov_k = np.diag(self.vars_[k])
            return np.exp(-0.5 * (x - self.mu_[k]).T @ np.inv(cov_k) @ (
                (x - self.mu_[k]))) / np.sqrt(
                np.det(cov_k) * (2 * np.pi) ** x.shape[0])

        return np.array([[calc_pdf(x, k) for k in self.classes_] for x in X])

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
        from ...metrics import misclassification_error
        return misclassification_error(y, self.predict(X))
