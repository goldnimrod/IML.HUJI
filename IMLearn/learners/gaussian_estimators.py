from __future__ import annotations

import math

import numpy as np
from numpy.linalg import inv, det, slogdet


class UnivariateGaussian:
    """
    Class for univariate Gaussian Distribution Estimator
    """

    def __init__(self, biased_var: bool = False) -> UnivariateGaussian:
        """
        Estimator for univariate Gaussian mean and variance parameters

        Parameters
        ----------
        biased_var : bool, default=False
            Should fitted estimator of variance be a biased or unbiased estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `UnivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `UnivariateGaussian.fit`
            function.

        var_: float
            Estimated variance initialized as None. To be set in `UnivariateGaussian.fit`
            function.
        """
        self.biased_ = biased_var
        self.fitted_, self.mu_, self.var_ = False, None, None

    def fit(self, X: np.ndarray) -> UnivariateGaussian:
        """
        Estimate Gaussian expectation and variance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.var_` attributes according to calculated estimation (where
        estimator is either biased or unbiased). Then sets `self.fitted_` attribute to `True`
        """
        self.mu_ = X.mean()
        self.var_ = self.get_variance(X)

        self.fitted_ = True
        return self

    def get_variance(self, X: np.ndarray):
        """
        Calculates the variance whether the estimator is biased

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        variance: float
            the calculated variance
        """
        if self.biased_:
            return sum([(x - self.mu_) ** 2 for x in X]) / X.size
        return sum([(x - self.mu_) ** 2 for x in X]) / (X.size - 1)

    def pdf(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, var_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError(
                "Estimator must first be fitted before calling `pdf` function")
        return np.array([math.exp(- ((x - self.mu_) ** 2) / (2 * self.var_)) /
                         math.sqrt(2 * math.pi * self.var_) for x in X])

    @staticmethod
    def log_likelihood(mu: float, sigma: float, X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        sigma : float
            Variance of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        return (math.log(1 / math.pow(2 * math.pi * sigma, X.size / 2)) -
                0.5 * sigma * sum([(x - mu) ** 2 for x in X]))


class MultivariateGaussian:
    """
    Class for multivariate Gaussian Distribution Estimator
    """

    def __init__(self):
        """
        Initialize an instance of multivariate Gaussian estimator

        Attributes
        ----------
        fitted_ : bool
            Initialized as false indicating current estimator instance has not been fitted.
            To be set as True in `MultivariateGaussian.fit` function.

        mu_: float
            Estimated expectation initialized as None. To be set in `MultivariateGaussian.ft`
            function.

        cov_: float
            Estimated covariance initialized as None. To be set in `MultivariateGaussian.ft`
            function.
        """
        self.mu_, self.cov_ = None, None
        self.fitted_ = False

    def fit(self, X: np.ndarray) -> MultivariateGaussian:
        """
        Estimate Gaussian expectation and covariance from given samples

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Training data

        Returns
        -------
        self : returns an instance of self.

        Notes
        -----
        Sets `self.mu_`, `self.cov_` attributes according to calculated estimation.
        Then sets `self.fitted_` attribute to `True`
        """
        self.mu_ = np.mean(X, axis=0)
        self.cov_ = self.get_covariance(X)

        self.fitted_ = True
        return self

    def get_covariance(self, X: np.ndarray):
        """
        Calculates the covariance matrix, whether the estimator is biased

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Training data

        Returns
        -------
        covariance: ndarray of shape (X.shape[1], X.shape[1])
            the calculated covariance matrix
        """
        centered_samples = self.get_centered_sample_matrix(X)
        return centered_samples.T @ centered_samples / (X.shape[0] - 1)

    def get_centered_sample_matrix(self, X: np.ndarray):
        """
        Calculates the centered samples matrix, according to mu_

        Parameters
        ----------
        X: ndarray of shape (n_samples, n_features)
            Training data

        Returns
        -------
        centered_samples: ndarray of shape (n_samples, n_features)
            the centered samples matrix
        """
        centered_samples = np.zeros(X.shape)
        for i in range(centered_samples.shape[0]):
            centered_samples[i, :] = X[i, :] - self.mu_
        return centered_samples

    def pdf(self, X: np.ndarray):
        """
        Calculate PDF of observations under Gaussian model with fitted estimators

        Parameters
        ----------
        X: ndarray of shape (n_samples, )
            Samples to calculate PDF for

        Returns
        -------
        pdfs: ndarray of shape (n_samples, )
            Calculated values of given samples for PDF function of N(mu_, cov_)

        Raises
        ------
        ValueError: In case function was called prior fitting the model
        """
        if not self.fitted_:
            raise ValueError(
                "Estimator must first be fitted before calling `pdf` function")
        pdfs = np.zeros(X.shape[0])

        for i, x in enumerate(X):
            pdfs[i] = math.exp(-0.5 * x.T @ np.linalg.inv(
                self.cov_) @ x) / math.sqrt(
                np.linalg.det(self.cov_) * (2 * math.pi) ** X.shape[1])

        return pdfs

    @staticmethod
    def log_likelihood(mu: np.ndarray, cov: np.ndarray,
                       X: np.ndarray) -> float:
        """
        Calculate the log-likelihood of the data under a specified Gaussian model

        Parameters
        ----------
        mu : float
            Expectation of Gaussian
        cov : float
            covariance matrix of Gaussian
        X : ndarray of shape (n_samples, )
            Samples to calculate log-likelihood with

        Returns
        -------
        log_likelihood: float
            log-likelihood calculated
        """
        return (math.log(
            1 / math.pow(np.linalg.det(cov) * (2 * math.pi) ** X.shape[1],
                         X.shape[0] / 2)) -
                0.5 * sum(
                    [(x - mu).T @ np.linalg.inv(cov) @ (x - mu) for x in X]))
