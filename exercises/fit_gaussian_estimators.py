import numpy.random

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.templates.default = "simple_white"


def test_univariate_gaussian():
    mu = 10
    sigma = 1

    # Question 1 - Draw samples and print fitted model
    samples = numpy.random.normal(mu, sigma, 1000)
    estimator = UnivariateGaussian().fit(samples)
    print(f"({estimator.mu_, estimator.var_})")

    # Question 2 - Empirically showing sample mean is consistent
    distance_from_mu = []
    sample_sizes = np.arange(10, 1010, 10)
    for m in sample_sizes:
        X = samples[:m + 1]
        estimator = UnivariateGaussian().fit(X)
        distance_from_mu.append(abs(estimator.mu_ - mu))

    go.Figure([go.Scatter(x=sample_sizes, y=distance_from_mu,
                          mode='markers+lines',
                          name=r'$\widehat\mu$')],
              layout=go.Layout(
                  title=r"$\text{(2) Distance from Expectation As Function Of Number Of Samples}$",
                  xaxis_title="$m\\text{ - number of samples}$",
                  yaxis_title=r'$\left|\hat\mu-\mu\right|$',
                  height=500)).show()

    # Question 3 - Plotting Empirical PDF of fitted model
    pdf = estimator.pdf(samples)

    go.Figure([go.Scatter(x=samples, y=pdf,
                          mode='markers',
                          name=r'$\widehat\mu$')],
              layout=go.Layout(
                  title=r"$\text{(3) PDF As Function Of Sample Value}$",
                  xaxis_title="$m\\text{ - sample value}$",
                  yaxis_title=r'$\text{PDF}$',
                  height=500)).show()
    print()


def test_multivariate_gaussian():
    # Question 4 - Draw samples and print fitted model
    raise NotImplementedError()

    # Question 5 - Likelihood evaluation
    raise NotImplementedError()

    # Question 6 - Maximum likelihood
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
