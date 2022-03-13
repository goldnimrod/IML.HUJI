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
                  xaxis_title="$\\text{sample value}$",
                  yaxis_title=r'$\text{PDF}$',
                  height=500)).show()


def test_multivariate_gaussian():
    mu = np.array([0, 0, 4, 0])
    sigma = np.array([[1, 0.2, 0, 0.5],
                      [0.2, 2, 0, 0],
                      [0, 0, 1, 0],
                      [0.5, 0, 0, 1]])
    # Question 4 - Draw samples and print fitted model
    samples = numpy.random.multivariate_normal(mu, sigma, 1000)
    estimator = MultivariateGaussian().fit(samples)

    print(estimator.mu_)
    print(estimator.cov_)

    # Question 5 - Likelihood evaluation
    mu_values = np.array(
        np.meshgrid(np.linspace(-10, 10, 200), [0], np.linspace(-10, 10, 200),
                    [0])).T.reshape(-1, 4)
    log_likelihoods = [MultivariateGaussian.log_likelihood(mu, sigma, samples)
                       for mu in mu_values]
    go.Figure(
        go.Heatmap(x=mu_values[:, 2], y=mu_values[:, 0], z=log_likelihoods),
        layout=go.Layout(title="Hi", height=300, width=200)).show()

    # Question 6 - Maximum likelihood
    raise NotImplementedError()


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
