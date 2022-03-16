import numpy.random

from IMLearn.learners import UnivariateGaussian, MultivariateGaussian
import numpy as np
import plotly.graph_objects as go
import plotly.io as pio

pio.renderers.default = "browser"
pio.templates.default = "simple_white"


def test_univariate_gaussian():
    mu = 10
    sigma = 1

    # Question 1 - Draw samples and print fitted model
    samples = numpy.random.normal(mu, sigma, 1000)
    estimator = UnivariateGaussian().fit(samples)
    print(f"Q1: {estimator.mu_, estimator.var_}")

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
    samples = np.random.multivariate_normal(mu, sigma, 1000)
    estimator = MultivariateGaussian().fit(samples)

    print("Q4:")
    print(f"mu: {estimator.mu_}")
    print(f"cov: {estimator.cov_}")

    # Question 5 - Likelihood evaluation
    models = np.array(
        np.meshgrid(np.linspace(-10, 10, 200), [0], np.linspace(-10, 10, 200),
                    [0])).T.reshape(-1, 4)
    log_likelihoods = [MultivariateGaussian.log_likelihood(mu, sigma, samples)
                       for mu in models]
    go.Figure(
        go.Heatmap(x=models[:, 2], y=models[:, 0], z=log_likelihoods),
        layout=go.Layout(
            title=r"$\text{(5) Heatmap of log-likelihood according to }\mu$",
            xaxis_title="$f_{3}$",
            yaxis_title=r'$f_{1}$',
            height=500, width=500)).show()

    # Question 6 - Maximum likelihood
    max_index = numpy.argmax(log_likelihoods)
    print("Q6: ")
    print(
        f"model with max log-likelihood: {models[max_index]}")
    print(f"log-likelihood value of the model: {log_likelihoods[max_index]}")


if __name__ == '__main__':
    np.random.seed(0)
    test_univariate_gaussian()
    test_multivariate_gaussian()
