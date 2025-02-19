import numpy as np
from typing import Tuple
from IMLearn.metalearners.adaboost import AdaBoost
from IMLearn.learners.classifiers import DecisionStump
from IMLearn.metrics.loss_functions import accuracy
from utils import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def generate_data(n: int, noise_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a dataset in R^2 of specified size

    Parameters
    ----------
    n: int
        Number of samples to generate

    noise_ratio: float
        Ratio of labels to invert

    Returns
    -------
    X: np.ndarray of shape (n_samples,2)
        Design matrix of samples

    y: np.ndarray of shape (n_samples,)
        Labels of samples
    """
    '''
    generate samples X with shape: (num_samples, 2) and labels y with shape (num_samples).
    num_samples: the number of samples to generate
    noise_ratio: invert the label for this ratio of the samples
    '''
    X, y = np.random.rand(n, 2) * 2 - 1, np.ones(n)
    y[np.sum(X ** 2, axis=1) < 0.5 ** 2] = -1
    y[np.random.choice(n, int(noise_ratio * n))] *= -1
    return X, y


def fit_and_evaluate_adaboost(noise, n_learners=250, train_size=5000,
                              test_size=500):
    (train_X, train_y), (test_X, test_y) = generate_data(train_size,
                                                         noise), generate_data(
        test_size, noise)

    # go.Figure([go.Scatter(x=train_X[:, 0], y=train_X[:, 1], mode="markers",
    #                       showlegend=False,
    #                       marker=dict(color=train_y))]).show()

    # Question 1: Train- and test errors of AdaBoost in noiseless case
    adaboost = AdaBoost(DecisionStump, n_learners).fit(train_X, train_y)
    fig = go.Figure()
    fig.add_traces([go.Line(x=list(range(1, n_learners + 1)),
                            y=[adaboost.partial_loss(train_X, train_y, i) for
                               i in range(1, n_learners + 1)],
                            name='Train'),
                    go.Line(x=list(range(1, n_learners + 1)),
                            y=[adaboost.partial_loss(test_X, test_y, i) for
                               i in range(1, n_learners + 1)],
                            name='Test')
                    ])
    fig.update_layout(xaxis_title="Learners Count",
                      yaxis_title="Error",
                      title=f"Error as Function of fitted learners count with noise {noise}")
    fig.show()

    # Question 2: Plotting decision surfaces
    T = [5, 50, 100, 250]
    lims = np.array([np.r_[train_X, test_X].min(axis=0),
                     np.r_[train_X, test_X].max(axis=0)]).T + np.array(
        [-.1, .1])
    fig = make_subplots(rows=2, cols=2,
                        subplot_titles=[rf"$\textbf{{{t} Iterations}}$" for t
                                        in T],
                        horizontal_spacing=0.01, vertical_spacing=.03)
    for i, t in enumerate(T):
        fig.add_traces(
            [decision_surface(lambda x: adaboost.partial_predict(x, t),
                              lims[0], lims[1],
                              showscale=False),
             go.Scatter(x=test_X[:, 0], y=test_X[:, 1],
                        mode="markers",
                        showlegend=False,
                        marker=dict(color=test_y,
                                    colorscale=[custom[0],
                                                custom[-1]],
                                    line=dict(color="black",
                                              width=1)))],
            rows=(i // 2) + 1, cols=(i % 2) + 1)

    fig.update_layout(
        title=rf"$\textbf{{Decision Boundaries On different iterations with noise {noise}}}$",
        margin=dict(t=100)).update_xaxes(visible=False).update_yaxes(
        visible=False)

    fig.show()

    # Question 3: Decision surface of best performing ensemble
    best_ensemble_size = np.vectorize(
        lambda i: adaboost.partial_loss(train_X, train_y, i))(
        np.arange(1, n_learners + 1)).argmin()
    fig = go.Figure()
    fig.add_traces([decision_surface(
        lambda x: adaboost.partial_predict(x, best_ensemble_size),
        lims[0], lims[1],
        showscale=False),
        go.Scatter(x=test_X[:, 0], y=test_X[:, 1],
                   mode="markers",
                   showlegend=False,
                   marker=dict(color=test_y, colorscale=[custom[0],
                                                         custom[-1]],
                               line=dict(color="black", width=1)))])
    fig.update_layout(
        title=rf"$\textbf{{Best performing Ensemble on noise {noise} - {best_ensemble_size} iterations - "
              rf"Accuracy: {accuracy(test_y, adaboost.partial_predict(test_X, best_ensemble_size))}}}$")
    fig.show()

    # Question 4: Decision surface with weighted samples
    fig = go.Figure()
    fig.add_traces([decision_surface(
        adaboost.predict, lims[0], lims[1], showscale=False),
        go.Scatter(x=train_X[:, 0], y=train_X[:, 1],
                   mode="markers",
                   showlegend=False,
                   marker=dict(color=test_y, colorscale=[custom[0],
                                                         custom[-1]],
                               line=dict(color="black", width=1),
                               size=(adaboost.D_ / np.max(adaboost.D_)) * 5))])
    fig.update_layout(
        title=rf"$\textbf{{Train Set probability on last adaboost iteration on noise {noise}}}$")
    fig.show()


if __name__ == '__main__':
    np.random.seed(0)
    fit_and_evaluate_adaboost(0)
    fit_and_evaluate_adaboost(0.4)
