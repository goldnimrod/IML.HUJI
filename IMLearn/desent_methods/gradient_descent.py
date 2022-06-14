from __future__ import annotations
from typing import Callable, NoReturn
import numpy as np

from IMLearn.base import BaseModule, BaseLR
from .learning_rate import FixedLR

OUTPUT_VECTOR_TYPE = ["last", "best", "average"]


def default_callback(solver: GradientDescent, **kwargs) -> NoReturn:
    pass


class GradientDescent:
    """
    Gradient Descent algorithm

    Attributes:
    -----------
    learning_rate_: BaseLR
        Learning rate strategy for retrieving the learning rate at each iteration t of the algorithm

    tol_: float
        The stopping criterion. Training stops when the Euclidean norm of w^(t)-w^(t-1) is less than
        specified tolerance

    max_iter_: int
        The maximum number of GD iterations to be performed before stopping training

    out_type_: str
        Type of returned solution:
            - `last`: returns the point reached at the last GD iteration
            - `best`: returns the point achieving the lowest objective
            - `average`: returns the average point over the GD iterations

    callback_: Callable[[GradientDescent, ...], None]
        A callable function to be called after each update of the model while fitting to given data
        Callable function should receive as input a GradientDescent instance, and any additional
        arguments specified in the `GradientDescent.fit` function
    """

    def __init__(self,
                 learning_rate: BaseLR = FixedLR(1e-3),
                 tol: float = 1e-5,
                 max_iter: int = 1000,
                 out_type: str = "last",
                 callback: Callable[
                     [GradientDescent, ...], None] = default_callback):
        """
        Instantiate a new instance of the GradientDescent class

        Parameters
        ----------
        learning_rate: BaseLR, default=FixedLR(1e-3)
            Learning rate strategy for retrieving the learning rate at each iteration t of the algorithm

        tol: float, default=1e-5
            The stopping criterion. Training stops when the Euclidean norm of w^(t)-w^(t-1) is less than
            specified tolerance

        max_iter: int, default=1000
            The maximum number of GD iterations to be performed before stopping training

        out_type: str, default="last"
            Type of returned solution. Supported types are specified in class attributes

        callback: Callable[[GradientDescent, ...], None], default=default_callback
            A callable function to be called after each update of the model while fitting to given data
            Callable function should receive as input a GradientDescent instance, and any additional
            arguments specified in the `GradientDescent.fit` function
        """
        self.learning_rate_ = learning_rate
        if out_type not in OUTPUT_VECTOR_TYPE:
            raise ValueError("output_type not supported")
        self.out_type_ = out_type
        self.tol_ = tol
        self.max_iter_ = max_iter
        self.callback_ = callback

    def fit(self, f: BaseModule, X: np.ndarray, y: np.ndarray):
        """
        Optimize module using Gradient Descent iterations over given input samples and responses

        Parameters
        ----------
        f : BaseModule
            Module of objective to optimize using GD iterations
        X : ndarray of shape (n_samples, n_features)
            Input data to optimize module over
        y : ndarray of shape (n_samples, )
            Responses of input data to optimize module over

        Returns
        -------
        solution: ndarray of shape (n_features)
            Obtained solution for module optimization, according to the specified self.out_type_

        Notes
        -----
        - Optimization is performed as long as self.max_iter_ has not been reached and that
        Euclidean norm of w^(t)-w^(t-1) is more than the specified self.tol_

        - At each iteration the learning rate is specified according to self.learning_rate_.lr_step

        - At the end of each iteration the self.callback_ function is called passing self and the
        following named arguments:
            - solver: GradientDescent
                self, the current instance of GradientDescent
            - weights: ndarray of shape specified by module's weights
                Current weights of objective
            - val: ndarray of shape specified by module's compute_output function
                Value of objective function at current point, over given data X, y
            - grad:  ndarray of shape specified by module's compute_jacobian function
                Module's jacobian with respect to the weights and at current point, over given data X,y
            - t: int
                Current GD iteration
            - eta: float
                Learning rate used at current iteration
            - delta: float
                Euclidean norm of w^(t)-w^(t-1)

        """
        solution_func = self.get_solution_func_()
        return_weights = f.weights
        best_obj_val = f.compute_output(X=X, y=y)
        self.callback_(solver=self, weights=f.weights,
                       val=best_obj_val, grad=0, t=0,
                       eta=0, delta=0)

        for t in range(self.max_iter_):
            prev_weights = f.weights
            grad = f.compute_jacobian(X=X, y=y)
            eta = self.learning_rate_.lr_step(t=t)
            f.weights = prev_weights - eta * grad
            delta = np.linalg.norm(f.weights - prev_weights)
            objective_val = f.compute_output(X=X, y=y)
            self.callback_(solver=self, weights=f.weights,
                           val=objective_val, grad=grad, t=t,
                           eta=eta, delta=delta)
            return_weights, best_obj_val = solution_func(weights=f.weights,
                                                         f=f, t=t, X=X, y=y,
                                                         prev_ret_weights=return_weights,
                                                         best_obj_val=best_obj_val)
            if delta < self.tol_:
                return return_weights

    def get_solution_func_(self):
        """
        Returns a function that calculates the solution according to self.out_type_

        Returns
        -------
        A function that will calculate the solution in each iteration
        """

        func = None
        if self.out_type_ == "last":
            return lambda weights, **kwargs: (weights, None)
        elif self.out_type_ == "best":
            def func(weights, f, prev_ret_weights, best_obj_val,
                     X, y, **kwargs):
                current_objective_val = f.compute_output(X=X, y=y)
                if current_objective_val < best_obj_val:
                    return weights, current_objective_val
                return prev_ret_weights, best_obj_val
        elif self.out_type_ == "average":
            def func(weights, prev_ret_weights, t, **kwargs):
                return np.sum(weights, prev_ret_weights * (t - 1)) / t, None
        return func
