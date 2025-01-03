"""Utils functions."""

import numpy as np
from fastcan import minibatch
from fastcan.narx import (
    _mask_missing_value,
    make_narx,
    make_poly_features,
    make_time_shift_features,
)
from scipy.integrate import odeint
from sklearn.cluster import MiniBatchKMeans
from sklearn.linear_model import LinearRegression


def get_dual_stable_equilibria_data(auto=False):
    """Get dual stable equilibria data."""

    def _nonautonomous_dual_stable_equilibria(y, t):
        y1, y2 = y
        u = 0.1 * np.cos(0.2 * np.pi * t)
        dydt = [y2, -(y1**3) - y1**2 + y1 - y2 + u]
        return dydt

    def _autonomous_dual_stable_equilibria(y, t):
        y1, y2 = y
        dydt = [y2, -(y1**3) - y1**2 + y1 - y2]
        return dydt

    if auto:
        func = _autonomous_dual_stable_equilibria
    else:
        func = _nonautonomous_dual_stable_equilibria
    n_samples = 100
    x0 = np.linspace(0, 2, 10)
    n_init = len(x0)
    y0_y = np.cos(np.pi * x0)
    y0_x = np.sin(np.pi * x0)
    y0 = np.c_[y0_x, y0_y]
    t = np.linspace(0, 10, n_samples)
    sol = np.zeros((n_init, n_samples, 2))
    u = np.zeros(0)
    y = np.zeros(0)
    for i in range(n_init):
        sol[i] = odeint(func, y0[i], t)
        u = np.r_[u, 0.1 * np.cos(0.2 * np.pi * t), np.nan]
        y = np.r_[y, sol[i, :, 0], np.nan]
    return u, y, sol


def get_narx_terms(u, y):
    """
    Generate NARX terms from input and output data.

    Parameters
    ----------
    u : array-like
        Input data.

    y : array-like
        Output data.

    Returns
    -------
    poly_terms : array-like
        Polynomial terms generated from the input and output data.

    narx : object
        Fitted NARX model.
    """

    narx = make_narx(
        u.reshape(-1, 1),
        y,
        n_features_to_select=10,
        max_delay=10,
        poly_degree=3,
        verbose=0,
    ).fit(
        u.reshape(-1, 1),
        y,
    )

    xy_hstack = np.c_[u, y]
    time_shift_vars = make_time_shift_features(xy_hstack, narx.time_shift_ids_)
    poly_terms = make_poly_features(time_shift_vars, narx.poly_ids_)
    return *_mask_missing_value(poly_terms, y), narx


def fastcan_pruned_narx(terms, y, n_samples_to_select: int, random_state: int):
    """
    Fit a NARX model with data pruned by FastCan.

    Parameters
    ----------
    terms : array-like
        Input data.

    y : array-like
        Output data.

    n_samples_to_select : int
        Number of samples to select.

    random_state : int
        Random state.

    Returns
    -------
    coef : array-like
        Coefficients of the linear regression.

    intercept : float
        Intercept of the linear regression.
    """
    kmeans = MiniBatchKMeans(
        n_clusters=100,
        random_state=random_state,
        batch_size=6,
        n_init="auto",
    ).fit(terms)
    atoms = kmeans.cluster_centers_
    ids_fastcan = minibatch(terms.T, atoms.T, n_samples_to_select, batch_size=7)
    pruned_narx = LinearRegression().fit(terms[ids_fastcan], y[ids_fastcan])
    return pruned_narx.coef_, pruned_narx.intercept_


def random_pruned_narx(terms, y, n_samples_to_select: int, random_state: int):
    """
    Fit a NARX model with data pruned by random selection.

    Parameters
    ----------
    terms : array-like
        Input data.

    y : array-like
        Output data.

    n_samples_to_select : int
        Number of samples to select.

    random_state : int
        Random state.

    Returns
    -------
    coef : array-like
        Coefficients of the linear regression.

    intercept : float
        Intercept of the linear regression.
    """
    rng = np.random.default_rng(random_state)
    ids_random = rng.choice(y.size, n_samples_to_select, replace=False)
    pruned_narx = LinearRegression().fit(terms[ids_random], y[ids_random])
    return pruned_narx.coef_, pruned_narx.intercept_
