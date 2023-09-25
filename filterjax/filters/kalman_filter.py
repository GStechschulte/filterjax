from typing import NamedTuple

import jax
from jax import numpy as jnp


class KalmanParams(NamedTuple):
    """Parameters for the Kalman filter.

    Parameters
    ----------
    m : jnp.ndarray
        Mean of the prior state estimate.
    F : jnp.ndarray
        State transition matrix.
    H : jnp.ndarray
        Observation (emission) matrix.
    R : jnp.ndarray
        Covariance matrix of the observation (emission) noise.
    Q : jnp.ndarray
        Covariance matrix of the process model (dynamics) noise.
    P : jnp.ndarray
        Covariance matrix of the prior state estimate.
    B : jnp.ndarray, optional
        Control transition matrix, by default None
    """

    m: jnp.ndarray 
    F: jnp.ndarray
    H: jnp.ndarray 
    R: jnp.ndarray
    Q: jnp.ndarray
    P: jnp.ndarray
    B: jnp.ndarray = None


class PosteriorFilter(NamedTuple):
    """Posterior state estimate and covariance matrix.

    Parameters
    ----------
    m : jnp.ndarray
        Mean of the posterior state estimate.
    P : jnp.ndarray
        Covariance of the posterior state estimate.
    """

    mean: jnp.ndarray
    covariance: jnp.ndarray


class KalmanFilter:
    def __init__(self, dim_m, dim_y):
        self.dim_m = dim_m
        self.dim_y = dim_y

    def initialize(self, params: KalmanParams):
        assert params.m.shape == (self.dim_m,)
        assert params.F.shape == (self.dim_m, self.dim_m)
        assert params.H.shape == (self.dim_y, self.dim_m)
        assert params.R.shape == (self.dim_y, self.dim_y)
        assert params.Q.shape == (self.dim_m, self.dim_m)
        assert params.P.shape == (self.dim_m, self.dim_m)

        if params.B is not None:
            assert params.B.shape == (self.dim_m, self.dim_m)

        return params

    def filter(self, params: KalmanParams, measurements: jnp.ndarray):
        return batch_filter(params, measurements)


def predict(F, Q, x, P, B=None, u=None):
    """
    Predict next state using the Kalman filter state propagation equations.
    """
    if B is not None and u is not None:
        x = F @ x + B @ u
    else:
        x = F @ x

    P = F @ P @ F.T + Q

    return x, P


def update(y, x, R, H, P):
    """
    Observe new measurement (emission) 'y' and update state.
    """
    error = y - (H @ x)
    # TODO: Use Cholesky decomposition for covariance
    S = H @ P @ H.T + R
    K = P @ H.T @ jnp.linalg.inv(S)
    x = x + K @ error
    P = P - K @ S @ K.T

    return x, P


def batch_filter(params: KalmanParams, y: jnp.ndarray):

    num_timesteps = len(y)

    def step(carry, t):
        m, P = carry

        m, P = update(y[t], m, params.R, params.H, P)
        m, P = predict(params.F, params.Q, m, P)

        return (m, P), (m, P)

    _, (ms, Ps) = jax.lax.scan(
        step, (params.m, params.P), jnp.arange(num_timesteps)
    )

    return PosteriorFilter(ms, Ps)
