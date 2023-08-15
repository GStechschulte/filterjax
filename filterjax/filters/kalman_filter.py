from typing import NamedTuple

import jax
from jax import numpy as jnp


class KalmanParams(NamedTuple):
    """
    Parameters for the Kalman filter
    """

    x: jnp.ndarray
    F: jnp.ndarray
    H: jnp.ndarray
    R: jnp.ndarray
    Q: jnp.ndarray
    P: jnp.ndarray


class KalmanFilter:
    def __init__(self, dim_x, dim_z):
        self.dim_z = dim_z
        self.dim_x = dim_x

    def initialize(self, params: KalmanParams):
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


def update(z, x, R, H, P):
    """
    Observe new measurement 'z' and update state.
    """
    error = z - (H @ x)
    S = H @ P @ H.T + R
    K = P @ H.T @ jnp.linalg.inv(S)
    x = x + K @ error
    P = P - K @ S @ K.T

    return x, P


def batch_filter(params: KalmanParams, z: jnp.ndarray):

    num_timesteps = len(z)

    def step(carry, t):
        x, P = carry

        x, P = update(z[t], x, params.R, params.H, P)
        x, P = predict(params.F, params.Q, x, P)

        return (x, P), (x, P)

    _, (xs, Ps) = jax.lax.scan(
        step, (params.x, params.P), jnp.arange(num_timesteps)
    )

    return xs, Ps
