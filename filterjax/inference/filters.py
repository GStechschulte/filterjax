from typing import NamedTuple

import jax
import jax.numpy as jnp

from filterjax.params import KalmanParams


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


def batch_filter(params: KalmanParams, emissions: jnp.ndarray):

    num_timesteps = len(emissions)

    def step(carry, t):
        m, P = carry

        # TODO: add and update log-likelihood

        m, P = update(emissions[t], m, params.R, params.H, P)
        m, P = predict(params.F, params.Q, m, P)

        return (m, P), (m, P)

    log_likelihood = 0.0
    carry = (params.m, params.P)
    _, (ms, Ps) = jax.lax.scan(step, carry, jnp.arange(num_timesteps))

    return PosteriorFilter(ms, Ps)