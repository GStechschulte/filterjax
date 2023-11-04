from typing import NamedTuple

import jax
import jax.numpy as jnp

from filterjax.models.params import KalmanParams


class PosteriorFilter(NamedTuple):
    """Posterior state estimate and covariance matrix.

    Parameters
    ----------
    m : jnp.ndarray
        Mean of the posterior state estimate.
    P : jnp.ndarray
        Covariance matrix of the posterior state estimate.
    """

    mean: jnp.ndarray
    covariance: jnp.ndarray


def _update(y, m, R, H, P):
    """Update (measurement) step conditions on a new linear Gaussian
    observation and computes the Kalman gain to compute the posterior state mean
    and covariance.

    $$error := y - \mathbf{H}m$$
    $$\mathbf{S} := \mathbf{H}\mathbf{P}\mathbf{H}^T + R$$
    $$\mathbf{K} := \mathbf{P}\mathbf{H}^T\mathbf{S}^{-1}$$
    $$m := m + \mathbf{K}y$$
    $$\mathbf{P} := (\mathbf{I} - \mathbf{K}\mathbf{H})\mathbf{P}$$

    Parameters
    ----------
    y : jnp.ndarray
        Emissions (measurement) vector.
    m : jnp.ndarray
        Mean of the prior state estimate.
    R : jnp.ndarray
        Covariance matrix of the emission (observation) noise.
    H : jnp.ndarray
        Emission (observation) matrix.
    P : jnp.ndarray
        Covariance matrix of the prior state estimate.

    Returns
    -------
    m : jnp.ndarray
        Mean of the posterior state estimate.
    P : jnp.ndarray
        Covariance matrix of the posterior state estimate.
    """
    # compute residual between emission and prediction
    error = y - (H @ m)

    # TODO: Use Cholesky decomposition for covariance
    # compute Kalman gain (scaling factor)
    S = H @ P @ H.T + R
    K = P @ H.T @ jnp.linalg.inv(S)

    # update belief in state mean and covariance based on how certain we are
    m = m + K @ error
    P = P - K @ S @ K.T

    return m, P


def _predict(F, Q, m, P, B=None, u=None):
    """Predict next state mean and covariance using the Kalman filter state
    propagation equations.

    $$m := \mathbf{F}m + \mathbf{B}u$$
    $$\mathbf{P} := \mathbf{F}\mathbf{P}\mathbf{F}^T + \mathbf{Q}$$

    Parameters
    ----------
    F : jnp.ndarray
        State transition matrix.
    Q : jnp.ndarray
        Covariance matrix of the process model (dynamics) noise.
    m : jnp.ndarray
        Mean of the prior state estimate.
    P : jnp.ndarray
        Covariance matrix of the prior state estimate.
    B : jnp.ndarray, optional
        Control transition matrix, by default None
    u : jnp.ndarray, optional
        Control vector, by default None

    Returns
    -------
    m : jnp.ndarray
        Mean of the posterior state estimate.
    P : jnp.ndarray
        Covariance matrix of the posterior state estimate.
    """
    # predict next state (prior) mean using the process model (state transition
    # matrix) and control inputs if available
    if B is not None and u is not None:
        m = F @ m + B @ u
    else:
        m = F @ m

    # predict next state (prior) covariance using the process model (state
    # transition matrix), state covariance, and process noise
    P = F @ P @ F.T + Q

    return m, P


def batch_filter(
    params: KalmanParams, emissions: jnp.ndarray
) -> PosteriorFilter:

    num_timesteps = len(emissions)

    def step(carry, t):
        m, P = carry

        # TODO: add and update log-likelihood

        m, P = _update(emissions[t], m, params.R, params.H, P)
        m, P = _predict(params.F, params.Q, m, P)

        return (m, P), (m, P)

    log_likelihood = 0.0
    carry = (params.m, params.P)
    _, (ms, Ps) = jax.lax.scan(step, carry, jnp.arange(num_timesteps))

    return PosteriorFilter(ms, Ps)
