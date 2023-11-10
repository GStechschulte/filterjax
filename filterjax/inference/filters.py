from typing import NamedTuple

import jax
import jax.numpy as jnp

from jax.scipy.stats import multivariate_normal

from filterjax.models.params import KalmanParams


class PosteriorFilter(NamedTuple):
    """Posterior state estimate and covariance matrix.

    Parameters
    ----------
    marginal_log_likelihood : jnp.ndarray
        Marginal log-likelihood of the observations.
    m : jnp.ndarray
        Mean of the posterior state estimate.
    P : jnp.ndarray
        Covariance matrix of the posterior state estimate.
    """

    marginal_log_likelihood: jnp.ndarray
    mean: jnp.ndarray
    covariance: jnp.ndarray


def _update(y, m, R, H, P):
    """Update step conditions on a new linear Gaussian observation and computes
    the Kalman gain to compute the posterior state mean and covariance.

    $$error := y - \mathbf{H}m$$
    $$\mathbf{S} := \mathbf{H}\mathbf{P}\mathbf{H}^T + R$$
    $$\mathbf{K} := \mathbf{P}\mathbf{H}^T\mathbf{S}^{-1}$$
    $$m := m + \mathbf{K}y$$
    $$\mathbf{P} := (\mathbf{I} - \mathbf{K}\mathbf{H})\mathbf{P}$$

    where $S$ is the innovation covariance and $K$ is the Kalman gain. Instead
    of directly inverting $S$, a Cholesky decomposition is used within
    `jax.scipy.linalg.solve` to solve the linear equation `a @ x == b`. Otherwise,
    $K$ could be computed as follows:

    .. code-block:: python

            K = P @ H.T @ jnp.linalg.inv(S)

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
    # compute innovation covariance
    S = H @ P @ H.T + R
    # Kalman gain (scaling factor)
    K = jax.scipy.linalg.solve(S, H @ P, assume_a="pos").T
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

    # predict next state (prior) covariance using the state transition matrix,
    # state covariance, and process noise
    P = F @ P @ F.T + Q

    return m, P


def _log_likelihood(y, m, R, H, P):
    """Computes the log-likelihood of a new linear Gaussian observation given
    the current state mean and covariance.

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
    log_likelihood : float
        Log-likelihood of the observation.
    """
    # map state space to observation space
    m = H @ m
    if R.ndim == 2:
        # map state covariance to observation covariance
        S = R + H @ P @ H.T
        return multivariate_normal.logpdf(y, m, S)

    # TODO: use low rank MVN if R.ndim > 2
    return NotImplementedError(
        f"Computing log-likelihood when R.ndim = {R.ndim} is not supported."
    )


def batch_filter(
    params: KalmanParams, emissions: jnp.ndarray
) -> PosteriorFilter:
    """Performs filtering on a batch of sequential emissions.

    Parameters
    ----------
    params : KalmanParams
        Kalman filter parameters.
    emissions : jnp.ndarray
        Batch of emissions (measurements).

    Returns
    -------
    PosteriorFilter
        Posterior state estimate of mean and covariance matrix.
    """

    num_timesteps = len(emissions)

    def step(carry, t):
        ll, m, P = carry

        m, P = _update(emissions[t], m, params.R, params.H, P)
        m, P = _predict(params.F, params.Q, m, P)
        ll += _log_likelihood(emissions[t], m, params.R, params.H, P)
        
        return (ll, m, P), (m, P)

    ll = 0.0
    carry = (ll, params.m, params.P)
    (ll, _, _), (ms, Ps) = jax.lax.scan(step, carry, jnp.arange(num_timesteps))

    return PosteriorFilter(ll, ms, Ps)
