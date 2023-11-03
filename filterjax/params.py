from typing import NamedTuple

import jax.numpy as jnp


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