import jax
from jax import numpy as jnp


def _update(x, P, z, R):
    """Updates mu and variance of the system."""
    x = (P * z + x * R) / (P + R)
    P = 1. / (1. / P + 1. / R)
    
    return x, P


def _predict(x, u, P, Q):
    """Predicts the next state of the system."""
    x += u
    P += Q

    return x, P


def filter_1d(z, x0, P, R, Q, u=0):
    """Implements Kalman filter for 1D systems.

    Parameters
    ----------
    z : array_like
        Measurements of the system.
    x0 : float
        Initial state of the system.
    P : float
        Initial variance of the state of the system.
    R : float
        Variance of the measurement noise.
    Q : float
        Variance of the movement noise.
    u : float, optional
        Movement of the system.

    Returns
    -------
    xs : array_like
        Estimated state of the system.
    Ps : array_like
        Estimated variance of the state of the system.
    """
    num_timesteps = len(z)

    def _step(carry, t):
        y = z[t]
        x, P = carry
        x, P = _predict(x, u, P, Q)
        x, P = _update(x, P, y, R)

        return (x, P), (x, P)

    _, (xs, Ps) = jax.lax.scan(_step, (x0, P), jnp.arange(num_timesteps))

    return xs, Ps
