import jax
from jax import numpy as jnp


class KalmanFilter1d:
    def __init__(self, x0, P, R, Q, u=0):
        """Implements Kalman filter for 1D systems.

        Parameters
        ----------
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
        """
        self.x = x0
        self.P = P
        self.R = R
        self.Q = Q
        self.u = u

    def _update(self, x, P, z, R):
        """
        Updates mu and variance of the system.
        """
        x = (P * z + x * R) / (P + R)
        P = 1.0 / (1.0 / P + 1.0 / R)

        return x, P

    def _predict(self, x, u, P, Q):
        """Predicts the next state of the system."""
        x += u
        P += Q

        return x, P

    def predict_update(self, z):
        num_timesteps = len(z)

        def _step(carry, t):
            # get measurement at time t
            y = z[t]
            x, P = carry
            x, P = self._predict(x, self.u, P, self.Q)
            x, P = self._update(x, P, y, self.R)

            return (x, P), (x, P)

        _, (xs, Ps) = jax.lax.scan(
            _step, (self.x0, self.P), jnp.arange(num_timesteps)
        )

        return xs, Ps


def _update(x, P, z, R):
    """
    Updates mu and variance of the system.
    """
    x = (P * z + x * R) / (P + R)
    P = 1.0 / (1.0 / P + 1.0 / R)

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
        # get measurement at time t
        y = z[t]
        x, P = carry
        x, P = _predict(x, u, P, Q)
        x, P = _update(x, P, y, R)

        return (x, P), (x, P)

    _, (xs, Ps) = jax.lax.scan(_step, (x0, P), jnp.arange(num_timesteps))

    return xs, Ps


class KalmanFilter:
    def __init__(self, dim_x, dim_z, dim_u=0):
        """Implements Kalman filter for general systems.

        Parameters
        ----------
        dim_x : int
            Dimension of the state of the system.
        dim_z : int
            Dimension of the measurements of the system.
        dim_u : int, optional
            Dimension of the movement of the system.
        """

        self.x = jnp.zeros((dim_x, 1))
        self.P = jnp.eye(dim_x)
        self.Q = jnp.eye(dim_x)
        self.u = jnp.zeros((dim_x, 1))
        self.B = 0
        self.F = 0
        self.H = 0
        self.R = jnp.eye(dim_z)

        self.I = jnp.eye(dim_x)

    def update(self, z, R=None, H=None):
        """Updates the state of the system.

        Parameters
        ----------
        Z : array_like
            Measurements for this update.
        R : array_like, optional
            New measurement noise for this update.

        Returns
        -------
        x : array_like
            Estimated state of the system.
        P : array_like
            Estimated variance of the state of the system.
        """
        if R is None:
            R = self.R
        elif jnp.isscalar(R):
            R = jnp.eye(self.dim_z) * R

        if H is None:
            z = jnp.reshape(z, self.dim_z, self.x.ndim)
            H = self.H

        # error (residual) between measurement and prediction
        y = z - jnp.dot(H, self.x)

        PHT = jnp.dot(self.P, H.T)

        return self.x, self.P

    # def predict(self, u=None, B=None, F=None, Q=None):

    # def predict_update(self)
    # """Implents predict-update cycle of Kalman filter in one step.
    # """
