from typing import NamedTuple, Union

import jax
from jax import numpy as jnp

from filterjax.models.params import KalmanParams
from filterjax.inference.filters import batch_filter, PosteriorFilter


class KalmanFilter:
    """Kalman filter for exact Bayesian filtering and smoothing.

    The model is defined as follows:

    # TODO: add model definition
    """

    def __init__(self, state_dim, emission_dim):
        self.state_dim = state_dim
        self.emission_dim = emission_dim

    def initialize(
        self,
        init_state: jnp.ndarray,
        init_transition: jnp.ndarray,
        init_emission: jnp.ndarray,
        init_emission_covariance: jnp.ndarray,
        init_process_covariance: jnp.ndarray,
        init_state_covariance: jnp.ndarray,
        init_control: Union[jnp.ndarray, None] = None,
    ) -> KalmanParams:
        """Sets the initial parameters of the Kalman filter.

        Parameters
        ----------
        init_state : jnp.ndarray
            Initial state mean.
        init_transition : jnp.ndarray
            Initial state transition matrix.
        init_emission : jnp.ndarray
            Initial emission matrix.
        init_emission_covariance : jnp.ndarray
            Initial emission covariance matrix.
        init_process_covariance : jnp.ndarray
            Initial process covariance matrix.
        init_state_covariance : jnp.ndarray
            Initial state covariance matrix.
        init_control : Union[jnp.ndarray, None], optional
            Initial control matrix, by default None

        Returns
        -------
        KalmanParams
            Named tuple of Kalman parameters.
        """

        params = KalmanParams(
            m=init_state,
            F=init_transition,
            H=init_emission,
            R=init_emission_covariance,
            Q=init_process_covariance,
            P=init_state_covariance,
            B=init_control,
        )

        self.check_dims(params)

        return params

    def check_dims(self, params: KalmanParams):
        """In order to perform matrix multiplication in inference, the
        dimensions of the Kalman parameters must be checked with the state
        and emission dimensions.

        Parameters
        ----------
        params : KalmanParams
            Kalman parameters.

        Raises
        ------
        AssertionError
            If the dimensions of the Kalman parameters do not match the state
            and emission dimensions.
        """
        if params.m.shape != (self.state_dim,):
            raise ValueError(
                f"m.shape {params.m.shape} and {self.state_dim,} are incompatible."
            )

        if params.F.shape != (self.state_dim, self.state_dim):
            raise ValueError(
                f"F.shape {params.F.shape} and {self.state_dim, self.state_dim} are incompatible."
            )

        if params.H.shape != (self.emission_dim, self.state_dim):
            raise ValueError(
                f"H.shape {params.H.shape} and {self.emission_dim, self.state_dim} are incompatible."
            )

        if params.R.shape != (self.emission_dim, self.emission_dim):
            raise ValueError(
                f"R.shape {params.R.shape} and {self.emission_dim, self.emission_dim} are incompatible."
            )

        if params.Q.shape != (self.state_dim, self.state_dim):
            raise ValueError(
                f"Q.shape {params.Q.shape} and {self.state_dim, self.state_dim} are incompatible."
            )

        if params.P.shape != (self.state_dim, self.state_dim):
            raise ValueError(
                f"P.shape {params.P.shape} and {self.state_dim, self.state_dim} are incompatible."
            )

        if params.B is not None:
            if params.B.shape != (self.state_dim, self.state_dim):
                raise ValueError(
                    f"B.shape {params.B.shape} and {self.state_dim, self.state_dim} are incompatible."
                )

    def filter(
        self, params: KalmanParams, emissions: jnp.ndarray
    ) -> PosteriorFilter:
        """Given a set (batch) emissions, performs online filtering.

        Parameters
        ----------
        params : KalmanParams
            Kalman parameters.
        emissions : jnp.ndarray
            Emissions (observations).

        Returns
        -------
        PosteriorFilter
            History (state) of the state mean and covariance.
        """
        return batch_filter(params, emissions)
