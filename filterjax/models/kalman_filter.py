from typing import NamedTuple, Union

import jax
from jax import numpy as jnp

from filterjax.params import KalmanParams
from filterjax.inference.filters import batch_filter


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

        # TODO: add a method that checks dims of params before initializing
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
        assert params.m.shape == (self.state_dim,)
        assert params.F.shape == (self.state_dim, self.state_dim)
        assert params.H.shape == (self.emission_dim, self.state_dim)
        assert params.R.shape == (self.emission_dim, self.emission_dim)
        assert params.Q.shape == (self.state_dim, self.state_dim)
        assert params.P.shape == (self.state_dim, self.state_dim)

        if params.B is not None:
            assert params.B.shape == (self.state_dim, self.state_dim)

    def filter(self, params: KalmanParams, emissions: jnp.ndarray):
        return batch_filter(params, emissions)

