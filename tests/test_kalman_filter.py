import jax
import jax.numpy as jnp
import pytest

from filterjax import KalmanFilter


## Ideas ##
# - test against known results (simulated data)
# - consistency checks (e.g. P is positive semi-definite)?
# - test ove a range of initial design parameters


@pytest.fixture(scope="module")
def sensing_data():
    """
    Simulates a 1D tracking problem using two states: position and velocity.
    Sensor noise is also added to the simulated data.
    """
    key = jax.random.PRNGKey(0)
    movement = 1.
    sensor_error = jnp.sqrt(10.)
    pos = (0, 500)
    timesteps = 50
    
    x = jnp.array(pos[0])
    velocity = jnp.array(movement)
    noise = jnp.array(sensor_error)

    emissions = []
    for _ in range(timesteps):
        x += velocity
        emissions.append(x)
    
    emissions = jnp.array(emissions) + jax.random.normal(key, (timesteps,)) * noise

    return emissions


def test_one_dimensional_tracking(sensing_data):

    state_dim, emission_dim = 2, 1

    def Q_DWPA(dt=1., sigma=1.):
        Q = jnp.array(
            [
                [.25*dt**4, .5*dt**3], 
                [.5*dt**3, dt**2]
            ],
                dtype=float
        )

        return Q * sigma

    F = jnp.array([[1, 1], [0, 1]])
    H = jnp.array([[1, 0]])
    Q = Q_DWPA(sigma=0.)
    R = jnp.eye(1) * 5
    P = jnp.eye(2) * 500.
    m = jnp.array([0., 0.])

    kf = KalmanFilter(state_dim=2, emission_dim=1)
    params = kf.initialize(m, F, H, R, Q, P)
    posterior = kf.filter(params, sensing_data)