import jax
import jax.numpy as jnp


# TODO: This shouldn't be a class? There is no state here?
#       Although, we should be able to remember the state of the system, i.e., 
#       the params. of the system at time `t` since it can be used for
#       online prediction.
class GHFilter:

    def __init__(self, x, dx, dt, g, h):
        self.x = x
        self.dx = dx
        self.dt = dt
        self.g = g
        self.h = h

    def update(self, z, g=None, h=None):
        """Performs predict-update step on the measurement 'z' and returns
        the state of 'x' and 'dx'.
        """

        if g is None:
            g = self.g
        if h is None:
            h = self.h

        def batch(carry, y):
            x, dx = carry
            x_est = x + (dx * self.dt)
            residual = y - x_est
            dx = dx + h * residual
            x = x_est + g * residual
        
            return (x, dx), (x, dx)

        _, (xs, dxs) = jax.lax.scan(batch, (self.x, self.dx), z)

        return xs, dxs
