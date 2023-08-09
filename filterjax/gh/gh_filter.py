import jax
import jax.numpy as jnp


class GHFilter:

    def __init__(self, x, dx, dt, g, h):
        self.x = x
        self.dx = dx
        self.dt = dt
        self.g = g
        self.h = h

    def update(self, z, g=None, h=None):
        """Performs predict-update step on the measurement 'z' and returns
        the state of 'x' and 'dx' as a tuple.
        """

        if g is None:
            g = self.g
        if h is None:
            h = self.h

        def compute(carry, y):
            x, dx = carry
            x_est = x + (dx * self.dt)
            residual = y - x_est
            dx = dx + h * residual
            x = x_est + g * residual
        
            return (x, dx), (x, dx)

        _, (xs, dxs) = jax.lax.scan(compute, (self.x, self.dx), z)

        return xs, dxs