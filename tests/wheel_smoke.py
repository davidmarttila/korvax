"""Exercise the installed CPU extension with a first-order impulse response."""

import jax.numpy as jnp
import numpy as np

from korvax.filter.ltv import allpole


def test_allpole_impulse():
    x = jnp.array([1.0, 0.0, 0.0, 0.0])
    a = jnp.full((4, 1), -0.5)
    np.testing.assert_allclose(allpole(x, a), [1.0, 0.5, 0.25, 0.125])


if __name__ == "__main__":
    test_allpole_impulse()
