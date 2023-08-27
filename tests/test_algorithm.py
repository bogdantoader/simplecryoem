import unittest
import numpy as np
import jax.numpy as jnp
from jax.config import config

from simplecryoem.optimization import conjugate_gradient

config.update("jax_enable_x64", True)


class TestAlgorithm(unittest.TestCase):
    def test_conjugate_gradient(self):
        N = 100
        iterations = 1000

        A = jnp.array(np.random.randn(N, N) + 1j * np.random.randn(N, N))
        H = jnp.transpose(jnp.conj(A)) @ A
        H = H + jnp.diag(np.random.randn(N) + 100)

        x = jnp.array(np.random.randn(N))
        b = H @ x
        x0 = jnp.array(np.random.randn(N))

        def op(x):
            return H @ x

        xcg, max_iter = conjugate_gradient(op, b, x0, iterations, verbose=False)

        assert max_iter < iterations - 1
