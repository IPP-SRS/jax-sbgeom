"""Sample-then-polish: the shared building block of `minimum_distance` and `Coil.bounding_box`."""
from __future__ import annotations

import jax
import jax.numpy as jnp
import optimistix as optx


def polish_stationary(f, starts, args=None, *, tol : float = 1e-10, max_steps : int = 12):
    """
    Newton-polish the stationary points of a smooth scalar f(x, args) from each row of `starts` (k, d).
    Sampling is the caller's job (it chooses the starts). Derivatives use optimistix' implicit adjoint, so nothing is
    differentiated through the iterations (first order: the envelope theorem).

    Returns (x, converged): the polished points (k, d) and a boolean (k,). Where Newton did not converge x is the
    last iterate: still a valid point, but derivatives through it are not.
    """
    solver = optx.Newton(rtol=tol, atol=tol)
    sol = jax.vmap(lambda x0: optx.root_find(jax.grad(f), solver, x0, args=args, throw=False, max_steps=max_steps))(starts)
    return sol.value, sol.result == optx.RESULTS.successful
