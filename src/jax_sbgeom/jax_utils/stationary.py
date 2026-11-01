"""Sample-then-polish: the shared building block of `minimum_distance` and `Coil.bounding_box`."""
from __future__ import annotations

from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import optimistix as optx


def _grad_objective(x, packed):
    """grad of f(x, args), with (f, args) packed into optimistix' `args` (module level: stable identity, so traces are cached)."""
    f, args = packed
    return jax.grad(f)(x, args)


def _root_from(x0, f, args, solver, max_steps):
    return optx.root_find(_grad_objective, solver, x0, args=(f, args), throw=False, max_steps=max_steps)


@eqx.filter_jit
def polish_stationary(f, starts, args=None, *, tol : float = 1e-10, max_steps : int = 12):
    """
    Newton-polish the stationary points of a smooth scalar f(x, args) from each row of `starts` (k, d).
    Sampling is the caller's job (it chooses the starts). Derivatives use optimistix' implicit adjoint.

    `f` should be a module-level function and everything it needs a pytree in `args` (no closures: a fresh function
    object per call defeats the trace cache). `args` may be any pytree (arrays are traced).

    Returns (x, converged): the polished points (k, d) and a boolean (k,). Where Newton did not converge x is the
    last iterate: still a valid point, but derivatives through it are not.
    """
    solver = optx.Newton(rtol=tol, atol=tol)
    sol = eqx.filter_vmap(_root_from, in_axes=(0, None, None, None, None))(starts, f, args, solver, max_steps)
    return sol.value, sol.result == optx.RESULTS.successful


@dataclass(frozen=True)
class Polish:
    """Newton tolerance and step limit of a polished extremum (static, hashable). Passing `None` instead means: sampled only."""
    tol : float
    max_steps : int


def _scaled(x, packed):
    f, args, scale = packed
    return f(x[0], args) / scale


def maximise_periodic(f, args, n : int, polish : Polish | None):
    """
    Maximum over s in [0, 1) of a smooth, 1-periodic scalar f(s, args) (a module-level function, everything it needs in args): the best of n equally spaced samples, polished by
    Newton on f' = 0 if `polish` is given. The polished value replaces the sample only where Newton converged and it is
    larger, so the result is never below the sampled maximum, and its derivatives with respect to args are exact (envelope
    theorem). Newton works on f divided by the spread of the samples, so `polish.tol` does not depend on the size of f.
    """
    v = jax.vmap(f, in_axes=(0, None))(jnp.arange(n) / n, args)
    k = jnp.argmax(v)
    if polish is None:
        return v[k]
    scale = jax.lax.stop_gradient((jnp.max(v) - jnp.min(v))) + 1e-30
    x, ok = polish_stationary(_scaled, jnp.array([[k / n]]), (f, args, scale), tol=polish.tol, max_steps=polish.max_steps)
    polished = f(x[0, 0], args)
    return jnp.where(ok[0] & (polished > v[k]), polished, v[k])
