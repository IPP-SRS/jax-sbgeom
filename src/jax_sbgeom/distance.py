"""
Minimum distance between smooth objects: a coil and a flux surface (`minimum_distance`), or two coils
(`minimum_distance_between_curves`).

Method: sample both objects coarsely, take the `n_starts` samples of the first object that are closest
to the second, polish each by a Newton root-find on the gradient of 0.5 d^2 (optimistix) and return the
smallest distance. Derivatives use optimistix' implicit adjoint: exact (first and higher order) and nothing
is differentiated through the sampling or the iterations (at first order: the envelope theorem).

Limits, stated plainly:

* The sampling only has to put the wanted minimum in the basin of a start. If two minima are nearly equally
  deep the polished value can still come from the wrong one, by at most the sampling error (O(h^2 curvature));
  refine `n_curve` / `n_theta` / `n_phi` or raise `n_starts` if that matters.
* `converged` reports whether Newton converged for the returned minimum. If it is False the value is still an
  upper bound on the distance (any pair of points is at least as far apart as the minimum), but derivatives
  are not valid.
* A continuum of minima (e.g. a ring coaxial with an axisymmetric surface) makes the Hessian singular; the result
  is NaN with converged=False.
* The result is piecewise smooth; smoothing across several coils (a soft minimum) is the caller's job.
* Warm start: pass `x0` (e.g. the previous `result.x`) to skip the sampling.

Curve parameters are in radians, t = 2 pi s with s in [0, 1) the coil parametrisation. The curve must be smooth
(FourierCoil, a spline coil); a piecewise-linear DiscreteCoil is not suitable.
"""
from __future__ import annotations

from typing import NamedTuple, Optional

import jax
import jax.numpy as jnp
import optimistix as optx

_TWO_PI = 2.0 * jnp.pi


class MinimumDistance(NamedTuple):
    distance : jnp.ndarray     # minimum distance [m]; differentiable
    x         : jnp.ndarray    # minimiser (no gradient): (t, theta, phi) or (t_a, t_b) [rad], wrapped to [0, 2 pi)
    converged : jnp.ndarray    # bool: Newton converged for the returned minimiser


def _solve(f, starts, args, tol, max_steps):
    solver = optx.Newton(rtol=tol, atol=tol)
    sol = jax.vmap(lambda x0: optx.root_find(jax.grad(f), solver, x0, args=args, throw=False, max_steps=max_steps))(starts)
    d = jax.vmap(lambda x: jnp.sqrt(2.0 * f(x, args) + 1e-300))(sol.value)
    ok = sol.result == optx.RESULTS.successful
    best = jnp.argmin(jnp.where(ok, d, jnp.inf))
    best = jnp.where(ok.any(), best, jnp.argmin(d))
    return MinimumDistance(d[best], jax.lax.stop_gradient(jnp.mod(sol.value[best], _TWO_PI)), ok[best])


def _closest_samples(points_a, points_b, k):
    """Indices of the k points of `a` closest to `b`, and the index of the nearest `b` point for each."""
    a, b = jax.lax.stop_gradient(points_a), jax.lax.stop_gradient(points_b)
    d2 = jnp.sum((a[:, None, :] - b[None, :, :]) ** 2, axis=-1)
    top = jnp.argsort(jnp.min(d2, axis=1))[:k]
    return top, jnp.argmin(d2[top], axis=1)


def _curve_surface_f(x, args):
    curve, surface, s = args
    return 0.5 * jnp.sum((curve.position(x[0] / _TWO_PI) - surface.cartesian_position(s, x[1], x[2])) ** 2)


def minimum_distance(curve, surface, *, s : float = 1.0, n_curve : int = 256, n_theta : int = 60, n_phi : int = 100,
                     n_starts : int = 2, x0 : Optional[jnp.ndarray] = None, tol : float = 1e-10,
                     max_steps : int = 12) -> MinimumDistance:
    """
    Minimum distance between a coil and the flux surface at label `s` (full torus: theta, phi in [0, 2 pi)).

    Parameters
    ----------
    curve : Coil (smooth, e.g. FourierCoil)
    surface : FluxSurface
    s : surface label (1.0: last closed flux surface of a multi-surface description)
    n_curve, n_theta, n_phi : coarse sampling used to find the starts
    n_starts : number of starts polished by Newton
    x0 : (3,) or (k, 3) warm start (t, theta, phi) [rad]; skips the sampling
    tol, max_steps : Newton tolerance on the gradient and step limit
    """
    args = (curve, surface, s)
    if x0 is None:
        th, ph = jnp.meshgrid(jnp.linspace(0, _TWO_PI, n_theta, endpoint=False),
                              jnp.linspace(0, _TWO_PI, n_phi, endpoint=False))
        tp = jnp.stack([th.ravel(), ph.ravel()], axis=-1)
        top, nearest = _closest_samples(curve.position(jnp.arange(n_curve) / n_curve),
                                        surface.cartesian_position(s, tp[:, 0], tp[:, 1]), n_starts)
        starts = jnp.concatenate([(_TWO_PI * top / n_curve)[:, None], tp[nearest]], axis=1)
    else:
        starts = jnp.atleast_2d(jnp.asarray(x0, dtype=float))
    return _solve(_curve_surface_f, starts, args, tol, max_steps)


def _curve_curve_f(x, args):
    a, b = args
    return 0.5 * jnp.sum((a.position(x[0] / _TWO_PI) - b.position(x[1] / _TWO_PI)) ** 2)


def minimum_distance_between_curves(curve_a, curve_b, *, n_curve : int = 256, n_starts : int = 2,
                                    x0 : Optional[jnp.ndarray] = None, tol : float = 1e-10,
                                    max_steps : int = 12) -> MinimumDistance:
    """
    Minimum distance between two different closed curves; x = (t_a, t_b) [rad]. Arguments as for
    `minimum_distance`. Do not pass a curve with itself (the minimum is the trivial zero).
    """
    args = (curve_a, curve_b)
    if x0 is None:
        t = jnp.arange(n_curve) / n_curve
        top, nearest = _closest_samples(curve_a.position(t), curve_b.position(t), n_starts)
        starts = jnp.stack([_TWO_PI * top / n_curve, _TWO_PI * nearest / n_curve], axis=1)
    else:
        starts = jnp.atleast_2d(jnp.asarray(x0, dtype=float))
    return _solve(_curve_curve_f, starts, args, tol, max_steps)
