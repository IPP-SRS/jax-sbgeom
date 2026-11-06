from dataclasses import dataclass, field
from functools import partial
import jax
import jax.numpy as jnp
from .base_coil import Coil
from .coilset import CoilSet
from .discrete_coil import DiscreteCoil
from jax_sbgeom.jax_utils.splines import periodic_bspline


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class SplineCoil(Coil):
    '''
    Coil defined by a periodic B-spline of degree k with n_ctrl control coefficients, x(s) = sum_i coeffs[i] B_i(s), s in [0, 1).
    Each coefficient only influences the curve on k + 1 of the n_ctrl knot intervals (local shape control).
    The coefficients are the spline coefficients, not interpolated points; for k = 3 they lie close to the curve.

    For creating SplineCoil objects from discrete positions, use `convert_to_spline_coil` or `convert_to_spline_coilset`.

    Parameters
    ----------
    coeffs : jnp.ndarray
        Control coefficients [n_ctrl, 3]. Requires n_ctrl > k. A CoilSet batches these, as for the other coil types.
    k : int
        Spline degree (static pytree metadata, not a leaf).
    '''
    coeffs : jnp.ndarray
    k      : int = field(default=3, metadata=dict(static=True))

    def position(self, s):
        return _spline_position(self, s)

    def tangent(self, s):
        return _spline_tangent(self, s)

    def normal(self, s):
        return _spline_normal(self, s)

    def centre(self):
        return jnp.mean(self.coeffs, axis=-2)

    def reverse_parametrisation(self):
        # x(-s): basis i is symmetric about (i - (k-1)/2) / n, so the new coefficient m is the old one at (k - 1 - m) mod n
        n = self.coeffs.shape[-2]
        return SplineCoil(coeffs=self.coeffs[..., (self.k - 1 - jnp.arange(n)) % n, :], k=self.k)


@jax.jit
def _spline_position(coil : SplineCoil, s):
    '''Position at s of any shape -> s.shape + (3,).'''
    s = jnp.asarray(s, dtype=float)
    position = jax.vmap(lambda c: periodic_bspline(s.ravel(), c, coil.k, 1.0), out_axes=-1)(coil.coeffs.T)
    return position.reshape(s.shape + (3,))


@jax.jit
def _spline_tangent(coil : SplineCoil, s):
    '''Unit tangent at s of any shape -> s.shape + (3,).'''
    s = jnp.asarray(s, dtype=float)
    derivative = jax.jvp(lambda x: _spline_position(coil, x), (s,), (jnp.ones_like(s),))[1]
    return derivative / jnp.linalg.norm(derivative, axis=-1, keepdims=True)


@jax.jit
def _spline_normal(coil : SplineCoil, s):
    '''Unit normal (Frenet) at s of any shape -> s.shape + (3,).'''
    s = jnp.asarray(s, dtype=float)
    tangent_derivative = jax.jvp(lambda x: _spline_tangent(coil, x), (s,), (jnp.ones_like(s),))[1]
    return tangent_derivative / jnp.linalg.norm(tangent_derivative, axis=-1, keepdims=True)


def _fit_coefficients(positions, n_ctrl : int, k : int):
    '''Least-squares coefficients [..., n_ctrl, 3] of closed-curve samples [..., M, 3] at s = j / M. All curves share one basis.'''
    M = positions.shape[-2]
    s = jnp.arange(M) / M
    basis = jax.vmap(lambda c: periodic_bspline(s, c, k, 1.0))(jnp.eye(n_ctrl)).T    # [M, n_ctrl]
    return jnp.linalg.pinv(basis) @ positions


@partial(jax.jit, static_argnums=(1, 2))
def convert_to_spline_coil(coil : DiscreteCoil, n_ctrl : int, k : int = 3):
    '''Convert a DiscreteCoil to a SplineCoil of degree k by a least-squares fit of n_ctrl coefficients to its positions.'''
    return SplineCoil(coeffs=_fit_coefficients(coil.positions, n_ctrl, k), k=k)


@partial(jax.jit, static_argnums=(1, 2))
def convert_to_spline_coilset(coilset : CoilSet, n_ctrl : int, k : int = 3):
    '''Convert a CoilSet of DiscreteCoils to a CoilSet of SplineCoils of degree k, as `convert_to_spline_coil`.'''
    return CoilSet(SplineCoil(coeffs=_fit_coefficients(coilset.coils.positions, n_ctrl, k), k=k))
