"""jax_sbgeom.distance: coil-surface and coil-coil minimum distances."""
import jax
import jax.numpy as jnp
import numpy as onp
import pytest

import jax_sbgeom as jsb
from jax_sbgeom.coils import FourierCoil
from jax_sbgeom.distance import minimum_distance, minimum_distance_between_curves
from jax_sbgeom.flux_surfaces import FluxSurface, FluxSurfaceSettings

jax.config.update("jax_enable_x64", True)

R0, A = 5.0, 1.2


def _torus(nfp=1):
    st = FluxSurfaceSettings(mpol=1, ntor=0, nfp=nfp)
    return FluxSurface.from_rmnc_zmns_settings(jnp.array([[R0, A]]), jnp.array([[0.0, A]]), st, make_normals_point_outwards=False)


def _stellarator():
    st = FluxSurfaceSettings(mpol=2, ntor=1, nfp=5)
    mv = onp.asarray(jsb.flux_surfaces.flux_surfaces_base._create_mpol_vector(st))
    nv = onp.asarray(jsb.flux_surfaces.flux_surfaces_base._create_ntor_vector(st))
    R = onp.zeros(len(mv)); Z = onp.zeros(len(mv))
    R[(mv == 0) & (nv == 0)] = 6.0; R[(mv == 1) & (nv == 0)] = 1.0; Z[(mv == 1) & (nv == 0)] = 1.0
    k = onp.where((mv == 1) & (nv == 5))[0][0]; R[k] = 0.3; Z[k] = 0.3
    return FluxSurface.from_rmnc_zmns_settings(jnp.asarray(R)[None], jnp.asarray(Z)[None], st, make_normals_point_outwards=False)


def _ring(radius, z=0.0, tilt=0.0):
    c = jnp.array([[radius, 0, 0]], dtype=float); s = jnp.array([[0, radius * jnp.cos(tilt), radius * jnp.sin(tilt)]], dtype=float)
    return FourierCoil(fourier_cos=c, fourier_sin=s, centre_i=jnp.array([0.0, 0.0, z]))


def _wavy_coil(scale=1.0):
    k = jnp.arange(1, 4)
    cos = jnp.stack([9.0 * (k == 1) + 0.3 / k, 0.4 / k ** 2, 0.8 * (k == 1) + 0.2 / k ** 2], -1) * scale
    sin = jnp.stack([0.2 / k ** 2, 9.0 * (k == 1) + 0.3 / k, 0.5 / k], -1) * scale
    return FourierCoil(cos, sin, jnp.array([0.0, 0.0, 0.5]))


def _xz_ring(centre_x, radius):
    """circle in the xz-plane (not coaxial with the torus, so the minimum is unique)"""
    return FourierCoil(fourier_cos=jnp.array([[radius, 0, 0]], dtype=float), fourier_sin=jnp.array([[0, 0, radius]], dtype=float),
                       centre_i=jnp.array([centre_x, 0.0, 0.0]))


@pytest.mark.parametrize("nfp", [1, 3])
def test_xz_ring_outside_torus_analytic(nfp):
    # nearest torus point to the ring is (R0 + A, 0, 0); nearest ring point is (X - rho, 0, 0)
    X, rho = R0 + A + 3.0, 1.0
    r = minimum_distance(_xz_ring(X, rho), _torus(nfp))
    assert r.converged and jnp.allclose(r.distance, X - rho - (R0 + A), rtol=1e-10)
    f = lambda x_: minimum_distance(_xz_ring(x_, rho), _torus(nfp)).distance
    assert jnp.allclose(jax.grad(f)(X), 1.0, rtol=1e-8)


def test_against_brute_force_and_gradient():
    surf, coil = _stellarator(), _wavy_coil()
    r = minimum_distance(coil, surf)
    assert r.converged
    s = jnp.arange(4000) / 4000
    th, ph = jnp.meshgrid(jnp.linspace(0, 2 * jnp.pi, 300, endpoint=False), jnp.linspace(0, 2 * jnp.pi, 600, endpoint=False))
    ps = surf.cartesian_position(1.0, th.ravel(), ph.ravel())
    pc = coil.position(s)
    brute = jnp.sqrt(jnp.min(jnp.sum((pc[::4, None] - ps[None]) ** 2, -1)))
    assert 0 <= brute - r.distance < 3e-3              # polished value lies below the sampled one
    # gradient wrt the coil coefficients vs finite differences
    f = lambda c: minimum_distance(FourierCoil(c, coil.fourier_sin, coil.centre_i), surf).distance
    g = jax.grad(f)(coil.fourier_cos)
    d = jax.random.normal(jax.random.PRNGKey(0), coil.fourier_cos.shape)
    h = 1e-6
    assert jnp.allclose(jnp.sum(g * d), (f(coil.fourier_cos + h * d) - f(coil.fourier_cos - h * d)) / (2 * h), rtol=1e-6)


def test_warm_start_and_flags():
    surf, coil = _stellarator(), _wavy_coil()
    r = minimum_distance(coil, surf)
    r2 = minimum_distance(coil, surf, x0=r.x)
    assert r2.converged and jnp.allclose(r2.distance, r.distance, rtol=1e-12)
    bad = minimum_distance(coil, surf, max_steps=1, x0=jnp.array([0.0, 0.0, 0.0]))
    assert not bad.converged                           # reports non-convergence instead of hiding it


def test_jit_vmap():
    surf = _stellarator()
    cos = jnp.stack([_wavy_coil(1.0).fourier_cos, _wavy_coil(1.05).fourier_cos])
    sin = jnp.stack([_wavy_coil(1.0).fourier_sin, _wavy_coil(1.05).fourier_sin])
    f = jax.jit(jax.vmap(lambda c, s_: minimum_distance(FourierCoil(c, s_, jnp.array([0.0, 0.0, 0.5])), surf).distance))
    d = f(cos, sin)
    assert d.shape == (2,) and jnp.all(jnp.isfinite(d)) and d[0] != d[1]


def test_coil_coil_analytic_and_degenerate_case_is_reported():
    # ring in the xy-plane (radius r1) and a ring in the xz-plane: unique nearest points on the x axis
    r1, X, rho = 4.0, 7.0, 1.0
    r = minimum_distance_between_curves(_ring(r1), _xz_ring(X, rho))
    assert r.converged and jnp.allclose(r.distance, X - rho - r1, rtol=1e-10)
    f = lambda x_: minimum_distance_between_curves(_ring(r1), _xz_ring(x_, rho)).distance
    assert jnp.allclose(jax.grad(f)(X), 1.0, rtol=1e-8)
    # coaxial rings: the minimum is a continuum (singular Hessian) -> must not be reported as converged
    deg = minimum_distance_between_curves(_ring(4.0), _ring(5.5, z=0.8))
    assert not deg.converged
