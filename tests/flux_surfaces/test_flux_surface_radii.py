"""FluxSurface.mean_cross_section_area / minor_radius / major_radius / aspect_ratio: analytic tori, brute force, gradients, VMEC data."""
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as onp
import pytest

import jax_sbgeom as jsb
from jax_sbgeom.flux_surfaces import FluxSurface, FluxSurfaceSettings

jax.config.update("jax_enable_x64", True)

HELIAS5_WOUT = Path(__file__).parent.parent / "data" / "flux_surfaces" / "helias5_wout.h5"


def _surface(R0, a, nfp, extra=None, mpol=1, ntor=0, b=None):
    """R = R0 + a cos(th) + sum extra, Z = b sin(th) + ...  extra = {(m, n): (rmnc, zmns)} (VMEC mode order)."""
    b = a if b is None else b
    st = FluxSurfaceSettings(mpol=mpol, ntor=ntor, nfp=nfp)
    mv = onp.asarray(jsb.flux_surfaces.flux_surfaces_base._create_mpol_vector(st))
    nv = onp.asarray(jsb.flux_surfaces.flux_surfaces_base._create_ntor_vector(st))
    R = onp.zeros(len(mv)); Z = onp.zeros(len(mv))
    R[(mv == 0) & (nv == 0)] = R0
    R[(mv == 1) & (nv == 0)] = a
    Z[(mv == 1) & (nv == 0)] = b
    for (m, n), (r, z) in (extra or {}).items():
        k = onp.where((mv == m) & (nv == n * nfp))[0][0]
        R[k] += r; Z[k] += z
    return FluxSurface.from_rmnc_zmns_settings(jnp.asarray(R)[None], jnp.asarray(Z)[None], st, make_normals_point_outwards=False)


STELL = {(1, 1): (0.35, 0.35), (1, -1): (0.0, 0.0), (2, 0): (0.1, -0.05), (2, 1): (0.04, 0.02)}


def _stellarator(nfp=5):
    return _surface(6.0, 1.0, nfp, extra=STELL, mpol=2, ntor=1)


@pytest.mark.parametrize("nfp", [1, 3, 5])
def test_torus_analytic(nfp):
    R0, a = 5.5, 1.3
    fs = _surface(R0, a, nfp)
    assert jnp.allclose(fs.mean_cross_section_area(1.0), jnp.pi * a ** 2, rtol=1e-13)
    assert jnp.allclose(fs.minor_radius(1.0), a, rtol=1e-13)
    assert jnp.allclose(fs.major_radius(1.0), R0, rtol=1e-12)
    assert jnp.allclose(fs.aspect_ratio(1.0), R0 / a, rtol=1e-12)


@pytest.mark.parametrize("nfp", [1, 4])
def test_elliptical_torus(nfp):
    R0, a, b = 7.0, 1.5, 0.8
    fs = _surface(R0, a, nfp, b=b)
    assert jnp.allclose(fs.mean_cross_section_area(1.0), jnp.pi * a * b, rtol=1e-13)
    assert jnp.allclose(fs.minor_radius(1.0), jnp.sqrt(a * b), rtol=1e-13)
    assert jnp.allclose(fs.major_radius(1.0), R0, rtol=1e-12)    # V = 2 pi^2 R0 a b


def _brute_force_mean_area(fs, n_theta=4000, n_phi=64):
    th = jnp.linspace(0, 2 * jnp.pi, n_theta, endpoint=False)
    areas = []
    for ph in onp.linspace(0, 2 * onp.pi / fs.nfp, n_phi, endpoint=False):
        RZ = onp.asarray(fs.cylindrical_position(1.0, th, jnp.full_like(th, ph)))[:, :2]
        R, Z = RZ[:, 0], RZ[:, 1]
        areas.append(abs(0.5 * onp.sum(R * onp.roll(Z, -1) - onp.roll(R, -1) * Z)))   # shoelace
    return float(onp.mean(areas))


def test_stellarator_vs_brute_force_shoelace():
    fs = _stellarator()
    ref = _brute_force_mean_area(fs)
    assert abs(float(fs.mean_cross_section_area(1.0)) / ref - 1) < 1e-6    # shoelace polygon error ~ h^2
    assert abs(float(fs.mean_cross_section_area(1.0)) / _brute_force_mean_area(fs, 8000, 128) - 1) < 3e-7


def test_quadrature_is_exact_with_nyquist_sampling():
    # the integrand is band limited; the default sampling must equal a heavily oversampled rule
    fs = _stellarator()
    th = jnp.linspace(0, 2 * jnp.pi, 97, endpoint=False)
    ph = jnp.linspace(0, 2 * jnp.pi / fs.nfp, 37, endpoint=False)
    tt, pp = jnp.meshgrid(th, ph, indexing="ij")
    rz = lambda t, p: fs.cylindrical_position(1.0, t, p)[:2]
    f = jax.vmap(jax.vmap(lambda t, p: rz(t, p)[0] * jax.jacfwd(rz)(t, p)[1]))(tt, pp)
    assert jnp.allclose(fs.mean_cross_section_area(1.0), jnp.mean(f) * 2 * jnp.pi, rtol=1e-13)


def test_gradients_vs_finite_differences():
    fs = _stellarator()
    d = jax.random.normal(jax.random.PRNGKey(1), fs.data.Rmnc.shape) * 1e-2
    for name in ("mean_cross_section_area", "minor_radius", "major_radius", "aspect_ratio"):
        def f(x):
            new = FluxSurface.from_rmnc_zmns_settings(fs.data.Rmnc + x * d, fs.data.Zmns, fs.settings, make_normals_point_outwards=False)
            return getattr(new, name)(1.0)
        g = jax.grad(f)(0.0)
        eps = 1e-6
        fd = (f(eps) - f(-eps)) / (2 * eps)
        assert jnp.allclose(g, fd, rtol=1e-6), name


@pytest.mark.skipif(not HELIAS5_WOUT.exists(), reason="helias5_wout.h5 not available")
def test_helias5_vmec_regression():
    fs = FluxSurface.from_hdf5(str(HELIAS5_WOUT))
    assert jnp.allclose(fs.minor_radius(1.0), 1.8020693189284707, rtol=1e-8)       # VMEC Aminor_p
    assert jnp.allclose(fs.major_radius(1.0), 22.193094912867643, rtol=1e-8)
    assert jnp.allclose(fs.aspect_ratio(1.0), 12.315339193535513, rtol=1e-8)
