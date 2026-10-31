"""FluxSurface.volume / .area: analytic tori, nfp consistency, independent mesh, convergence, gradients."""
import jax
import jax.numpy as jnp
import numpy as onp
import pytest

import jax_sbgeom as jsb
from jax_sbgeom.flux_surfaces import FluxSurface, FluxSurfaceSettings, ToroidalExtent

jax.config.update("jax_enable_x64", True)


def _surface(R0, a, nfp, extra=None, mpol=1, ntor=0):
    """R = R0 + a cos(th) + sum extra, Z = a sin(th) + ...  extra = {(m, n): (rmnc, zmns)} (VMEC mode order)."""
    st = FluxSurfaceSettings(mpol=mpol, ntor=ntor, nfp=nfp)
    mv = onp.asarray(jsb.flux_surfaces.flux_surfaces_base._create_mpol_vector(st))
    nv = onp.asarray(jsb.flux_surfaces.flux_surfaces_base._create_ntor_vector(st))
    R = onp.zeros(len(mv)); Z = onp.zeros(len(mv))
    R[(mv == 0) & (nv == 0)] = R0
    R[(mv == 1) & (nv == 0)] = a
    Z[(mv == 1) & (nv == 0)] = a
    for (m, n), (r, z) in (extra or {}).items():
        k = onp.where((mv == m) & (nv == n * nfp))[0][0]
        R[k] += r; Z[k] += z
    return FluxSurface.from_rmnc_zmns_settings(jnp.asarray(R)[None], jnp.asarray(Z)[None], st, make_normals_point_outwards=False)


@pytest.mark.parametrize("nfp", [1, 3, 5])
def test_torus_analytic_and_nfp_consistent(nfp):
    R0, a = 5.5, 1.3
    fs = _surface(R0, a, nfp)
    assert jnp.allclose(fs.volume(1.0), 2 * jnp.pi ** 2 * R0 * a ** 2, rtol=1e-12)
    assert jnp.allclose(fs.area(1.0), 4 * jnp.pi ** 2 * R0 * a, rtol=1e-12)


STELL = {(1, 1): (0.35, 0.35), (1, -1): (0.0, 0.0), (2, 0): (0.1, -0.05), (2, 1): (0.04, 0.02)}


def _stellarator(nfp=5):
    return _surface(6.0, 1.0, nfp, extra=STELL, mpol=2, ntor=1)


def test_stellarator_vs_independent_mesh():
    fs = _stellarator()
    pos, tri = jsb.flux_surfaces.mesh_surface(fs, 1.0, ToroidalExtent.full(), 200, 600)
    pos, tri = onp.asarray(pos), onp.asarray(tri)
    a, b, c = pos[tri[:, 0]], pos[tri[:, 1]], pos[tri[:, 2]]
    mesh_area = 0.5 * onp.linalg.norm(onp.cross(b - a, c - a), axis=1).sum()
    mesh_vol = abs(onp.einsum("ij,ij->i", a, onp.cross(b - a, c - a)).sum() / 6.0)
    assert abs(float(fs.area(1.0)) / mesh_area - 1) < 5e-4      # mesh is a lower bound with O(h^2) error
    assert abs(float(fs.volume(1.0)) / mesh_vol - 1) < 5e-4


def test_quadrature_converges_and_nfp_independent():
    fs5 = _stellarator(5)
    a2, a4, a6 = fs5.area(1.0, 2), fs5.area(1.0), fs5.area(1.0, 6)
    assert 1e-9 < abs(a2 / a6 - 1) < 1e-6      # geometric convergence in the sampling
    assert abs(a4 / a6 - 1) < 1e-11            # the default
    # the same geometry written with nfp=1 (modes scaled by 5 in n) is the same surface
    st = FluxSurfaceSettings(mpol=2, ntor=5, nfp=1)
    mv = onp.asarray(jsb.flux_surfaces.flux_surfaces_base._create_mpol_vector(st))
    nv = onp.asarray(jsb.flux_surfaces.flux_surfaces_base._create_ntor_vector(st))
    st5 = fs5.settings
    mv5 = onp.asarray(jsb.flux_surfaces.flux_surfaces_base._create_mpol_vector(st5))
    nv5 = onp.asarray(jsb.flux_surfaces.flux_surfaces_base._create_ntor_vector(st5))
    R = onp.zeros(len(mv)); Z = onp.zeros(len(mv))
    for k5 in range(len(mv5)):
        k = onp.where((mv == mv5[k5]) & (nv == nv5[k5]))[0][0]
        R[k] = fs5.data.Rmnc[0, k5]; Z[k] = fs5.data.Zmns[0, k5]
    fs1 = FluxSurface.from_rmnc_zmns_settings(jnp.asarray(R)[None], jnp.asarray(Z)[None], st, make_normals_point_outwards=False)
    assert jnp.allclose(fs1.volume(1.0), fs5.volume(1.0), rtol=1e-10)
    assert jnp.allclose(fs1.area(1.0, 6), a6, rtol=1e-9)


def test_gradients_vs_finite_differences():
    fs = _stellarator()
    d = jax.random.normal(jax.random.PRNGKey(1), fs.data.Rmnc.shape) * 1e-2
    for name in ("volume", "area"):
        def f(x):
            new = FluxSurface.from_rmnc_zmns_settings(fs.data.Rmnc + x * d, fs.data.Zmns, fs.settings, make_normals_point_outwards=False)
            return getattr(new, name)(1.0)
        g = jax.grad(f)(0.0)
        h = 1e-5
        fd = (f(h) - f(-h)) / (2 * h)
        assert jnp.allclose(g, fd, rtol=1e-7), name
