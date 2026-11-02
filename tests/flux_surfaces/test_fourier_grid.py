"""The separable tensor-grid evaluator (analytic derivatives) used by volume / area / mean cross section area, against the
lax.scan-over-modes reference with autodiff derivatives."""
import jax
import jax.numpy as jnp
import numpy as onp
import pytest

from jax_sbgeom.flux_surfaces import FluxSurface, FluxSurfaceSettings
from jax_sbgeom.flux_surfaces import flux_surfaces_base as fsb
from jax_sbgeom.jax_utils import interpolate_array

jax.config.update("jax_enable_x64", True)


def _scan_interpolated(fs, s, theta, phi):
    def step(vals, i):
        R, Z = vals
        a = fs.modes.mpol_vector[i] * theta - fs.modes.ntor_vector[i] * phi
        return (R + interpolate_array(fs.data.Rmnc[..., i], s) * jnp.cos(a), Z + interpolate_array(fs.data.Zmns[..., i], s) * jnp.sin(a)), None
    s_, th, ph = jnp.broadcast_arrays(s, theta, phi)
    R, Z = jax.lax.scan(step, (jnp.zeros_like(th), jnp.zeros_like(th)), jnp.arange(fs.data.Rmnc.shape[1]))[0]
    return jnp.stack([R, Z, ph], axis=-1)


@pytest.fixture(scope="module")
def fs():
    st = FluxSurfaceSettings(mpol=5, ntor=4, nfp=3)
    n_modes = len(fsb._create_mpol_vector(st))
    rng = onp.random.default_rng(0)
    mv = onp.asarray(fsb._create_mpol_vector(st)); nv = onp.asarray(fsb._create_ntor_vector(st))
    decay = 1.0 / (1.0 + mv ** 2 + (nv / 3) ** 2)
    sc = onp.linspace(0.2, 1.0, 7)[:, None]
    R = sc * decay * rng.normal(size=(7, n_modes)) * 0.3; Z = sc * decay * rng.normal(size=(7, n_modes)) * 0.3
    R[:, (mv == 0) & (nv == 0)] = 6.0; R[:, (mv == 1) & (nv == 0)] = sc * 1.0; Z[:, (mv == 1) & (nv == 0)] = sc * 1.0
    return FluxSurface.from_rmnc_zmns_settings(jnp.asarray(R), jnp.asarray(Z), st, make_normals_point_outwards=False)


@pytest.mark.parametrize("s", [0.0, 0.63, 1.0 / 6.0, 1.0])
def test_grid_evaluator_matches_autodiff(fs, s):
    # tensor-grid (separable) evaluator with analytic derivatives, used by volume / area / mean cross section area
    theta, phi = jnp.linspace(0, 6, 9), jnp.linspace(0, 2, 7)
    th, ph = jnp.meshgrid(theta, phi, indexing="ij")
    r, n = fsb._grid_position_and_normal(fs, s, theta, phi)
    assert jnp.allclose(r, fs.cartesian_position(s, th, ph), atol=1e-12)
    assert jnp.allclose(n, fsb._dx_dphi_cross_dx_dtheta(fs, s, th, ph), rtol=1e-11, atol=1e-11)
    R, Z, R_t, R_p, Z_t, Z_p = fsb._grid_rz_derivatives(fs, s, theta, phi)
    J = jax.vmap(jax.vmap(jax.jacfwd(lambda x: _scan_interpolated(fs, s, x[0], x[1])[:2])))(jnp.stack([th, ph], -1))   # (.., 2 [R,Z], 2 [theta,phi])
    assert jnp.allclose(jnp.stack([R_t, R_p, Z_t, Z_p], -1), J.reshape(J.shape[:2] + (4,)), atol=1e-11)


def test_scalars_match_scan_reference_quadrature(fs):
    # volume / mean cross section area from the scan reference evaluator (autodiff derivatives) on the same grids
    nth, nph = 6 * fs.settings.mpol + 1, 6 * fs.settings.ntor + 1
    theta = jnp.linspace(0, 2 * jnp.pi, nth, endpoint=False); phi = jnp.linspace(0, 2 * jnp.pi / fs.nfp, nph, endpoint=False)
    pts = jnp.stack(jnp.meshgrid(theta, phi, indexing="ij"), -1)

    def ref_volume(s):
        xyz = lambda x: fsb._cylindrical_to_cartesian(_scan_interpolated(fs, s, x[0], x[1]))
        J = jax.vmap(jax.vmap(jax.jacfwd(xyz)))(pts)
        r = jax.vmap(jax.vmap(xyz))(pts)
        f = jnp.einsum('...i,...i->...', r, jnp.cross(J[..., 1], J[..., 0]))
        return jnp.sum(f) * (2 * jnp.pi / nth) * (2 * jnp.pi / fs.nfp / nph) / 3 * fs.nfp

    def ref_mean_area(s):
        rz = lambda x: _scan_interpolated(fs, s, x[0], x[1])[:2]
        J = jax.vmap(jax.vmap(jax.jacfwd(rz)))(pts)
        R = jax.vmap(jax.vmap(rz))(pts)[..., 0]
        return jnp.abs(jnp.mean(R * J[..., 1, 0]) * 2 * jnp.pi)

    for s in [0.7, 1.0 / 6.0, 1.0]:
        assert jnp.allclose(fs.volume(s), ref_volume(s), rtol=1e-12)
        assert jnp.allclose(fs.mean_cross_section_area(s), ref_mean_area(s), rtol=1e-12)
        assert jnp.allclose(fs.aspect_ratio(s), fs.major_radius(s) / fs.minor_radius(s), rtol=1e-12)
    assert jnp.allclose(jax.grad(fs.volume)(0.7), jax.grad(ref_volume)(0.7), rtol=1e-10)
    assert jnp.allclose(jax.grad(fs.mean_cross_section_area)(0.7), jax.grad(ref_mean_area)(0.7), rtol=1e-10)
