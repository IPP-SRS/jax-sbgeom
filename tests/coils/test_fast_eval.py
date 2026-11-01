"""The einsum Fourier evaluator against the original lax.scan version (kept here as the reference), the chunked
Biot-Savart against the all-at-once form, and the precomputed-element / vmapped field routines."""
import jax
import jax.numpy as jnp
import numpy as onp
import pytest

from jax_sbgeom.coils import FourierCoil, CoilSet, filament_field, self_field, inductance_matrix
from jax_sbgeom.coils import fields
from jax_sbgeom.coils.biot_savart import biot_savart, biot_savart_single, _biot_savart_multiple

jax.config.update("jax_enable_x64", True)


def scan_position(coil, s):
    """The original lax.scan over the modes (single coil)."""
    n_modes = coil.fourier_cos.shape[-2]
    n = jnp.arange(1.0, n_modes + 1.0) * 2 * jnp.pi
    init = jnp.zeros(jnp.array(s).shape + (3,))

    def step(xyz, i):
        return xyz + coil.fourier_cos[i, :] * jnp.cos(n[i] * s)[..., None] + coil.fourier_sin[i, :] * jnp.sin(n[i] * s)[..., None], None

    return jax.lax.scan(step, init, jnp.arange(n_modes))[0] + coil.centre_i


def random_coil(key, n_modes=12, scale=1.0):
    k1, k2, k3 = jax.random.split(key, 3)
    decay = 1.0 / jnp.arange(1, n_modes + 1)[:, None] ** 2
    return FourierCoil(fourier_cos=scale * decay * jax.random.normal(k1, (n_modes, 3)),
                       fourier_sin=scale * decay * jax.random.normal(k2, (n_modes, 3)), centre_i=jax.random.normal(k3, (3,)))


def random_set(n=4, n_modes=12):
    return CoilSet.from_list([random_coil(jax.random.PRNGKey(i), n_modes, 2.0) for i in range(n)])


S = jnp.linspace(0.0, 1.0, 37)


def test_position_value_and_shapes():
    c = random_coil(jax.random.PRNGKey(0))
    onp.testing.assert_allclose(c.position(S), scan_position(c, S), atol=1e-13, rtol=1e-13)
    onp.testing.assert_allclose(c.position(0.3), scan_position(c, 0.3), atol=1e-13)
    assert c.position(jnp.ones((2, 5)) * 0.1).shape == (2, 5, 3)


def test_derivatives_match_scan_autodiff():
    c = random_coil(jax.random.PRNGKey(1))
    d1 = jax.vmap(jax.jacfwd(lambda s: scan_position(c, s)))(S)
    d2 = jax.vmap(jax.jacfwd(jax.jacfwd(lambda s: scan_position(c, s))))(S)
    onp.testing.assert_allclose(jax.vmap(jax.jacfwd(c.position))(S), d1, atol=1e-11, rtol=1e-12)
    onp.testing.assert_allclose(jax.vmap(jax.jacfwd(jax.jacfwd(c.position)))(S), d2, atol=1e-10, rtol=1e-12)
    onp.testing.assert_allclose(c.tangent(S), d1 / jnp.linalg.norm(d1, axis=-1, keepdims=True), atol=1e-12)


def test_batched_coilset_jit_vmap_grad():
    cs = random_set()
    ref = jax.vmap(scan_position, in_axes=(0, None))(cs.coils, S)
    onp.testing.assert_allclose(cs.position(S), ref, atol=1e-13, rtol=1e-13)
    onp.testing.assert_allclose(jax.jit(cs.position)(S), ref, atol=1e-13, rtol=1e-13)
    onp.testing.assert_allclose(jax.vmap(lambda c: c.position(S))(cs.coils), ref, atol=1e-13, rtol=1e-13)
    # different s per coil
    s2 = jnp.stack([S * (i + 1) / 4 for i in range(4)])
    onp.testing.assert_allclose(cs.position_different_s(s2), jax.vmap(scan_position)(cs.coils, s2), atol=1e-13)
    # jacobian and second derivative of a batched coil set
    onp.testing.assert_allclose(jax.vmap(lambda c: jax.vmap(jax.jacfwd(jax.jacfwd(c.position)))(S))(cs.coils),
                                jax.vmap(lambda c: jax.vmap(jax.jacfwd(jax.jacfwd(lambda s: scan_position(c, s))))(S))(cs.coils), atol=1e-10, rtol=1e-12)
    # gradient with respect to the coefficients
    loss = lambda c, f: jnp.sum(jnp.sin(f(c, S)) ** 2)
    one = jax.tree_util.tree_map(lambda a: a[0], cs.coils)
    g = jax.grad(lambda c: loss(c, lambda cc, s: cc.position(s)))(one)
    g_ref = jax.grad(lambda c: loss(c, scan_position))(one)
    for a, b in zip(jax.tree_util.tree_leaves(g), jax.tree_util.tree_leaves(g_ref)):
        onp.testing.assert_allclose(a, b, atol=1e-12, rtol=1e-11)


# ---------------------------------------------------------------------------------------------- Biot-Savart / fields
def old_biot_savart(cur, pos, dl, q):
    return _biot_savart_multiple(cur, pos, dl, q)


@pytest.mark.parametrize("batch_size", [1, 7, 50, 1000])
def test_biot_savart_chunked_equals_all_at_once(batch_size):
    k = jax.random.split(jax.random.PRNGKey(3), 4)
    pos, dl, cur, q = jax.random.normal(k[0], (300, 3)), 0.05 * jax.random.normal(k[1], (300, 3)), jax.random.normal(k[2], (300,)), jax.random.normal(k[3], (50, 3))
    onp.testing.assert_allclose(biot_savart(cur, pos, dl, q, batch_size), old_biot_savart(cur, pos, dl, q), rtol=1e-12, atol=1e-18)


def test_field_of_nodes_of_other_coils():
    cs = random_set()
    cur = jnp.array([1.0, 2.0, -1.0, 0.5]) * 1e5
    pts = jax.random.normal(jax.random.PRNGKey(5), (20, 3)) * 3
    pos, dl = fields.coil_elements(cs, 64)
    onp.testing.assert_allclose(filament_field(cs, cur, pts, 64, 7), fields._field_of_nodes(pos, dl, cur, pts, 20), rtol=1e-13)
    j = fields._others(4, 2)                                    # the nodes of the other coils == the field of the others
    onp.testing.assert_allclose(fields._field_of_nodes(pos[j], dl[j], cur[j], pts, 20), filament_field(CoilSet(cs[j]), cur[j], pts, 64, 7), rtol=1e-12)


def test_elements_use_analytic_derivatives_equal_autodiff():
    cs = random_set()
    pos, dl = fields.coil_elements(cs, 32)
    pos_ref = jax.vmap(lambda c: jax.vmap(lambda s: scan_position(c, s))(jnp.arange(32) / 32))(cs.coils)
    dl_ref = jax.vmap(lambda c: jax.vmap(jax.jacfwd(lambda s: scan_position(c, s)))(jnp.arange(32) / 32) / 32)(cs.coils)
    onp.testing.assert_allclose(pos, pos_ref, atol=1e-13)
    onp.testing.assert_allclose(dl, dl_ref, atol=1e-12, rtol=1e-12)


def test_batch_size_of_the_inductance_rows_does_not_change_results():
    cs = random_set()
    L1 = inductance_matrix(cs, 0.2, 0.15, n_rows=4, n_mut=64, n_self=128, batch_size=4)
    L2 = inductance_matrix(cs, 0.2, 0.15, n_rows=4, n_mut=64, n_self=128, batch_size=3)
    onp.testing.assert_allclose(L1, L2, rtol=1e-12)


def _bs_loss(pos, cur, dl, q, batch_size):
    return jnp.sum(biot_savart(cur, pos, dl, q, batch_size) ** 2)


def test_biot_savart_gradient_chunked_equals_all_at_once():
    k = jax.random.split(jax.random.PRNGKey(4), 4)
    pos, dl, cur, q = jax.random.normal(k[0], (300, 3)), 0.05 * jax.random.normal(k[1], (300, 3)), jax.random.normal(k[2], (300,)), jax.random.normal(k[3], (50, 3))
    g = jax.grad(_bs_loss, argnums=(0, 3))
    for a, b in zip(g(pos, cur, dl, q, 7), g(pos, cur, dl, q, 50)):     # summation order differs between chunk sizes
        onp.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12 * onp.max(onp.abs(b)))
