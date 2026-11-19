import jax
import jax.numpy as jnp
import numpy as onp
import pytest

import jax_sbgeom as jsb

jax.config.update("jax_enable_x64", True)


def _saddle(n, uneven=0.0):
    """Closed saddle-shaped curve sampled at n points; `uneven` > 0 spaces the points non-uniformly (smoothly)."""
    u = onp.arange(n) / n
    t = 2 * onp.pi * (u + uneven * onp.sin(6 * onp.pi * u))
    positions = onp.stack([3 * onp.cos(t), 2 * onp.sin(t), 0.5 * onp.cos(2 * t)], axis=-1)
    tangent = onp.stack([-3 * onp.sin(t), 2 * onp.cos(t), -1.0 * onp.sin(2 * t)], axis=-1)
    return positions, tangent / onp.linalg.norm(tangent, axis=-1, keepdims=True)


def _max_angle_error(tangent, tangent_ref):
    return onp.arccos(onp.clip(onp.sum(tangent * tangent_ref, axis=-1), -1.0, 1.0)).max()


def test_tangent_exact_on_uniform_circle():
    n = 64
    t = 2 * onp.pi * onp.arange(n) / n
    positions = 2.0 * onp.stack([onp.cos(t), onp.sin(t), onp.zeros(n)], axis=-1)
    expected = onp.stack([-onp.sin(t), onp.cos(t), onp.zeros(n)], axis=-1)

    coil = jsb.coils.DiscreteCoil.from_positions(jnp.asarray(positions))
    onp.testing.assert_allclose(coil.tangent(jnp.arange(n) / n), expected, atol=1e-12)


def test_tangent_second_order_on_uneven_spacing():
    errors = []
    for n in (48, 96, 192):
        positions, expected = _saddle(n, uneven=0.04)
        coil = jsb.coils.DiscreteCoil.from_positions(jnp.asarray(positions))
        errors.append(_max_angle_error(onp.asarray(coil.tangent(jnp.arange(n) / n)), expected))

    # halving the spacing reduces the error by ~4 for a second-order tangent (the forward segment gives ~2)
    assert errors[0] / errors[1] > 3.3
    assert errors[1] / errors[2] > 3.3
    assert errors[2] < 2e-3


def test_tangent_is_continuous_across_discrete_points():
    n = 40
    positions, _ = _saddle(n, uneven=0.04)
    coil = jsb.coils.DiscreteCoil.from_positions(jnp.asarray(positions))
    eps = 1e-9
    for k in range(1, n, 7):
        before = coil.tangent(jnp.asarray(k / n - eps))
        after = coil.tangent(jnp.asarray(k / n + eps))
        onp.testing.assert_allclose(before, after, atol=1e-6)


def test_radial_vector_frame_is_orthonormal_at_every_s():
    # radial vectors that are not perpendicular to the coil: the frame must still be orthonormal for any s,
    # including between the samples of the radial vectors
    n = 60
    positions, _ = _saddle(n, uneven=0.04)
    coil = jsb.coils.DiscreteCoil.from_positions(jnp.asarray(positions))
    angle = 2 * onp.pi * onp.arange(23) / 23
    radial = onp.stack([onp.cos(angle), onp.sin(angle), 0.4 * onp.ones(23)], axis=-1)
    finite_size_coil = jsb.coils.FiniteSizeCoil.from_coil(coil, jsb.coils.RadialVectorFrame, jnp.asarray(radial))

    s = jnp.asarray(onp.random.default_rng(0).uniform(0.0, 1.0, 200))
    tangent = onp.asarray(finite_size_coil.tangent(s))
    e_radial, e_phi = onp.moveaxis(onp.asarray(finite_size_coil.finite_size_frame(s)), -2, 0)

    onp.testing.assert_allclose(onp.linalg.norm(tangent, axis=-1), 1.0, atol=1e-12)
    onp.testing.assert_allclose(onp.linalg.norm(e_radial, axis=-1), 1.0, atol=1e-12)
    onp.testing.assert_allclose(onp.linalg.norm(e_phi, axis=-1), 1.0, atol=1e-12)
    onp.testing.assert_allclose(onp.sum(tangent * e_radial, axis=-1), 0.0, atol=1e-12)
    onp.testing.assert_allclose(onp.sum(tangent * e_phi, axis=-1), 0.0, atol=1e-12)
    onp.testing.assert_allclose(onp.sum(e_radial * e_phi, axis=-1), 0.0, atol=1e-12)
