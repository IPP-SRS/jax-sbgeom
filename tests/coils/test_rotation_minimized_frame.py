import jax
import jax.numpy as jnp
import numpy as onp
import pytest

import jax_sbgeom as jsb

jax.config.update("jax_enable_x64", True)


def _poloidal_ring_coil(k, n_coils=8, n=60):
    """Planar circle in a vertical plane through the torus axis, at toroidal angle k."""
    phi = 2 * onp.pi * (k + 0.5) / n_coils
    t = onp.linspace(0, 2 * onp.pi, n, endpoint=False)
    r = 5.0 + onp.cos(t)
    return onp.stack([r * onp.cos(phi), r * onp.sin(phi), onp.sin(t)], axis=-1)


@pytest.mark.parametrize("k", range(8))
def test_rotation_minimized_frame_of_planar_coil_is_finite(k):
    # for planar coils the frame closes without twist, and the dot product of the first and last radial vector can
    # round to just above 1: arccos used to return NaN for the whole frame
    coil = jsb.coils.DiscreteCoil.from_positions(jnp.asarray(_poloidal_ring_coil(k)))
    finite_size_coil = jsb.coils.FiniteSizeCoil.from_coil(coil, jsb.coils.RotationMinimizedFrame, 60)

    frame = onp.asarray(finite_size_coil.finite_size_frame(jnp.linspace(0.0, 1.0, 97, endpoint=False)))
    assert onp.isfinite(frame).all()
