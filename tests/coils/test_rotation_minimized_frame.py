import jax
import jax.numpy as jnp
import numpy as onp
import pytest
from pathlib import Path

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


DATA_INPUT = Path(__file__).parent.parent / "data" / "coils"


def _outward_cosine(finite_size_coil, coil, n=128):
    """Cosine between the radial vector and the outward direction (from the coil centre, normal to the tangent)."""
    s = jnp.linspace(0.0, 1.0, n, endpoint=False)
    radial = onp.asarray(finite_size_coil.finite_size_frame(s))[:, 0, :]
    outward = onp.asarray(jsb.coils.FiniteSizeCoil.from_coil(coil, jsb.coils.CentroidFrame).finite_size_frame(s))[:, 0, :]
    return onp.sum(radial * outward, axis=-1)


@pytest.mark.parametrize("k", range(8))
def test_rotation_minimized_frame_of_planar_coil_points_outward(k):
    # planar coil: the transported start vector stays in the plane, outward at every point
    coil = jsb.coils.DiscreteCoil.from_positions(jnp.asarray(_poloidal_ring_coil(k)))
    finite_size_coil = jsb.coils.FiniteSizeCoil.from_coil(coil, jsb.coils.RotationMinimizedFrame, 60)
    onp.testing.assert_allclose(_outward_cosine(finite_size_coil, coil), 1.0, atol=1e-6)


@pytest.mark.parametrize("data_file", sorted(DATA_INPUT.glob("*_input.npy")))
def test_rotation_minimized_frame_starts_outward_and_is_outward_on_average(data_file):
    # orientation convention (see FiniteSizeMethod): radial vector outward at s=0, and on average
    positions = onp.load(data_file)
    for i in range(positions.shape[0]):
        coil = jsb.coils.DiscreteCoil.from_positions(jnp.asarray(positions[i]))
        finite_size_coil = jsb.coils.FiniteSizeCoil.from_coil(coil, jsb.coils.RotationMinimizedFrame, positions.shape[1])
        cosine = _outward_cosine(finite_size_coil, coil)
        assert cosine[0] > 0.99
        assert cosine.mean() > 0.8


def _coil_with_undefined_outward_at_start():
    # circle, except that the first point is the apex of a kink: its neighbours are mirror images about the line
    # through the first point along y, so the tangent at s=0 is along x, and the centre is on the line through the
    # first point along x (the tangent line)
    t = onp.linspace(0, 2 * onp.pi, 60, endpoint=False)
    positions = onp.stack([5.0 + onp.cos(t), onp.sin(t), 0.0 * t], axis=-1)
    positions[1] = [6.5, 0.2, 0.0]
    positions[-1] = [5.5, 0.2, 0.0]
    positions[0, 1] = positions[1:, 1].mean()
    return jsb.coils.DiscreteCoil.from_positions(jnp.asarray(positions))


def test_rotation_minimized_frame_raises_when_outward_at_start_is_undefined():
    coil = _coil_with_undefined_outward_at_start()
    with pytest.raises(ValueError, match="cannot determine the starting radial vector"):
        jsb.coils.FiniteSizeCoil.from_coil(coil, jsb.coils.RotationMinimizedFrame, 60)


def test_rotation_minimized_frame_coilset_raises_and_names_the_coil():
    good = jsb.coils.DiscreteCoil.from_positions(jnp.asarray(_poloidal_ring_coil(0)))
    with pytest.raises(ValueError, match=r"coil index \[1\]"):
        jsb.coils.FiniteSizeCoilSet.from_coils([good, _coil_with_undefined_outward_at_start()],
                                               jsb.coils.RotationMinimizedFrame, 60)


@pytest.mark.parametrize("k", range(8))
def test_frenet_serret_frame_of_convex_planar_fourier_coil_points_outward(k):
    # the curve normal points to the centre of curvature (inward); the radial vector is minus the normal
    discrete = jsb.coils.DiscreteCoil.from_positions(jnp.asarray(_poloidal_ring_coil(k)))
    fourier_set = jsb.coils.convert_to_fourier_coilset(jsb.coils.CoilSet.from_list([discrete]))
    coil = fourier_set[0]
    finite_size_coil = jsb.coils.FiniteSizeCoil.from_coil(coil, jsb.coils.FrenetSerretFrame)
    assert _outward_cosine(finite_size_coil, coil).min() > 0.99


def test_rotation_minimized_frame_raises_inside_jit_when_outward_at_start_is_undefined():
    # under jit the eager check is skipped (traced values), the check inside the graph catches it
    coil = _coil_with_undefined_outward_at_start()
    with pytest.raises(Exception, match="Outward direction is \\(almost\\) parallel to tangent"):
        jax.jit(lambda c: jsb.coils.FiniteSizeCoil.from_coil(c, jsb.coils.RotationMinimizedFrame, 60).finite_size_frame(0.3))(coil)
