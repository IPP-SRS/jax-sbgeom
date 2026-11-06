import jax
import jax.numpy as jnp
import numpy as onp
import pytest
import jax_sbgeom as jsb
from jax_sbgeom.coils import SplineCoil, DiscreteCoil, CoilSet, convert_to_spline_coil, convert_to_spline_coilset

jax.config.update("jax_enable_x64", True)

R = 2.0


def _circle(M=200, R=R, z0=0.0):
    t = 2 * onp.pi * onp.arange(M) / M
    return jnp.array(onp.stack([R * onp.cos(t), R * onp.sin(t), z0 + 0 * t], axis=-1))


def _circle_coil(k=3, n_ctrl=16, **kw):
    return convert_to_spline_coil(DiscreteCoil.from_positions(_circle(**kw)), n_ctrl, k)


def _wobbly_coil(k=3, n_ctrl=12, seed=0):
    rng = onp.random.default_rng(seed)
    t = 2 * onp.pi * onp.arange(n_ctrl) / n_ctrl
    c = onp.stack([R * onp.cos(t), R * onp.sin(t), 0.2 * onp.cos(2 * t)], axis=-1) + 0.05 * rng.normal(size=(n_ctrl, 3))
    return SplineCoil(coeffs=jnp.array(c), k=k)


@pytest.mark.parametrize("k", [3, 4, 5])
def test_circle_fit(k):
    coil = _circle_coil(k=k, n_ctrl=16)
    s = jnp.linspace(0, 1, 301)
    ref = onp.stack([R * onp.cos(2 * onp.pi * s), R * onp.sin(2 * onp.pi * s), 0 * s], axis=-1)
    onp.testing.assert_allclose(coil.position(s), ref, atol=2e-3)
    onp.testing.assert_allclose(coil.length(), 2 * onp.pi * R, rtol=1e-3)
    onp.testing.assert_allclose(coil.curvature(s), 1 / R, rtol=2e-2)


@pytest.mark.parametrize("k", [3, 4, 5])
def test_derivatives_vs_finite_differences(k):
    coil = _wobbly_coil(k=k)
    s = jnp.linspace(0.03, 0.97, 41)
    h = 1e-6
    fd = (coil.position(s + h) - coil.position(s - h)) / (2 * h)
    onp.testing.assert_allclose(coil.derivative(s), fd, rtol=1e-6, atol=1e-6)
    onp.testing.assert_allclose(coil.tangent(s), fd / jnp.linalg.norm(fd, axis=-1, keepdims=True), rtol=1e-6, atol=1e-6)
    fd2 = (coil.derivative(s + h) - coil.derivative(s - h)) / (2 * h)
    onp.testing.assert_allclose(coil.second_derivative(s), fd2, rtol=1e-5, atol=1e-5)
    n = coil.normal(s)
    onp.testing.assert_allclose(jnp.linalg.norm(n, axis=-1), 1.0, atol=1e-10)
    onp.testing.assert_allclose(jnp.sum(n * coil.tangent(s), axis=-1), 0.0, atol=1e-8)


@pytest.mark.parametrize("k", [3, 4, 5])
def test_periodicity(k):
    coil = _wobbly_coil(k=k)
    onp.testing.assert_allclose(coil.position(0.0), coil.position(1.0), atol=1e-12)
    eps = 1e-7
    onp.testing.assert_allclose(coil.position(1.0 - eps), coil.position(0.0 - eps), atol=1e-12)
    onp.testing.assert_allclose(coil.position(1.0 - eps), coil.position(0.0), atol=1e-5)
    # derivatives are continuous across the seam
    onp.testing.assert_allclose(coil.derivative(1.0 - eps), coil.derivative(eps), atol=1e-4)
    onp.testing.assert_allclose(coil.tangent(1.0 - eps), coil.tangent(eps), atol=1e-5)


def test_coilset_batching():
    coils = [_wobbly_coil(seed=i) for i in range(3)]
    cs = jsb.coils.CoilSet.from_list(coils)
    s = jnp.linspace(0, 1, 17)
    assert cs.coils.coeffs.shape == (3, 12, 3)
    assert cs.position(s).shape == (3, 17, 3)
    assert cs.tangent(s).shape == (3, 17, 3)
    assert cs.normal(s).shape == (3, 17, 3)
    assert cs.centre().shape == (3, 3)
    for i in range(3):
        onp.testing.assert_allclose(cs[i].position(s), coils[i].position(s), atol=1e-12)
        onp.testing.assert_allclose(cs[i].length(), coils[i].length(), rtol=1e-12)


def test_grad_and_jit():
    coil = _wobbly_coil()
    s = jnp.linspace(0, 1, 64, endpoint=False)

    def loss(c):
        return jnp.sum(SplineCoil(coeffs=c, k=3).curvature(s))

    g = jax.jit(jax.grad(loss))(coil.coeffs)
    assert g.shape == coil.coeffs.shape
    h = 1e-6
    for idx in [(0, 0), (3, 1), (11, 2)]:
        e = jnp.zeros_like(coil.coeffs).at[idx].set(h)
        fd = (loss(coil.coeffs + e) - loss(coil.coeffs - e)) / (2 * h)
        onp.testing.assert_allclose(g[idx], fd, rtol=1e-5)
    onp.testing.assert_allclose(jax.jit(lambda c: c.position(s))(coil), coil.position(s), atol=1e-12)
    # k is metadata, not a leaf
    assert len(jax.tree_util.tree_leaves(coil)) == 1


@pytest.mark.parametrize("k", [3, 4, 5])
def test_locality(k):
    n_ctrl = 20
    coil = _wobbly_coil(k=k, n_ctrl=n_ctrl)
    j = 7
    pert = SplineCoil(coeffs=coil.coeffs.at[j].add(jnp.array([0.1, -0.1, 0.1])), k=k)
    s = jnp.linspace(0, 1, 4000, endpoint=False)
    changed = s[jnp.linalg.norm(pert.position(s) - coil.position(s), axis=-1) > 1e-12]
    width = float(changed.max() - changed.min())
    assert 0 < width <= (k + 1) / n_ctrl + 1e-3
    # and the curve is untouched outside that window
    assert changed.size < (k + 1) / n_ctrl * s.size + 2


@pytest.mark.parametrize("k", [3, 4, 5])
def test_reverse_parametrisation(k):
    coil = _wobbly_coil(k=k)
    rev = coil.reverse_parametrisation()
    s = jnp.linspace(0, 1, 23)
    onp.testing.assert_allclose(rev.position(s), coil.position(1.0 - s), atol=1e-12)
    onp.testing.assert_allclose(rev.tangent(s), -coil.tangent(1.0 - s), atol=1e-10)


@pytest.mark.parametrize("method,args", [(jsb.coils.CentroidFrame, ()), (jsb.coils.RotationMinimizedFrame, (64,))])
def test_finite_size_frames(method, args):
    coil = _circle_coil()
    s = jnp.linspace(0, 1, 9)
    fs = jsb.coils.FiniteSizeCoil.from_coil(coil, method, *args)
    frame = fs.finite_size_frame(s)
    assert frame.shape == (9, 2, 3)
    assert jnp.all(jnp.isfinite(frame))
    # circle in the xy plane: the centroid radial vector is the in-plane radial direction
    if method is jsb.coils.CentroidFrame:
        onp.testing.assert_allclose(fs.radial_vector(s), coil.position(s) / R, atol=5e-3)


def _wavy_positions(M=400, phase=0.0, z0=0.0):
    t = 2 * onp.pi * onp.arange(M) / M
    r = R + 0.25 * onp.cos(3 * t + phase)
    return jnp.array(onp.stack([r * onp.cos(t), r * onp.sin(t), z0 + 0.3 * onp.sin(2 * t + phase)], axis=-1))


@pytest.mark.parametrize("k", [3, 4])
def test_convert_coilset_matches_each_coil(k):
    discrete = [DiscreteCoil.from_positions(_wavy_positions(phase=p)) for p in (0.0, 0.7)]
    converted = convert_to_spline_coilset(CoilSet.from_list(discrete), 24, k)
    assert isinstance(converted, CoilSet) and converted.coils.k == k
    for i, coil in enumerate(discrete):
        onp.testing.assert_allclose(converted.coils.coeffs[i], convert_to_spline_coil(coil, 24, k).coeffs, atol=1e-12)
    s = jnp.arange(400) / 400
    onp.testing.assert_allclose(converted.position(s)[0], _wavy_positions(), atol=2e-3)


def _same_shape_pair(k, n_ctrl=120):
    '''The same two curves as a Fourier and as a fine spline coil set.'''
    discrete = CoilSet.from_list([DiscreteCoil.from_positions(_wavy_positions(z0=z)) for z in (0.0, 1.5)])
    return jsb.coils.convert_to_fourier_coilset(discrete, 20), convert_to_spline_coilset(discrete, n_ctrl, k)


@pytest.mark.parametrize("k", [3, 4])
def test_geometry_matches_fourier(k):
    fourier, spline = _same_shape_pair(k)
    s = jnp.linspace(0, 1, 37, endpoint=False)
    onp.testing.assert_allclose(spline.position(s), fourier.position(s), atol=1e-5)
    onp.testing.assert_allclose(spline.tangent(s), fourier.tangent(s), atol=1e-4)
    onp.testing.assert_allclose(spline.normal(s), fourier.normal(s), atol=1e-2)
    onp.testing.assert_allclose(spline.centre(), fourier.centre(), atol=1e-5)


@pytest.mark.parametrize("k", [3, 4])
@pytest.mark.parametrize("method,args", [(jsb.coils.CentroidFrame, ()), (jsb.coils.FrenetSerretFrame, ()), (jsb.coils.RotationMinimizedFrame, (64,))])
def test_finite_size_frames_match_fourier(k, method, args):
    fourier, spline = _same_shape_pair(k)
    s = jnp.linspace(0, 1, 37, endpoint=False)
    frame = lambda cs: jsb.coils.FiniteSizeCoilSet.from_coilset(cs, method, *args).finite_size_frame(s)
    onp.testing.assert_allclose(frame(spline), frame(fourier), atol=1e-2)


@pytest.mark.parametrize("k", [3, 4])
def test_force_density_matches_fourier(k):
    fourier, spline = _same_shape_pair(k)
    currents = jnp.array([1e7, 1e7])
    force = lambda cs, s: jsb.coils.force_density_on_filament(cs, currents, 0.3, 0.3, 0, s, 256, 128)
    for s in (0.1, 0.37, 0.8):
        onp.testing.assert_allclose(force(spline, s), force(fourier, s), rtol=1e-3)
