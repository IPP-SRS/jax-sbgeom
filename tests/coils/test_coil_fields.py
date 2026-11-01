"""Finite-cross-section coil fields and inductances against independent references (elliptic integrals, the analytic
ring inductance, the energy/force identity for a ring) plus convergence, gradient and transform checks."""
import jax
import jax.numpy as jnp
import numpy as onp
import pytest
from scipy.special import ellipk, ellipe

from jax_sbgeom.coils import (FourierCoil, CoilSet, FiniteSizeCoil, CentroidFrame, filament_field, self_field,
                              max_force_density_on_filament, inductance_matrix, total_inductance)
from jax_sbgeom.coils.fields import MU0, _others

jax.config.update("jax_enable_x64", True)


def ring(radius=3.0, centre=(0.0, 0.0, 0.0), e1=(1.0, 0.0, 0.0), e2=(0.0, 1.0, 0.0)):
    """Circle centre + R (cos t e1 + sin t e2); FourierCoil layout (mode, xyz)."""
    return FourierCoil(fourier_cos=radius * jnp.array([e1], dtype=float), fourier_sin=radius * jnp.array([e2], dtype=float),
                       centre_i=jnp.array(centre, dtype=float))


def fs(coil):
    return FiniteSizeCoil.from_coil(coil, CentroidFrame)


def stack(*coils):
    return CoilSet.from_list(list(coils))


def wavy(scale=1.0):
    k = jnp.arange(1, 4)
    cos = jnp.stack([3.0 * (k == 1) + 0.2 / k ** 3, 0.3 / k ** 2 * (k == 2), 0.4 / k ** 3], axis=-1) * scale
    sin = jnp.stack([0.1 / k ** 3, 3.0 * (k == 1) + 0.15 / k ** 3, 0.5 * (k == 1)], axis=-1) * scale
    return FourierCoil(fourier_cos=cos, fourier_sin=sin, centre_i=jnp.array([0.3, -0.2, 0.1]))


# ---------------------------------------------------------------------------------------------- (1) filament field
def loop_field_elliptic(I, R, rho, z):
    """Exact field of a circular filament (Jackson / Smythe) from complete elliptic integrals; ellipk takes m = k^2."""
    m = 4 * R * rho / ((R + rho) ** 2 + z ** 2)
    K, E = ellipk(m), ellipe(m)
    pre = MU0 * I / (2 * onp.pi * onp.sqrt((R + rho) ** 2 + z ** 2))
    bz = pre * (K + (R ** 2 - rho ** 2 - z ** 2) / ((R - rho) ** 2 + z ** 2) * E)
    brho = pre * z / rho * (-K + (R ** 2 + rho ** 2 + z ** 2) / ((R - rho) ** 2 + z ** 2) * E)
    return brho, bz


def test_filament_on_axis():
    I, R, z = 1e6, 3.0, 1.5
    b = filament_field(stack(ring(R)), jnp.array([I]), jnp.array([[0.0, 0.0, z]]), n=128)[0]
    expect = MU0 * I * R ** 2 / (2 * (R ** 2 + z ** 2) ** 1.5)
    onp.testing.assert_allclose(b, [0, 0, expect], atol=1e-11 * expect)


@pytest.mark.parametrize("rho,z", [(0.5, 0.3), (1.0, -1.0), (2.0, 0.7), (2.9, 0.2), (4.0, 1.0), (6.0, -2.0)])
def test_filament_off_axis_elliptic(rho, z):
    I, R, phi = 2.5e5, 3.0, 0.7
    pt = jnp.array([[rho * onp.cos(phi), rho * onp.sin(phi), z]])
    b = onp.asarray(filament_field(stack(ring(R)), jnp.array([I]), pt, n=512)[0])
    brho, bz = loop_field_elliptic(I, R, rho, z)
    onp.testing.assert_allclose(b, [brho * onp.cos(phi), brho * onp.sin(phi), bz], rtol=1e-9, atol=1e-9 * abs(bz))


def test_others_removes_one_coil():
    cs = stack(ring(3.0), ring(2.0, (0, 0, 1.0)), ring(1.5, (0, 0, -1.0)))
    cur = jnp.array([1e6, 5e5, 2e5])
    on_coil = jnp.array([[3.0, 0.0, 0.0]])
    for i in range(3):
        j = jax.jit(_others, static_argnums=0)(3, i)                  # i may be traced
        keep = [k for k in range(3) if k != i]
        assert sorted(j.tolist()) == keep                                # cyclic order, the sum does not care
        kk = jnp.array(keep)
        ref = filament_field(stack(*[(ring(3.0), ring(2.0, (0, 0, 1.0)), ring(1.5, (0, 0, -1.0)))[k] for k in keep]), cur[kk], on_coil, n=64)
        onp.testing.assert_allclose(filament_field(jax.tree.map(lambda a: a[j], cs), cur[j], on_coil, n=64), ref, rtol=1e-12)


# ---------------------------------------------------------------------------------------------- (2) self field
def _ring_inductance(R, a_eff):
    return MU0 * R * (jnp.log(8 * R / a_eff) - 2.0 + 0.25)


def _filament_bundle_bz(R, w, u, v, m, I=1.0):
    """B_z at section point (u, v) of a ring of radius R, modelled (as in Landreman & Hudson) as m x m equal-current
    filament rings filling the w x w section, evaluated with the exact elliptic-integral loop field. In the
    coil frame q = T x p = -z at s=0, hence z = -v w/2."""
    g = (onp.arange(m) + 0.5) / m * w - w / 2
    r = ((R + g)[:, None] * onp.ones((1, m))).ravel()
    z = (onp.ones((m, 1)) * g[None, :]).ravel()
    return loop_field_elliptic(I / m ** 2, r, R + u * w / 2, -v * w / 2 - z)[1].sum()


@pytest.mark.parametrize("u,v", [(0.0, 0.0), (0.36, -0.24), (-0.8, 0.5), (0.9, 0.9)])
def test_self_field_vs_filament_bundle(u, v):
    """Independent: the LH field at a section point equals the field of a dense bundle of equal-current filaments
    (exact elliptic-integral loops, 100 x 100 of them), up to O((w/R)^2) and the discretisation of the bundle."""
    R, w, I = 3.0, 0.04, 1.0
    b = float(self_field(fs(ring(R)), I, w, w, jnp.array([0.0]), u=u, v=v, n_src=2048)[0, 2])
    ref = _filament_bundle_bz(R, w, u, v, 100, I)
    assert abs(b - ref) < 2e-3 * abs(ref) + 1e-3 * MU0 * I / (4 * onp.pi * R), (b, ref)


@pytest.mark.parametrize("width,height", [(0.04, 0.04), (0.06, 0.03)])
def test_self_field_ring_matches_hoop_force(width, height):
    """Independent from energy: the net hoop force of a ring, I^2/2 dL/dR with L = mu0 R (ln(8R/GMD) - 2) (GMD =
    0.2235 (w + h) of a rectangle), is F = int J_phi R_j B_z dV; with equal current per filament (the LH model),
    <(R_j/R) B_z> over the section = mu0 I / (4 pi R) (ln(8R/GMD) - 1). The LH field averaged with Gauss-Legendre over
    (u, v), weighted by R_j / R = 1 + u w / (2R), must reproduce it (residual O((w/R)^2) and the 4-digit GMD constant)."""
    R, I = 3.0, 1e6
    coil = ring(R)
    x, g = onp.polynomial.legendre.leggauss(16)
    U, V = onp.meshgrid(x, x)
    W = onp.outer(g, g).ravel() / 4
    b = jax.vmap(lambda u, v: self_field(fs(coil), I, width, height, jnp.array([0.0]), u=u, v=v, n_src=2048)[0, 2])(
        jnp.array(U.ravel()), jnp.array(V.ravel()))
    avg = float(jnp.sum(W * (1 + U.ravel() * width / (2 * R)) * b))
    expect = MU0 * I / (4 * onp.pi * R) * (onp.log(8 * R / (0.2235 * (width + height))) - 1.0)
    assert abs(avg - expect) < 1e-3 * expect, (avg, expect)


def test_self_field_ring_direction_and_centre_values():
    """Field at the section centre is along the ring axis (+z for counter-clockwise current), and is larger at the
    inboard edge (u=-1) than at the outboard edge (u=+1), as for any toroidal current."""
    R, I, w = 3.0, 1e6, 0.05
    s = jnp.array([0.13])
    b0 = self_field(fs(ring(R)), I, w, w, s, u=0.0, v=0.0, n_src=1024)[0]
    bin_, bout = self_field(fs(ring(R)), I, w, w, s, u=-1.0, v=0.0, n_src=1024)[0], self_field(fs(ring(R)), I, w, w, s, u=1.0, v=0.0, n_src=1024)[0]
    assert abs(b0[0]) < 1e-10 * b0[2] and abs(b0[1]) < 1e-10 * b0[2] and b0[2] > 0
    assert bin_[2] > b0[2] > bout[2]


def test_self_field_convergence_n_src_and_spectral_in_n_eval():
    coil, I, w, h = wavy(), 1e6, 0.1, 0.07
    ref = self_field(fs(coil), I, w, h, jnp.arange(32) / 32, u=0.0, v=0.0, n_src=4096)
    for n_src, tol in [(512, 5e-3), (1024, 5e-5), (2048, 1e-8)]:      # geometric (spectral) convergence in n_src
        b = self_field(fs(coil), I, w, h, jnp.arange(32) / 32, u=0.0, v=0.0, n_src=n_src)
        assert float(jnp.max(jnp.abs(b - ref)) / jnp.max(jnp.abs(ref))) < tol, n_src
    # B(s) is a smooth periodic function of s: its Fourier spectrum decays to round-off (n_eval resolves it)
    b = self_field(fs(coil), I, w, h, jnp.arange(128) / 128, u=0.0, v=0.0, n_src=4096)
    spec = onp.abs(onp.fft.rfft(onp.asarray(b), axis=0))
    assert spec[30:].max() < 1e-12 * spec.max()


def test_max_force_density_mutual_part():
    """Two coaxial rings (current I each): |I T x B|/(w h) on ring 0 with B = B_reg(ring 0) + B(ring 1). The ring-1
    field is the exact elliptic-integral one; B_reg(ring 0) (along z, azimuthally symmetric) is taken from the
    isolated-ring result, so this checks the combination and the exclusion of the own coil, not B_reg itself."""
    R, I, w, h, z1 = 3.0, 1e6, 0.1, 0.1, 1.0
    f_iso = float(max_force_density_on_filament(stack(ring(R)), jnp.array([I]), w, h, jnp.arange(1), n_eval=4, n_src=1024, n_mut=128))
    bz_reg = f_iso * w * h / I
    f = float(max_force_density_on_filament(stack(ring(R), ring(2.0, (0, 0, z1))), jnp.array([I, I]), w, h, jnp.array([0]),
                                n_eval=4, n_src=1024, n_mut=512))
    brho, bz = loop_field_elliptic(I, 2.0, R, 0 - z1)
    expect = I / (w * h) * onp.hypot(bz_reg + bz, brho)
    onp.testing.assert_allclose(f, expect, rtol=1e-8)


# ---------------------------------------------------------------------------------------------- (3) inductance
@pytest.mark.parametrize("R,w,h", [(3.0, 0.1, 0.1), (3.0, 0.2, 0.1), (1.0, 0.02, 0.03)])
def test_ring_inductance_analytic(R, w, h):
    """L = mu0 R (ln(8R/GMD) - 2), GMD = 0.2235 (w + h) (Grover). This is the full inductance of a uniform current
    (for a circular wire it is Maxwell's mu0 R (ln(8R/a) - 7/4), GMD = a exp(-1/4)): it already contains the internal
    inductance, so the softened kernel alone must reproduce it. O((w/R)^2) and 4-digit GMD constant."""
    L = float(total_inductance(stack(ring(R)), w, h, n_rows=1, n_mut=128, n_self=2048))
    expect = float(MU0 * R * (jnp.log(8 * R / (0.2235 * (w + h))) - 2.0))
    assert abs(L - expect) < 1e-3 * expect, (L, expect)


def test_ring_inductance_vs_filament_bundle():
    """Independent: inductance of the m x m equal-current filament-ring bundle (pairwise elliptic Neumann mutuals plus
    the GMD self term of each tiny cell) converges to the same value as the softened-kernel ring."""
    R, w, m = 3.0, 0.2, 48
    g = (onp.arange(m) + 0.5) / m * w - w / 2
    r = ((R + g)[:, None] * onp.ones((1, m))).ravel()
    z = (onp.ones((m, 1)) * g[None, :]).ravel()
    Ri, Rj, d = r[:, None], r[None, :], z[:, None] - z[None, :]
    off = ~onp.eye(len(r), dtype=bool)
    k2 = 4 * Ri * Rj / ((Ri + Rj) ** 2 + d ** 2)
    k2 = onp.where(off, k2, 0.5)
    Mij = MU0 * onp.sqrt(Ri * Rj) * ((2 / onp.sqrt(k2) - onp.sqrt(k2)) * ellipk(k2) - 2 / onp.sqrt(k2) * ellipe(k2))
    cell = w / m
    L_ref = onp.sum(onp.where(off, Mij, 0.0)) / m ** 4 + onp.mean(MU0 * r * (onp.log(8 * r / (0.44705 * cell)) - 2.0)) / m ** 2
    L = float(total_inductance(stack(ring(R)), w, w, n_rows=1, n_mut=128, n_self=2048))
    assert abs(L - L_ref) < 5e-4 * L_ref, (L, L_ref)


def test_two_coaxial_rings_mutual_matches_elliptic_neumann():
    """Mutual inductance of coaxial circles, M = mu0 sqrt(R1 R2) [(2/k - k) K - 2/k E], k^2 = 4 R1 R2 / ((R1+R2)^2 + d^2)."""
    R1, R2, d = 3.0, 2.0, 1.0
    L = onp.asarray(inductance_matrix(stack(ring(R1), ring(R2, (0, 0, d))), 0.05, 0.05, n_rows=2, n_mut=512, n_self=1024))
    k2 = 4 * R1 * R2 / ((R1 + R2) ** 2 + d ** 2)
    M = MU0 * onp.sqrt(R1 * R2) * ((2 / onp.sqrt(k2) - onp.sqrt(k2)) * ellipk(k2) - 2 / onp.sqrt(k2) * ellipe(k2))
    onp.testing.assert_allclose(L[0, 1], M, rtol=1e-10)
    onp.testing.assert_allclose(L[1, 0], M, rtol=1e-10)


def test_periodicity_option_matches_full_sum():
    n, R0, r = 6, 5.0, 1.0
    coils = []
    for k in range(n):
        ph = 2 * onp.pi * k / n
        er = (onp.cos(ph), onp.sin(ph), 0.0)
        coils.append(ring(r, centre=(R0 * er[0], R0 * er[1], 0.0), e1=er, e2=(0.0, 0.0, 1.0)))
    cs = stack(*coils)
    full = total_inductance(cs, 0.05, 0.05, n_rows=n, n_mut=128, n_self=512)
    part = total_inductance(cs, 0.05, 0.05, n_rows=1, n_mut=128, n_self=512)
    onp.testing.assert_allclose(part, full, rtol=1e-10)
    L = onp.asarray(inductance_matrix(cs, 0.05, 0.05, n_rows=n, n_mut=128, n_self=512))
    onp.testing.assert_allclose(L, L.T, rtol=1e-10)


# ---------------------------------------------------------------------------------------------- (4) gradients
def _fd(f, x, h):
    return (f(x + h) - f(x - h)) / (2 * h)


def test_gradients_vs_finite_differences():
    coil = wavy()
    cs2 = stack(wavy(), wavy(1.3))
    s = jnp.arange(4) / 4

    # current (linear): d/dI of the self field
    f = lambda I: jnp.sum(self_field(fs(coil), I, 0.1, 0.07, s, u=-1.0, v=0.3, n_src=512) ** 2)
    onp.testing.assert_allclose(jax.grad(f)(1e5), _fd(f, 1e5, 10.0), rtol=1e-6)

    # width
    f = lambda w: jnp.sum(self_field(fs(coil), 1e5, w, 0.07, s, u=-1.0, v=0.0, n_src=512))
    onp.testing.assert_allclose(jax.grad(f)(0.1), _fd(f, 0.1, 1e-5), rtol=1e-6)
    f = lambda w: total_inductance(cs2, w, 0.07, n_rows=2, n_mut=64, n_self=256)
    onp.testing.assert_allclose(jax.grad(f)(0.1), _fd(f, 0.1, 1e-5), rtol=1e-6)

    # a Fourier coefficient
    def f(c):
        co = FourierCoil(fourier_cos=coil.fourier_cos.at[0, 0].set(c), fourier_sin=coil.fourier_sin, centre_i=coil.centre_i)
        return jnp.sum(self_field(fs(co), 1e5, 0.1, 0.07, s, u=-1.0, v=0.0, n_src=512))
    onp.testing.assert_allclose(jax.grad(f)(3.2), _fd(f, 3.2, 1e-5), rtol=1e-6)

    def f(c):
        co = FourierCoil(fourier_cos=cs2.coils.fourier_cos.at[1, 0, 0].set(c), fourier_sin=cs2.coils.fourier_sin, centre_i=cs2.coils.centre_i)
        co = CoilSet(co)
        return max_force_density_on_filament(co, jnp.array([1e5, 2e5]), 0.1, 0.07, jnp.arange(2), n_eval=32, n_src=256, n_mut=64) + total_inductance(co, 0.1, 0.07, n_rows=2, n_mut=64, n_self=256)
    onp.testing.assert_allclose(jax.grad(f)(3.9), _fd(f, 3.9, 1e-5), rtol=1e-5)


# ---------------------------------------------------------------------------------------------- (5) jit / vmap
def test_jit_and_vmap():
    cs2 = stack(wavy(), wavy(1.3))
    pts = jnp.array([[0.0, 0.0, 0.5], [1.0, 0.2, -0.3]])
    cur = jnp.array([1e5, 2e5])
    b = filament_field(cs2, cur, pts, 64)
    onp.testing.assert_allclose(jax.jit(filament_field, static_argnums=(3,))(cs2, cur, pts, 64), b, rtol=1e-12)
    bs = jax.vmap(lambda scale: filament_field(cs2, scale * cur, pts, 64))(jnp.array([1.0, 2.0]))
    onp.testing.assert_allclose(bs[1], 2 * b, rtol=1e-12)

    widths = jnp.array([0.05, 0.1])
    Ls = jax.jit(jax.vmap(lambda w: total_inductance(cs2, w, 0.07, n_rows=2, n_mut=64, n_self=256)))(widths)
    onp.testing.assert_allclose(Ls[1], total_inductance(cs2, 0.1, 0.07, n_rows=2, n_mut=64, n_self=256), rtol=1e-12)
    sf = jax.jit(jax.vmap(lambda I: self_field(fs(wavy()), I, 0.1, 0.07, jnp.arange(4) / 4, u=0.0, v=0.0, n_src=256)))(cur)
    onp.testing.assert_allclose(sf[1], 2 * sf[0], rtol=1e-12)
