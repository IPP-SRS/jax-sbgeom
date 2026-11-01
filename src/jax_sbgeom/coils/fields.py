'''
Magnetic fields and inductances of coils with a finite rectangular cross-section.

Filament fields use the periodic trapezoid rule on the analytic curve (spectrally convergent for smooth coils) and
`jax_sbgeom.coils.biot_savart.biot_savart`. The self field of a finite-size coil follows

    Landreman & Hudson, "Magnetic fields of finite-cross-section coils" (2018): the regularised Biot-Savart integral
    (kernel softened by delta * a * b) plus local terms for the rectangular cross-section (equation numbers refer to the paper).
'''

import jax
import jax.numpy as jnp
from .base_coil import FiniteSizeCoil
from .biot_savart import biot_savart, Constant
from .coilset import CoilSet

MU0 = Constant.MU_0


def _nodes(coil, n: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    '''Positions x(s_i) and line elements dl_i = x'(s_i) / n [n, 3] of n equally spaced nodes of one coil.'''
    s = jnp.arange(n) / n
    return coil.position(s), coil.derivative(s) / n


def coil_elements(coils: CoilSet, n: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    '''
    Trapezoid-rule nodes of a coil set.

    Parameters
    ----------
    coils : CoilSet
    n : int
        Number of equally spaced nodes per coil.

    Returns
    -------
    (jnp.ndarray, jnp.ndarray)
        Positions and line elements dl = x'(s) / n, each of shape [n_coils, n, 3].
    '''
    return jax.vmap(_nodes, in_axes=(0, None))(coils.coils, n)


# ---------------------------------------------------------------------------------------------------------------------
#  Filament field
# ---------------------------------------------------------------------------------------------------------------------
def filament_field(coils: CoilSet, currents: jnp.ndarray, points: jnp.ndarray, n: int) -> jnp.ndarray:
    '''
    Magnetic field of all coils as infinitely thin filaments (periodic trapezoid rule with n nodes per coil).
    The points must not lie on a coil (the field of a filament diverges there).

    Parameters
    ----------
    coils : CoilSet
    currents : jnp.ndarray
        Current in every coil [n_coils]
    points : jnp.ndarray
        Evaluation points [M, 3]
    n : int
        Number of nodes per coil.

    Returns
    -------
    jnp.ndarray
        Field [M, 3]
    '''
    return _field_of_nodes(*coil_elements(coils, n), currents, points)


def _field_of_nodes(pos, dl, currents, points):
    '''Filament field of coils given by their nodes pos, dl [n_coils, n, 3].'''
    return biot_savart(jnp.repeat(currents, pos.shape[1]), pos.reshape(-1, 3), dl.reshape(-1, 3), points)


def _others(n: int, i) -> jnp.ndarray:
    '''Indices of all coils but i, cyclic from i + 1 [n - 1] (static shape; i may be traced).'''
    return (i + 1 + jnp.arange(n - 1)) % n


# ---------------------------------------------------------------------------------------------------------------------
#  Landreman & Hudson (2018) self field of a rectangular cross-section
# ---------------------------------------------------------------------------------------------------------------------
def _t(x, y):
    return 4.0 * y / (3.0 * x) * jnp.arctan(x / y)


def _ln(x, y):
    return y ** 2 / (6.0 * x ** 2) * jnp.log(y / x)


def k_delta(a, b):
    '''k (eq. 13) and delta = exp(-25/6 + k) (eq. 12) of a rectangular cross-section a x b.'''
    sq = (a ** 4 - 6.0 * a ** 2 * b ** 2 + b ** 4) / (6.0 * a ** 2 * b ** 2) * jnp.log(a / b + b / a)
    k = _t(a, b) + _t(b, a) + _ln(a, b) + _ln(b, a) - sq
    return k, jnp.exp(-25.0 / 6.0 + k)


def _G(x, y):
    '''Eq. (18) with its removable singularities at x = 0 and y = 0 handled.'''
    ys, xs = jnp.where(y != 0.0, y, 1.0), jnp.where(x != 0.0, x, 1.0)
    return jnp.where(y != 0.0, y * jnp.arctan(x / ys), 0.0) + jnp.where(x != 0.0, 0.5 * x * jnp.log1p(y ** 2 / xs ** 2), 0.0)


def _b0(u, v, a, b, p, q, current):
    '''Local straight-conductor field B_0 at cross-section point (u, v) (eq. 17).'''
    acc = 0.0
    for su in (1.0, -1.0):
        for sv in (1.0, -1.0):
            gq, gp = _G(b * (v - sv), a * (u - su)), _G(a * (u - su), b * (v - sv))
            acc = acc + su * sv * (gq[..., None] * q - gp[..., None] * p)
    return MU0 * current / (4.0 * jnp.pi * a * b) * acc


def _bkappa(u, v, a, b, p, q, current, k1, k2):
    '''Curvature correction B_kappa at cross-section point (u, v) (eq. 19-20); k1 = kappa . p, k2 = kappa . q.'''
    k1, k2 = k1[..., None], k2[..., None]
    acc = 0.0
    for su in (1.0, -1.0):
        for sv in (1.0, -1.0):
            U, V = u - su, v - sv
            w = a * U ** 2 / b + b * V ** 2 / a
            lw = jnp.log(jnp.where(w > 0.0, w, 1.0))
            Us, Vs = jnp.where(U != 0.0, U, 1.0), jnp.where(V != 0.0, V, 1.0)
            at_uv = jnp.where(U != 0.0, jnp.arctan(b * V / (a * Us)), 0.0)
            at_vu = jnp.where(V != 0.0, jnp.arctan(a * U / (b * Vs)), 0.0)
            c_log, c_al = jnp.where(w > 0.0, U * V * lw, 0.0), jnp.where(w > 0.0, w * lw, 0.0)
            acc = acc + su * sv * (-2.0 * c_log[..., None] * (k1 * q - k2 * p) + c_al[..., None] * (k2 * q - k1 * p)
                                   + (4.0 * a * U ** 2 / b * at_uv)[..., None] * (k2 * p)
                                   - (4.0 * b * V ** 2 / a * at_vu)[..., None] * (k1 * q))
    return MU0 * current / (64.0 * jnp.pi) * acc


def regularised_field(points, xs, dl, current, eps2):
    '''
    Landreman-Hudson regularised Biot-Savart field B_reg (eq. 15) of one coil. The kernel 1/|r|^3 is replaced by
    (|r|^2 + eps2)^(-3/2) with eps2 = delta * a * b, which is smooth, so the trapezoid rule converges spectrally once
    the nodes resolve the length scale sqrt(eps2).

    Parameters
    ----------
    points : jnp.ndarray [M, 3]
    xs, dl : jnp.ndarray [n, 3]
        Nodes and line elements of the coil (see `coil_elements`).
    current : float
    eps2 : float
    '''
    r = points[:, None, :] - xs[None, :, :]
    den = (jnp.sum(r * r, -1) + eps2) ** 1.5
    return MU0 * current / (4.0 * jnp.pi) * jnp.sum(jnp.cross(dl[None], r) / den[..., None], axis=1)


def self_field(coil: FiniteSizeCoil, current: float, width: float, height: float, s: jnp.ndarray,
               u: float, v: float, n_src: int) -> jnp.ndarray:
    '''
    Self field B_reg + B_0 + B_kappa + B_b of a rectangular-cross-section coil (Landreman & Hudson 2018) at the
    point (u, v) of the cross-section at each parameter s. Valid for cross-section dimensions small compared with the
    radius of curvature.

    Parameters
    ----------
    coil : FiniteSizeCoil (single)
        The coil and the frame (`coil.finite_size_frame`) that orients the cross-section.
    current : float
        Total current.
    width, height : float
        Cross-section size along the first (p, radial) and second (q) direction of the frame.
    s : jnp.ndarray
        Evaluation parameters [M]
    u, v : float
        Position in the cross-section in units of half the width/height, u along p and v along q, in [-1, 1].
        (u, v) = (0, 0) is the centre; u = -1 is the edge at -p (for `CentroidFrame` the side facing the centre).
    n_src : int
        Trapezoid nodes of the B_reg integral. The kernel scale is sqrt(delta * width * height), so n_src has to
        resolve that (converges spectrally beyond roughly length / scale).

    Returns
    -------
    jnp.ndarray
        Field [M, 3]
    '''
    x, d1 = coil.position(s), coil.derivative(s)
    d2 = coil.second_derivative(s)
    speed = jnp.linalg.norm(d1, axis=-1, keepdims=True)
    T = d1 / speed
    kvec = (d2 - jnp.sum(d2 * T, -1, keepdims=True) * T) / speed ** 2         # dT/dl
    p, q = jnp.moveaxis(coil.finite_size_frame(s), -2, 0)
    xs, dl = _nodes(coil, n_src)
    _, delta = k_delta(width, height)
    b_reg = regularised_field(x, xs, dl, current, delta * width * height)
    b0 = _b0(u, v, width, height, p, q, current)
    bk = _bkappa(u, v, width, height, p, q, current, jnp.sum(kvec * p, -1), jnp.sum(kvec * q, -1))
    bb = MU0 * current / (8.0 * jnp.pi) * jnp.cross(T, kvec) * (4.0 + 2.0 * jnp.log(2.0) + jnp.log(delta))
    return b_reg + b0 + bk + bb


def max_force_density_on_filament(coils: CoilSet, currents: jnp.ndarray, width: float, height: float, indices: jnp.ndarray,
                                  n_eval: int, n_src: int, n_mut: int) -> jnp.ndarray:
    '''
    Maximum of |I (T x B)| / (width * height) [N/m^3] over the selected coils and n_eval points on their centreline
    (the force on the current filament, spread over the cross-section). B is the regularised self field B_reg of the
    coil (without the local cross-section terms of `self_field`) plus the filament field of all other coils.
    All selected coils are evaluated at once (vmap): the transient memory is about
    n_eval * (n_src + (n_coils - 1) * n_mut) * 48 bytes per coil (measured for 50 coils with n_mut = 256: 8.3 GB all
    at once, 0.17 GB one at a time).

    Parameters
    ----------
    coils : CoilSet
    currents : jnp.ndarray [n_coils]
    width, height : float
    indices : jnp.ndarray
        Coils to evaluate.
    n_eval, n_src, n_mut : int
        Evaluation points per coil, nodes of the self field, nodes of the other coils. The trapezoid rule converges
        geometrically: for the 50 Helias coils (closest approach about 1 m) n_mut = 64, 128, 256 gives relative errors of
        8e-5, 2e-7, 1e-12 against n_mut = 512. Closer coils need more nodes. The maximum is taken over the n_eval
        sampled points, so it is a lower bound of the true maximum (halving n_eval = 256 changed it by 1e-4).
    '''
    _, delta = k_delta(width, height)
    consts = (coils, currents, coil_elements(coils, n_mut), jnp.arange(n_eval) / n_eval, delta * width * height, width * height)
    return jnp.max(jax.vmap(_force_one, in_axes=(0, None, None))(indices, consts, n_src))


def _force_one(i, consts, n_src):
    coils, currents, (pos, dl_all), sn, eps2, area = consts
    c, j = coils[i], _others(currents.shape[0], i)
    x, (xs, dl) = c.position(sn), _nodes(c, n_src)
    b = regularised_field(x, xs, dl, currents[i], eps2) + _field_of_nodes(pos[j], dl_all[j], currents[j], x)
    return jnp.max(jnp.linalg.norm(currents[i] * jnp.cross(c.tangent(sn), b) / area, axis=-1))


# ---------------------------------------------------------------------------------------------------------------------
#  Inductance
# ---------------------------------------------------------------------------------------------------------------------
def inductance_matrix(coils: CoilSet, width: float, height: float, n_rows: int, n_mut: int, n_self: int) -> jnp.ndarray:
    '''
    Rows of the inductance matrix [H]. The off-diagonal entries are Neumann integrals
    L_ij = 1e-7 int int dl_i . dl_j / |x_i - x_j| (trapezoid rule, n_mut nodes). The diagonal uses the softened kernel
    1 / sqrt(|x_i - x_j|^2 + delta * width * height) (n_self nodes). sqrt(delta * width * height) is the geometric mean
    distance of the cross-section, which already contains the internal inductance of a uniform current, so e.g. a ring
    gives mu0 R (ln(8R/GMD) - 2).

    Parameters
    ----------
    coils : CoilSet
    width, height : float
        Rectangular cross-section (all coils identical).
    n_rows : int
        Only compute the rows of the first n_rows coils (all columns): n_rows = n_coils for the full matrix, fewer for
        a periodic coil set (see `total_inductance`).
    n_mut, n_self : int
        Nodes per coil of the mutual and self terms. For the 50 Helias coils n_mut = 64, 128, 256 gives relative errors of
        7.6e-5, 2.6e-7, 1e-12 against n_mut = 512, and n_self = 128, 256 gives 1.1e-6, 4e-11 against 1024.

    Returns
    -------
    jnp.ndarray
        [n_rows, n_coils]
    '''
    _, delta = k_delta(width, height)
    pos, dl = coil_elements(coils, n_mut)
    pos_s, dl_s = coil_elements(CoilSet(coils[:n_rows]), n_self)
    consts = (pos, dl, pos_s, dl_s, delta * width * height)
    return jax.vmap(_inductance_row, in_axes=(0, None))(jnp.arange(n_rows), consts)


def _mutual(j, i, pos, dl):
    d2 = jnp.sum((pos[i][:, None] - pos[j][None]) ** 2, -1)          # safe sqrt: j == i has d2 = 0
    return 1e-7 * jnp.sum((dl[i] @ dl[j].T) / jnp.sqrt(jnp.where(d2 > 0, d2, 1.0)))


def _inductance_row(i, consts):
    pos, dl, pos_s, dl_s, eps2 = consts
    x, d_ = pos_s[i], dl_s[i]
    d2 = jnp.sum((x[:, None] - x[None]) ** 2, -1) + eps2
    own = 1e-7 * jnp.sum((d_ @ d_.T) / jnp.sqrt(d2))
    n_c = pos.shape[0]
    return jnp.where(jnp.arange(n_c) == i, own, jax.vmap(_mutual, in_axes=(0, None, None, None))(jnp.arange(n_c), i, pos, dl))


def total_inductance(coils: CoilSet, width: float, height: float, n_rows: int, n_mut: int, n_self: int) -> jnp.ndarray:
    '''
    Sum of all entries of the inductance matrix [H], i.e. the inductance of the coils connected in series (equal
    currents). The first n_rows coils have to be mapped onto all the others by a symmetry of the coil set (n_rows
    divides n_coils; n_rows = n_coils for no symmetry): the rows of the first n_rows coils are summed and multiplied by
    n_coils / n_rows. See `inductance_matrix`.
    '''
    assert coils.n_coils % n_rows == 0, "n_rows must divide the number of coils"
    return (coils.n_coils // n_rows) * jnp.sum(inductance_matrix(coils, width, height, n_rows, n_mut, n_self))
