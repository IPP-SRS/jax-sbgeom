'''
Magnetic fields and inductances of coils with a finite rectangular cross-section.

Filament fields use the periodic trapezoid rule on the analytic curve (spectrally convergent for smooth coils) and
`jax_sbgeom.coils.biot_savart.biot_savart`. The self field of a finite-size coil follows

    Landreman & Hudson, "Magnetic fields of finite-cross-section coils" (2018): the regularised Biot-Savart integral
    (kernel softened by delta * a * b) plus local terms for the rectangular cross-section (equation numbers refer to the paper).

Coils are a `CoilSet`. The orientation of the rectangular cross-section is the `finite_size_frame` of a
`FiniteSizeCoil` / `FiniteSizeCoilSet`, so the user chooses the frame (`CentroidFrame`, `RotationMinimizedFrame`, ...).
All quantities are SI and differentiable.
'''
from functools import partial

import jax
import jax.numpy as jnp
from .base_coil import FiniteSizeCoil
from .biot_savart import biot_savart, Constant
from .coilset import CoilSet, FiniteSizeCoilSet

MU0 = Constant.MU_0


def _nodes(coil, n: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    '''Positions x(s_i) and line elements dl_i = x'(s_i) / n [n, 3] of n equally spaced nodes of one coil.'''
    s = jnp.arange(n) / n
    return jax.vmap(coil.position)(s), jax.vmap(jax.jacfwd(coil.position))(s) / n


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
def filament_field(coils: CoilSet, currents: jnp.ndarray, points: jnp.ndarray, n: int = 256) -> jnp.ndarray:
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
               u: float = 0.0, v: float = 0.0, n_src: int = 1024) -> jnp.ndarray:
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
    x, d1, d2 = (jax.vmap(f)(s) for f in (coil.position, jax.jacfwd(coil.position), jax.jacfwd(jax.jacfwd(coil.position))))
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


def max_force_density(coils: CoilSet, currents: jnp.ndarray, width: float, height: float, indices=None,
                      n_eval: int = 256, n_src: int = 1024, n_mut: int = 256) -> jnp.ndarray:
    '''
    Maximum of |I (T x B)| / (width * height) [N/m^3] over the selected coils and n_eval points on their centreline.
    B is the regularised self field B_reg of the coil plus the filament field of all other coils.

    Parameters
    ----------
    coils : CoilSet
    currents : jnp.ndarray [n_coils]
    width, height : float
    indices : jnp.ndarray, optional
        Coils to evaluate (default: all).
    n_eval, n_src, n_mut : int
        Evaluation points per coil, nodes of the self field, nodes of the other coils.
    '''
    indices = jnp.arange(coils.n_coils) if indices is None else indices
    _, delta = k_delta(width, height)
    consts = (coils, currents, coil_elements(coils, n_mut), jnp.arange(n_eval) / n_eval, delta * width * height, width * height)
    return jnp.max(jax.lax.map(partial(_force_one, consts, n_src), indices))


def _force_one(consts, n_src, i):
    coils, currents, (pos, dl_all), sn, eps2, area = consts
    c, j = coils[i], _others(currents.shape[0], i)
    x, (xs, dl) = c.position(sn), _nodes(c, n_src)
    b = regularised_field(x, xs, dl, currents[i], eps2) + _field_of_nodes(pos[j], dl_all[j], currents[j], x)
    return jnp.max(jnp.linalg.norm(currents[i] * jnp.cross(c.tangent(sn), b) / area, axis=-1))


def edge_field_norm(coils: FiniteSizeCoilSet, currents: jnp.ndarray, indices: jnp.ndarray, width: float,
                    height: float, n_eval: int = 256, n_src: int = 1024, n_mut: int = 256) -> jnp.ndarray:
    '''
    |B| [len(indices), n_eval] at the plasma-facing edge (u = -1, v = 0) of the winding pack of the selected coils: the
    self field (`self_field`) plus the filament field of all other coils at x - p * width / 2, where p is the first
    direction of the finite-size frame of the coils.

    Parameters
    ----------
    coils : FiniteSizeCoilSet
    currents : jnp.ndarray [n_coils]
    indices : jnp.ndarray
        Coils to evaluate.
    width, height : float
    n_eval, n_src, n_mut : int
        Evaluation points per coil, nodes of the self field, nodes of the other coils.
    '''
    consts = (coils, currents, coil_elements(coils, n_mut), jnp.arange(n_eval) / n_eval, width, height)
    return jax.lax.map(partial(_edge_one, consts, n_src), indices)


def _edge_one(consts, n_src, i):
    coils, currents, (pos, dl), s, width, height = consts
    c, j = coils[i], _others(currents.shape[0], i)
    edge = c.position(s) - c.finite_size_frame(s)[:, 0] * width / 2.0
    bs = self_field(c, currents[i], width, height, s, u=-1.0, n_src=n_src)
    return jnp.linalg.norm(bs + _field_of_nodes(pos[j], dl[j], currents[j], edge), axis=-1)


# ---------------------------------------------------------------------------------------------------------------------
#  Inductance
# ---------------------------------------------------------------------------------------------------------------------
def inductance_matrix(coils: CoilSet, width: float, height: float, n_rows: int | None = None, n_mut: int = 256,
                      n_self: int = 1024, internal_term: bool = False) -> jnp.ndarray:
    '''
    Rows of the inductance matrix [H]. The off-diagonal entries are Neumann integrals
    L_ij = 1e-7 int int dl_i . dl_j / |x_i - x_j| (trapezoid rule, n_mut nodes). The diagonal uses the softened kernel
    1 / sqrt(|x_i - x_j|^2 + delta * width * height) (n_self nodes). sqrt(delta * width * height) is the geometric mean
    distance of the cross-section, which already contains the internal inductance of a uniform current, so e.g. a ring
    gives mu0 R (ln(8R/GMD) - 2). `internal_term=True` adds 1e-7 * length / 2 (mu0 R / 4 for a ring) on top, which
    double counts it (this was done in the code this was ported from).

    Parameters
    ----------
    coils : CoilSet
    width, height : float
        Rectangular cross-section (all coils identical).
    n_rows : int, optional
        Only compute the rows of the first n_rows coils (default: all). Use for periodic coil sets.
    n_mut, n_self : int
        Nodes per coil of the mutual and self terms.
    internal_term : bool
        Add the internal inductance 1e-7 * length / 2 to the diagonal (default False, see above).

    Returns
    -------
    jnp.ndarray
        [n_rows, n_coils]
    '''
    n_rows = coils.n_coils if n_rows is None else n_rows
    _, delta = k_delta(width, height)
    pos, dl = coil_elements(coils, n_mut)
    pos_s, dl_s = coil_elements(CoilSet(coils[:n_rows]), n_self)
    consts = (pos, dl, pos_s, dl_s, delta * width * height)
    return jax.lax.map(partial(_inductance_row, consts, internal_term), jnp.arange(n_rows))


def _mutual(j, i, pos, dl):
    d2 = jnp.sum((pos[i][:, None] - pos[j][None]) ** 2, -1)          # safe sqrt: j == i has d2 = 0
    return 1e-7 * jnp.sum((dl[i] @ dl[j].T) / jnp.sqrt(jnp.where(d2 > 0, d2, 1.0)))


def _inductance_row(consts, internal_term, i):
    pos, dl, pos_s, dl_s, eps2 = consts
    x, d_ = pos_s[i], dl_s[i]
    d2 = jnp.sum((x[:, None] - x[None]) ** 2, -1) + eps2
    own = 1e-7 * jnp.sum((d_ @ d_.T) / jnp.sqrt(d2))
    if internal_term:
        own = own + 0.5e-7 * jnp.sum(jnp.linalg.norm(d_, axis=-1))
    n_c = pos.shape[0]
    return jnp.where(jnp.arange(n_c) == i, own, jax.vmap(_mutual, in_axes=(0, None, None, None))(jnp.arange(n_c), i, pos, dl))


def total_inductance(coils: CoilSet, width: float, height: float, n_rows: int | None = None, repeats: int = 1,
                     n_mut: int = 256, n_self: int = 1024, internal_term: bool = False) -> jnp.ndarray:
    '''
    Sum of all entries of the inductance matrix [H], i.e. the inductance of the coils connected in series (equal
    currents). For a coil set invariant under a symmetry that maps the first n_rows coils onto all the others, pass
    n_rows and repeats = n_coils / n_rows: the rows of the first n_rows coils are summed and multiplied by repeats.
    See `inductance_matrix`.
    '''
    return repeats * jnp.sum(inductance_matrix(coils, width, height, n_rows, n_mut, n_self, internal_term))
