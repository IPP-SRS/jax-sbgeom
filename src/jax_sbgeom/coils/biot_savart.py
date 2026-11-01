### Experimental!!
import jax.numpy as jnp
import jax
from functools import partial
from jax_sbgeom.jax_utils.utils import map_chunked
class Constant:
    MU_0 = 4e-7 * jnp.pi  # Permeability of free space

@jax.jit
def _biot_savart_internal(current : float, position : jnp.ndarray, delta_line_segment : jnp.ndarray, query_point : jnp.ndarray):
    '''
    Biot-Savart law for a single line element and single query point

    Parameters
    ----------
    current : float
        Current in the coil
    position : jnp.ndarray
        Position of the line element [3,]
    tangent : jnp.ndarray
        Tangent vector of the line element [3,]
    query_point : jnp.ndarray
        Point at which to evaluate the field [3,]

    Returns
    -------
    jnp.ndarray
        Magnetic field at the query point [3,]
    '''
    r_vec = query_point - position
    r_mag = jnp.linalg.norm(r_vec, axis=-1)
    dl_x_r = jnp.cross(delta_line_segment, r_vec, axis=-1)    
    B = (Constant.MU_0 * current) / (4.0 * jnp.pi) * dl_x_r / (r_mag**3)[..., None]    
    return B

_vmapped_biot_savart = jax.jit(jax.vmap(_biot_savart_internal, in_axes=(0, 0, 0, None)))


def biot_savart_single(currents : jnp.ndarray, positions : jnp.ndarray, delta_line_segments : jnp.ndarray, query_point : jnp.ndarray):
    '''
    Biot-Savart law for multiple line elements and a single query point

    Parameters
    ----------
    currents : jnp.ndarray
        Currents in the coil segments [N,]
    positions : jnp.ndarray
        Positions of the line elements [N, 3]
    delta_line_segments : jnp.ndarray
        Line element vectors [N, 3]
    query_point : jnp.ndarray
        Point at which to evaluate the field [3,]

    Returns
    -------
    jnp.ndarray
        Magnetic field at the query point [3,]
    '''    
    B_segments = _vmapped_biot_savart(currents, positions, delta_line_segments, query_point) # shape (N, 3)
    B_total = jnp.sum(B_segments, axis=0) # shape (3,)
    return B_total

_biot_savart_multiple = jax.jit(jax.vmap(biot_savart_single, in_axes=(None, None, None, 0)))

def _field_at_point(currents, positions, delta_line_segments, delta_cross_positions, x):
    '''Field at ONE point x from all elements: sum_j w_j dl_j x (x - p_j) = (sum_j w_j dl_j) x x - sum_j w_j (dl_j x p_j),
    w_j = I_j / |x - p_j|^3. Under the vmap of map_chunked the two sums are matrix products. Coordinates must be centred
    (see biot_savart): the rounding error is ~1e-16 |x| / min_j |x - p_j| relative.'''
    r2 = (x[0] - positions[:, 0]) ** 2 + (x[1] - positions[:, 1]) ** 2 + (x[2] - positions[:, 2]) ** 2
    w = currents / (r2 * jnp.sqrt(r2))
    return Constant.MU_0 / (4.0 * jnp.pi) * (jnp.cross(w @ delta_line_segments, x) - w @ delta_cross_positions)


def _field_at_point_consts(x, consts):
    return _field_at_point(*consts, x)


@partial(jax.jit, static_argnums = (4,))
def biot_savart(currents : jnp.ndarray, positions : jnp.ndarray, delta_line_segments : jnp.ndarray, query_points : jnp.ndarray, batch_size : int):
    '''
    Biot-Savart law for multiple line elements and multiple query points

    The query points are processed in chunks of `batch_size` (`map_chunked`), so only [batch_size, N] intermediates exist,
    in the forward and in the backward pass. Both point sets are shifted by the centroid of the elements first, which keeps
    the rounding error of the matrix-product form small (see `_field_at_point`); points on a coil are not supported.

    Parameters
    ----------
    currents : jnp.ndarray
        Currents in the coil segments [N,]
    positions : jnp.ndarray
        Positions of the line elements [N, 3]
    delta_line_segments : jnp.ndarray
        Line element vectors [N, 3]
    query_points : jnp.ndarray
        Points at which to evaluate the field [M, 3]
    batch_size : int
        Query points per chunk (static, see `map_chunked`; M: all at once). Transient memory is about 1 (forward) and 9
        (gradient) arrays of batch_size * N * 8 bytes (N = 12800, compiled, CPU).

    Returns
    -------
    jnp.ndarray
        Magnetic field at the query points [M, 3]
    '''
    centre = jnp.sum(positions, axis=0) / max(positions.shape[0], 1)     # no elements (e.g. the other coils of a single coil): zero field
    p = positions - centre
    return map_chunked(_field_at_point_consts, query_points - centre,
                       (currents, p, delta_line_segments, jnp.cross(delta_line_segments, p)), batch_size)


def create_coilset_total_arrays(jax_coilset, currents, number_of_samples_per_coil):
    coil_samples     = jax_coilset.position(jnp.linspace(0,1, number_of_samples_per_coil, endpoint=False))    
    currents_stacked = jnp.stack([currents] * number_of_samples_per_coil, axis=-1)    
    coil_diff        = jnp.roll(coil_samples, -1, axis=1) - coil_samples
    return currents_stacked.reshape(-1) , coil_samples.reshape(-1,3), coil_diff.reshape(-1,3)