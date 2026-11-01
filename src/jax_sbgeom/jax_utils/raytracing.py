import jax 
import jax.numpy as jnp
from typing import Type
from dataclasses import dataclass
from functools import partial

def _get_norm_centroids(positions : jnp.ndarray, connectivity : jnp.ndarray) -> jnp.ndarray:
    '''
    Get normalized centroids for the triangles defined by the connectivity on the positions.

    Parameters
    ------------
    positions : jnp.ndarray
        (N, 3) array of vertex positions
    connectivity : jnp.ndarray
        (M, 3) array of connectivity indices
    Returns
    ------------
    normalized_centroids : jnp.ndarray
        (M, 3) array of normalized centroids
    '''
    centroids = jnp.mean(positions[connectivity], axis=1)  # (M, 3)

    r_min     = jnp.min(centroids, axis=0)
    r_max     = jnp.max(centroids, axis=0)  

    safe_divisor = jnp.where(r_max - r_min == 0, 1.0, r_max - r_min)

    normalized_centroids = (centroids - r_min) / safe_divisor    
    return normalized_centroids


def _create_morton_codes(normalized_positions : jnp.ndarray) -> jnp.ndarray:
    '''
    Create 32 bit morton codes for the triangles defined by the connectivity on the positions.

    See https://developer.nvidia.com/blog/thinking-parallel-part-iii-tree-construction-gpu/

    Parameters
    ------------
    positions : jnp.ndarray
        (N, 3) array of vertex positions
    connectivity : jnp.ndarray
        (M, 3) array of connectivity indices
    Returns
    ------------
    morton_codes : jnp.ndarray
        (M,) array of morton codes
    '''    
    return _morton_3d(normalized_positions)

_N_BIT       = 10
_MORTON_CLIP = 2 ** _N_BIT - 1
_MORTON_SCALE = 2.0 ** _N_BIT

def _expand_bits(v):
    # ensure unsigned 32-bit
    v = jnp.asarray(v, dtype=jnp.uint32)

    # Use JAX integer constants (avoid raw Python ints in bitwise expressions)
    c1 = jnp.uint32(0x00010001)  # 65537
    m1 = jnp.uint32(0xFF0000FF)
    c2 = jnp.uint32(0x00000101)  # 257
    m2 = jnp.uint32(0x0F00F00F)
    c3 = jnp.uint32(0x00000011)  # 17
    m3 = jnp.uint32(0xC30C30C3)
    c4 = jnp.uint32(0x00000005)  # 5
    m4 = jnp.uint32(0x49249249)

    v = (v * c1) & m1
    v = (v * c2) & m2
    v = (v * c3) & m3
    v = (v * c4) & m4

    return v

def _morton_3d(norm_centroids):
    int_coords = jnp.clip(jnp.floor(norm_centroids * _MORTON_SCALE).astype(jnp.uint32), 0, jnp.astype(_MORTON_CLIP, jnp.uint32))
    return _expand_bits(int_coords[:, 0]) * 4 + _expand_bits(int_coords[:, 1]) * 2 + _expand_bits(int_coords[:, 2])

def _common_prefix_length(a : jnp.ndarray, b : jnp.ndarray) -> jnp.ndarray:
    '''
    Compute the common prefix length between two unsigned integers a and b.

    Uses jax.lax.clz

    Parameters
    -----------
    a : jnp.ndarray
        Unsigned integer array
    b : jnp.ndarray
        Unsigned integer array
    Returns
    ----------- 
    jnp.ndarray
        Common prefix length array (scalar)

    '''
    return jax.lax.clz(jnp.bitwise_xor(a,b))


def _delta_ij(sorted_morton_codes : jnp.ndarray, i : int, j : int, int_dtype : jnp.dtype = jnp.int32):
    '''
    Compute the delta_ij function as defined in [1].


    [1] Karras, T. (2012, June). Maximizing parallelism in the construction of BVHs, octrees, and k-d trees. In Proceedings of the Fourth ACM SIGGRAPH/Eurographics Conference on High-Performance Graphics (pp. 33-37).

    Parameters 
    -----------
    sorted_morton_codes : jnp.ndarray
        Sorted morton codes array
    i : int
        Index i
    j : int
        Index j
    int_dtype : jnp.dtype
        Integer dtype for the output
    Returns
    -----------
    jnp.ndarray
        Delta_ij value (scalar)    
    '''

    # To ensure no OOB access, first clip the indices
    i_safe = jnp.clip(i, 0, sorted_morton_codes.shape[0]-1)
    j_safe = jnp.clip(j, 0, sorted_morton_codes.shape[0]-1)

    # However, we need to have a mask to indicate whether the original indices were OOB. If so, the result should be -1
    safe_mask = jnp.logical_and( jnp.logical_and( i >= 0, i < sorted_morton_codes.shape[0]), 
                                 jnp.logical_and( j >= 0, j < sorted_morton_codes.shape[0]))
    
    # Furthermore, if the morton codes are the same, we should fallback to the _common_prefix_length + the base common prefix length of the indices themselves.
    # (jnp.where computes eagerly anyway, so need to compute all of them)
    equal_mask                   = jnp.equal(sorted_morton_codes[i_safe], sorted_morton_codes[j_safe])
    common_prefix_length_morton  = jnp.astype(_common_prefix_length(sorted_morton_codes[i_safe],              sorted_morton_codes[j_safe]), int_dtype)
    common_prefix_length_indices = jnp.astype(_common_prefix_length(i_safe.astype(sorted_morton_codes.dtype), j_safe.astype(sorted_morton_codes.dtype)), int_dtype)
    
    return jnp.where(safe_mask,                 
                        jnp.where(equal_mask, 
                                  common_prefix_length_morton + common_prefix_length_indices,
                                  common_prefix_length_morton
                                 )
                     , -1
                     )

def _lmax_condition(state):
    l, step, idx, d, delta_min, sorted_morton_codes, max_doublings = state
    return jnp.logical_and(_delta_ij(sorted_morton_codes, idx, idx + d * l) > delta_min, step < max_doublings)

def _lmax_body(state):
    l, step, *rest = state
    return (l * 2, step + 1, *rest)

def _binary_search_condition(state):
    return state[0] > 0

def _binary_search_body(state):
    t, l, idx, d, delta_min, sorted_morton_codes = state
    l_carry = jnp.where(_delta_ij(sorted_morton_codes, idx, idx + d * (l+t)) > delta_min, l + t, l)
    return t // 2, l_carry, idx, d, delta_min, sorted_morton_codes

def _split_condition(state):
    return ~state[2] # this continues including the t=1 case just once

def _split_body(state):
    t, s, _, idx, d, delta_node, sorted_morton_codes = state
    s_carry = jnp.where(_delta_ij(sorted_morton_codes, idx, idx + d * (s + t)) > delta_node, s + t, s)
    t_carry = (t+1)  // 2
    return t_carry, s_carry, jnp.equal(t, 1), idx, d, delta_node, sorted_morton_codes

def _radix_node(idx, sorted_morton_codes):
    '''Children (left, right) of the internal node idx of the radix tree (Karras 2012).'''
    N               = sorted_morton_codes.shape[0]
    morton_order_type = idx.dtype
    max_doublings   = jnp.ceil(jnp.log2(N)).astype(morton_order_type) + 2
    l0              = jnp.astype(1, morton_order_type)

    d = jnp.sign(_delta_ij(sorted_morton_codes, idx + 1, idx) - _delta_ij(sorted_morton_codes, idx, idx - 1))
    delta_min = _delta_ij(sorted_morton_codes, idx, idx - d)

    # Exponential search to find upper bound on lmax
    l_final = jax.lax.while_loop(_lmax_condition, _lmax_body, (l0, 0, idx, d, delta_min, sorted_morton_codes, max_doublings))[0]

    t_final, l_final = jax.lax.while_loop(_binary_search_condition, _binary_search_body,
                                          (l_final // 2, jnp.astype(0, morton_order_type), idx, d, delta_min, sorted_morton_codes))[:2]

    j_idx = idx + d * l_final

    # Split position
    delta_node = _delta_ij(sorted_morton_codes, idx, j_idx)
    s0 = jnp.astype(0, morton_order_type)

    s_final = jax.lax.while_loop(_split_condition, _split_body, ((l_final+1) // 2, s0, False, idx, d, delta_node, sorted_morton_codes))[1]
    gamma   = idx + s_final * d + jnp.minimum(d, 0)

    left_idx  = jnp.where(jnp.minimum(idx, j_idx) == gamma, gamma, gamma + N)
    right_idx = jnp.where(jnp.maximum(idx, j_idx) == gamma + 1, gamma + 1, gamma + 1 + N)
    return left_idx, right_idx

@jax.jit
def _create_parallel_binary_radix_tree(morton_codes : jnp.ndarray):
    '''
    Create a parallel binary radix tree as defined in [1].
     
    [1] Karras, T. (2012, June). Maximizing parallelism in the construction of BVHs, octrees, and k-d trees. In Proceedings of the Fourth ACM SIGGRAPH/Eurographics Conference on High-Performance Graphics (pp. 33-37).

    Parameters
    '''
    # See https://developer.nvidia.com/blog/parallelforall/wp-content/uploads/2012/11/karras2012hpg_paper.pdf

    N = morton_codes.shape[0]                      
    morton_order        = jnp.argsort(morton_codes) # int_dtype
    sorted_morton_codes = morton_codes[morton_order]# uint_dtype

    morton_order_type   = morton_order.dtype        # type(morton_order): used for indexing
    
    internal_nodes = N - 1 # size_type
    internal_idx   = jnp.arange(0, internal_nodes, dtype = morton_order_type) # size_type

    left_idx, right_idx = jax.vmap(_radix_node, in_axes=(0, None))(internal_idx, sorted_morton_codes)

    total_left_idx = jnp.concatenate([jnp.arange(N, dtype = morton_order_type), left_idx])
    total_right_idx = jnp.concatenate([jnp.arange(N, dtype = morton_order_type), right_idx])
    return total_left_idx, total_right_idx, morton_order

def _check_tree_condition(state):
    total_computed, i_loop, left_idx, _ = state
    N_leaves = (left_idx.shape[0] +1) //2
    return jnp.any(~total_computed[N_leaves:]) & (i_loop < 2 * N_leaves)

def _check_tree_body(state):
    computed, i_loop, left_idx, right_idx = state
    computed = computed[left_idx] & computed[right_idx]
    return computed, i_loop + 1, left_idx, right_idx

@jax.jit
def _check_binary_radix_tree(left_idx, right_idx):
    '''
    Convenience function to check whether a binary radix tree is actually valid (i.e. one can visit all nodes from the leafs to the root).

    Parameters
    left_idx : jnp.ndarray
        Left child indices for each node
    right_idx : jnp.ndarray
        Right child indices for each node
    '''
    N_leaves = (left_idx.shape[0] +1) //2
    
    initial_available = jnp.zeros((2 *  N_leaves - 1), dtype = jnp.bool) 
    initial_available = initial_available.at[:N_leaves].set(True)

    total_computed = jax.lax.while_loop(_check_tree_condition, _check_tree_body, (initial_available, 0, left_idx, right_idx))[0]
    return jnp.sum(total_computed) == total_computed.shape[0]

@jax.jit
def _create_aabb(primitive_coordinates : jnp.ndarray) -> jnp.ndarray:
    '''
    Create axis-aligned bounding box for a given primitive.

    Parameters
    ----------
    primitive:  jnp.ndarray of shape (N, M, 3)
        N coordinates of a primitive with shape M

    Returns
    -------
    jnp.ndarray of shape (N, 2, 3)
      representing min and max corners of the AABB.
    '''
    min_corner = jnp.min(primitive_coordinates, axis=-2)
    max_corner = jnp.max(primitive_coordinates, axis=-2)
    return jnp.stack([min_corner, max_corner], axis = 1)

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class BVH:
    '''
    Dataclass representing a Bounding Volume Hierarchy (BVH).

    Build with `build_lbvh`.
    '''
    left_idx      : jnp.ndarray # left child indices for each node
    right_idx     : jnp.ndarray # right child indices for each node
    aabb          : jnp.ndarray # AABB for each node
    leafs         : jnp.ndarray # bool array
    order         : jnp.ndarray # order from primitives -> BVH
    inverse_order : jnp.ndarray # order from BVH -> primitives


def _aabb_condition(state):
    return ~jnp.all(state[1])

def _compute_aabb_compound(aabb, left_i, right_i):
    return jnp.stack([
        jnp.minimum(aabb[left_i,0], aabb[right_i,0]),
        jnp.maximum(aabb[left_i,1], aabb[right_i,1])
    ], axis = 1)

def _aabb_compute(state):
    aabb, computed, aabb_leaves_sorted, left_internal, right_internal = state
    N              = aabb_leaves_sorted.shape[0]
    new_aabb       = _compute_aabb_compound(aabb, left_internal, right_internal)
    new_computed   = computed[left_internal] & computed[right_internal]
    return (jnp.concatenate([aabb_leaves_sorted, new_aabb], axis=0), jnp.concatenate([jnp.ones((N,), dtype=bool), new_computed], axis=0),
            aabb_leaves_sorted, left_internal, right_internal)

@jax.jit
def build_lbvh(positions, connectivity):
    '''
    Build a Linear Bounding Volume Hierarchy (LBVH) for a given set of 3D positions and connectivity.

    Parameters
    ----------
    positions : jnp.ndarray
        Array of shape (N, 3) with point coordinates.
    connectivity : jnp.ndarray
        Array of shape (M, K) with primitive connectivity indices.

    Returns
    -------
    BVH
        LBVH structure with child indices, AABBs, leaf mask, and ordering maps.
    '''
    normalized_centroids  = _get_norm_centroids(positions, connectivity)
    morton_codes          = _create_morton_codes(normalized_centroids)
    N                     = morton_codes.shape[0]
    

    left_idx, right_idx, morton_order  = _create_parallel_binary_radix_tree(morton_codes)        
    aabb_leaves                        = _create_aabb(positions[connectivity])
    aabb_leaves_sorted                 = aabb_leaves[morton_order] 

    left_internal  = left_idx [N:]
    right_internal = right_idx[N:]

    initial_aabb        = jnp.concatenate([aabb_leaves_sorted, jnp.zeros((N-1,2,3))], axis=0)
    initial_available   = jnp.concatenate([jnp.ones((N,), dtype=bool), jnp.zeros((N-1,), dtype=bool)], axis=0)
    
    aabb_total          = jax.lax.while_loop(_aabb_condition, _aabb_compute, (initial_aabb, initial_available, aabb_leaves_sorted, left_internal, right_internal))[0]
    
    return BVH(left_idx, right_idx, 
               aabb_total, initial_available, 
               morton_order, jnp.argsort(morton_order))

def _point_in_aabb(point : jnp.ndarray, aabb : jnp.ndarray) -> jnp.ndarray:
    """Return whether a point lies inside an axis-aligned bounding box."""
    min_corner = aabb[0]
    max_corner = aabb[1]
    inside = jnp.all((point >= min_corner) & (point <= max_corner))
    return inside


points_in_aabbs   = jax.vmap(jax.vmap(_point_in_aabb, in_axes=(0, None)), in_axes = (None, 0))

points_in_aabb    = jax.vmap(_point_in_aabb, in_axes=(0, None))

points_in_aabbvec = jax.vmap(_point_in_aabb, in_axes=(0, 0))

def _traversal_condition(state):
    stack_idx, _, _, _, loop_idx, bvh = state[:6]
    N_leafs = (bvh.leafs.shape[0] + 1) // 2
    return jnp.any(stack_idx >= 0) & (loop_idx < N_leafs)

def _probe_loop(state):
    stack_idx, stack, hits, no_hits, loop_idx, bvh, points = state
    n_points_arange = jnp.arange(points.shape[0])

    current_idx  = stack[n_points_arange, stack_idx]                


    left_idx     = bvh.left_idx[ current_idx]
    right_idx    = bvh.right_idx[current_idx]

    left_contains  = points_in_aabbvec(points, bvh.aabb[left_idx])    # shape (N_points, )
    right_contains = points_in_aabbvec(points, bvh.aabb[right_idx])  # shape (N_points, )

    hits_with_left = jnp.where(jnp.logical_and(left_contains, bvh.leafs[left_idx])[:,None], 
         hits.at[n_points_arange, no_hits].set(left_idx), hits)

    no_hits_with_left = jnp.where(jnp.logical_and(left_contains, bvh.leafs[left_idx]),
        no_hits + 1, no_hits)

    hits_with_both    = jnp.where(jnp.logical_and(right_contains, bvh.leafs[right_idx])[:,None], 
         hits_with_left.at[n_points_arange, no_hits_with_left].set(right_idx), hits_with_left)

    no_hits_with_both = jnp.where(jnp.logical_and(right_contains, bvh.leafs[right_idx]),
        no_hits_with_left + 1, no_hits_with_left)


    traverse_left    = jnp.logical_and(left_contains, jnp.logical_not(bvh.leafs[left_idx]))
    traverse_right   = jnp.logical_and(right_contains, jnp.logical_not(bvh.leafs[right_idx]))

    transverse_any = jnp.logical_or(traverse_left, traverse_right)
    transverse_both = jnp.logical_and(traverse_left, traverse_right)

    new_stack_idx = stack_idx + transverse_both.astype(jnp.int32) - (~transverse_any).astype(jnp.int32)

    new_stack = jnp.where(
        transverse_any[:, None],
        jnp.where(
            transverse_both[:, None],
            stack.at[n_points_arange, stack_idx + 1].set(right_idx).at[n_points_arange, stack_idx].set(left_idx),
            jnp.where(
                traverse_left[:, None],
                stack.at[n_points_arange, stack_idx].set(left_idx),
                stack.at[n_points_arange, stack_idx].set(right_idx)
            )
        ),
        stack.at[n_points_arange, stack_idx].set(-1)
    )

    return new_stack_idx, new_stack, hits_with_both, no_hits_with_both, loop_idx + 1, bvh, points



@partial(jax.jit, static_argnums = (2,3))
def _probe_bvh_imp(bvh : BVH, points : jnp.ndarray, stack_size : int = 64, max_hit_size : int = 64):
    '''
    Probe a BVH with a set of points to find which AABBs contain the points.

    Parameters
    -----------
    bvh : BVH
        The BVH to probe.
    points : jnp.ndarray
        The points to probe the BVH with. Shape (N_points, 3)
    stack_size : int
        The size of the stack to use for traversal.
    max_hit_size : int
        The maximum number of hits to record per point.
    Returns
    -----------
    jnp.ndarray
        An array of shape (N_points, max_hit_size) containing the indices of the AABBs that contain each point. 
        If a point hits fewer than max_hit_size AABBs, the remaining entries are filled with -1.
    int 
        The number of traversal loops performed.
    '''

    N_leafs = (bvh.leafs.shape[0] + 1) // 2

    N_points = points.shape[0]
    stack = jnp.full((N_points, stack_size,  ), -1) 
    hits  = jnp.full((N_points, max_hit_size,), -1)    
    
    initial_stack = stack.at[..., 0].set(N_leafs)  # start with root node
    initial_state = (jnp.zeros(N_points, dtype=int), initial_stack, hits,jnp.zeros(N_points, dtype= int),0, bvh, points)
    
    final_stack_idx, final_stack, final_hits, final_no_hits, n_loops, _, _ = jax.lax.while_loop(
        _traversal_condition,
        _probe_loop,
        initial_state)

    return final_hits, n_loops

def probe_bvh(bvh : BVH, points : jnp.ndarray, stack_size : int = 64, max_hit_size : int = 64):
    '''
    Probe a BVH with a set of points to find which AABBs contain the points.

    Parameters
    -----------
    bvh : BVH
        The BVH to probe.
    points : jnp.ndarray
        The points to probe the BVH with. Shape (N_points, 3)
    stack_size : int
        The size of the stack to use for traversal.
    max_hit_size : int
        The maximum number of hits to record per point.
    Returns
    -----------
    jnp.ndarray
        An array of shape (N_points, max_hit_size) containing the indices of the AABBs of the original mesh. (not in the BVH order, but in the original order)
        In other words, bvh.order is indexed by the resulting output of _probe_bvh_imp. If no hit, -1 is returned.

    
    '''    
    final_hits, n_loops = _probe_bvh_imp(bvh, points, stack_size, max_hit_size)
    return jnp.where(final_hits >=0, bvh.order[final_hits], -1)##[:,0]  #[hit_i[hit_i >=0] for hit_i in final_hits] if we want variable length outputs
def ray_intersects_aabb(origin : jnp.ndarray, direction : jnp.ndarray, aabb : jnp.ndarray):
    '''
    Vectorized function to check ray-AABB intersections using the slab method.

    Parameters
    -----------
    origin : jnp.ndarray
        Ray origin of shape (..., 3)
    direction : jnp.ndarray
        Ray direction of shape (..., 3)
    aabb : jnp.ndarray
        Axis-aligned bounding box of shape (..., 2, 3)  
    Returns
    -----------
    jnp.ndarray
        Boolean array indicating whether the ray intersects the AABB.

    '''
    inv_dir = jnp.where(direction == 0.0, jnp.inf, 1.0 / direction)  # precompute to avoid divides
    tmin = (aabb[..., 0, :] - origin) * inv_dir
    tmax = (aabb[..., 1, :] - origin) * inv_dir
    
    # if direction is negative, swap
    t1 = jnp.minimum(tmin, tmax)
    t2 = jnp.maximum(tmin, tmax)

    t_enter = jnp.max(t1, axis=-1)
    t_exit  = jnp.min(t2, axis=-1)
    hit = (t_exit >= jnp.maximum(t_enter, 0.0))    
    hit = jnp.logical_or(hit, jnp.any(jnp.isnan(jnp.array([t_enter, t_exit])))) 
    # ugly; but for slab parallel it ensures it traverses the BVH further and does not miss triangles.
    # this only happens in very very rare cases anyway; almost never is the direction slab parallel (i.e. exactly 0.0 in one of the components).
    
    
    return hit#, t_enter, t_exit

def _ray_loop(state):
    stack_idx, stack, hits, no_hits, loop_idx, bvh, points, directions = state
    n_points_arange = jnp.arange(points.shape[0])

    current_idx  = stack[n_points_arange, stack_idx]                

    left_idx     = bvh.left_idx[ current_idx]
    right_idx    = bvh.right_idx[current_idx]

    left_contains  = ray_intersects_aabb(points, directions, bvh.aabb[left_idx])    # shape (N_points, )
    right_contains = ray_intersects_aabb(points, directions, bvh.aabb[right_idx])   # shape (N_points, )

    hits_with_left = jnp.where(jnp.logical_and(left_contains, bvh.leafs[left_idx])[:,None], 
         hits.at[n_points_arange, no_hits].set(left_idx), hits)

    no_hits_with_left = jnp.where(jnp.logical_and(left_contains, bvh.leafs[left_idx]),
        no_hits + 1, no_hits)

    hits_with_both    = jnp.where(jnp.logical_and(right_contains, bvh.leafs[right_idx])[:,None], 
         hits_with_left.at[n_points_arange, no_hits_with_left].set(right_idx), hits_with_left)

    no_hits_with_both = jnp.where(jnp.logical_and(right_contains, bvh.leafs[right_idx]),
        no_hits_with_left + 1, no_hits_with_left)


    traverse_left    = jnp.logical_and(left_contains, jnp.logical_not(bvh.leafs[left_idx]))
    traverse_right   = jnp.logical_and(right_contains, jnp.logical_not(bvh.leafs[right_idx]))

    transverse_any = jnp.logical_or(traverse_left, traverse_right)
    transverse_both = jnp.logical_and(traverse_left, traverse_right)

    new_stack_idx = stack_idx + transverse_both.astype(jnp.int32) - (~transverse_any).astype(jnp.int32)

    new_stack = jnp.where(
        transverse_any[:, None],
        jnp.where(
            transverse_both[:, None],
            stack.at[n_points_arange, stack_idx + 1].set(right_idx).at[n_points_arange, stack_idx].set(left_idx),
            jnp.where(
                traverse_left[:, None],
                stack.at[n_points_arange, stack_idx].set(left_idx),
                stack.at[n_points_arange, stack_idx].set(right_idx)
            )
        ),
        stack.at[n_points_arange, stack_idx].set(-1)
    )

    return new_stack_idx, new_stack, hits_with_both, no_hits_with_both, loop_idx + 1, bvh, points, directions



@jax.jit
def ray_traversal_bvh(bvh : BVH, points : jnp.ndarray, directions : jnp.ndarray, stack_size : int = 64, max_hit_size : int = 64):
    '''
    Traverse a BVH with a set of rays defined by points and directions.

    Parameters
    -----------
    bvh : BVH
        The BVH to traverse.
    points : jnp.ndarray
        The ray origins. Shape (N_points, 3)
    directions : jnp.ndarray
        The ray directions. Shape (N_points, 3)
    stack_size : int
        The size of the stack to use for traversal.
    max_hit_size : int
        The maximum number of hits to record per ray.
    Returns
    -----------
    jnp.ndarray
        An array of shape (N_points, max_hit_size) containing the indices of the AABBs that the rays intersect. 
        If a ray hits fewer than max_hit_size AABBs, the remaining entries are filled with -1.  
    '''

    N_leafs = (bvh.leafs.shape[0] + 1) // 2

    N_points = points.shape[0]
    stack = jnp.full((N_points, stack_size,  ), -1) 
    hits  = jnp.full((N_points, max_hit_size,), -1)    
    
    initial_stack = stack.at[..., 0].set(N_leafs)  # start with root node
    initial_state = (jnp.zeros(N_points, dtype=int), initial_stack, hits,jnp.zeros(N_points, dtype= int),0, bvh, points, directions)
    
    final_stack_idx, final_stack, final_hits, final_no_hits, n_loops, _, _, _ = jax.lax.while_loop(
        _traversal_condition,
        _ray_loop,
        initial_state)
    return final_hits


def _ray_loop_single(state):
    stack_idx, stack, hits, no_hits, loop_idx, bvh, point, direction = state

    current_idx  = stack[stack_idx]                

    left_idx     = bvh.left_idx[ current_idx]
    right_idx    = bvh.right_idx[current_idx]

    left_contains  = ray_intersects_aabb(point, direction, bvh.aabb[left_idx])    # shape (N_points, )
    right_contains = ray_intersects_aabb(point, direction, bvh.aabb[right_idx])   # shape (N_points, )

    hits_with_left = jnp.where(jnp.logical_and(left_contains, bvh.leafs[left_idx]), 
         hits.at[no_hits].set(left_idx), hits)

    no_hits_with_left = jnp.where(jnp.logical_and(left_contains, bvh.leafs[left_idx]),
        no_hits + 1, no_hits)

    hits_with_both    = jnp.where(jnp.logical_and(right_contains, bvh.leafs[right_idx]), 
         hits_with_left.at[no_hits_with_left].set(right_idx), hits_with_left)

    no_hits_with_both = jnp.where(jnp.logical_and(right_contains, bvh.leafs[right_idx]),
        no_hits_with_left + 1, no_hits_with_left)


    traverse_left    = jnp.logical_and(left_contains, jnp.logical_not(bvh.leafs[left_idx]))
    traverse_right   = jnp.logical_and(right_contains, jnp.logical_not(bvh.leafs[right_idx]))

    transverse_any = jnp.logical_or(traverse_left, traverse_right)
    transverse_both = jnp.logical_and(traverse_left, traverse_right)

    new_stack_idx = stack_idx + transverse_both.astype(jnp.int32) - (~transverse_any).astype(jnp.int32)

    new_stack = jnp.where(
        transverse_any,
        jnp.where(
            transverse_both,
            stack.at[stack_idx + 1].set(right_idx).at[stack_idx].set(left_idx),
            jnp.where(
                traverse_left,
                stack.at[ stack_idx].set(left_idx),
                stack.at[ stack_idx].set(right_idx)
            )
        ),
        stack.at[ stack_idx].set(-1)
    )

    return new_stack_idx, new_stack, hits_with_both, no_hits_with_both, loop_idx + 1, bvh, point, direction        


@jax.jit
def ray_traversal_bvh_single(bvh : BVH, point : jnp.ndarray, direction : jnp.ndarray, stack_size : int = 128, max_hit_size : int = 128):
    '''
    Traverse a BVH with a single ray defined by point and direction.

    Use ray_traversal_bvh_vectorized to handle multiple rays. This is only ~10% slower than ray_traversal_bvh for large number of rays.

    Parameters
    -----------
    bvh : BVH
        The BVH to traverse.
    point : jnp.ndarray
        The ray origin. Shape (3,)
    direction : jnp.ndarray
        The ray direction. Shape (3,)
    stack_size : int
        The size of the stack to use for traversal.
    max_hit_size : int
        The maximum number of hits to record.
    Returns
    -----------
    jnp.ndarray
        An array of shape (max_hit_size,) containing the indices of the AABBs that the ray intersects. 
        If the ray hits fewer than max_hit_size AABBs, the remaining entries are filled with -1.    
    '''

    N_leafs = (bvh.leafs.shape[0] + 1) // 2

    N_points = point.shape[0]
    stack = jnp.full(( stack_size,  ), -1) 
    hits  = jnp.full(( max_hit_size,), -1)    
    
    initial_stack = stack.at[..., 0].set(N_leafs)  # start with root node
    initial_state = (0, initial_stack, hits,0,0, bvh, point, direction)
    
    final_stack_idx, final_stack, final_hits, final_no_hits, n_loops, _, _, _ = jax.lax.while_loop(
        _traversal_condition,
        _ray_loop_single,
        initial_state)
    return final_hits

ray_traversal_bvh_vectorized = jax.jit(jnp.vectorize(ray_traversal_bvh_single, excluded=(0,), signature='(3),(3)->(max_hit_size)'))
# =========================================================================================================

@jax.jit
def ray_triangle_intersection_single(point : jnp.ndarray, direction : jnp.ndarray, triangle : jnp.ndarray, eps=1e-8, eps_bary_center = 1e-8):
    """
    Compute intersections of one ray with one trianlge

    Parameters
    ----------
    point:     jnp.ndarray
      Ray origin. Shape (3,)
    direction: jnp.ndarray
      Ray direction. Shape (3,)
    triangle:  jnp.ndarray
      Triangle vertices. Shape (3, 3)

    Returns
    -------
    t:  
        distance along ray (jnp.inf if no hit)    
    """
    v0 = triangle[0, :]
    v1 = triangle[1, :]
    v2 = triangle[2, :]

    e1 = v1 - v0 # (3,)
    e2 = v2 - v0 # (3,)

    pvec = jnp.cross(direction, e2) #(3,)
    det  = jnp.dot(e1, pvec)  #(,)

    valid_det = jnp.abs(det) > eps #(,)
    inv_det   = jnp.where(valid_det, 1.0 / det, 0.0) #(,)

    tvec      = point - v0 # (3,)
    u         = jnp.dot(tvec, pvec) * inv_det # (,)

    qvec = jnp.cross(tvec, e1) # (3,)
    v = jnp.dot(direction, qvec) * inv_det # (,)

    t =  jnp.dot(e2, qvec) * inv_det #(,)

    mask = (valid_det &
            (u >= -eps_bary_center) &
            (v >= -eps_bary_center) &
            ((u + v) <= 1.0 + eps_bary_center) &
            (t > eps))

    t = jnp.where(mask, t, jnp.inf)
    return t

ray_triangle_intersection_vectorized = jax.jit(jnp.vectorize(ray_triangle_intersection_single, signature='(3),(3),(3,3)->()'))

@jax.jit
def find_minimum_distance_to_mesh(points : jnp.ndarray, directions : jnp.ndarray, mesh):
    '''
    Convenience function to find the minimum distance from rays to a triangle mesh.

    Parameters
    -----------
    points : jnp.ndarray
        Ray origins. Shape (N_points, 3)
    directions : jnp.ndarray
        Ray directions. Shape (N_points, 3)
    mesh : tuple of (positions, connectivity)
        positions:    jnp.ndarray of shape (M, 3) representing the 3D coordinates of mesh vertices.
        connectivity: jnp.ndarray of shape (K, 3) representing the triangle connectivity of the mesh.
    Returns
    -----------
    jnp.ndarray
        An array of shape (N_points,) containing the minimum distance from each ray to the mesh. If no intersection occurs, the distance is jnp.inf.
    '''
    bvh = build_lbvh(mesh[0], mesh[1]) # BVH
    hits_possible = ray_traversal_bvh_vectorized(bvh, points, directions)    
    mesh_total = jnp.moveaxis(mesh[0][mesh[1][bvh.order[hits_possible]]], -3, 0) # we move the possible hits to the front.
    return jnp.nanmin(ray_triangle_intersection_vectorized(points, directions, mesh_total), axis=0)


# ==========================================================================================================
#                                       Closest point on triangle
#===========================================================================================================

def closest_point_on_triangle(p : jnp.ndarray, a : jnp.ndarray, b : jnp.ndarray, c : jnp.ndarray):
    '''
    Compute the closest point on triangle abc to point p.
    Not vectorized itself. Reimplements the Embree reference algorithm
    without explicit if conditionals.

    Parameters
    ----------
    p : jnp.ndarray
        (3,) array of point coordinates
    a : jnp.ndarray
        (3,) array of triangle vertex a coordinates
    b : jnp.ndarray
        (3,) array of triangle vertex b coordinates
    c : jnp.ndarray
        (3,) array of triangle vertex c coordinates
    Returns
    -------
    closest_point : jnp.ndarray
        (3,) array of closest point coordinates
    '''
    # Edge vectors
    ab = b - a
    ac = c - a
    ap = p - a
    
    d1 = jnp.dot(ab, ap)
    d2 = jnp.dot(ac, ap)

    d1_d2_cond = (d1 <= 0.0) & (d2 <= 0.0)
    d1_d2_cond_vec = a

    bp = p - b
    d3 = jnp.dot(ab, bp)
    d4 = jnp.dot(ac, bp)

    d3_d4_cond     = (d3 >= 0.0) & (d4 <= d3)
    d3_d4_cond_vec = b

    cp = p - c

    d5 = jnp.dot(ab, cp)
    d6 = jnp.dot(ac, cp)

    d5_d6_cond     = (d6 >= 0.0) & (d5 <= d6)
    d5_d6_cond_vec = c
    
    # Edge conditions
    vc = d1 * d4 - d3 * d2    
    ab_edge_cond = (vc <= 0.0) & (d1 >= 0.0) & (d3 <= 0.0)
    ab_edge_vec  = a + d1 / (d1 - d3) * ab

    vb = d5 * d2 - d1 * d6
    ac_edge_cond = (vb <= 0.0) & (d2 >= 0.0) & (d6 <= 0.0)
    ac_edge_vec  = a + d2 /(d2-d6) * ac

    va = d3 * d6 - d5 * d4
    bc_edge_cond = (va <=0.0) & ( (d4-d3) >=0.0) & ( (d5 -d6) >=0.0)
    bc_edge_vec  = b + (d4 - d3) /((d4 - d3) + (d5 - d6)) * (c-b)

    # Inside Triangle    
    denom = 1.0 / (va + vb + vc)
    v = vb * denom
    w = vc * denom
    triangle_vec = a + v * ab + w * ac

    return jnp.where(
        d1_d2_cond, d1_d2_cond_vec, 
        jnp.where(
            d3_d4_cond, d3_d4_cond_vec, 
            jnp.where(
                d5_d6_cond, d5_d6_cond_vec, 
                jnp.where(
                    ab_edge_cond, ab_edge_vec,
                    jnp.where(
                        ac_edge_cond, ac_edge_vec,
                        jnp.where(
                            bc_edge_cond, bc_edge_vec, 
                            triangle_vec
                        )
                    )
                )
            )
        )
    )

def _point_aabb_distance(p : jnp.ndarray, aabb : jnp.ndarray):
    '''
    Compute squared distance from point to AABB. Returns 0 if point is inside AABB.
    Parameters
    -----------
    p : jnp.ndarray
        (3,) array of point coordinates
    aabb : jnp.ndarray  
        (2, 3) array of AABB min and max corner coordinates
    Returns
    -----------
    distance : jnp.ndarray
        Squared distance from point to AABB
    '''
    # Distance from point to box (0 if inside)
    diff = jnp.maximum(0.0, jnp.maximum(aabb[..., 0, :] - p, p - aabb[..., 1, :]))
    return jnp.sum(diff**2)

def _set_if_leaf(bvh, point, surface_points, d_min, d_idx, idx, is_leaf):
    N_leafs = (bvh.leafs.shape[0] + 1) // 2
    # if it is clipped it just returns the distance to the last triangle.
    # this gets masked out anyway by is_leaf.
    index_in_mesh = bvh.order[jnp.clip(idx, 0, N_leafs -1)]
    leaf          = surface_points[index_in_mesh]
    closest_leaf  = closest_point_on_triangle(point, leaf[0], leaf[1], leaf[2])
    d_leaf        = jnp.sum((closest_leaf - point)**2)
    d_idx_new     = jnp.where(is_leaf & (d_leaf < d_min), index_in_mesh, d_idx)
    d_min_new     = jnp.where(is_leaf & (d_leaf < d_min), d_leaf, d_min)
    return d_min_new, d_idx_new

def _closest_point_loop(state):
    stack_idx, stack, d_min, d_idx, loop_idx, bvh, point, surface_points = state
    N_leafs = (bvh.leafs.shape[0] + 1) // 2
    current_idx  = stack[stack_idx]                                
    left_idx     = bvh.left_idx[ current_idx]
    right_idx    = bvh.right_idx[current_idx]

    left_aabb_dist = _point_aabb_distance(point, bvh.aabb[left_idx] )
    right_aabb_dist= _point_aabb_distance(point, bvh.aabb[right_idx])

    left_better  =  left_aabb_dist< d_min  # shape (N_points, )
    right_better =  right_aabb_dist< d_min  # shape (N_points, )        

    left_is_leaf = bvh.leafs[left_idx]
    right_is_leaf= bvh.leafs[right_idx]

    d_min_left, d_idx_left   = _set_if_leaf(bvh, point, surface_points, d_min,      d_idx,   left_idx, left_is_leaf)
    d_min_both, d_idx_both   = _set_if_leaf(bvh, point, surface_points, d_min_left, d_idx_left, right_idx, right_is_leaf)

    traverse_left    = jnp.logical_and(left_better,  jnp.logical_not(bvh.leafs[left_idx]))
    traverse_right   = jnp.logical_and(right_better, jnp.logical_not(bvh.leafs[right_idx]))

    transverse_any   = jnp.logical_or(traverse_left, traverse_right)
    transverse_both  = jnp.logical_and(traverse_left, traverse_right)

    # if we hit one, stack stays the same, if we hit both, we add one, if we hit none, we go one lower
    new_stack_idx = stack_idx + transverse_both.astype(jnp.int32) - (~transverse_any).astype(jnp.int32)        

    new_stack = jnp.where(
        transverse_any,
        jnp.where(
            transverse_both,
            stack.at[stack_idx + 1].set(right_idx).at[stack_idx].set(left_idx),
            jnp.where(
                traverse_left,
                stack.at[stack_idx].set(left_idx),
                stack.at[stack_idx].set(right_idx)
            )
        ),
        stack.at[stack_idx].set(-1)
    )
    return new_stack_idx, new_stack, d_min_both, d_idx_both, loop_idx + 1, bvh, point, surface_points        


@jax.jit
def _bvh_closest_point(bvh : BVH, point: jnp.ndarray, mesh, stack_size : int = 64, max_hit_size : int = 64):
    '''
    Get closest point on triangle mesh for a single point using BVH traversal.

    For multiple points, use bvh_closest_point_vectorized (handles arbitrary point shapes)

    Parameters
    ----------
    bvh : BVH
        BVH structure.
    point : jnp.ndarray
        Query point with shape (3,).
    mesh : tuple
        Mesh tuple `(positions, connectivity)` with triangle connectivity.

    Returns
    --------
    closest_point : jnp.ndarray
        Closest point on the mesh.
    distance : jnp.ndarray
        Distance from query point to closest point.
    triangle_index : jnp.ndarray
        Triangle index of the closest point.
    '''

    N_leafs = (bvh.leafs.shape[0] + 1) // 2

    surface_points = mesh[0][mesh[1]]  # (N_triangles, 3, 3)

    
    stack = jnp.full((stack_size,  ), -1)     
    
    initial_stack = stack.at[..., 0].set(N_leafs)  # start with root node
    initial_state = (0, initial_stack,jnp.inf, -1, 0, bvh, point, surface_points)  # stack_idx, stack, d_min, d_idx, loop_idx, constants
    
    final_stack_idx, final_stack, d_min, d_idx, n_loops, _, _, _ = jax.lax.while_loop(
        _traversal_condition,
        _closest_point_loop,
        initial_state)
    
    closest_point = closest_point_on_triangle(point, 
                              surface_points[d_idx,0], 
                              surface_points[d_idx,1], 
                              surface_points[d_idx,2])
    return closest_point, jnp.sqrt(d_min), d_idx

bvh_closest_point_vectorized = jax.jit(jnp.vectorize(_bvh_closest_point, excluded=(0,2), signature='(3)->(3),(),()'))

@jax.jit
def get_closest_points_on_mesh(points : jnp.ndarray, mesh) -> jnp.ndarray:
    '''
    Get closest points on triangle mesh for a set of points. 

    Parameters
    -----------
        points: jnp.ndarray [N,3]
            Array of points
        mesh: tuple of (positions, connectivity)
            positions:    jnp.ndarray of shape (M, 3) representing the 3D coordinates of mesh vertices.
            connectivity: jnp.ndarray of shape (K, 3) representing the triangle connectivity of the mesh.
    Returns
    -----------
        closest_points: jnp.ndarray [N,3]
            Array of closest points on the mesh for each input point.
        distances: jnp.ndarray [N,]
            Squared distances from each input point to its closest point on the mesh.
        triangle_indices: jnp.ndarray [N,]
            Indices of the triangles on the mesh corresponding to the closest points.
    '''
    bvh = build_lbvh(mesh[0], mesh[1]) # BVH
    return bvh_closest_point_vectorized(bvh, points, mesh)
    