from .flux_surfaces_base import FluxSurface, FluxSurfaceSettings, ToroidalExtent, ParametrisedSurface

import jax
import jax.numpy as jnp
from functools import partial
import equinox as eqx
from typing import List

# ===================================================================================================================================================================================
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#                                                                           Triangle Meshing
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# ===================================================================================================================================================================================

def _concatenate_connectivity(connectivity_list : List[jnp.ndarray], points_per_mesh : List[int]):
    '''
    Internal function to concatenate multiple mesh connectivities into a single connectivity.

    Parameters
    ----------
    connectivity_array : List[jnp.ndarray]
        A list of arrays of shape (n_points_per_element, 3) containing the indices of the vertices for each triangle for each surface.        
    '''
    added_connectivity = jnp.cumulative_sum(jnp.array(points_per_mesh), include_initial=True)
    connectivity       = jnp.concatenate([conn + added_connectivity[i] for i, conn in enumerate(connectivity_list)], axis=0)
    return connectivity

@partial(jax.jit, static_argnums = (0))
def _build_closed_strip(n_strip : int, offset_index_0 : int, offset_index_1 : int, normals_orientation : bool, stride_0: int = 1, stride_1 : int = 1):
    '''
    This functions connects two closed strips together. Each strip has n_strip vertices, and the first vertex of each strip is at offset_index_0 and offset_index_1 respectively.
    Normals_orientation determines whether the normals should face outwards (True) or inwards (False). Strides allow meshing of non-adjacent vertices.    

    Parameters
    ----------
    n_strip : int
        The number of vertices in each strip.
    offset_index_0 : int
        The starting index of the first strip.
    offset_index_1 : int
        The starting index of the second strip.
    normals_orientation : bool
        Whether the normals should face outwards (right-hand rule).
    stride_0 : int, optional
        The stride between vertices in the first strip. Default is 1.
    stride_1 : int, optional
        The stride between vertices in the second strip. Default is 1.  
    Returns
    -------
    triangles : jnp.ndarray
        An array of shape (n_triangles, 3) containing the indices of the vertices for each triangle.

    '''    
    u_i_block = jnp.arange(n_strip)
    
    bottom_left  =  u_i_block * stride_0  + offset_index_0
    bottom_right = ( (u_i_block + 1) % n_strip ) * stride_0 + offset_index_0 

    top_left     = u_i_block * stride_1 + offset_index_1 
    top_right    = ((u_i_block + 1) % n_strip ) * stride_1 + offset_index_1
    

    tri1_out, tri2_out = _strip_triangles_out(bottom_left, bottom_right, top_left, top_right)
    tri1_in,  tri2_in  = _strip_triangles_in(bottom_left, bottom_right, top_left, top_right)
    tri1 = jnp.where(normals_orientation, tri1_out, tri1_in)
    tri2 = jnp.where(normals_orientation, tri2_out, tri2_in)
    triangles = jnp.stack([tri1, tri2],axis=-2) # last axis is vertices, second last is triangle index (0 or 1)    
    return triangles

def _strip_triangles_in(bottom_left, bottom_right, top_left, top_right):
    return jnp.stack([top_left, bottom_left, bottom_right], axis=-1), jnp.stack([top_right, top_left, bottom_right], axis=-1)

def _strip_triangles_out(bottom_left, bottom_right, top_left, top_right):
    return jnp.stack([bottom_left, top_left, bottom_right], axis=-1), jnp.stack([top_left, top_right, bottom_right], axis=-1)

def _build_open_strip(n_strip : int, offset_index_0 : int, offset_index_1 : int, normals_orientation : bool, stride_0: int, stride_1 : int):
    return _build_closed_strip(n_strip, offset_index_0, offset_index_1, normals_orientation, stride_0, stride_1)[:-1, ...]

@partial(jax.jit, static_argnums = (0))
def _build_closed_wedges(n_wedge : int, offset_index_0 : int, offset_index_1 : int, normals_orientation : bool, stride_1):
    '''
    This functions connects a wedge of triangles to a center point. Each wedge has n_wedge vertices, and the first vertex of the wedge is at offset_index_1.

    Parameters
    ----------
    n_wedge : int
        The number of vertices in the wedge.
    offset_index_0 : int
        The index of the center point.
    offset_index_1 : int
        The starting index of the wedge.
    normals_orientation : bool
        Whether the normals should face outwards (right-hand rule).
    stride_1 : int
        The stride between vertices in the wedge.
    Returns
    -------
    triangles : jnp.ndarray
        An array of shape (n_triangles, 3) containing the indices of the vertices for each triangle.        
    '''
    u_i_block = jnp.arange(n_wedge)
    center     = jnp.zeros_like(u_i_block) + offset_index_0
    wedge_1    = u_i_block * stride_1 + offset_index_1
    wedge_2    = ( (u_i_block + 1) % n_wedge ) * stride_1 + offset_index_1

    triangles = jnp.where(normals_orientation, jnp.stack([center, wedge_1, wedge_2], axis=-1), jnp.stack([wedge_1, center, wedge_2], axis=-1))
    return triangles


_build_closed_strips = jax.vmap(_build_closed_strip, in_axes=(None,0,0,None, None, None))
_build_open_strips   = jax.vmap(_build_open_strip,   in_axes=(None,0,0,None, None, None))

# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#                                                                        Single Surface Meshing 
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

@partial(jax.jit, static_argnums = (0,1,2,3))
def _build_triangles_surface(n_theta : int, theta_blocks : int, n_phi : int, phi_blocks : int, normals_orientation : bool):
    '''
    Build the triangle connectivity for a surface mesh given the number of poloidal and toroidal points/blocks.

    Assumes the points are ordered implicitly as:
        [n_theta, n_phi, ndim] (reshaped to [-1, ndim])

    Builds triangles in increasing second dimension first.
    In other words, we have the following [implicit] shape:
        [phi_blocks, theta_blocks, 2, 3] (2 is from 2 triangles per quad, 3 is from 3 vertices per triangle)

    Parameters
    ----------
    n_theta : int
        The number of poloidal points.
    theta_blocks : int
        The number of poloidal blocks.
    n_phi : int
        The number of toroidal points.
    phi_blocks : int
        The number of toroidal blocks.
    normals_orientation : bool
        Whether the normals should face outwards (right-hand rule).
    Returns
    -------
    triangles : jnp.ndarray
        An array of shape (n_triangles, 3) containing the indices of the vertices for each triangle.

    '''

    # Points are ordered with phi first, then theta
    # Therefore, given a particular strip:
    # - Starting index of strip i in theta direction is i * n_phi
    # - Starting index of next strip is i * n_phi + n_theta
    # - Stride between points in strip is n_phi
    # - Stride between points in next strip is n_phi

    # Since the stride multiplies the index, we can just pass n_phi as stride and use 0, 1, 2, ... as the indices
    # Since phi_blocks can be n_phi or n_phi - 1, we build the arrays of starting indices first
    
    triangles_shaped = _build_closed_strips(n_theta, jnp.arange(phi_blocks), (jnp.arange(phi_blocks) +  1) % n_phi, normals_orientation, n_phi,n_phi)
    return jnp.moveaxis(triangles_shaped, 0,1).reshape(-1,3) # Move triangle index to first axis so we have triangle order first in phi.


def _build_open_triangle_surface(n_theta : int, n_phi :int, normals_orientation : bool):
    '''
    Build the triangle connectivity for an open surface mesh given the number of poloidal and toroidal points.
    Note that the triangles are built in increasing second dimension first. 
    
    Assumed is that the points are ordered with the implicit ordering:
       [n_theta, n_phi, ndim] (reshaped to [-1, ndim])
           
    We have the following [implicit] shape:
        [n_theta - 1, n_phi - 1, 2, 3] (2 is from 2 triangles per quad, 3 is from 3 vertices per triangle)

    n_triangles is thus 2 * (n_phi - 1) * (n_theta - 1)

    Parameters
    ----------
    n_theta : int
        The number of poloidal points.  
    n_phi : int 
        The number of toroidal points.
    normals_orientation : bool
        Whether the normals should face outwards (right-hand rule).
    Returns
    -------
    triangles : jnp.ndarray [n_triangles = 2* (n_phi - 1) * (n_theta - 1), 3]
        An array of shape  containing the indices of the vertices for each triangle.
    '''
    triangles_shaped = _build_open_strips(n_theta, jnp.arange(n_phi -1), jnp.arange(1, n_phi), normals_orientation, n_phi, n_phi)        
    return jnp.moveaxis(triangles_shaped, 0,1).reshape(-1,3) # Move triangle index to first axis so we have triangle order first in phi.


def _phi_grid(phi_start, phi_end, full_angle, n_phi : int):
    '''n_phi toroidal angles from phi_start to phi_end; the end point is excluded for the full angle (periodic).'''
    return jnp.where(full_angle, jnp.linspace(phi_start, phi_end, n_phi, endpoint=False), jnp.linspace(phi_start, phi_end, n_phi, endpoint=True))

def _mesh_surface_points(flux_surfaces : ParametrisedSurface, s : float, phi_start : float, phi_end : bool, full_angle : bool,  n_theta : int, n_phi : int):
    '''
    Obtain the mesh points on a flux surface at normalized radius s with n_theta poloidal and n_phi toroidal points
    Full angle defines whether the end point is included.

    Parameters
    ----------
    flux_surfaces : ParametrisedSurface
        The parametrized  surface object containing the parameterization.
    s : float
        The normalized radius of the flux surface to mesh.
    phi_start : float
        The starting toroidal angle (in radians) if not full_angle.
    phi_end : float
        The ending toroidal angle (in radians) if not full_angle.
    full_angle : bool
        Whether to mesh the full 0 to 2π toroidal angle or a segment.
    n_theta : int
        The number of poloidal points.
    n_phi : int
        The number of toroidal points.
    Returns
    -------
    positions : jnp.ndarray
        An array of shape (n_points, 3) containing the Cartesian coordinates of the mesh points.    
    '''
    theta = jnp.linspace(0, 2 * jnp.pi, n_theta, endpoint=False)
    phi   = _phi_grid(phi_start, phi_end, full_angle, n_phi)
    t, p  = jnp.meshgrid(theta, phi, indexing='ij')
    return flux_surfaces.cartesian_position(s,t,p).reshape(-1,3)

_mesh_surface_points_multiple = jax.jit(jax.vmap(_mesh_surface_points, in_axes=(None,0,None,None,None,None,None)), static_argnums=(4,5,6))

def _mesh_surface_connectivity(n_theta : int, n_phi : int, full_angle : bool, normals_facing_outwards : bool):
    '''
    Obtain the mesh connectivity on a flux surface with n_theta poloidal and n_phi toroidal points
    Full angle defines whether the end point is included.

    Parameters
    ----------
    n_theta : int
        The number of poloidal points.
    n_phi : int
        The number of toroidal points.
    full_angle : bool
        Whether to mesh the full 0 to 2π toroidal angle or a segment.
    normals_facing_outwards : bool
        Whether the normals should face outwards (right-hand rule).
    Returns
    -------
    triangles : jnp.ndarray
        An array of shape (n_triangles, 3) containing the indices of the vertices for each triangle.    
    '''
    phi_blocks = n_phi if full_angle else n_phi - 1 # full_angle is static so this is possible
    theta_blocks = n_theta

    return _build_triangles_surface(n_theta, theta_blocks, n_phi,  phi_blocks, normals_facing_outwards)

_mesh_surface_connectivity_multiple = jax.jit(jax.vmap(_mesh_surface_connectivity, in_axes=(None,None,None,0)), static_argnums=(0,1,2))

@partial(jax.jit, static_argnums = (4,5,6))
def _mesh_surface(flux_surfaces : ParametrisedSurface, s : float, phi_start : float, phi_end : bool, full_angle : bool,  n_theta : int, n_phi : int, normals_facing_outwards  : bool):
    '''
    Mesh a flux surface at normalized radius s with n_theta poloidal and n_phi toroidal points.

    full_angle defines not the extent, only the connectivity. It only makes sense to set it to true if phi_start = 0 and phi_end = 2π.
    This is separate here because the number of trianlges depends on full_angle:

    2 * n_phi * n_theta if full_angle else 2 * (n_phi - 1) * n_theta

    Therefore, to allow jitting, it is a separate, static, argument. Use the convenience function mesh_surface below to ensure consistency.

    The number of points is always n_phi * n_theta, regardless of full_angle. 

    Parameters
    ----------
    flux_surfaces : ParametrisedSurface
        The parametrized surface object containing the parameterization.
    s : float
        The normalized radius of the flux surface to mesh.
    phi_start : float
        The starting toroidal angle (in radians) if not full_angle.
    phi_end : float
        The ending toroidal angle (in radians) if not full_angle.
    full_angle : bool
        Whether to mesh the full 0 to 2π toroidal angle or a segment.
    n_theta : int
        The number of poloidal points.
    n_phi : int
        The number of toroidal points.
    normals_facing_outwards : bool, optional
        Whether the normals should face outwards (right-hand rule). Default is True.
    Returns
    -------
    positions : jnp.ndarray
        An array of shape (n_points, 3) containing the Cartesian coordinates of the mesh points.
    triangles : jnp.ndarray
        An array of shape (n_triangles, 3) containing the indices of the vertices for each triangle.

    '''

    positions  = _mesh_surface_points(flux_surfaces, s, phi_start, phi_end, full_angle, n_theta, n_phi)
    connectivity = _mesh_surface_connectivity(n_theta, n_phi, full_angle, normals_facing_outwards)
    return positions, connectivity

_mesh_multiple_surfaces = jax.jit(jax.vmap(_mesh_surface, in_axes=(None, 0, None, None, None, None, None, 0)), static_argnums=(4,5,6)) # this is only for convenience: internally, do not use since we want to separate connectivity and points.


# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#                                                                        Closed Surface Meshing 
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

@partial(jax.jit, static_argnums= (0,1,2,3,4))
def _mesh_surfaces_closed_connectivity(include_axis : bool, full_angle : bool, n_theta : int, n_phi : int, n_cap : int):    
    # We have the following possibilities:
    # 1. include_axis = True & full_angle = True: We have one surface only. This is a closed surface, so we can just return it.
    # 2. include_axis = False & full_angle = True: We have two surfaces. These are already closed so we can just return them.
    # 3. include_axis = False & full_angle = False: We have two surfaces. These are not closed surfaces, so we need to add caps at the edges from s_0 to s_end
    # 4. include_axis = True & full_angle = False: We have one surface only. This is not a closed surface, so we need to add caps at the edges from 0 ( = s_0 but wedge) to s_end

    # They are all compile-time constants (and in fact need to be so as output shape depends on their value)
    if include_axis and full_angle:
        # 1. include_axis = True & full_angle = True: We have one surface only. This is a closed surface, so we can just return it.
        return _mesh_surface_connectivity(n_theta, n_phi, full_angle, normals_facing_outwards=True)
    elif (not include_axis) and full_angle:
        # 2. include_axis = False & full_angle = True: We have two surfaces. These are already closed so we can just return them.
        normals_facing_outwards         = jnp.array([False, True])        
        multiple_surface_connectivity   =  _mesh_surface_connectivity_multiple(n_theta, n_phi, full_angle, normals_facing_outwards)
        n_points_first_mesh             = n_theta * n_phi # always the same
        return _concatenate_connectivity(multiple_surface_connectivity, [n_points_first_mesh, n_points_first_mesh])
    elif (not include_axis) and (not full_angle):
        normals_facing_outwards         = jnp.array([False, True])        
        multiple_surface_connectivity   =  _mesh_surface_connectivity_multiple(n_theta, n_phi, full_angle, normals_facing_outwards)
        n_points_first_mesh             = n_theta * n_phi # always the same
        connectivity_base_surfaces      =  _concatenate_connectivity(multiple_surface_connectivity, [n_points_first_mesh, n_points_first_mesh])
        
        start_surface_0 = 0 
        start_surface_1 = n_theta * n_phi 
        end_surface_1   = 2 * n_theta * n_phi

        start_cap_0      =  end_surface_1
        start_cap_0_edge =  end_surface_1 + (n_cap - 1)

        start_cap_1      =  end_surface_1 + n_cap * n_theta
        start_cap_1_edge =  end_surface_1 + n_cap * n_theta + (n_cap - 1)
        # However, we have to modify the first and last rings of the outer mesh to connect to the outer surfaces
                
        connectivity_start_f  = _build_closed_strip(n_theta, start_surface_0, start_cap_0,      normals_orientation=True, stride_0 = n_phi, stride_1 = n_cap).reshape(-1,3)
        connectivity_start_m  = _build_triangles_surface(n_theta, n_theta, n_cap, n_cap - 1, normals_orientation=True) + start_cap_0
        connectivity_start_e  = _build_closed_strip(n_theta, start_cap_0_edge, start_surface_1, normals_orientation=True, stride_0 = n_cap, stride_1 = n_phi).reshape(-1,3)
        
        connectivity_start    = jnp.concatenate([connectivity_start_f, connectivity_start_m, connectivity_start_e], axis=0) 

        connectivity_end_f    = _build_closed_strip(n_theta, start_surface_0 + (n_phi - 1) , start_cap_1, normals_orientation=False, stride_0 = n_phi, stride_1 = n_cap).reshape(-1,3)
        connectivity_end_m    = _build_triangles_surface(n_theta, n_theta, n_cap, n_cap - 1,  normals_orientation=False) + start_cap_1
        connectivity_end_e    = _build_closed_strip(n_theta, start_cap_1_edge, start_surface_1 + (n_phi - 1), normals_orientation=False, stride_0 = n_cap, stride_1 = n_phi).reshape(-1,3)

        connectivity_end      = jnp.concatenate([connectivity_end_f, connectivity_end_m, connectivity_end_e], axis=0)
        
        connectivity_total    = jnp.concatenate([connectivity_base_surfaces, connectivity_start, connectivity_end], axis=0)
        
        return connectivity_total
    
    elif include_axis and (not full_angle):

        edge_surface_connectivity = _mesh_surface_connectivity(n_theta, n_phi, full_angle, normals_facing_outwards=True)
        
        n_points_first_mesh       = n_theta * n_phi # always the same

        start_surface = 0 
        
        end_surface   = n_theta * n_phi
        n_axis        = 2

        start_cap_0      =  end_surface + n_axis
        start_cap_0_edge =  start_cap_0 + (n_cap - 1) 

        start_cap_1        =  start_cap_0 + n_cap * n_theta
        start_cap_1_edge   =  start_cap_1 + (n_cap - 1)

        # This 'hack' uses a zero stride to basically set the number of points in the axis to 1, effectively connecting to the axis point.
        connectivity_start_axis    = _build_closed_wedges(n_theta, end_surface, start_cap_0, True, stride_1 = n_cap).reshape(-1,3)
        connectivity_start_m       = _build_triangles_surface(n_theta, n_theta, n_cap, n_cap - 1, normals_orientation=True) + start_cap_0
        connectivity_start_e       = _build_closed_strip(n_theta, start_cap_0_edge, start_surface, normals_orientation=True, stride_0 = n_cap, stride_1 = n_phi).reshape(-1,3)

        connectivity_end_axis      = _build_closed_wedges(n_theta, end_surface + 1, start_cap_1, False, stride_1 = n_cap).reshape(-1,3)
        connectivity_end_m         = _build_triangles_surface(n_theta, n_theta, n_cap, n_cap - 1, normals_orientation=False) + start_cap_1
        connectivity_end_e         = _build_closed_strip(     n_theta,  start_cap_1_edge, start_surface + (n_phi - 1), normals_orientation=False, stride_0 = n_cap, stride_1 =  n_phi).reshape(-1,3)

        return jnp.concatenate([edge_surface_connectivity, connectivity_start_axis, connectivity_start_m, connectivity_start_e, connectivity_end_axis, connectivity_end_m, connectivity_end_e], axis=0)

    else:
        return 0
    
@partial(jax.jit, static_argnums= (3, 6, 7,8,9))
def _mesh_surfaces_closed_points(flux_surfaces: ParametrisedSurface,
                                 s_values_start : float, s_value_end : float, include_axis : bool,
                                 phi_start : float, phi_end : float, full_angle : bool,
                                 n_theta : int, n_phi : int, n_cap : int):
        
    
        # We have the following possibilities:
        # 1. include_axis = True & full_angle = True: We have one surface only. This is a closed surface, so we can just return it.
        # 2. include_axis = False & full_angle = True: We have two surfaces. These are already closed so we can just return them.
        # 3. include_axis = False & full_angle = False: We have two surfaces. These are not closed surfaces, so we need to add caps at the edges from s_0 to s_end
        # 4. include_axis = True & full_angle = False: We have one surface only. This is not a closed surface, so we need to add caps at the edges from 0 ( = s_0 but wedge) to s_end
    
        # They are all compile-time constants (and in fact need to be so as output shape depends on their value)
    
        if include_axis and full_angle:        
            # 1. include_axis = True & full_angle = True: We have one surface only. This is a closed surface, so we can just return it.
            return _mesh_surface_points(flux_surfaces, s = s_value_end, phi_start = phi_start, phi_end = phi_end, full_angle = full_angle, n_theta = n_theta, n_phi = n_phi)
        elif (not include_axis) and full_angle:
            # 2. include_axis = False & full_angle = True: We have two surfaces. These are already closed so we can just return them.
            s_values_outer            = jnp.array([s_values_start, s_value_end])                    
            multiple_surface_points   =  _mesh_surface_points_multiple(flux_surfaces, s_values_outer, phi_start, phi_end,  full_angle, n_theta,  n_phi)
            return jnp.concatenate(multiple_surface_points, axis=0)
        elif (not include_axis) and (not full_angle):            
            s_values = jnp.array([s_values_start, s_value_end])
            multiple_surface_points         =  _mesh_surface_points_multiple(flux_surfaces, s_values,phi_start, phi_end, full_angle,  n_theta, n_phi).reshape(-1,3)

            s_cap = jnp.linspace(s_values_start, s_value_end, n_cap + 2)[1:-1] # Exclude first and last
            t_cap = jnp.linspace(0, 2 * jnp.pi, n_theta, endpoint=False)

            t_cap_mg, s_cap_mg = jnp.meshgrid(t_cap, s_cap, indexing='ij')
                
            positions_cap_start = flux_surfaces.cartesian_position(s_cap_mg, t_cap_mg, phi_start).reshape(-1,3)
            positions_cap_end   = flux_surfaces.cartesian_position(s_cap_mg, t_cap_mg, phi_end  ).reshape(-1,3) 

            return jnp.concatenate([multiple_surface_points, positions_cap_start, positions_cap_end], axis=0)
        elif include_axis and (not full_angle):
            surface_points = _mesh_surface_points(flux_surfaces, s = s_value_end, phi_start = phi_start, phi_end = phi_end, full_angle = full_angle, n_theta = n_theta, n_phi = n_phi)
            axis_points    = flux_surfaces.cartesian_position(jnp.array([0.0, 0.0]), 0.0, jnp.array([phi_start, phi_end])).reshape(-1,3)

            s_cap = jnp.linspace(s_values_start, s_value_end, n_cap + 2)[1:-1] # Exclude first and last
            t_cap = jnp.linspace(0, 2 * jnp.pi, n_theta, endpoint=False)
            t_cap_mg, s_cap_mg = jnp.meshgrid(t_cap, s_cap, indexing='ij')
            
            positions_cap_start = flux_surfaces.cartesian_position(s_cap_mg, t_cap_mg, phi_start).reshape(-1,3)
            positions_cap_end   = flux_surfaces.cartesian_position(s_cap_mg, t_cap_mg, phi_end  ).reshape(-1,3) 

            return jnp.concatenate([surface_points, axis_points, positions_cap_start, positions_cap_end], axis=0)
        else: 
            return 0

@partial(jax.jit, static_argnums= (3, 6, 7,8,9))
def _mesh_surfaces_closed(flux_surfaces: ParametrisedSurface,
                          s_values_start : float, s_value_end : float, include_axis : bool,
                          phi_start : float, phi_end : float, full_angle : bool,
                          n_theta : int, n_phi : int, n_cap : int):
    '''
    Mesh closed flux surfaces between s_values_start and s_value_end with n_theta poloidal and n_phi toroidal points.

    Internal function. Use the convenience function mesh_closed_surfaces below to automatically set include_axis and full angle based on the flux surface settings.

    Parameters
    ----------
    flux_surfaces : ParametrisedSurface
        The parametrized surface object containing the parameterization.
    s_values_start : float
        The normalized radius of the inner flux surface to mesh.    
    s_value_end : float
        The normalized radius of the outer flux surface to mesh.
    include_axis : bool
        Whether to include the magnetic axis in the mesh.
    phi_start : float
        The starting toroidal angle (in radians) if not full_angle.
    phi_end : float : float
        The ending toroidal angle (in radians) if not full_angle.
    full_angle : bool
        Whether to mesh the full 0 to 2π toroidal angle or a segment.
    n_theta : int
        The number of poloidal points.
    n_phi : int
        The number of toroidal points.
    n_cap : int
        The number of radial points in the caps if needed.
    Returns
    -------
    points : jnp.ndarray    
        An array of shape (n_points, 3) containing the Cartesian coordinates of the mesh points.
    connectivity : jnp.ndarray
        An array of shape (n_triangles, 3) containing the indices of the vertices for each triangle.
    '''
    connectivity = _mesh_surfaces_closed_connectivity(include_axis, full_angle, n_theta, n_phi, n_cap)
    points       = _mesh_surfaces_closed_points(flux_surfaces, s_values_start, s_value_end, include_axis, phi_start, phi_end, full_angle, n_theta, n_phi, n_cap)
    return points, connectivity

@partial(jax.jit, static_argnums= (0,1,2))
def _mesh_poloidal_connectivity(n_layers : int, n_theta : int, include_axis : bool):
    
    if not include_axis:        
        return _build_closed_strips(n_theta, jnp.arange(n_layers - 1) * n_theta, (jnp.arange(n_layers - 1) +  1) * n_theta, True, 1,1).reshape(-1,3)
    else:
        closed_strips = _build_closed_strips(n_theta, jnp.arange(n_layers-2) * n_theta, (jnp.arange(n_layers - 2) + 1) * n_theta, True, 1,1).reshape(-1,3) + 1 # axis point
        axis_wedge = _build_closed_wedges(n_theta, 0, 1, True, 1)
        return jnp.concatenate([axis_wedge, closed_strips])
    
@partial(jax.jit, static_argnums = (2,3,4))
def _mesh_poloidal_points_coordinates(flux_surfaces : ParametrisedSurface, s_layers : jnp.ndarray, phi : float, n_theta : int, include_axis : bool):        
    if not include_axis:        
        theta = jnp.linspace(0, 2 * jnp.pi, n_theta, endpoint=False)    
        s_mg, theta_mg  = jnp.meshgrid(s_layers, theta, indexing='ij')
        return s_mg.reshape(-1), theta_mg.reshape(-1), phi * jnp.ones(s_mg.size)        
    else:
        theta = jnp.linspace(0, 2 * jnp.pi, n_theta, endpoint=False)    
        s_mg, theta_mg  = jnp.meshgrid(s_layers[1:], theta, indexing='ij')
        positions_no_axis = flux_surfaces.cartesian_position(s_mg, theta_mg, phi).reshape(-1,3)
        return jnp.concatenate([jnp.array([0.0]), s_mg.reshape(-1)]), jnp.concatenate([jnp.array([0.0]), theta_mg.reshape(-1)]), jnp.concatenate([jnp.array([phi]), jnp.ones(positions_no_axis.size) * phi])        

@partial(jax.jit, static_argnums = (2,3,4))
def _mesh_poloidal_points(flux_surfaces : ParametrisedSurface, s_layers : jnp.ndarray, phi : float, n_theta : int, include_axis : bool):        
    if not include_axis:        
        theta = jnp.linspace(0, 2 * jnp.pi, n_theta, endpoint=False)    
        s_mg, theta_mg  = jnp.meshgrid(s_layers, theta, indexing='ij')
        positions_no_axis = flux_surfaces.cartesian_position(s_mg, theta_mg, phi).reshape(-1,3)
        return positions_no_axis
    else:
        theta = jnp.linspace(0, 2 * jnp.pi, n_theta, endpoint=False)    
        s_mg, theta_mg  = jnp.meshgrid(s_layers[1:], theta, indexing='ij')
        positions_no_axis = flux_surfaces.cartesian_position(s_mg, theta_mg, phi).reshape(-1,3)
        return jnp.concatenate([flux_surfaces.cartesian_position(0.0, 0.0, phi)[None, :], positions_no_axis], axis=0)

    
def mesh_poloidal_plane(flux_surface : ParametrisedSurface, s_layers : jnp.ndarray, phi : float, n_theta : int):
    '''
    Mesh a poloidal plane of a flux surface at the given s_layers and toroidal angle phi with n_theta poloidal points.

    Vertices are ordered first axis if present (s[0]==0), then increasing theta, then increasing s.

    Connectivity is ordered first axis wedges if present, then each closed theta strip, then increasing s.

    Parameters
    flux_surface : ParametrisedSurface
        The parametrized surface object containing the parameterization.
    s_layers : jnp.ndarray
        The normalized radii of the flux surfaces to mesh. Shape (n_layers,)
    phi : float
        The toroidal angle (in radians) at which to mesh the poloidal plane.
    n_theta : int
        The number of poloidal points.
    Returns
    -------
    points : jnp.ndarray    
        An array of shape (n_points, 3) containing the Cartesian coordinates of the mesh points.
    connectivity : jnp.ndarray
        An array of shape (n_triangles, 3) containing the indices of the vertices for each triangle.
    '''

    include_axis = bool(s_layers[0] == 0.0) 
    points = _mesh_poloidal_points(flux_surface, s_layers, phi, n_theta, include_axis)
    connectivity = _mesh_poloidal_connectivity(s_layers.shape[0], n_theta, include_axis)
    return points, connectivity

def _build_layer_strips(n_layers, n_points_mesh, n_theta, n_phi):
    # Connecting layers
    n_offsets = jnp.arange(n_layers - 1) * n_points_mesh

    first_closed_strips = _build_closed_strips(n_theta, n_offsets, n_offsets + n_points_mesh, True, n_phi, n_phi).reshape(n_layers - 1, -1, 3)
    end_closed_strips   = _build_closed_strips(n_theta, n_offsets + (n_phi - 1), n_offsets + n_points_mesh + (n_phi-1), False, n_phi, n_phi).reshape(n_layers - 1, -1, 3)

    return jnp.stack([first_closed_strips, end_closed_strips], axis=1).reshape(n_layers -1, -1, 3)

@partial(jax.jit, static_argnums= (0,1,2,3))
def _mesh_watertight_layers_connectivity(n_layers : int, full_angle : bool,  n_theta : int, n_phi : int):

    normals_facing_outwards         = jnp.ones(n_layers, dtype=bool)        
    normals_facing_outwards = normals_facing_outwards.at[0].set(False)    
    multiple_surface_connectivity   =  _mesh_surface_connectivity_multiple(n_theta, n_phi, full_angle, normals_facing_outwards)
    n_points_mesh                   = n_theta * n_phi # always the same
    connectivity_base_surfaces      =  _concatenate_connectivity(multiple_surface_connectivity, jnp.full(n_layers, n_points_mesh)).reshape(n_layers,-1, 3)

    if not full_angle:
        closed_strips = _build_layer_strips(n_layers, n_points_mesh, n_theta, n_phi)

        # connectivity surface 1, side layer 1, connectivity surface 2, side layer 1, ...
        # ... n_layers is static, so this control flow is possible ...
        total_array = []
        for i in range(n_layers):
            total_array.append(connectivity_base_surfaces[i])
            if i < n_layers - 1:
                total_array.append(closed_strips[i])
        return total_array
    else:        
        # connectivity surface 1, side layer 1, connectivity surface 2, side layer 1, ...
        # ... n_layers is static, so this control flow is possible ...
        total_array = []
        for i in range(n_layers):
            total_array.append(connectivity_base_surfaces[i])    
        return total_array

@partial(jax.jit, static_argnums = (4,5,6))
def _mesh_watertight_layers_points(flux_surfaces : ParametrisedSurface, s_values : jnp.ndarray, phi_start : float, phi_end : float, full_angle : bool, n_theta : int, n_phi : int):
    multiple_surface_points         =  _mesh_surface_points_multiple(flux_surfaces, s_values, phi_start, phi_end, full_angle, n_theta, n_phi)
    return jnp.concatenate(multiple_surface_points, axis=0)

@partial(jax.jit, static_argnums= (4,5,6))
def _mesh_watertight_layers(flux_surfaces : ParametrisedSurface, s_values : jnp.ndarray, phi_start : float, phi_end : float, full_angle : bool, n_theta : int, n_phi : int):
    '''
    Mesh watertight flux surface layers at the given s_values with n_theta poloidal and n_phi toroidal points.

    Parameters
    ----------
    flux_surfaces : ParametrisedSurface
        The parametrized surface object containing the parameterization.
    s_values : jnp.ndarray
        The normalized radii of the flux surfaces to mesh. Shape (n_layers,)
    phi_start : float
        The starting toroidal angle (in radians)
    phi_end : float
        The ending toroidal angle (in radians) 
    full_angle : bool
        Whether to mesh the full 0 to 2π toroidal angle or a segment.
    n_theta : int
        The number of poloidal points.
    n_phi : int
        The number of toroidal points.
    Returns
    -------
    points : jnp.ndarray    
        An array of shape (n_points, 3) containing the Cartesian coordinates of the mesh points.
    connectivity : jnp.ndarray
        An array of shape (n_triangles, 3) containing the indices of the vertices for each triangle.
    ''' 
    connectivity = _mesh_watertight_layers_connectivity(s_values.shape[0], full_angle, n_theta, n_phi)
    points       = _mesh_watertight_layers_points(flux_surfaces, s_values, phi_start, phi_end, full_angle, n_theta, n_phi)
    return points, connectivity

#===================================================================================================================================================================================
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#                                                                           Tetrahedra Meshing
# ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#===================================================================================================================================================================================



def _strip_tets_true(vertex_c, vertex_d, vertex_e, vertex_f, vertex_g, vertex_h, vertex_i, vertex_j):
    t1 =  jnp.stack([vertex_c, vertex_e, vertex_f, vertex_i], axis = -1)
    t2 =  jnp.stack([vertex_c, vertex_f, vertex_i, vertex_j], axis = -1)
    t3 =  jnp.stack([vertex_c, vertex_g, vertex_j, vertex_i], axis = -1)
    t4 =  jnp.stack([vertex_c, vertex_d, vertex_f, vertex_j], axis = -1)
    t5 =  jnp.stack([vertex_c, vertex_d, vertex_g, vertex_j], axis = -1)
    t6 =  jnp.stack([vertex_h, vertex_d, vertex_g, vertex_j], axis = -1)
    return jnp.stack([t1, t2, t3, t4, t5, t6], axis=-2) # last axis is vertices, second last is tetrahedron index (0 to 5)

def _strip_tets_false(vertex_c, vertex_d, vertex_e, vertex_f, vertex_g, vertex_h, vertex_i, vertex_j):
    t1 =  jnp.stack([vertex_c, vertex_e, vertex_i, vertex_f], axis=-1)
    t2 =  jnp.stack([vertex_c, vertex_f, vertex_j, vertex_i], axis=-1)
    t3 =  jnp.stack([vertex_c, vertex_g, vertex_i, vertex_j], axis=-1)
    t4 =  jnp.stack([vertex_c, vertex_d, vertex_j, vertex_f], axis=-1)
    t5 =  jnp.stack([vertex_c, vertex_d, vertex_j, vertex_g], axis=-1)
    t6 =  jnp.stack([vertex_h, vertex_d, vertex_j, vertex_g], axis=-1)
    return jnp.stack([t1, t2, t3, t4, t5, t6], axis=-2)

@partial(jax.jit, static_argnums = (0))
def _tetrahedral_closed_strip(n_strip : int, offset_index_0 : int, offset_index_1 : int, offset_index_2 : int, offset_index_3 : int, volume_orientation : bool, stride_0 : int, stride_1: int, stride_2 : int, stride_3 : int):

    # Very similar to triangular closed strips, but instead of 2 lines, we have 4. These 'lines' can have different strides and be offset by a different index.
    u_i_block = jnp.arange(n_strip)
    
    vertex_c = u_i_block * stride_0  + offset_index_0 # first line    
    vertex_d = u_i_block * stride_1  + offset_index_1 # second line

    vertex_e = ( (u_i_block + 1) % n_strip ) * stride_0 + offset_index_0  # first line next
    vertex_f = ( (u_i_block + 1) % n_strip ) * stride_1 + offset_index_1  # second line next
    
    vertex_g = u_i_block * stride_2 + offset_index_2  # third line
    vertex_h = u_i_block * stride_3 + offset_index_3  # fourth line

    vertex_i = ( (u_i_block + 1) % n_strip ) * stride_2 + offset_index_2  # third line next
    vertex_j = ( (u_i_block + 1) % n_strip ) * stride_3 + offset_index_3  # fourth line next

    verts = (vertex_c, vertex_d, vertex_e, vertex_f, vertex_g, vertex_h, vertex_i, vertex_j)
    vertices = jnp.where(volume_orientation, _strip_tets_true(*verts), _strip_tets_false(*verts))

    return vertices

def _tetrahedral_open_strip(n_strip : int, offset_index_0 : int, offset_index_1 : int, offset_index_2 : int, offset_index_3 : int, volume_orientation : bool, stride_0 : int, stride_1: int, stride_2 : int, stride_3 : int):
    return _tetrahedral_closed_strip(n_strip, offset_index_0, offset_index_1, offset_index_2, offset_index_3, volume_orientation, stride_0, stride_1, stride_2, stride_3)[:-1, ...] # remove last strip which would wrap around

_tetrahedral_open_strip_vmap = jax.jit(jax.vmap(_tetrahedral_open_strip, in_axes=(None,0,0,0,0,None,None,None,None,None), out_axes=0), static_argnums=(0))

def _open_cube_layer(j, n_x, n_y, n_z):
    arange_nz = jnp.arange(n_z - 1) + j * n_z
    return _tetrahedral_open_strip_vmap(n_x, arange_nz, 1 + arange_nz, n_z + arange_nz, n_z + 1 + arange_nz, True,  n_y * n_z, n_y * n_z, n_y * n_z, n_y * n_z)

def _tetrahedral_open_cube(n_x : int, n_y : int, n_z : int):
    '''
    Creates a tetrahedral mesh of an open cube with n_x, n_y, n_z points in each direction.

    Assumes the points have implicit ordering:
        [n_x, n_y, n_z]

    The resulting tetrahedra are implicitly ordered as:
        [n_x -1, n_y -1, n_z -1, 6, 4] where the last two dimensions are the tetrahedron index (0 to 5) and the vertices (0 to 3).
    
    Parameters
    n_x : int
        Number of points in x direction.
    n_y : int
        Number of points in y direction.
    n_z : int
        Number of points in z direction.
    Returns
    -------
    jnp.ndarray [n_tetrahedra = (n_z - 1) * (n_y -1 ) * (n_x - 1) * 6,  4]
        An array containing the tetrahedral connectivity.        
    '''    
    result =  jax.vmap(_open_cube_layer, in_axes=(0, None, None, None))(jnp.arange(n_y - 1), n_x, n_y, n_z)     #[n_y -1, n_z -1, n_x -1, 6, 4]    
    result_rs = jnp.moveaxis(result, 2,0)    
    return result_rs.reshape(-1,4)

def _wedge_tets_true(vertex_a, vertex_b, vertex_c, vertex_d, vertex_e, vertex_f):
    t1 =  jnp.stack([vertex_a, vertex_b, vertex_c, vertex_e], axis = -1)
    t2 =  jnp.stack([vertex_b, vertex_c, vertex_e, vertex_f], axis = -1)
    t3 =  jnp.stack([vertex_b, vertex_c, vertex_d, vertex_f], axis = -1)
    return jnp.stack([t1, t2, t3], axis=-2) # last axis is vertices, second last is tetrahedron index (0 to 3)

def _wedge_tets_false(vertex_a, vertex_b, vertex_c, vertex_d, vertex_e, vertex_f):
    t1 =  jnp.stack([vertex_a, vertex_b, vertex_e, vertex_c], axis = -1)
    t2 =  jnp.stack([vertex_b, vertex_c, vertex_f, vertex_e], axis = -1)
    t3 =  jnp.stack([vertex_b, vertex_c, vertex_f, vertex_d], axis = -1)
    return jnp.stack([t1, t2, t3], axis=-2)

def _tetrahedral_wedge(n_strip : int, offset_wedge_0 : int, offset_wedge_1 : int, offset_index_0 : int, offset_index_1 : int, volume_orientation : bool, stride_0 : int, stride_1 : int):   
    u_i_block = jnp.arange(n_strip)

    # Wedge centers are simply the two points
    vertex_a = offset_wedge_0 + jnp.zeros_like(u_i_block) 
    vertex_b = offset_wedge_1 + jnp.zeros_like(u_i_block)  

    vertex_c = u_i_block * stride_0  + offset_index_0 # first line    
    vertex_d = u_i_block * stride_1  + offset_index_1 # second line

    vertex_e = ( (u_i_block + 1) % n_strip ) * stride_0 + offset_index_0  # first line next
    vertex_f = ( (u_i_block + 1) % n_strip ) * stride_1 + offset_index_1  # second line next

    verts = (vertex_a, vertex_b, vertex_c, vertex_d, vertex_e, vertex_f)
    vertices = jnp.where(volume_orientation, _wedge_tets_true(*verts), _wedge_tets_false(*verts))
    return vertices


@partial(jax.jit, static_argnums = (0, 1, 2, 3))
def _tetrahedral_mesh_layers(n_layers : int, n_theta : int, n_phi : int, full_angle : bool):
    # Points ordering is layer, theta, phi
    # Therefore, given a particular strip:
    # - Starting index of strip i in phi direction is i  + layer * n_theta * n_phi
    # - Starting index of next strip is i * n_phi + n_theta + layer * n_theta * n_phi
    # - Stride between points in strip is n_phi
    # - Stride between points in next strip is n_phi

    phi_blocks     = n_phi if full_angle else n_phi - 1 # full_angle is static so this is possible
    theta_blocks   = n_theta
    n_layer_blocks = n_layers - 1
    
    # First line (will never wrap around):  
    offset_index_0 = jnp.arange(phi_blocks)[None, :]                   + jnp.arange(n_layer_blocks)[:, None] * n_theta * n_phi

    #  Second line (will wrap around phi if full angle):   
    offset_index_1 = ( (jnp.arange(phi_blocks) + 1) % n_phi )[None, :] +  jnp.arange(n_layer_blocks)[:, None] * n_theta * n_phi

    # Third line (will never wrap around phi):
    offset_index_2 = jnp.arange(phi_blocks)[None, :]                   + (jnp.arange(n_layer_blocks)[:, None] + 1) * n_theta * n_phi

    # Fourth line (will wrap around phi if full angle):
    offset_index_3 = ((jnp.arange(phi_blocks) + 1) % n_phi )[None,:]   + (jnp.arange(n_layer_blocks)[:, None] + 1) * n_theta * n_phi
    
    tetrahedra_shaped_layers = jax.vmap(jax.vmap(_tetrahedral_closed_strip, in_axes=(None, 0,0,0,0, None, None, None, None, None)), in_axes=(None, 0,0,0,0, None, None, None, None, None))(
        n_theta,
        offset_index_0,
        offset_index_1,
        offset_index_2,
        offset_index_3,
        True,
        n_phi,
        n_phi,
        n_phi,
        n_phi
    )  # shape (n_layer_blocks, phi_blocks, theta_blocks, 6, 4)        
    return tetrahedra_shaped_layers

@partial(jax.jit, static_argnums = (0, 1, 2))
def _tetrahedral_mesh_axis(n_theta : int, n_phi : int, full_angle : bool):

    phi_blocks     = n_phi if full_angle else n_phi - 1 # full_angle is static so this is possible
    theta_blocks   = n_theta

    # Axis points are always before the first layer points
    axis_end       = n_phi
    
    # First line (will never wrap around):  
    offset_index_0 = jnp.arange(phi_blocks) + axis_end

    #  Second line (will wrap around phi if full angle):   
    offset_index_1 = ( (jnp.arange(phi_blocks) + 1) % n_phi ) + axis_end

    wedge_center_0 = jnp.arange(phi_blocks)
    wedge_center_1 = ( (jnp.arange(phi_blocks) + 1) % n_phi )

    tetrahedra_shaped_layers = jax.vmap(_tetrahedral_wedge, in_axes=(None, 0,0,0,0, None, None, None))(
        n_theta,
        wedge_center_0,
        wedge_center_1,
        offset_index_0,
        offset_index_1,
        True,
        n_phi,
        n_phi
    )  # shape (phi_blocks, theta_blocks, 3, 4)        
    return tetrahedra_shaped_layers

@partial(jax.jit, static_argnums = (0,1,2,3,4))
def _mesh_tetrahedra_connectivity(n_layers : int, include_axis : bool, full_angle : bool,  n_theta : int, n_phi : int):
    '''
    Create a tetrahedral mesh between layers of flux surfaces. 

    Note that this function only creates the connectivity of the tetrahedra. The points must be created separately.

    The points are spaced as follows:

    if axis is included:
        - First n_phi points are the axis points (one per toroidal angle)
    Then,
        - For each layer from 1 to n_layers - 1:
            - For each theta from 0 to n_theta - 1:
                - For each phi from 0 to n_phi - 1:
                    - Point at (s_layer, theta, phi)

    n_layers is the number of flux surfaces, not the number of resulting layers.

    Parameters
    ----------
    n_layers : int
        The number of flux surface layers (including axis if include_axis is True).
    n_theta : int
        The number of poloidal points.
    n_phi : int
        The number of toroidal points.
    include_axis : bool
        Whether to include the magnetic axis in the mesh.
    full_angle : bool
        Whether the toroidal angle is full (0 to 2π) or a segment.
    Returns
    -------
    tetrahedra : jnp.ndarray
        An array of shape (n_tetrahedra, 4) containing the indices of the vertices for each tetrahedron.
    '''
    # We have two options: 
    # - include_axis = True: We have n_layers - 1 layers of tetrahedra + wedges connecting to axis
    #    However, in this case the first layer is from axis to layer 1, so we have n_layers - 1 layers of tetrahedra only
    #    If this is 2, we only have the axis and wedge and the separate layers should not be called.
    # 
    #    

    # - include_axis = False: We have n_layers - 1 layers of tetrahedra only


    if include_axis:
        start_mesh = _tetrahedral_mesh_axis(n_theta, n_phi, full_angle) # shape (phi_blocks, theta_blocks, 3, 4)
        if n_layers > 2:
            # We have to add the n_phi axis points to the non-axis connectivity
            layer_mesh = _tetrahedral_mesh_layers(n_layers - 1, n_theta, n_phi, full_angle) + n_phi # shape (n_layer_blocks, phi_blocks, theta_blocks, 6, 4)
            total_connectivity = jnp.concatenate([start_mesh.reshape(-1,4), layer_mesh.reshape(-1,4)])
            return total_connectivity
        else:
            total_connectivity = start_mesh.reshape(-1,4)
            return total_connectivity
    else:
        layer_mesh = _tetrahedral_mesh_layers(n_layers, n_theta, n_phi, full_angle)  # shape (n_layer_blocks, phi_blocks, theta_blocks, 6, 4)
        total_connectivity = layer_mesh.reshape(-1,4)
        return total_connectivity

@partial(jax.jit, static_argnums=(2,5,6,7))
def _mesh_tetrahedra_points(flux_surfaces : ParametrisedSurface, s_layers, include_axis : bool,  phi_start : float, phi_end : float, full_angle : bool, n_theta : int, n_phi : int):

    s_coords, theta_coords, phi_coords = _mesh_tetrahedra_points_coordinates(s_layers, include_axis, phi_start, phi_end, full_angle, n_theta, n_phi) # shape (n_points,)

    return flux_surfaces.cartesian_position(s_coords, theta_coords, phi_coords)

@eqx.filter_jit
def _mesh_tetrahedra_points_coordinates(s_layers, include_axis : bool,  phi_start : float, phi_end : float, full_angle : bool, n_theta : int, n_phi : int):
    '''
    Internal function that returns the flux surface coordinates of the point in the tetrahedral mesh obtained by the same arguments as _mesh_tetrahedra.

    Returns
    -------
    s_coords : jnp.ndarray
        The radial coordinates of the points in the mesh. Shape (n_points,)
    theta_coords : jnp.ndarray
        The poloidal angle coordinates of the points in the mesh. Shape (n_points,)
    phi_coords : jnp.ndarray
        The toroidal angle coordinates of the points in the mesh. Shape (n_points,)
    ''' 
    
    assert s_layers.shape[0] >= 2, "At least two layers are required to create a tetrahedral mesh. Only {} were provided.".format(s_layers.shape[0])
        
    theta = jnp.linspace(0, 2 * jnp.pi, n_theta, endpoint=False)
    phi   = _phi_grid(phi_start, phi_end, full_angle, n_phi)
    if include_axis:        
        s_mg_layers, theta_mg_layers, phi_mg_layers = jnp.meshgrid(s_layers[1:], theta, phi, indexing='ij')
        axis_s_theta = jnp.zeros(n_phi)
        return jnp.concatenate([axis_s_theta, s_mg_layers.ravel()]), jnp.concatenate([axis_s_theta, theta_mg_layers.ravel()]), jnp.concatenate([phi, phi_mg_layers.ravel()])
    else:
        s_mg_layers, theta_mg_layers, phi_mg_layers = jnp.meshgrid(s_layers, theta, phi, indexing='ij')
        
        return s_mg_layers.ravel(), theta_mg_layers.ravel(), phi_mg_layers.ravel()

@partial(jax.jit, static_argnums=(2, 5, 6, 7))
def _mesh_tetrahedra(flux_surfaces : ParametrisedSurface, s_layers, include_axis : bool,  phi_start : float, phi_end : float, full_angle : bool, n_theta : int, n_phi : int):
    connectivity = _mesh_tetrahedra_connectivity(len(s_layers), include_axis, full_angle, n_theta, n_phi)
    points       = _mesh_tetrahedra_points(flux_surfaces, s_layers, include_axis, phi_start, phi_end, full_angle, n_theta, n_phi)
    return points, connectivity


# ===================================================================================================================================================================================
#                                                                           Convenience Functions
# ===================================================================================================================================================================================

# These functions are for exposing a simple interface to users. They are not jitted and cannot be because the toroidal extent determines whether it is closed, which determines the number of triangles. 
# Similarly, the closed surface meshing function cannot be jitted because of the same reason (plus whether to include the axis or not).
# Functions should be built on the internal jitted functions above.

def mesh_surface(flux_surfaces: ParametrisedSurface, s : float, toroidal_extent : ToroidalExtent, n_theta : int, n_phi : int, normals_facing_outwards : bool = True):
    """
    Create a mesh of points on a flux surface at normalized radius s, with n_theta poloidal and n_phi toroidal points.

    This cannot be jitted because the toroidal extent determines whether it is closed, which determines the number of triangles. Therefore, the 
    size of the arrays is unknown at compile time (which cannot be jitted unless toroidal_extent is static, but this is inconvenient because then the function would recompile for every different extent).

    This is therefore a convenience function only. Internal functions should not build on this function.

    Parameters
    ----------
    flux_surfaces : ParametrisedSurface
        The parametrized surface object containing the parameterization.
    s : float
        The normalized radius of the flux surface to mesh.
    toroidal_extent : ToroidalExtent
        The toroidal extent of the mesh. 
    n_theta : int
        The number of poloidal points.
    n_phi : int
        The number of toroidal points.
    normals_facing_outwards : bool, optional
        Whether the normals should face outwards. Default is True.    
    Returns
    -------
    positions : jnp.ndarray
        An array of shape (n_points, 3) containing the Cartesian coordinates of the mesh points.
    triangles : jnp.ndarray
        An array of shape (n_triangles, 3) containing the indices of the vertices for each triangle.

    """

    return _mesh_surface(flux_surfaces, s, *toroidal_extent, n_theta, n_phi,  normals_facing_outwards)

def mesh_surfaces_closed(flux_surfaces: ParametrisedSurface,
                         s_values_start : float, s_value_end : float,
                         toroidal_extent : ToroidalExtent,
                         n_theta : int, n_phi : int, n_cap : int):
    '''
    Create a closed mesh of points on flux surfaces between normalized radius s_values_start and s_value_end, with n_theta poloidal and n_phi toroidal points.

    This cannot be jitted because the toroidal extent determines whether it is closed, which determines the number of triangles. Therefore, the 
    size of the arrays is unknown at compile time (which cannot be jitted unless toroidal_extent is static, but this is inconvenient because then the function would recompile for every different extent).

    This is therefore a convenience function only. Internal functions should not build on this function.

    Parameters
    ----------
    flux_surfaces : ParametrisedSurface
        The parametrized surface object containing the parameterization.
    s_values_start : float
        The starting normalized radius of the flux surfaces to mesh.
    s_value_end : float
        The ending normalized radius of the flux surfaces to mesh.
    include_axis : bool
        Whether to include the magnetic axis in the mesh.
    toroidal_extent : ToroidalExtent
        The toroidal extent of the mesh. 
    n_theta : int
        The number of poloidal points.
    n_phi : int
        The number of toroidal points.
    n_cap : int
        The number of radial points in the caps if not full toroidal extent.
    Returns
    -------
    positions : jnp.ndarray
        An array of shape (n_points, 3) containing the Cartesian coordinates of the mesh points.
    triangles : jnp.ndarray
        An array of shape (n_triangles, 3) containing the indices of the vertices for each triangle.

    '''
    include_axis = s_values_start == 0.0
    return _mesh_surfaces_closed(flux_surfaces, s_values_start, s_value_end, include_axis, *toroidal_extent, n_theta, n_phi, n_cap)

def mesh_tetrahedra(flux_surfaces : ParametrisedSurface, s_values : jnp.ndarray, toroidal_extent : ToroidalExtent, n_theta : int, n_phi : int):
    '''
    Create a tetrahedral mesh between layers of flux surfaces at normalized radii s_values, with n_theta poloidal and n_phi toroidal points.

    This cannot be jitted because the toroidal extent determines whether it is closed, which determines the number of tetrahedra. Therefore, the 
    size of the arrays is unknown at compile time (which cannot be jitted unless toroidal_extent is static, but this is inconvenient because then the function would recompile for every different extent).

    This is therefore a convenience function only. Internal functions should not build on this function.

    Parameters
    ----------
    flux_surfaces : FluxSurface
        The flux surface object containing the parameterization.
    s_values : jnp.ndarray
        An array of normalized radii of the flux surfaces to mesh.
    toroidal_extent : ToroidalExtent
        The toroidal extent of the mesh. 
    n_theta : int
        The number of poloidal points.
    n_phi : int
        The number of toroidal points.
    Returns
    -------
    positions : jnp.ndarray
        An array of shape (n_points, 3) containing the Cartesian coordinates of the mesh points.
    tetrahedra : jnp.ndarray
        An array of shape (n_tetrahedra, 4) containing the indices of the vertices for each tetrahedron.

    '''
    include_axis = bool(s_values[0] == 0.0)
    return _mesh_tetrahedra(flux_surfaces, s_values, include_axis, *toroidal_extent, n_theta, n_phi)

def mesh_watertight_layers(flux_surfaces : ParametrisedSurface, s_values : jnp.ndarray, toroidal_extent : ToroidalExtent, n_theta : int, n_phi : int):
    '''
    Create a watertight mesh of points on flux surfaces at normalized radii s_values, with n_theta poloidal and n_phi toroidal points.

    This cannot be jitted because the toroidal extent determines whether it is closed, which determines the number of triangles. Therefore, the 
    size of the arrays is unknown at compile time (which cannot be jitted unless toroidal_extent is static, but this is inconvenient because then the function would recompile for every different extent).

    This is therefore a convenience function only. Internal functions should not build on this function.

    Parameters
    ----------
    flux_surfaces : ParametrisedSurface
        The parametrized surface object containing the parameterization.
    s_values : jnp.ndarray
        An array of normalized radii of the flux surfaces to mesh.
    toroidal_extent : ToroidalExtent
        The toroidal extent of the mesh. 
    n_theta : int
        The number of poloidal points.
    n_phi : int
        The number of toroidal points.
    Returns
    -------
    positions : jnp.ndarray
        An array of shape (n_points, 3) containing the Cartesian coordinates of the mesh points.
    triangles : jnp.ndarray
        An array of shape (n_triangles, 3) containing the indices of the vertices for each triangle.

    '''
    return _mesh_watertight_layers(flux_surfaces, s_values, *toroidal_extent, n_theta, n_phi)

