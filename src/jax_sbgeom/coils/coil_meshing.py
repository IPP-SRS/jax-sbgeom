import jax 
import jax.numpy as jnp
from jax_sbgeom.flux_surfaces.flux_surface_meshing import _build_triangles_surface, _build_closed_strips
from .base_coil import FiniteSizeCoil
from functools import partial
from .coilset import FiniteSizeCoilSet

def _mesh_finite_sized_lines_connectivity(n_samples : int, n_lines_per_coil : int, normal_orientation : bool):
    ''''
    Connectivity of a surface spanned by some finite sized lines

    Uses _build_triangles_surface to build the connectivity.

    Parameters
    ----------
    n_samples : int
        Number of samples along the coil
    normal_orientation : bool
        Whether to orient the normals outwards (True) or inwards (False)
    Returns
    -------
    jnp.ndarray
        Connectivity array of the meshed coil surface
    '''

    return _build_triangles_surface(n_samples, n_samples, n_lines_per_coil, n_lines_per_coil, normal_orientation)

def _mesh_rectangular_finite_sized_coils_connectivity(n_samples : int, normal_orientation : bool):
    ''''
    Connectivity of a surface spanned by 4 finite sized lines

    Parameters
    ----------
    n_samples : int
        Number of samples along the coil
    normal_orientation : bool
        Whether to orient the normals outwards (True) or inwards (False)
    Returns
    -------
    jnp.ndarray
        Connectivity array of the meshed coil surface

    '''
    return _mesh_finite_sized_lines_connectivity(n_samples, 4, normal_orientation)

@partial(jax.jit, static_argnums = 1)
def mesh_coil_surface(coil : FiniteSizeCoil, n_s : int, width_radial : float, width_phi : float):
    '''
    Mesh the surface of a coil using a defined number of samples and coil width.

    Parameters
    ----------
    coil : Coil
        Coil to mesh
    n_s : int
        Number of samples along the coil
    width_radial : float
        Radial width of the finite sized coil
    width_phi : float
        Toroidal width of the finite sized coil
    Returns
    -------
    jnp.ndarray
        Vertices of the meshed coil surface
    jnp.ndarray
        Connectivity array of the meshed coil surface
    '''
    finite_size_lines = coil.finite_size(jnp.linspace(0, 1.0, n_s, endpoint=False), width_radial, width_phi)    
    connectivity = _mesh_rectangular_finite_sized_coils_connectivity(n_s, normal_orientation=True)
    return finite_size_lines.reshape(-1, 3), connectivity

@partial(jax.jit, static_argnums = (0,1,2,3))
def _mesh_rectangular_finite_sized_coilset_connectivity(n_coils, n_samples : int, n_lines_per_coil : int, normal_orientation : bool):
    '''
    Connectivity of a coilset surface spanned by some finite sized lines per coil
    Includes an offset for each coil in the coilset to ensure a consistent coilset mesh

    Assumes the vertices array has a shape 
    [n_coils, n_samples, n_lines_per_coil, 3] but then flattened.

    Parameters
    ----------
    n_coils : int
        Number of coils in the coilset
    n_samples : int
        Number of samples along the coil
    normal_orientation : bool
        Whether to orient the normals outwards (True) or inwards (False)
    Returns
    -------
    jnp.ndarray
        Connectivity array of the meshed coil surface
    '''
    connectivity_base = _mesh_finite_sized_lines_connectivity(n_samples, n_lines_per_coil, normal_orientation)
    offsets = jnp.arange(n_coils) * (n_samples * n_lines_per_coil)    
    return (connectivity_base[None, :, :] + offsets[:, None, None]).reshape(-1, connectivity_base.shape[1])


def mesh_coilset_surface(coils : FiniteSizeCoilSet, n_s : int, width_radial : float, width_phi : float):
    '''
    Mesh the surface of a coilset

    The coils vertices are originally:
    [n_coils, n_s, 4, 3] (4 lines per coil)
    
    The coils connectivity is originally:
    [n_coils, n_s, 4, 2, 3] (4 lines per coil, 2 triangles per quad)
    Both are reshaped to (-1,3) to facilate easier post processing.
    

    Parameters
    ----------
    coil : Coil
        Coil to mesh
    n_s : int
        Number of samples along the coil
    width_radial : float
        Radial width of the finite sized coil
    width_phi : float
        Toroidal width of the finite sized coil
    Returns
    -------
    jnp.ndarray
        Vertices of the meshed coil surface
    jnp.ndarray
        Connectivity array of the meshed coil surface
    '''
    finite_size_lines = coils.finite_size(jnp.linspace(0, 1.0, n_s, endpoint=False), width_radial, width_phi)            
    connectivity = _mesh_rectangular_finite_sized_coilset_connectivity(coils.n_coils, n_s, 4, True)
    return finite_size_lines.reshape(-1, 3), connectivity


def _generate_vertices_from_finite_sized_lines(finite_size_lines : jnp.ndarray, n_grid : int):
    '''
    Generate vertices of a coil volume from finite sized lines

    Parameters
    ----------
    finite_size_lines : jnp.ndarray [n_samples, 4, 3]
        Finite sized lines of the coil
    n_grid : int
        Number of grid points in the radial and toroidal directions
    Returns
    -------
    jnp.ndarray
        Vertices of the meshed coil volume
    '''
    # finite_size_lines has shape [n_samples, 4, 3]
    # We want to generate a grid of points between each line in the radial and toroidal directions
    # The resulting vertices array should have shape [n_samples, n_grid, n_grid, 3]
    # where the first n_grid corresponds to the radial direction and the second n_grid corresponds to the toroidal direction

    vertices = jnp.zeros((finite_size_lines.shape[0], n_grid, n_grid, 3))

    # Generate grid points in the radial and toroidal directions (for now they are the same, but we can change this later)
    radial_grid = jnp.linspace(0, 1, n_grid)
    toroidal_grid = jnp.linspace(0, 1, n_grid)

    # Generate grid of points between each line
    radial_grid, toroidal_grid = jnp.meshgrid(radial_grid, toroidal_grid)
    
    for i, points_on_surface in enumerate(finite_size_lines):
        # surface has 4 points with 3D coordinates (shape [4, 3]) corresponding to the 4 lines of the finite sized coil

        # Combine grid of points into a single array of vertices
        # The vertices are generated by linearly interpolating between 4 corners of the surface defined by the finite sized lines
        vertices = vertices.at[i].set(
            points_on_surface[0] * (1 - radial_grid[..., None]) * (1 - toroidal_grid[..., None]) + 
            points_on_surface[3] * radial_grid[..., None] * (1 - toroidal_grid[..., None]) + 
            points_on_surface[1] * (1 - radial_grid[..., None]) * toroidal_grid[..., None] + 
            points_on_surface[2] * radial_grid[..., None] * toroidal_grid[..., None])

        # This is equivalent to flipping the array
        # vertices_v2 = vertices.at[i].set(
        #     points_on_surface[1] * jnp.flip(radial_grid * toroidal_grid)[..., None] + 
        #     points_on_surface[3] * jnp.flip(radial_grid * toroidal_grid, axis=0)[..., None] + 
        #     points_on_surface[2] * jnp.flip(radial_grid * toroidal_grid, axis=1)[..., None] + 
        #     points_on_surface[0] * (radial_grid * toroidal_grid)[..., None])
    
    return vertices


def _build_hexahedrals_from_vertices(n_samples : int, n_grid : int):
    '''
    Build hexahedral connectivity from vertices of a coil volume

    Parameters
    ----------
    n_samples : int
        Number of samples along the coil
    n_grid : int
        Number of grid points in the radial and toroidal directions
    Returns
    -------
    jnp.ndarray
        Connectivity array of the meshed coil volume (hexahedrals)
    '''
    # The vertices array has shape [n_samples, n_grid, n_grid, 3]
    # We want to generate a connectivity array of shape [n_samples*(n_grid-1)*(n_grid-1), 8]
    # where each hexahedral element is defined by 8 vertices corresponding to the corners of the hexahedron
    # The last layer wraps around to the first layer for cyclic connectivity

    connectivity = jnp.zeros((n_samples, (n_grid-1)*(n_grid-1), 8), dtype=jnp.int32)

    for i in range(n_samples):
        i_next = (i + 1) % n_samples
        for j in range(n_grid - 1):
            for k in range(n_grid - 1):
                # vertexes created in intuitive order, proper order for cell definition is ensured when writing them into connectvity matrix 
                v0 = i * n_grid * n_grid + j * n_grid + k
                v1 = v0 + 1
                v2 = v0 + n_grid
                v3 = v2 + 1
                v4 = i_next * n_grid * n_grid + j * n_grid + k
                v5 = v4 + 1
                v6 = v4 + n_grid
                v7 = v6 + 1

                # vertex order for cell definition is implemented at this step
                connectivity = connectivity.at[i, j*(n_grid-1) + k].set(jnp.array([v0, v2, v3, v1, v4, v6, v7, v5]))

    return connectivity.reshape(-1, 8)


def refine_longitudinal_mesh(finite_size_lines, refinement_array : jnp.ndarray):
    # AI written - probably needs refactoring. But first the question is if we really need this...
    # We want to refine the mesh in the longitudinal direction according to the provided refinement array
    # The refinement array should have a length equal to n_longitudinal and values between 0 and 1, where 0 means no refinement and 1 means maximum refinement (doubling the number of points)
    # We can achieve this by generating additional points between the original points according to the refinement array

    # Generate refined finite sized lines
    refined_finite_size_lines = []
    for i in range(n_longitudinal):
        refined_finite_size_lines.append(finite_size_lines[i])
        if refinement_array[i] > 0:
            # Generate additional point between this point and the next point
            next_i = (i + 1) % n_longitudinal
            additional_point = (finite_size_lines[i] + finite_size_lines[next_i]) / 2
            refined_finite_size_lines.append(additional_point)

    finite_size_lines = jnp.array(refined_finite_size_lines)
    n_longitudinal = finite_size_lines.shape[0]
    return finite_size_lines, n_longitudinal


def mesh_coil_volumetric(coil : FiniteSizeCoil, 
                         n_longitudinal : int, 
                         n_grid : int, 
                         width_radial : float, 
                         width_phi : float,
                         n_radial: int = None,
                         n_toroidal: int = None,
                         longitudinal_refinement: jnp.ndarray = None):
    '''
    Mesh the volume of a coil using a defined number of samples and coil width.

    Parameters
    ----------
    coil : Coil
        Coil to mesh
    n_longitudinal : int
        Number of samples along the coil
    n_grid : int
        Number of grid points in the radial and toroidal directions
    width_radial : float
        Radial width of the finite sized coil
    width_phi : float
        Toroidal width of the finite sized coil
    Returns
    -------
    jnp.ndarray
        Vertices of the meshed coil volume
    jnp.ndarray
        Connectivity array of the meshed coil volume (tetrahedra)
    '''
    finite_size_lines = coil.finite_size(jnp.linspace(0, 1.0, n_longitudinal, endpoint=False), width_radial, width_phi)

    if longitudinal_refinement is not None:
        finite_size_lines, n_longitudinal = refine_longitudinal_mesh(finite_size_lines, longitudinal_refinement)
    
    vertices = _generate_vertices_from_finite_sized_lines(finite_size_lines, n_grid)
    connectivity = _build_hexahedrals_from_vertices(n_longitudinal, n_grid)
    return vertices.reshape(-1, 3), connectivity


def _find_closest_lines_to_point(lines, ref_point, n_closest=2, take_mean='True'):
    """
    Given a list of densely-sampled lines (each an [n, 3] array of points)
    and a reference point (shape [3]), return the indices of the
    `n_closest` lines that lie nearest to the point on average.

    Parameters
    ----------
    lines : array-like, shape [grid, n_lines, 3]
    point : array-like, shape [3]
    n_closest : int, number of closest lines to return (default 2)
    take_mean : bool, decide if the average disatnce should be taken into account, 
    local minimum if false (default True)
    """
    ref_point = jnp.asarray(ref_point, dtype=float)

    # distance from every point on every line to the reference point
    point_distances = jnp.linalg.norm(
        lines - ref_point[None, None, :], axis=-1
    )  # [grid, n_lines]

    if take_mean:
        # average distance of each line to the reference point
        distances = point_distances.mean(axis=0)  # [n_lines]
    else:
        # closest distance of each line to the reference point
        distances = point_distances.min(axis=0)  # [n_lines]

    order = jnp.argsort(distances)[:n_closest]

    closest_indices = order.tolist()
    closest_distances = distances[order].tolist()

    return closest_indices, closest_distances


def intercoil_lines (coil1: FiniteSizeCoil, 
                              coil2: FiniteSizeCoil,
                              n_longitudinal : int,
                              coil_width_radial: float,
                              coil_width_phi : float,):
    '''
    Mesh the volume between two coils using a defined number of samples and coil width.

    TODO: Take into account diffrent widths for the two coils, and different number of samples for the two coils.

    Parameters
    ----------
    coil1 : FiniteSizeCoil
        First coil
    coil2 : FiniteSizeCoil
        Second coil
    n_longitudinal : int
        Number of samples along the coil
    n_grid : int
        Number of grid points in the toroidal direction
    width_radial : float
        Radial width of the finite sized coil
    width_phi : float
        Toroidal width of the finite sized coil

    Returns
    -------
    jnp.ndarray
        Vertices of the meshed inter-coil volume
    jnp.ndarray
        Connectivity array of the meshed inter-coil volume (tetrahedra)
    '''

    finite_size_lines_1 = coil1.finite_size(jnp.linspace(0, 1.0, n_longitudinal, endpoint=False), coil_width_radial, coil_width_phi)
    finite_size_lines_2 = coil2.finite_size(jnp.linspace(0, 1.0, n_longitudinal, endpoint=False), coil_width_radial, coil_width_phi)

    # Find the closest lines to the midpoint between the two coils (inner corners between the two coils)
    idx1, _ = _find_closest_lines_to_point(finite_size_lines_1, (coil1.centre() + coil2.centre()) / 2, n_closest=1)
    idx2, _ = _find_closest_lines_to_point(finite_size_lines_2, (coil1.centre() + coil2.centre()) / 2, n_closest=1)

    # The above function returns lists of indices, but we want it as int
    idx1 = idx1[0]
    idx2 = idx2[0]

    # This should work as long as the coils have the same order of the lines (which they should have)
    finite_size_lines = jnp.stack([
        finite_size_lines_1[:, idx1, :],
        finite_size_lines_2[:, idx2, :],
        finite_size_lines_1[:, (idx2+2)%4, :],
        finite_size_lines_2[:, (idx1+2)%4, :],
    ], axis=1)  # shape [n_longitudinal, 4, 3]

    return finite_size_lines
