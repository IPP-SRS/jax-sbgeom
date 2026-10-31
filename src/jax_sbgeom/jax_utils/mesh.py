
import jax
import jax.numpy as jnp
import numpy as onp
# ===================================================================================================================================================================================
#                                                                           Pyvista mesh conversion
# ===================================================================================================================================================================================
def mesh_to_pyvista_mesh(pts, conn = None):    
    '''
    Convert a mesh defined by pts and conn to a pyvista mesh

    Either pass a tuple (pts, conn) or pts and conn separately.

    Parameters
    ----------
    pts : jnp.ndarray [N_points, 3]
        Points of the mesh
    conn : jnp.ndarray [N_elements, M] optional
        Connectivity of the mesh (triangles or tetrahedra)
    Returns
    -------
    pyvista mesh

    '''
    if type(pts) is tuple:  
        conn = pts[1]
        pts = pts[0]
        
    import pyvista as pv
    if conn.shape[-1] ==2:
        points_onp = onp.array(pts)
        conn_onp   = onp.array(conn)
        lines = onp.hstack([onp.full((conn_onp.shape[0], 1), 2), conn_onp]).astype(onp.int64)
        lines = lines.flatten()
        mesh = pv.PolyData(points_onp, lines=lines)
        return mesh
    elif conn.shape[-1] == 3:   
        points_onp = onp.array(pts)
        conn_onp   = onp.array(conn)
        faces_pv = onp.hstack([onp.full((conn_onp.shape[0], 1), 3), conn_onp]).astype(onp.int64)
        faces_pv = faces_pv.flatten()
        mesh = pv.PolyData(points_onp, faces_pv)
        return mesh
    elif conn.shape[-1] == 4:
        points_onp = onp.array(pts)
        conn_onp   = onp.array(conn)
        cells = onp.hstack([onp.full((conn_onp.shape[0], 1), 4), conn_onp]).astype(onp.int64)
        cells = cells.flatten()
        mesh = pv.UnstructuredGrid(cells, onp.full(conn_onp.shape[0], 10), points_onp)
        return mesh
    elif conn.shape[-1] == 8:
        points_onp = onp.array(pts)
        conn_onp   = onp.array(conn)
        cells = onp.hstack([onp.full((conn_onp.shape[0], 1), 8), conn_onp]).astype(onp.int64)
        cells = cells.flatten()
        mesh = pv.UnstructuredGrid(cells, onp.full(conn_onp.shape[0], 12), points_onp)
        return mesh

    else:
        raise ValueError("Connectivity must be triangles, tetrahedra, or hexahedra")
    
def vertices_to_pyvista_polyline(pts : jnp.ndarray):
    '''
    Convert a set of points to a pyvista polyline

    Parameters
    ----------
    pts : jnp.ndarray [N_points, 3]
        Points of the polyline
    Returns
    -------
    pyvista PolyData line
    '''
    import pyvista as pv
    points_onp = onp.array(pts)
   # Create a PolyData line from the points
    lines   = onp.hstack([[points_onp.shape[0], *range(points_onp.shape[0])]]).astype(onp.int64)
    polyline = pv.PolyData(points_onp, lines=lines)

    return polyline

# ===================================================================================================================================================================================
#                                                                           Mesh utilities
# ===================================================================================================================================================================================
def surface_normals_from_mesh(mesh):
    '''
    Compute surface normals from triangular mesh

    Parameters
    ----------
    mesh : tuple (pts, conn)
        pts : jnp.ndarray [N_points, 3]
            Points of the mesh
        conn : jnp.ndarray [N_triangles, 3]
            Connectivity of the mesh (triangles)
    Returns
    -------
    jnp.ndarray [N_faces, 3]
        Normals at each face of the mesh
    '''
    assert mesh[1].shape[-1] == 3, "Mesh must be triangular"    
    positions_surface = mesh[0][mesh[1]]    
    e_1 = positions_surface[..., 1, :] - positions_surface[...,0, :]
    e_2 = positions_surface[..., 2, :] - positions_surface[...,0, :]
    normal = jnp.cross(e_1, e_2, axis=-1)
    normals = normal / jnp.linalg.norm(normal, axis=-1, keepdims=True)
    return normals

def boundary_normal_vectors_from_tetrahedron(tetrahedron : jnp.ndarray):
    '''
    Create a boundary vector from tetrahedron

    Parameters
    -----------
    tetrahedron : jnp.ndarray [..., 4,3] 
        Tetrahedron vertex locations
    Returns
    -------
    boundary : jnp.ndarray [..., 4,3]
        Normal
    '''
    assert tetrahedron.shape[-2] == 4 and tetrahedron.shape[-1] == 3, "Tetrahedron must have shape [...,4,3]"
    v0 = tetrahedron[...,1,:] - tetrahedron[...,0,:]
    v1 = tetrahedron[...,2,:] - tetrahedron[...,0,:]
    v2 = tetrahedron[...,3,:] - tetrahedron[...,0,:]
    v3 = tetrahedron[...,2,:] - tetrahedron[...,1,:]
    v4 = tetrahedron[...,3,:] - tetrahedron[...,1,:]
    n0 = jnp.cross(v0, v1)
    n1 = jnp.cross(v2, v0)
    n2 = jnp.cross(v1, v2)
    n3 = jnp.cross(v4, v3)
    normals = jnp.stack([n0, n1, n2, n3], axis=-2)
    positive_volume = jnp.sign(jnp.sum( (v0 * jnp.cross(v2, v1)), axis= -1)) # ...
    normals = normals / jnp.linalg.norm(normals, axis=-1, keepdims=True)  #[...,4,3]
    return normals * positive_volume[..., None, None]


def boundary_centroids_from_tetrahedron(tetrahedron : jnp.ndarray):
    '''
    Create the centroids of all boundaries from tetrahedron

    Parameters
    -----------
    tetrahedra : jnp.ndarray [..., 4,3] 
        Tetrahedron vertex locations
    Returns
    ------- 
    centroids : jnp.ndarray [..., 4,3]
        Centroids
    
    '''
    assert tetrahedron.shape[-2] == 4 and tetrahedron.shape[-1] == 3, "Tetrahedron must have shape [...,4,3]"
    v0 = tetrahedron[...,0,:]
    v1 = tetrahedron[...,1,:]
    v2 = tetrahedron[...,2,:]
    v3 = tetrahedron[...,3,:]
    c0 = (v0 + v1 + v2) / 3.0
    c1 = (v0 + v1 + v3) / 3.0
    c2 = (v0 + v2 + v3) / 3.0
    c3 = (v1 + v2 + v3) / 3.0
    centroids = jnp.stack([c0, c1, c2, c3,], axis=-2)
    return centroids


# ===================================================================================================================================================================================
#                                                                           Functions on meshes
# ===================================================================================================================================================================================
@jax.jit
def volumes_tetrahedra(positions : jnp.ndarray, connectivity : jnp.ndarray):
    assert connectivity.shape[-1] == 4, "Connectivity must have shape (n_tetrahedra, 4) for tetrahedral meshes."
    a = positions[connectivity[:,0], :]
    b = positions[connectivity[:,1], :]
    c = positions[connectivity[:,2], :]
    d = positions[connectivity[:,3], :]
    ab = b - a
    ac = c - a
    ad = d - a
    volumes = jnp.abs(jnp.einsum('ij,ij->i', ab, jnp.cross(ac, ad))) / 6.0
    return volumes

@jax.jit
def volume_of_mesh(positions : jnp.ndarray, connectivity : jnp.ndarray):    
    if connectivity.shape[-1] == 3:
        # Triangle mesh calculation

        a = positions[connectivity[:,0], :]
        b = positions[connectivity[:,1], :]
        c = positions[connectivity[:,2], :]
        cross_prod = jnp.cross(b - a, c - a)
        volume = jnp.sum(jnp.einsum('ij,ij->i', a, cross_prod)) / 6.0
        return volume
    elif connectivity.shape[-1] == 4:
        # Tetrahedral mesh calculation        
        return jnp.abs(volumes_tetrahedra(positions, connectivity)).sum()
    else:
        raise ValueError("Connectivity must have shape (n_elements, 3) for triangles or (n_elements, 4) for tetrahedra.")


@jax.jit
def area_of_mesh(positions : jnp.ndarray, connectivity : jnp.ndarray):
    '''
    Total area of a triangle mesh (sum of the triangle areas).

    Parameters
    ----------
    positions : jnp.ndarray
        Vertex positions, shape (n_vertices, 3)
    connectivity : jnp.ndarray
        Triangle vertex indices, shape (n_triangles, 3)
    Returns
    -------
    float
        Sum of the triangle areas
    '''
    if connectivity.shape[-1] != 3:
        raise ValueError("Connectivity must have shape (n_elements, 3) for triangles.")
    a = positions[connectivity[:,0], :]
    b = positions[connectivity[:,1], :]
    c = positions[connectivity[:,2], :]
    return 0.5 * jnp.sum(jnp.linalg.norm(jnp.cross(b - a, c - a), axis=-1))
