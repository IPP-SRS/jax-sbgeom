import os

import jax_sbgeom as jsb
import jax.numpy as jnp
import numpy as onp

import jax
jax.config.update("jax_enable_x64", True)


import time
from functools import partial
from jax_sbgeom.flux_surfaces.flux_surfaces_base import _check_whether_make_normals_point_outwards_required, ToroidalExtent
import pytest
from functools import lru_cache

from jax_sbgeom.jax_utils import mesh_to_pyvista_mesh
from pathlib import Path


DATA_INPUT_FLUX_SURFACES = Path(__file__).parent.parent / "data" / "flux_surfaces"


def _data_file_to_data_output(data_file : Path, extra_text : str = "") -> Path:
    filename_stem = data_file.stem
    
    output_filename = filename_stem.replace("_input", extra_text + "_output.npy")
    return DATA_INPUT_FLUX_SURFACES / output_filename

def _check_vectorized(fun):
    s_1 = jnp.array([0.5264, 0.567837])
    fun(1.0, 0.2 ,0.3)
    
    fun(s_1, 0.2, 0.3)
    fun(0.2, s_1, 0.2)
    fun(0.2, 0.2, s_1)

    fun(0.2, s_1, s_1)
    fun(s_1, 0.2, s_1)
    fun(s_1, s_1, 0.3)
    
    fun(s_1, s_1, s_1)
        
    b_0, b_1 = jnp.meshgrid(s_1, s_1, indexing='ij')

    fun(1.0, b_0, b_1)
    fun(b_0, 1.0, b_1)
    fun(b_0, b_1, 1.0)
    fun(b_0, b_1, b_1)


def n_theta_n_phi():
    return 11, 15

def _sampling_grid(fs_jax : jsb.flux_surfaces.FluxSurface, n_s : int = 6, n_theta : int = 7, n_phi : int = 5, include_axis = True):

    ss = jax.lax.cond(include_axis, lambda x : jnp.linspace(0,1,n_s), lambda x : jnp.linspace(0,1, n_s + 1)[1:], None)
    
    tt = jnp.linspace(0, 2 * jnp.pi, n_theta, endpoint=False)
    pp = jnp.linspace(0, 2 * jnp.pi / fs_jax.nfp, n_phi, endpoint=True)
    return jnp.meshgrid(ss, tt, pp, indexing='ij')


def _1d_sampling_grid(fs_jax : jsb.flux_surfaces.FluxSurface, n_s : int = 6, n_theta : int = 7, n_phi : int = 5, include_axis = True, reverse_theta : bool = False):
    ss, tt, pp = _sampling_grid(fs_jax, n_s, n_theta, n_phi, include_axis)
    if reverse_theta:
        tt = - onp.array(tt)
    return onp.array(ss).ravel(), onp.zeros(ss.shape).ravel(), onp.array(tt).ravel(), onp.array(pp).ravel()

def _get_flux_surface(vmec_file):
    fs_jax    = jsb.flux_surfaces.FluxSurface.from_hdf5(vmec_file)
    return fs_jax


# ===================================================================================================================================================================================
#                                                                          Base Flux Surface functions
# ===================================================================================================================================================================================

def _get_positions_jsb(fs_jax, sampling_func):
    return fs_jax.cartesian_position(*sampling_func(fs_jax))

def _get_positions_sbgeom(fs_jax, fs_sbgeom, sampling_func_1d):
    pos_sbgeom = fs_sbgeom.Return_Position(*sampling_func_1d(fs_jax, reverse_theta = fs_sbgeom.du_x_dv_sign() == 1.0))
    return pos_sbgeom.reshape(_sampling_grid(fs_jax)[0].shape + (3,))

def _get_normals_jsb(fs_jax, sampling_func):
    return fs_jax.normal(*sampling_func(fs_jax, include_axis=False))

def _get_normals_sbgeom(fs_jax, fs_sbgeom, sampling_func_1d):
    norm_sbgeom = fs_sbgeom.Return_Normal(*sampling_func_1d(fs_jax, include_axis=False, reverse_theta=fs_sbgeom.du_x_dv_sign() == 1.0))
    return norm_sbgeom.reshape(_sampling_grid(fs_jax, include_axis=False)[0].shape + (3,))

def _get_principal_curvatures_jsb(fs_jax, sampling_func):
    return fs_jax.principal_curvatures(*sampling_func(fs_jax, include_axis=False))

def _get_principal_curvatures_sbgeom(fs_jax, fs_sbgeom, sampling_func_1d):
    def return_all_principal_curvatures(s, d, theta, phi):
        # SBGeom returns does not have a vectorized function. we do the inefficient thing here.
        assert s.ndim == 1 and theta.ndim == 1 and phi.ndim == 1 and s.shape == theta.shape and s.shape == phi.shape
        k1 = onp.zeros(s.shape)
        k2 = onp.zeros(s.shape)
        for i in range(s.shape[0]):
            p_curv = fs_sbgeom.Return_Principal_Curvatures(s[i], d[i], theta[i], phi[i])
            k1[i] = p_curv[0]
            k2[i] = p_curv[1]
        return onp.stack([k1, k2], axis=-1)
    curv_sbgeom = return_all_principal_curvatures(*sampling_func_1d(fs_jax, include_axis=False, reverse_theta=fs_sbgeom.du_x_dv_sign() == 1.0))
    return curv_sbgeom.reshape(_sampling_grid(fs_jax, include_axis=False)[0].shape + (2,))

@pytest.mark.parametrize("data_file", list(DATA_INPUT_FLUX_SURFACES.glob("*_input.h5")))
def test_position(data_file):    
    fs_jax = _get_flux_surface(data_file)
    pos_jsb = _get_positions_jsb(fs_jax, _sampling_grid)    
    onp.testing.assert_allclose(pos_jsb, onp.load(_data_file_to_data_output(data_file, "_position")), atol=1e-13)


@pytest.mark.parametrize("data_file", list(DATA_INPUT_FLUX_SURFACES.glob("*_input.h5")))
def test_normal(data_file):    
    fs_jax = _get_flux_surface(data_file)
    pos_jsb = _get_normals_jsb(fs_jax, _sampling_grid)    
    onp.testing.assert_allclose(pos_jsb, onp.load(_data_file_to_data_output(data_file, "_normal")), atol=1e-13)

@pytest.mark.parametrize("data_file", list(DATA_INPUT_FLUX_SURFACES.glob("*_input.h5")))
def test_principal_curvatures(data_file):
    fs_jax = _get_flux_surface(data_file)
    pos_jsb = _get_principal_curvatures_jsb(fs_jax, _sampling_grid)    
    onp.testing.assert_allclose(pos_jsb, onp.load(_data_file_to_data_output(data_file, "_curvature")), atol=1e-13)

@pytest.mark.parametrize("data_file", list(DATA_INPUT_FLUX_SURFACES.glob("*_input.h5")))
@pytest.mark.slow
def test_vectorization(data_file):
    fs_jax = _get_flux_surface(data_file)
    _check_vectorized(fs_jax.cartesian_position)
    _check_vectorized(fs_jax.normal)
    _check_vectorized(fs_jax.principal_curvatures)
# ===================================================================================================================================================================================
#                                                                           Meshing
# ===================================================================================================================================================================================
def _flip_vertices_theta(positions, n_theta, n_phi):
    positions_rs = positions.reshape(n_theta, n_phi, 3)

    first = jnp.take(positions_rs, indices = 0, axis = 0)
    rest = jnp.flip(jnp.take(positions_rs, indices = jnp.arange(1, n_theta), axis = 0), axis = 0)
    positions_rs_flipped = jnp.concatenate([first[None, :, :], rest], axis = 0)
    return positions_rs_flipped.reshape(-1, 3)

def _flip_vertices_theta_tetrahedral(positions, n_theta, n_phi, axis_included : bool):
    vmap_flip = jax.vmap(partial(_flip_vertices_theta, n_theta = n_theta, n_phi=n_phi), in_axes=0, out_axes=0)

    if axis_included:        
        pos_surfs      = positions[n_phi:].reshape(-1, n_theta, n_phi, 3)        
        pos_theta_flip = vmap_flip( pos_surfs ).reshape(-1,3)
        total_flip     = jnp.vstack([positions[:n_phi], pos_theta_flip])
    else:
        total_flip     = vmap_flip(positions.reshape(-1, n_theta, n_phi, 3)).reshape(-1,3)
    return total_flip

def _get_meshing_surface_jsb(fs_jax, smax, tor_extent_str : str = 'half_module', n_theta=50, n_phi=60):
    if tor_extent_str == 'half_module':
        tor_extent = jsb.flux_surfaces.flux_surfaces_base.ToroidalExtent.half_module(fs_jax)
    elif tor_extent_str == 'full':
        tor_extent = jsb.flux_surfaces.flux_surfaces_base.ToroidalExtent.full()
    else:
        raise ValueError(f"Unknown toroidal extent: {tor_extent_str}")
    
    pos_jax, tri_jax = jsb.flux_surfaces.flux_surface_meshing.mesh_surface(fs_jax, smax, tor_extent, n_theta, n_phi, True)
    return pos_jax, tri_jax

def _get_meshing_surface_sbgeom(fs_jax, fs_sbgeom, smax, tor_extent_str : str = 'half_module', n_theta=50, n_phi=60):
    if tor_extent_str == 'half_module':
        tor_extent = jsb.flux_surfaces.flux_surfaces_base.ToroidalExtent.half_module(fs_jax)
    elif tor_extent_str == 'full':
        tor_extent = jsb.flux_surfaces.flux_surfaces_base.ToroidalExtent.full()
    else:
        raise ValueError(f"Unknown toroidal extent: {tor_extent_str}")
    
    mesh_sbgeom = fs_sbgeom.Mesh_Surface(min(smax, 1), max(smax - 1, 0.0), n_phi, n_theta, tor_extent.start, tor_extent.end, True)

    # The sampling cannot be directly influenced. 
    # Instead, we just reverse the theta direction by flipping all vertices if required 
    # This also takes care of the fact that SBGeom does not have normals facing outwards: they get flipped as well so will be equal again.
    reverse_theta = fs_sbgeom.du_x_dv_sign() == 1.0
    if reverse_theta:
        pos_jsb = _flip_vertices_theta(mesh_sbgeom.vertices, n_theta, n_phi)
    else:
        pos_jsb = mesh_sbgeom.vertices

    return pos_jsb, mesh_sbgeom.connectivity    



def _get_mesh_surfaces_closed(flux_surfaces: jsb.flux_surfaces.FluxSurface,
                          s_values_start : float, s_value_end : float,
                          phi_start : float, phi_end : float,
                          n_theta : int, n_phi : int, n_cap : int):
    
    tor_extent = ToroidalExtent(phi_start, phi_end)
    meshes =  jsb.flux_surfaces.flux_surface_meshing.mesh_surfaces_closed(flux_surfaces,
                                                                        s_values_start,
                                                                        s_value_end,
                                                                        tor_extent,                                                                        
                                                                        n_theta,
                                                                        n_phi,
                                                                        n_cap)
    
    pts     = meshes[0]
    conn    = meshes[1]

    assert conn.min() >= 0 and conn.max() < pts.shape[0]
    
    return pts, conn

def _get_all_closed_surfaces(fs_jax):
    
    single_surface  = _get_mesh_surfaces_closed(fs_jax, 0.0, 1.0,  0.0, 2.0 * jnp.pi,  50, 60, 10)    
    two_surfaces    = _get_mesh_surfaces_closed(fs_jax, 0.2, 1.0,  0.0, 2.0 * jnp.pi,  50, 60, 10)
    closed_no_axis  = _get_mesh_surfaces_closed(fs_jax, 0.2, 1.0,  0.0, 0.3 * jnp.pi,  50, 60, 10)    
    closed_axis     = _get_mesh_surfaces_closed(fs_jax, 0.0, 1.0,  0.0, 0.3 * jnp.pi,  50, 60, 10)

    return [single_surface, two_surfaces, closed_no_axis, closed_axis]


def _get_all_closed_surfaces_jsb(fs_jax):
    return _get_all_closed_surfaces(fs_jax)

def _check_closed_surface_volumes(surfaces, atol=1e-10):
    for surf in surfaces:
        points, connectivity = surf
        mesh = mesh_to_pyvista_mesh(points, connectivity)  
        onp.testing.assert_allclose(mesh.volume, jsb.jax_utils.mesh.volume_of_mesh(points, connectivity), atol=atol)

    
def _get_tetrahedral_mesh_jsb(fs_jax, s_values, phi_end, n_theta=20, n_phi=31):
    mesh_j = jsb.flux_surfaces.flux_surface_meshing.mesh_tetrahedra(fs_jax, s_values, ToroidalExtent(0, phi_end), n_theta, n_phi)
    return mesh_j[0], mesh_j[1]

def _get_tetrahedral_mesh_sbgeom(fs_sbgeom, s_values, phi_end, n_theta=20, n_phi=31):
    mesh = fs_sbgeom.Mesh_Tetrahedrons(onp.array(s_values), onp.zeros(len(s_values)), n_phi, n_theta, 0.0, float(phi_end))

    if fs_sbgeom.du_x_dv_sign() == 1.0:
        vertices = _flip_vertices_theta_tetrahedral(mesh.vertices, n_theta, n_phi, axis_included=s_values[0] == 0)
    else:
        vertices = mesh.vertices        
    return vertices, mesh.connectivity

def _get_all_tetrahedral_test_cases():
    s_disc = 5
    s_values = [
        onp.linspace(0.0, 1.0, s_disc),
        onp.linspace(0.0, 1.0, s_disc),
        onp.linspace(0.1, 1.0, s_disc),
        onp.linspace(0.1, 1.0, s_disc)
    ]
    phi_ends = [
        onp.pi,
        2 * onp.pi,
        onp.pi,
        2 * onp.pi
    ]
    return s_values, phi_ends
    
        
def _get_watertight_mesh_jsb(fs_jax, s_values, phi_end, n_theta=20, n_phi=31):
    mesh_j = jsb.flux_surfaces.flux_surface_meshing.mesh_watertight_layers(fs_jax, s_values, ToroidalExtent(0, phi_end), n_theta, n_phi)
    return mesh_j[0], mesh_j[1]

def _get_watertight_mesh_sbgeom(fs_sbgeom, s_values, phi_end, n_theta=20, n_phi=31):
    sdiff = onp.diff(s_values)
    mesh = fs_sbgeom.Mesh_Watertight_Flux_Surfaces(float(s_values[0]), 0.0, sdiff, onp.zeros_like(sdiff), n_phi, n_theta, 0.0, float(phi_end))
    reverse_theta = fs_sbgeom.du_x_dv_sign() == 1.0
    if reverse_theta:
        pos_jsb = []
        for i in range(len(sdiff) - 1):
            pos_jsb_i = _flip_vertices_theta(mesh[0][i * n_theta * n_phi : (i+1)*n_theta*n_phi], n_theta, n_phi)
            pos_jsb.append(pos_jsb_i)
        pos_jsb = jnp.vstack(pos_jsb)
        pos_jsb = pos_jsb.reshape(-1,3)
    else:
        pos_jsb = mesh[0]
    return pos_jsb, mesh[1]    


@pytest.mark.parametrize("data_file", list(DATA_INPUT_FLUX_SURFACES.glob("*_input.h5")))
def test_meshing_surface_half_mod(data_file):    
    fs_jax = _get_flux_surface(data_file)
    n_theta, n_phi = n_theta_n_phi()
    
    
    pos_jsb, tri_jsb = _get_meshing_surface_jsb(fs_jax, 1.0, tor_extent_str="half_module", n_theta=n_theta, n_phi=n_phi)    

    pos_sbgeom = onp.load(_data_file_to_data_output(data_file, "_meshing_halfmod_position"))
    tri_sbgeom = onp.load(_data_file_to_data_output(data_file, "_meshing_halfmod_connectivity"))
        
    onp.testing.assert_allclose(pos_jsb, pos_sbgeom, atol=1e-13)
    onp.testing.assert_allclose(tri_jsb, tri_sbgeom, atol=1e-13)

@pytest.mark.parametrize("data_file", list(DATA_INPUT_FLUX_SURFACES.glob("*_input.h5")))
def test_meshing_surface_full(data_file):
    fs_jax = _get_flux_surface(data_file)
    n_theta, n_phi = n_theta_n_phi()
    
    pos_jsb, tri_jsb = _get_meshing_surface_jsb(fs_jax, 1.0, tor_extent_str="full", n_theta=n_theta, n_phi=n_phi)    

    pos_sbgeom = onp.load(_data_file_to_data_output(data_file, "_meshing_full_position"))
    tri_sbgeom = onp.load(_data_file_to_data_output(data_file, "_meshing_full_connectivity"))
        
    
    onp.testing.assert_allclose(pos_jsb, pos_sbgeom, atol=1e-13)
    onp.testing.assert_allclose(tri_jsb, tri_sbgeom, atol=1e-13)


@pytest.mark.parametrize("data_file", list(DATA_INPUT_FLUX_SURFACES.glob("*_input.h5")))
def test_all_closed_surfaces(data_file):    
    fs_jax = _get_flux_surface(data_file)
    surfaces = _get_all_closed_surfaces_jsb(fs_jax)
    _check_closed_surface_volumes(surfaces)

def _get_all_cases_jsb(fs_jax):
    
    s_values, phi_ends = _get_all_tetrahedral_test_cases()

    n_theta, n_phi = n_theta_n_phi()

    mesh_pos = [] 
    mesh_con = [] 
    for s_i, phi_i in zip(s_values, phi_ends):
        mesh_i =  _get_tetrahedral_mesh_jsb(fs_jax, s_i, phi_i, n_theta=n_theta, n_phi=n_phi)

        mesh_pos.append(mesh_i[0])
        mesh_con.append(mesh_i[1])
    return onp.concatenate(mesh_pos, axis=0), onp.concatenate(mesh_con, axis=0)

    
def _get_all_cases_sbgeom(fs_sbgeom):    

    s_values, phi_ends = _get_all_tetrahedral_test_cases()

    n_theta, n_phi = n_theta_n_phi()

    mesh_pos = [] 
    mesh_con = [] 
    for s_i, phi_i in zip(s_values, phi_ends):
        mesh_i =  _get_tetrahedral_mesh_sbgeom(fs_sbgeom, s_i, phi_i, n_theta=n_theta, n_phi=n_phi)

        mesh_pos.append(mesh_i[0])
        mesh_con.append(mesh_i[1])
    return onp.concatenate(mesh_pos, axis=0), onp.concatenate(mesh_con, axis=0)


@pytest.mark.parametrize("data_file", list(DATA_INPUT_FLUX_SURFACES.glob("*_input.h5")))
def test_all_tetrahedral_meshes(data_file):
    fs_jax = _get_flux_surface(data_file)
    s_values, phi_ends = _get_all_tetrahedral_test_cases()

    n_theta, n_phi = n_theta_n_phi()

    mesh_pos, mesh_conn = _get_all_cases_jsb(fs_jax)

    
    pos_sbgeom = onp.load(_data_file_to_data_output(data_file, "_tetrahedral_all_position"))
    conn_sbgeom = onp.load(_data_file_to_data_output(data_file, "_tetrahedral_all_connectivity"))
    onp.testing.assert_allclose(mesh_pos, pos_sbgeom, atol=1e-13, err_msg="Tetrahedral mesh points do not match SBGeom")
    onp.testing.assert_allclose(mesh_conn, conn_sbgeom, err_msg="Tetrahedral mesh connectivity do not match SBGeom")
  
@pytest.mark.parametrize("data_file", list(DATA_INPUT_FLUX_SURFACES.glob("*_input.h5")))
def test_watertight_surfaces(data_file):    
    fs_jax = _get_flux_surface(data_file)

    n_theta, n_phi = n_theta_n_phi()
    s_disc = 5

    s_values = jnp.linspace(0.1, 1.0, s_disc)

    mesh_jsb = _get_watertight_mesh_jsb(fs_jax, s_values, 0.2 * jnp.pi, n_theta=n_theta, n_phi=n_phi)

    pos_sbgeom = onp.load(_data_file_to_data_output(data_file, "_watertight_position"))
    conn_sbgeom = onp.load(_data_file_to_data_output(data_file, "_watertight_connectivity"))

    onp.testing.assert_allclose(mesh_jsb[0], pos_sbgeom, atol=1e-13, err_msg="Watertight mesh points do not match SBGeom")
    onp.testing.assert_allclose(onp.concatenate(mesh_jsb[1],axis=0), conn_sbgeom, err_msg="Watertight mesh connectivity do not match SBGeom")



# ===================================================================================================================================================================================
#                                                                           FluxSurface base functions
# ===================================================================================================================================================================================

def _get_volume_half_mod_jsb(fs_jax, s=0.5357):
    return jsb.flux_surfaces.flux_surfaces_base._volume_from_fourier_half_mod(fs_jax, s)

def _get_volume_full_jsb(fs_jax, s=0.5357):
    return jsb.flux_surfaces.flux_surfaces_base._volume_from_fourier(fs_jax, s)

@pytest.mark.parametrize("data_file", list(DATA_INPUT_FLUX_SURFACES.glob("*_input.h5")))
def test_volumes(data_file):
    fs_jax = _get_flux_surface(data_file)
    s      = 0.5357    

    vol_half_mod = _get_volume_half_mod_jsb(fs_jax, s)
    vol_full     = _get_volume_full_jsb(fs_jax, s)
    
    onp.testing.assert_allclose(vol_half_mod, vol_full, atol=1e-13)

# ===================================================================================================================================================================================
#                                                                          Conversion to Fourier
# ===================================================================================================================================================================================

@pytest.mark.parametrize("data_file", list(DATA_INPUT_FLUX_SURFACES.glob("*_input.h5")))
def test_RZ_to_VMEC_lcfs(data_file):
    fs_jax = _get_flux_surface(data_file)

    n_theta = fs_jax.settings.mpol * 2 + 1 # nyquist
    n_phi   = fs_jax.settings.ntor * 2 + 1 # nyquist

     
    theta = jnp.linspace(0, 2*jnp.pi, n_theta, endpoint=False)
    phi   = jnp.linspace(0, 2*jnp.pi / fs_jax.nfp, n_phi, endpoint=False)
    theta_mg, phi_mg = jnp.meshgrid(theta, phi, indexing='ij')

    positions_jax = fs_jax.cylindrical_position(1.0, theta_mg, phi_mg)

    RZ            = jsb.flux_surfaces.convert_to_vmec._rz_to_vmec_representation(positions_jax[..., 0], positions_jax[..., 1])

    onp.testing.assert_allclose(RZ.Rmnc, fs_jax.data.Rmnc[-1,:], rtol=1e-12, atol=1e-12)
    onp.testing.assert_allclose(RZ.Zmns, fs_jax.data.Zmns[-1,:], rtol=1e-12, atol=1e-12)