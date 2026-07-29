import jax_sbgeom as jsb
import pytest
import jax
import jax.numpy as jnp 
import numpy as onp
from jax_sbgeom.jax_utils import raytracing as RT
from jax_sbgeom.jax_utils import mesh_to_pyvista_mesh
jax.config.update("jax_enable_x64", True)
from .test_flux_surface_base_data import DATA_INPUT_FLUX_SURFACES



def _get_coil_file(data_file):
    
    
    return data_file.parent.parent / "coils" / (data_file.stem + ".npy")



def _get_all_discrete_coils(request):        
    positions = onp.load(request)
    return jsb.coils.CoilSet.from_list([jsb.coils.DiscreteCoil.from_positions(positions[i]) for i in range(positions.shape[0])])



def _get_cws_and_flux_surface(data_file):
    
    fs_jax = jsb.flux_surfaces.FluxSurfaceNormalExtendedNoPhi.from_hdf5(data_file)
    coilset_jax = _get_all_discrete_coils(_get_coil_file(data_file=data_file))
    
    cws_mesh = jsb.coils.coil_winding_surface.create_optimized_coil_winding_surface(coilset_jax, 100, 200, 'spline')
    return cws_mesh, fs_jax

def check_cws_ray_tracing(cws_mesh, fs_jax):
    n_theta            = 210
    n_phi              = 100
    theta              = jnp.linspace(0, 2 * jnp.pi, n_theta)
    phi                = jnp.linspace(0, 2 * jnp.pi / fs_jax.nfp, n_phi)
    theta, phi         = jnp.meshgrid(theta, phi, indexing='ij')
    positions_lcfs_mg  = fs_jax.cartesian_position(1.0,  theta, phi)
    directions_lcfs_mg = fs_jax.cartesian_position(2.0, theta, phi) - positions_lcfs_mg 
    mesh_rt            = mesh_to_pyvista_mesh(*cws_mesh)
    trimesh_mesh       = mesh_rt.extract_surface()
    
    final_points, final_rays, final_cells = trimesh_mesh.multi_ray_trace(onp.array(positions_lcfs_mg).reshape(-1,3), onp.array(directions_lcfs_mg).reshape(-1,3), first_point = True,retry = True)
    final_points                          = jnp.array(final_points).reshape(positions_lcfs_mg.shape)
    dmesh_trimesh                         = jnp.linalg.norm(final_points - positions_lcfs_mg, axis=-1).reshape(theta.shape)
    dmesh_jax                             = RT.find_minimum_distance_to_mesh(positions_lcfs_mg, directions_lcfs_mg, cws_mesh)
    dmesh_utility                         = jsb.flux_surfaces.flux_surfaces_utilities.generate_thickness_matrix(fs_jax, cws_mesh, n_theta, n_phi)
    reinterpolated_positions              = fs_jax.cartesian_position(1.0 + dmesh_jax, theta, phi)
    onp.testing.assert_allclose(onp.array(dmesh_utility[2]), onp.array(dmesh_jax), atol=1e-10)
    onp.testing.assert_allclose(onp.array(dmesh_trimesh), onp.array(dmesh_jax), atol=1e-10)
    onp.testing.assert_allclose(onp.array(reinterpolated_positions), onp.array(final_points), atol=1e-10)


@pytest.mark.parametrize("data_file", list(DATA_INPUT_FLUX_SURFACES.glob("*_input.h5")))
def test_cws_ray_tracing(data_file):
    cws_mesh, fs_jax =_get_cws_and_flux_surface(data_file)
    check_cws_ray_tracing(cws_mesh, fs_jax)
    

@pytest.mark.parametrize("data_file", list(DATA_INPUT_FLUX_SURFACES.glob("*_input.h5")))
def test_half_to_full_module_conversion(data_file):
    '''
    This test ensures the conversion of full and half module meshes work. It test by
    
    1. Generating a half module mesh and a full module mesh using the mesh_surface function, and then converting the half module mesh to a full module mesh using the convert_half_module_points_to_full_module function. The resulting full module mesh is then compared to the original full module mesh to ensure they are the same.
    2. If the number of field periods is greater than 4, we also test the convert_full_module_points_multiple_full_module function by generating a mesh for a toroidal extent that covers one before and two after the full module, and comparing it to the result of converting the half module mesh to a full module mesh and then extending it by one full module in both directions.
    3. We also test the convert_full_module_points_multiple_full_module function by generating a mesh for a toroidal extent that covers zero before and one after the full module, and comparing it to the result of converting the half module mesh to a full module mesh and then extending it by one full module in the positive direction.
    4. We also test the convert_full_module_points_multiple_full_module function by generating a mesh for a toroidal extent that covers one before and zero after the full module, and comparing it to the result of converting the half module mesh to a full module mesh and then extending it by one full module in the negative direction.

    '''
    fs_jax = jsb.flux_surfaces.FluxSurfaceNormalExtendedNoPhi.from_hdf5(data_file)

    half_module = jsb.flux_surfaces.ToroidalExtent.half_module(fs_jax)
    full_module = jsb.flux_surfaces.ToroidalExtent.full_module(fs_jax)

    n_theta = 20
    n_phi   = 14

    half_module_surface = jsb.flux_surfaces.mesh_surface(fs_jax, 1.0, half_module, n_theta, n_phi, True)
    full_module_surface = jsb.flux_surfaces.mesh_surface(fs_jax, 1.0, full_module, n_theta, 2 * n_phi - 1, True)
    points_half_mod = jsb.flux_surfaces.flux_surfaces_utilities.convert_half_module_points_to_full_module(half_module_surface[0].reshape(n_theta, n_phi, 3))
    onp.testing.assert_allclose(full_module_surface[0], points_half_mod.reshape(-1,3), atol=1e-10, rtol = 1e-10)

    
    if fs_jax.nfp > 4:
        one_two_half_module_surface      = jsb.flux_surfaces.ToroidalExtent(- 2 * jnp.pi / fs_jax.nfp, 2 * jnp.pi / fs_jax.nfp * 3)
        one_two_half_module_surface_mesh = jsb.flux_surfaces.mesh_surface(fs_jax, 1.0, one_two_half_module_surface, n_theta,  (2 * n_phi - 1) + 3 *(2 * n_phi - 2), True)

        points_extended = jsb.flux_surfaces.flux_surfaces_utilities.convert_full_module_points_multiple_full_module(points_half_mod, full_module, 1, 2)

        onp.testing.assert_allclose(one_two_half_module_surface_mesh[0], points_extended.reshape(-1,3), atol=1e-10, rtol = 1e-10)

    one_before_one_half_module_surface      = jsb.flux_surfaces.ToroidalExtent(- 2 * jnp.pi / fs_jax.nfp, 2 * jnp.pi / fs_jax.nfp)
    one_before_one_half_module_surface_mesh = jsb.flux_surfaces.mesh_surface(fs_jax, 1.0, one_before_one_half_module_surface, n_theta,  (2 * n_phi - 1) + 1 * (2 * n_phi - 2), True)
    points_extended = jsb.flux_surfaces.flux_surfaces_utilities.convert_full_module_points_multiple_full_module(points_half_mod, full_module, 1, 0)
    onp.testing.assert_allclose(one_before_one_half_module_surface_mesh[0], points_extended.reshape(-1,3), atol=1e-10, rtol = 1e-10)

    one_before_one_half_module_surface      = jsb.flux_surfaces.ToroidalExtent(0, 2 * 2 * jnp.pi / fs_jax.nfp)
    one_before_one_half_module_surface_mesh = jsb.flux_surfaces.mesh_surface(fs_jax, 1.0, one_before_one_half_module_surface, n_theta,  (2 * n_phi - 1) + 1 * (2 * n_phi - 2), True)
    points_extended = jsb.flux_surfaces.flux_surfaces_utilities.convert_full_module_points_multiple_full_module(points_half_mod, full_module, 0, 1)
    onp.testing.assert_allclose(one_before_one_half_module_surface_mesh[0], points_extended.reshape(-1,3), atol=1e-10, rtol = 1e-10)
