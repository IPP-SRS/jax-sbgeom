import jax_sbgeom as jsb
import pytest
import jax
import jax.numpy as jnp 
import numpy as onp
jax.config.update("jax_enable_x64", True)

from pathlib import Path
DATA_INPUT_FLUX_SURFACES = Path(__file__).parent.parent / "data" / "flux_surfaces"





def _get_flux_surface_jax(data_file):    
    fs_jax    = jsb.flux_surfaces.FluxSurface.from_hdf5(data_file)
    return fs_jax




@pytest.mark.parametrize("data_file", list(DATA_INPUT_FLUX_SURFACES.glob("*_input.h5")))
def test_bvh_probing(data_file):
    fs_jax = _get_flux_surface_jax(data_file)
    N_test = 300
    surface_mesh = jsb.flux_surfaces.flux_surface_meshing.mesh_surface(fs_jax, 1.0, jsb.flux_surfaces.ToroidalExtent.full(), 65,62)

    lbvh         = jsb.jax_utils.raytracing.build_lbvh(surface_mesh[0], surface_mesh[1])

    hits_bvh_probing = jsb.jax_utils.raytracing._probe_bvh_imp(lbvh, surface_mesh[0][0:N_test])[0]
    
    @jax.jit
    def bulk_all(aabb, points):
        final_hits = jax.vmap(jax.vmap(jsb.jax_utils.raytracing._point_in_aabb, in_axes=(None, 0)), in_axes=(0, None))(points, aabb)  
        return final_hits
    
    aabb_total = jsb.jax_utils.raytracing._create_aabb(surface_mesh[0][surface_mesh[1]])
    hits_bulk = bulk_all(aabb_total, surface_mesh[0][0:N_test])

    for i in range(N_test):
        hi = hits_bvh_probing[i]
        hits_bvh = lbvh.order[hi[hi>0]]

        hits_original = jnp.where(hits_bulk[i])[0]
        sorted_hits = jnp.unique(hits_bvh)
        sorted_original_hits = jnp.unique(hits_original)
        onp.testing.assert_allclose(sorted_hits, sorted_original_hits)


@pytest.mark.parametrize("data_file", list(DATA_INPUT_FLUX_SURFACES.glob("*_input.h5")))
def test_bvh_closest_point(data_file):

    fs_jax = _get_flux_surface_jax(data_file)
    N_test = 300
    surface_mesh = jsb.flux_surfaces.flux_surface_meshing.mesh_surface(fs_jax, 1.0, jsb.flux_surfaces.ToroidalExtent.full(), 65,62)

    points = fs_jax.cartesian_position(0.5, jnp.linspace(0,2 * jnp.pi, N_test, endpoint=False), 0.5)    


    closest_points_jax, dmin, d_idx = jsb.jax_utils.raytracing.get_closest_points_on_mesh(points, surface_mesh)

    import trimesh
    tmesh = trimesh.Trimesh(*surface_mesh)
    closest_point_trimesh, dmin_trimesh, d_idx_trimesh = trimesh.proximity.closest_point_naive(tmesh, onp.array(points))

    onp.testing.assert_allclose(closest_point_trimesh, closest_points_jax, rtol=1e-5, atol=1e-5)
    onp.testing.assert_allclose(dmin_trimesh, dmin, rtol=1e-5, atol=1e-5)
    