import jax_sbgeom as jsb
import pytest
import jax
import jax.numpy as jnp 
import numpy as onp
from jax_sbgeom.jax_utils import raytracing as RT
from jax_sbgeom.jax_utils import mesh_to_pyvista_mesh, bilinear_interp
jax.config.update("jax_enable_x64", True)
from .test_flux_surface_base_data import DATA_INPUT_FLUX_SURFACES


@pytest.mark.parametrize("data_file", list(DATA_INPUT_FLUX_SURFACES.glob("*_input.h5")))
def test_d_matrix(data_file):
    '''
    This test checks the consistency of the distance matrix interpolation in the FluxSurfaceExtendedDistanceMatrix class. 
    It compares the cartesian positions obtained from the extended flux surface with those obtained from the original flux surface plus the interpolated distance matrix for points that are inside, between 1.0 and 2.0, and between 2.0 and 3.0.
    '''
    from jax_sbgeom.flux_surfaces.flux_surfaces_base import _normalize_theta_phi_full_mod
    fs_jax = jsb.flux_surfaces.FluxSurfaceNormalExtendedNoPhi.from_hdf5(data_file)

    thetad = jnp.linspace(0, 2*jnp.pi, 35, endpoint=False)
    phid   = jnp.linspace(0, 2*jnp.pi / fs_jax.nfp, 39, endpoint=False)
    thetad, phid = jnp.meshgrid(thetad, phid, indexing='ij')
    dmatrix = 0.5 * jnp.cos(thetad) * jnp.sin(phid * fs_jax.nfp)**2 + 1.05
    dmatrix2 = 0.4 * jnp.sin(thetad) * jnp.cos(phid * fs_jax.nfp)**2 + 1.98

    theta_s = jnp.linspace(0, 2*jnp.pi, 17, endpoint=True)
    phi_s = jnp.linspace(0, 2*jnp.pi / fs_jax.nfp * 2, 13, endpoint=False)

    theta_ss, phi_ss = jnp.meshgrid(theta_s, phi_s, indexing='ij')

    fs_ext = jsb.flux_surfaces.FluxSurfaceExtendedDistanceMatrix(fs_jax, jnp.stack([dmatrix, dmatrix2], axis=0))    
    # 3 cases:
    
    # 1. all inside
    s = jnp.linspace(0, 1, 10)
    points_inside = fs_jax.cartesian_position(s[:, None, None], theta_ss[None, :, :], phi_ss[None, :, :])    
    points_inside_2 = fs_ext.cartesian_position(s[:, None, None], theta_ss[None, :, :], phi_ss[None, :, :])  
    onp.testing.assert_allclose(points_inside, points_inside_2, atol=1e-12)

    #2. between 1.0 & 2.0

    s_beyond = jnp.linspace(1.0, 2.0, 10)

    points_between = fs_jax.cartesian_position(1.0 + (s_beyond[:, None, None] - 1.0) * bilinear_interp(*_normalize_theta_phi_full_mod(theta_ss[None, : , :], phi_ss[None, :, :], fs_jax.nfp), dmatrix), theta_ss[None, :, :], phi_ss[None, :, :])
    
    points_between_2 = fs_ext.cartesian_position(s_beyond[:, None, None], theta_ss[None, :, :], phi_ss[None, :, :])
    onp.testing.assert_allclose(points_between, points_between_2, atol=1e-12)
    
    #3. between 2.0 & 3.0
    s_beyond_2 = jnp.linspace(2.0, 3.0, 10)
    dinterp = bilinear_interp(*_normalize_theta_phi_full_mod(theta_ss[None, : , :], phi_ss[None, :, :], fs_jax.nfp), dmatrix) +  (s_beyond_2[:, None, None]- 2.0) * bilinear_interp(*_normalize_theta_phi_full_mod(theta_ss[None, : , :], phi_ss[None, :, :], fs_jax.nfp), dmatrix2 - dmatrix)    
    points_between = fs_jax.cartesian_position(1.0 + dinterp, theta_ss[None, :, :], phi_ss[None, :, :])
    points_between_2 = fs_ext.cartesian_position(s_beyond_2[:, None, None], theta_ss[None, :, :], phi_ss[None, :, :])
    onp.testing.assert_allclose (points_between, points_between_2, atol=1e-12)

    