import jax_sbgeom as jsb
import jax.numpy as jnp
import numpy as onp

import jax
jax.config.update("jax_enable_x64", True)

import time
from functools import partial
import pytest
from .test_flux_surface_base_data import DATA_INPUT_FLUX_SURFACES, _data_file_to_data_output, _get_positions_jsb, _get_positions_sbgeom, _get_normals_jsb, _get_principal_curvatures_jsb
from .test_flux_surface_base_extension import _extended_1d_sampling_grid, _extension_s_max, _extended_1d_sampling_grid_stop_1, _extended_sampling_grid, _get_extended_flux_surface


def _extension_s_max():
    return 2.0

# ======================================================================================================================================================================================
#                                                                   Tests for Conversion to Fourier Representation
# ======================================================================================================================================================================================

@pytest.mark.parametrize("data_file", list(DATA_INPUT_FLUX_SURFACES.glob("*_input.h5")))
def test_conversion_to_fourier_representation(data_file):

    fs_jax = _get_extended_flux_surface(data_file, jsb.flux_surfaces.FluxSurfaceNormalExtendedNoPhi)
    
    n_theta = fs_jax.settings.mpol * 2 + 3 # nyquist
    n_phi   = fs_jax.settings.ntor * 2 + 51 # nyquist
    symm    = fs_jax.nfp

    
    theta   = jnp.linspace(0, 2*jnp.pi, n_theta, endpoint=False)
    phi     = jnp.linspace(0, 2*jnp.pi / symm, n_phi, endpoint=False)

    theta_mg, phi_mg = jnp.meshgrid(theta, phi, indexing='ij')    

    d_edge           = 1.12

    fs_data, new_settings = jsb.flux_surfaces.convert_to_vmec.create_fourier_representation(fs_jax, 1.0 + d_edge, theta_mg)    
    
    fs_jax_converted = jsb.flux_surfaces.FluxSurface.from_rmnc_zmns_settings(fs_data.Rmnc[None,:], fs_data.Zmns[None, :], new_settings, False) 

    # It should match the original flux surface in the extended region on the sampled grid
    original_sampled_grid = fs_jax.cartesian_position(1.0 + d_edge, theta_mg, phi_mg)
    resampled_grid        = fs_jax_converted.cartesian_position(1.0, theta_mg, phi_mg)

    onp.testing.assert_allclose(onp.array(original_sampled_grid), onp.array(resampled_grid), rtol=1e-12, atol=1e-12)

    

