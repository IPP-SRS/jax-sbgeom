import jax_sbgeom as jsb
import jax.numpy as jnp
import numpy as onp

import jax
jax.config.update("jax_enable_x64", True)

import time
from functools import partial
import pytest
from .test_flux_surface_base_data import _check_vectorized

from .test_flux_surface_base_data import DATA_INPUT_FLUX_SURFACES, _data_file_to_data_output, _get_positions_jsb, _get_positions_sbgeom, _get_normals_jsb, _get_principal_curvatures_jsb


def _get_extended_flux_surface(data_file, extension_type):
    return extension_type.from_hdf5(data_file)

def _extension_s_max():
    return 2.0

def _extended_sampling_grid(fs_jax : jsb.flux_surfaces.FluxSurface, n_s : int = 6, n_theta : int = 7, n_phi : int = 5, include_axis = True):

    ss = jax.lax.cond(include_axis, lambda x : jnp.linspace(0,_extension_s_max(),n_s), lambda x : jnp.linspace(0, _extension_s_max(), n_s + 1)[1:], None)
    
    tt = jnp.linspace(0, 2 * jnp.pi, n_theta, endpoint=False)
    pp = jnp.linspace(0, 2 * jnp.pi / fs_jax.nfp, n_phi, endpoint=True)
    return jnp.meshgrid(ss, tt, pp, indexing='ij')

def _extended_1d_sampling_grid(fs_jax : jsb.flux_surfaces.FluxSurface, n_s : int = 6, n_theta : int = 7, n_phi : int = 5, include_axis = True, reverse_theta : bool = False):
    ss, tt, pp = _extended_sampling_grid(fs_jax, n_s, n_theta, n_phi, include_axis)
    if reverse_theta:
        tt = - onp.array(tt)
    s = onp.minimum(onp.array(ss).ravel(), 1.0)
    d = onp.maximum(onp.array(ss).ravel() - 1.0, 0.0)
    t = onp.array(tt).ravel()
    p = onp.array(pp).ravel()
    return s, d, t, p

def _extended_1d_sampling_grid_stop_1(fs_jax : jsb.flux_surfaces.FluxSurface, n_s : int = 6, n_theta : int = 7, n_phi : int = 5, include_axis = True, reverse_theta : bool = False):
    ss, tt, pp = _extended_sampling_grid(fs_jax, n_s, n_theta, n_phi, include_axis)
    if reverse_theta:
        tt = - onp.array(tt)
    s = onp.minimum(onp.array(ss).ravel(), 1.0)
    d = 0.0 * onp.array(ss).ravel()
    t = onp.array(tt).ravel()
    p = onp.array(pp).ravel()
    return s, d, t, p

def extension_types():
    return [
        jsb.flux_surfaces.FluxSurfaceNormalExtended,
        jsb.flux_surfaces.FluxSurfaceNormalExtendedNoPhi,
        jsb.flux_surfaces.FluxSurfaceNormalExtendedConstantPhi
    ]

def extension_type_to_name(extension_type):
    if extension_type == jsb.flux_surfaces.FluxSurfaceNormalExtended:
        return "normal"
    elif extension_type == jsb.flux_surfaces.FluxSurfaceNormalExtendedNoPhi:
        return "normal_no_phi"
    elif extension_type == jsb.flux_surfaces.FluxSurfaceNormalExtendedConstantPhi:
        return "normal_constant_phi"
    else:
        raise ValueError("Unknown extension type")

@pytest.mark.parametrize("extension_type", extension_types())
@pytest.mark.parametrize("data_file", list(DATA_INPUT_FLUX_SURFACES.glob("*_input.h5")))
def test_extension_position(data_file, extension_type):
    fs_jax = _get_extended_flux_surface(data_file, extension_type)
    pos_jsb = _get_positions_jsb(fs_jax, _extended_sampling_grid)    
    onp.testing.assert_allclose(pos_jsb, onp.load(_data_file_to_data_output(data_file, f"_{extension_type_to_name(extension_type)}_position")), atol=2e-7)


@pytest.mark.slow
@pytest.mark.parametrize("extension_type", extension_types())
@pytest.mark.parametrize("data_file", list(DATA_INPUT_FLUX_SURFACES.glob("*_input.h5")))
def test_extension_vectorization(data_file, extension_type):
    fs_jax = _get_extended_flux_surface(data_file, extension_type=extension_type)
    _check_vectorized(fs_jax.cartesian_position)
    _check_vectorized(fs_jax.normal)
    _check_vectorized(fs_jax.principal_curvatures)

@pytest.mark.parametrize("extension_type", extension_types())
@pytest.mark.parametrize("data_file", list(DATA_INPUT_FLUX_SURFACES.glob("*_input.h5")))
def test_extension_normals(data_file, extension_type):
    fs_jax = _get_extended_flux_surface(data_file, extension_type)
    pos_jsb = _get_normals_jsb(fs_jax, _extended_sampling_grid)    
    onp.testing.assert_allclose(pos_jsb, onp.load(_data_file_to_data_output(data_file, f"_{extension_type_to_name(extension_type)}_normals")), atol=2e-7)


@pytest.mark.parametrize("extension_type", extension_types())
@pytest.mark.parametrize("data_file", list(DATA_INPUT_FLUX_SURFACES.glob("*_input.h5")))
def test_extension_curvature(data_file, extension_type):
    fs_jax = _get_extended_flux_surface(data_file, extension_type)
    pos_jsb = _get_principal_curvatures_jsb(fs_jax, _extended_sampling_grid)    
    onp.testing.assert_allclose(pos_jsb, onp.load(_data_file_to_data_output(data_file, f"_{extension_type_to_name(extension_type)}_curvatures")), atol=2e-7)