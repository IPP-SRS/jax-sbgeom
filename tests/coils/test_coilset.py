import os
import jax_sbgeom as jsb
import jax.numpy as jnp
import numpy as onp
import jax
import time

from functools import partial

import pytest
from functools import lru_cache
from typing import Type, List

from .test_coils import _get_all_discrete_coils
from .test_coils import _get_all_fourier_coils
from .test_coils import data_input

jax.config.update("jax_enable_x64", True)


#=================================================================================================================================================
#                                                  CoilSet Tests
#=================================================================================================================================================

#-------------------------------------------------------------------------------------------------------------------------------------------------
#                                                  CoilSet Tests
#-------------------------------------------------------------------------------------------------------------------------------------------------
def check_vector_coilset(coils_jax):
    s =  jnp.linspace(0, 1, 1000)

    pos_base = []
    tan_base = []
    centre_base = []
    normal_base = []

    for i in coils_jax:
        pos_base.append(i.position(s))
        tan_base.append(i.tangent(s))
        centre_base.append(i.centre())
        normal_base.append(i.normal(s))

    pos_base = jnp.array(pos_base)
    tan_base = jnp.array(tan_base)

    coilsetv = jsb.coils.CoilSet.from_list(coils_jax)


    pos_vec = coilsetv.position(s)
    tan_vec = coilsetv.tangent(s)
    centre_vec = coilsetv.centre()
    normal_vec = coilsetv.normal(s)

    onp.testing.assert_allclose(pos_base, pos_vec)
    onp.testing.assert_allclose(tan_base, tan_vec)
    onp.testing.assert_allclose(centre_base, centre_vec)
    onp.testing.assert_allclose(normal_base, normal_vec)

@pytest.mark.parametrize("data_file", list(data_input.glob("*_input.npy")))
def test_vector_coilset_discrete(data_file):
    coils_jaxsbgeom = _get_all_discrete_coils(data_file)
    check_vector_coilset(coils_jaxsbgeom)   

@pytest.mark.parametrize("data_file", list(data_input.glob("*_input.npy")))
def test_vector_coilset_fourier(data_file):
    coils_jax  = _get_all_fourier_coils(data_file)
    check_vector_coilset(coils_jax)


#-------------------------------------------------------------------------------------------------------------------------------------------------
#                                                 FiniteSize Tests
#-------------------------------------------------------------------------------------------------------------------------------------------------
classes          = [jsb.coils.CentroidFrame, jsb.coils.RotationMinimizedFrame, jsb.coils.RadialVectorFrame, jsb.coils.FrenetSerretFrame]
classes_discrete = classes[:-1]  # Frenet-Serret frame not implemented for discrete coils

def _radial_vector(coil_i, n_coils):
    x = jnp.cos(jnp.linspace(0, 2*jnp.pi, 100) + coil_i  / n_coils * 2 * jnp.pi)
    y = jnp.sin(jnp.linspace(0, 2*jnp.pi, 100) + coil_i  / n_coils * 2 * jnp.pi)
    z = jnp.zeros_like(x)
    return jnp.stack([x, y, z], axis=-1)

_radial_vectors = jax.vmap(_radial_vector, in_axes=(0, None))

def additional_arguments_per_coil(frame_class : Type[jsb.coils.base_coil.FiniteSizeMethod], coil_i : int, ncoils : int):
    if frame_class == jsb.coils.RotationMinimizedFrame:
        return (10,)
    elif frame_class == jsb.coils.RadialVectorFrame:
        return (_radial_vector(coil_i, ncoils), )
    else:
        return ()
def additional_arguments(frame_class : Type[jsb.coils.base_coil.FiniteSizeMethod], coils_list : List):
    if frame_class == jsb.coils.RotationMinimizedFrame:
        return (10,)
    elif frame_class == jsb.coils.RadialVectorFrame:
        ncoils = len(coils_list)
        radial_vectors = _radial_vectors(jnp.arange(ncoils), ncoils)
        return (radial_vectors, )
    else:
        return ()        


def check_finitesize_coilset(coils_jax, frame_class : Type[jsb.coils.base_coil.FiniteSizeMethod]):

    finite_size_coils_list = [jsb.coils.FiniteSizeCoil.from_coil(coil, frame_class, *additional_arguments_per_coil(frame_class, i, len(coils_jax))) for i, coil in enumerate(coils_jax)]

    coils_jax_finite_size_set = jsb.coils.FiniteSizeCoilSet.from_coils(coils_jax, frame_class, *additional_arguments(frame_class, coils_jax))
    
    s = jnp.linspace(0, 1, 313)
    width_radial = 0.432
    width_phi    = 0.21
    
    radial_vector_vec    = coils_jax_finite_size_set.radial_vector(s) # shape (ncoils, ..., 3)
    finite_size_frame_vec = coils_jax_finite_size_set.finite_size_frame(s) # shape (ncoils, ..., 3, 3)
    finite_size_vec       = coils_jax_finite_size_set.finite_size(s, width_radial, width_phi) #

    for i, fs_coil in enumerate(finite_size_coils_list):
        radial_vector_base     = fs_coil.radial_vector(s)
        finite_size_frame_base = fs_coil.finite_size_frame(s)
        finite_size_base       = fs_coil.finite_size(s, width_radial, width_phi)

        onp.testing.assert_allclose(radial_vector_base, radial_vector_vec[i])
        onp.testing.assert_allclose(finite_size_frame_base, finite_size_frame_vec[i])
        onp.testing.assert_allclose(finite_size_base, finite_size_vec[i])


@pytest.mark.parametrize("data_file", list(data_input.glob("*_input.npy")))
@pytest.mark.parametrize("frame_class", classes_discrete)
def test_finitesize_coilset_discrete(data_file, frame_class):
    coils_jaxsbgeom = _get_all_discrete_coils(data_file)
    check_finitesize_coilset(coils_jaxsbgeom, frame_class)

@pytest.mark.parametrize("data_file", list(data_input.glob("*_input.npy")))
@pytest.mark.parametrize("frame_class", classes)
def test_finitesize_coilset_fourier(data_file, frame_class):
    coils_jax = _get_all_fourier_coils(data_file)
    check_finitesize_coilset(coils_jax, frame_class)

#-------------------------------------------------------------------------------------------------------------------------------------------------
#                                                 Coil Winding Surface Optimization Test
#-------------------------------------------------------------------------------------------------------------------------------------------------

def check_optimization(coilset_jax):
    import optax
    coilset_jaxsbgeom = jsb.coils.CoilSet.from_list(coilset_jax)
    (xarr, optimizer_state), coilset_ordered = jsb.coils.coil_winding_surface.optimize_coil_surface(coilset_jaxsbgeom, optimization_settings=jsb.jax_utils.optimize.OptimizationSettings(max_iterations=500, tolerance=1e-4))
    grad     = optax.tree.get(optimizer_state, 'grad')
    err      = optax.tree.norm(grad)
    assert err < 1e-4
    


@pytest.mark.slow
@pytest.mark.parametrize("data_file", list(data_input.glob("*_input.npy")))
def test_cws_optimization_discrete(data_file):
   
    coils_jaxsbgeom= _get_all_discrete_coils(data_file)
    check_optimization(coils_jaxsbgeom)

@pytest.mark.slow
@pytest.mark.parametrize("data_file", list(data_input.glob("*_input.npy")))
def test_cws_optimization_fourier(data_file):
    coils_jax = _get_all_fourier_coils(data_file)
    check_optimization(coils_jax)   
   

    