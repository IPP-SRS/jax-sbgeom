import os
import jax_sbgeom as jsb
import jax.numpy as jnp
import numpy as onp

import jax
import time

from functools import partial
from typing import Type, List

import pytest


jax.config.update("jax_enable_x64", True)


def test_backend():
    import jax
    

def _check_single_vectorized(fun):
    s_0 = 0.1
    s_1 = jnp.linspace(0.0, 1.0, 24)
    s_2 = s_1.reshape((4,6))
    s_3 = s_1.reshape((2,3,4))


    fun(s_0)
    r_1 = fun(s_1)
    r_2 = fun(s_2)
    r_3 = fun(s_3)

    onp.testing.assert_allclose(r_1, r_2.reshape(r_1.shape))
    onp.testing.assert_allclose(r_1, r_3.reshape(r_1.shape))

from pathlib import Path
data_input = Path(__file__).parent.parent / "data" / "coils"



def _get_all_discrete_coils(request):        
    positions = onp.load(request)
    return [jsb.coils.DiscreteCoil.from_positions(positions[i]) for i in range(positions.shape[0])]

def _get_all_fourier_coils(request):
    coilset_fourier = jsb.coils.convert_to_fourier_coilset(jsb.coils.CoilSet.from_list(_get_all_discrete_coils(request)))
    return [coilset_fourier[i] for i in range(coilset_fourier.n_coils)]


def _data_file_to_data_output(data_file : Path, extra_text : str = "") -> Path:
    filename_stem = data_file.stem
    
    output_filename = filename_stem.replace("_input", extra_text + "_output.npy")
    return data_input / output_filename


#=================================================================================================================================================
#                                                   Position & Tangent tests
#=================================================================================================================================================

def _sampling_s(n_s : int = 27):
    return jnp.linspace(0.0, 1.0, n_s)

def _get_positions(coilset_jsb):
    positions = []
    for i in range(len(coilset_jsb)):
        positions.append(coilset_jsb[i].position(_sampling_s()))
    return onp.array(positions)

def _get_tangents(coilset_jsb):
    tangents = []
    for i in range(len(coilset_jsb)):
        tangents.append(coilset_jsb[i].tangent(_sampling_s()))
    return onp.array(tangents)

@pytest.mark.parametrize("data_file", list(data_input.glob("*_input.npy")))
def test_discrete_coil_position(data_file):    
    coils = _get_all_discrete_coils(data_file)    

    positions_this =  _get_positions(coils)
    expected_positions = onp.load(_data_file_to_data_output(data_file, extra_text="_discrete_coil_position"))
    onp.testing.assert_allclose(positions_this, expected_positions, atol=1e-12, rtol=1e-12)


@pytest.mark.parametrize("data_file", list(data_input.glob("*_input.npy")))
def test_discrete_coil_position(data_file):    
    coils              = _get_all_discrete_coils(data_file)    
    positions_this      =  _get_tangents(coils)
    expected_positions = onp.load(_data_file_to_data_output(data_file, extra_text="_discrete_coil_tangent"))
    onp.testing.assert_allclose(positions_this, expected_positions, atol=1e-12, rtol=1e-12)

@pytest.mark.parametrize("data_file", list(data_input.glob("*_input.npy")))
def test_fourier_coil_position(data_file):    
    coils = _get_all_fourier_coils(data_file)    

    positions_this =  _get_positions(coils)
    expected_positions = onp.load(_data_file_to_data_output(data_file, extra_text="_fourier_coil_position"))
    onp.testing.assert_allclose(positions_this, expected_positions, atol=1e-12, rtol=1e-12)

@pytest.mark.parametrize("data_file", list(data_input.glob("*_input.npy")))
def test_fourier_coil_tangent(data_file):    
    coils              = _get_all_fourier_coils(data_file)    
    positions_this      =  _get_tangents(coils)
    expected_positions = onp.load(_data_file_to_data_output(data_file, extra_text="_fourier_coil_tangent"))
    onp.testing.assert_allclose(positions_this, expected_positions, atol=1e-12, rtol=1e-12)


#=================================================================================================================================================
#                                                   Finite Size Tests
#=================================================================================================================================================
def _sampling_s_finite_size(n_s : int = 27):
    return jnp.linspace(0.0, 1.0, n_s, endpoint=False)

def _get_finite_size_arguments():
    return {
        jsb.coils.CentroidFrame        : (),
        jsb.coils.FrenetSerretFrame    : (),
        jsb.coils.RotationMinimizedFrame: (50,),
        jsb.coils.RadialVectorFrame    : (jnp.array([jnp.sin(jnp.linspace(0, 2 * jnp.pi, 11)), jnp.cos(jnp.linspace(0, 2 * jnp.pi, 11)), jnp.zeros(11)]).T,)                                                   
    }


def _get_all_finite_sizes(coilset_discrete):
    width_phi = 0.25
    width_R   = 0.33

    finitesizes = _get_finite_size_arguments()

  
    fs_coils = []
    for coil_i in range(len(coilset_discrete)):
        coil_i_fs = []
        for method_class in finitesizes.keys():            
            fs_coil = jsb.coils.FiniteSizeCoil.from_coil(coilset_discrete[coil_i], method_class, *finitesizes[method_class])
            coil_i_fs.append(fs_coil.finite_size(s = _sampling_s_finite_size(), width_phi = width_phi, width_radial = width_R)  )
        fs_coils.append(coil_i_fs)


    return onp.array(fs_coils)
    

@pytest.mark.parametrize("data_file", list(data_input.glob("*_input.npy")))
def test_discrete_coil_finite_size(data_file):
    coils_discrete = _get_all_discrete_coils(data_file)    
    finitesizes_this =  _get_all_finite_sizes(coils_discrete)
    expected_finitesizes = onp.load(_data_file_to_data_output(data_file, extra_text="_discrete_coil_finite_size"))
    onp.testing.assert_allclose(finitesizes_this, expected_finitesizes, atol=1e-12, rtol=1e-12)

@pytest.mark.parametrize("data_file", list(data_input.glob("*_input.npy")))
def test_fourier_coil_finite_size(data_file):
    coils_fourier = _get_all_fourier_coils(data_file)    
    finitesizes_this =  _get_all_finite_sizes(coils_fourier)
    expected_finitesizes = onp.load(_data_file_to_data_output(data_file, extra_text="_fourier_coil_finite_size"))
    onp.testing.assert_allclose(finitesizes_this, expected_finitesizes, atol=1e-12, rtol=1e-12)
#=================================================================================================================================================
#                                                  Meshing
#=================================================================================================================================================

def _get_coil_mesh(coilset_jsb):
    from jax_sbgeom.coils import mesh_coil_surface
    width_phi = 0.25
    width_R   = 0.33

    finitesizes = _get_finite_size_arguments()

  
    coil_pos  = []
    coil_conn = []

    for coil_i in range(len(coilset_jsb)):
        coil_i_pos = []
        coil_i_conn = []
        for method_class in finitesizes.keys():            
            fs_coil = jsb.coils.FiniteSizeCoil.from_coil(coilset_jsb[coil_i], method_class, *finitesizes[method_class])

            mesh_i = mesh_coil_surface(fs_coil, 23, width_R, width_phi)
            coil_i_pos.append(mesh_i[0])
            coil_i_conn.append(mesh_i[1])
        coil_pos.append(coil_i_pos)
        coil_conn.append(coil_i_conn)
    
    return jnp.concatenate([onp.array(coil_pos).reshape(-1), onp.array(coil_conn).reshape(-1)], axis=0)


def test_generate_vertices_from_finite_sized_lines_bilinear_spacing():
    # AI written test
    from jax_sbgeom.coils.coil_meshing import _generate_vertices_from_finite_sized_lines

    finite_size_lines = jnp.array([[[1.0, 1.0, 0.0],
                                   [-1.0, 1.0, 0.0],
                                   [-1.0, -1.0, 0.0],
                                   [1.0, -1.0, 0.0]]])

    vertices = _generate_vertices_from_finite_sized_lines(finite_size_lines, 7)  # [1,7,7,3]
    mid_row = onp.asarray(vertices[0, 3, :, :])

    # mid-row should produce 7 distinct points along radial direction (not identical)
    assert onp.unique(mid_row, axis=0).shape[0] == 7
    assert not onp.allclose(mid_row, mid_row[0:1])

    # corners should match the given patch orientation
    assert onp.allclose(onp.asarray(vertices[0, 0, 0]), [-1.0, -1.0, 0.0])
    assert onp.allclose(onp.asarray(vertices[0, 0, -1]), [1.0, -1.0, 0.0])
    assert onp.allclose(onp.asarray(vertices[0, -1, 0]), [-1.0, 1.0, 0.0])
    assert onp.allclose(onp.asarray(vertices[0, -1, -1]), [1.0, 1.0, 0.0])


@pytest.mark.parametrize("data_file", list(data_input.glob("*_input.npy")))
def test_discrete_coil_mesh(data_file):
    coil_posconn = _get_coil_mesh(_get_all_discrete_coils(data_file))

    expected_posconn = onp.load(_data_file_to_data_output(data_file, extra_text="_discrete_coil_mesh"))
    onp.testing.assert_allclose(coil_posconn, expected_posconn, atol=1e-12, rtol=1e-12)

@pytest.mark.parametrize("data_file", list(data_input.glob("*_input.npy")))
def test_fourier_coil_mesh(data_file):
    coil_posconn = _get_coil_mesh(_get_all_fourier_coils(data_file))

    expected_posconn = onp.load(_data_file_to_data_output(data_file, extra_text="_fourier_coil_mesh"))
    onp.testing.assert_allclose(coil_posconn, expected_posconn, atol=1e-12, rtol=1e-12)

# #=================================================================================================================================================
# #                                                  Vectorization Tests
# #=================================================================================================================================================

@pytest.mark.slow
@pytest.mark.parametrize("data_file", list(data_input.glob("*_input.npy")))
def test_discrete_coil_vectorized_position(data_file):
    coilset_jaxsbgeom = _get_all_discrete_coils(data_file)
    
    _check_single_vectorized(coilset_jaxsbgeom[0].position)
    _check_single_vectorized(coilset_jaxsbgeom[0].tangent)

    def centroid(s):
        finitesize_coil = jsb.coils.base_coil.FiniteSizeCoil(coilset_jaxsbgeom[0], jsb.coils.base_coil.CentroidFrame())
        return finitesize_coil.finite_size(s, 0.3, 0.5)
    _check_single_vectorized(centroid)
    def frenet_serret(s):
        finitesize_coil = jsb.coils.base_coil.FiniteSizeCoil(coilset_jaxsbgeom[0], jsb.coils.base_coil.FrenetSerretFrame())
        return finitesize_coil.finite_size(s, 0.3, 0.5)
    _check_single_vectorized(frenet_serret)
    def rmf(s):
        finitesize_coil = jsb.coils.base_coil.FiniteSizeCoil(coilset_jaxsbgeom[0], jsb.coils.base_coil.RotationMinimizedFrame.from_coil(coilset_jaxsbgeom[0], 111))
        return finitesize_coil.finite_size(s, 0.3, 0.5)
    _check_single_vectorized(rmf)
    

@pytest.mark.slow
@pytest.mark.parametrize("data_file", list(data_input.glob("*_input.npy")))
def test_fourier_coil_vectorized_position(data_file):
    coilset_jax = _get_all_fourier_coils(data_file)
    
    _check_single_vectorized(coilset_jax[0].position)
    _check_single_vectorized(coilset_jax[0].tangent)

    def centroid(s):
        finitesize_coil = jsb.coils.base_coil.FiniteSizeCoil(coilset_jax[0], jsb.coils.base_coil.CentroidFrame())
        return finitesize_coil.finite_size(s, 0.3, 0.5)
    _check_single_vectorized(centroid)
    def frenet_serret(s):
        finitesize_coil = jsb.coils.base_coil.FiniteSizeCoil(coilset_jax[0], jsb.coils.base_coil.FrenetSerretFrame())
        return finitesize_coil.finite_size(s, 0.3, 0.5)
    _check_single_vectorized(frenet_serret)
    def rmf(s):
        finitesize_coil = jsb.coils.base_coil.FiniteSizeCoil(coilset_jax[0], jsb.coils.base_coil.RotationMinimizedFrame.from_coil(coilset_jax[0],111))
        return finitesize_coil.finite_size(s, 0.3, 0.5)
    _check_single_vectorized(rmf)

# #=================================================================================================================================================
# #                                                  Converting Tests
# #=================================================================================================================================================

def _get_fourier_transformation(coilset, n_ftrunc ):
    fourier_cos = []
    fourier_sin = []
    centres    = []    

    for i in range(len(coilset)):
        coil = jsb.coils.convert_to_fourier_coil(coilset[i], n_modes=n_ftrunc)
        
        fourier_cos.append(coil.fourier_cos)
        fourier_sin.append(coil.fourier_sin)
        centres.append(coil.centre_i)
    return onp.concatenate([onp.array(fourier_cos).ravel(), onp.array(fourier_sin).ravel(), onp.array(centres).ravel()], axis=0)

@pytest.mark.parametrize("data_file", list(data_input.glob("*_input.npy")))
@pytest.mark.parametrize("n_ftrunc", [None, 11])
def test_converting_fourier_coils(data_file, n_ftrunc):
    coilset_jaxsbgeom = _get_all_discrete_coils(data_file)

    fourier_transform = _get_fourier_transformation(coilset_jaxsbgeom, n_ftrunc)
    filename_output = _data_file_to_data_output(data_file, extra_text=f"_fourier_coil_transformation_nft{n_ftrunc}") if n_ftrunc is not None else _data_file_to_data_output(data_file, extra_text=f"_fourier_coil_transformation")
    onp.testing.assert_allclose(
        fourier_transform, 
        onp.load(filename_output), 
        atol=1e-12, rtol=1e-12
    )

@pytest.mark.parametrize("data_file", list(data_input.glob("*_input.npy")))
def test_equal_arclength(data_file):
    coilset_jax = _get_all_fourier_coils(data_file)

    n_points_sample  = 2531
    n_points_desired = 1513
    n_points_ea      = 511

    arclength_f = jax.vmap(jsb.coils.fourier_coil._arc_length_fourier, in_axes=(0,None))

    fourier_coilset = jsb.coils.CoilSet.from_list(coilset_jax)

    coilset_new = jsb.coils.fourier_coil.convert_fourier_coilset_to_equal_arclength(fourier_coilset, n_points_sample, n_points_desired, method='pchip')
    coilset_new_lin = jsb.coils.fourier_coil.convert_fourier_coilset_to_equal_arclength(fourier_coilset, n_points_sample, n_points_desired, method='linear')

    def check_arclength_uniformity(coilset, n_points_ea, threshold):
        arclength_f_ea = arclength_f(coilset, jnp.linspace(0.0, 1.0, n_points_ea))
        arclength_avg =  jnp.mean(arclength_f_ea, axis=1) # average over the sampling points
        uniformity_max = jnp.max(jnp.abs((arclength_f_ea - arclength_avg[:, None]) / arclength_avg[:, None]))        
        assert uniformity_max < threshold
        
    check_arclength_uniformity(coilset_new.coils, n_points_ea, threshold=1e-3)
    check_arclength_uniformity(coilset_new_lin.coils, n_points_ea, threshold=5e-3)


# #=================================================================================================================================================
# #                                                  Reversal
# #=================================================================================================================================================    
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

def check_reverse_finite_size(finitesize_coil):
    if isinstance(finitesize_coil.coil, jsb.coils.DiscreteCoil):
        # discretecoils do not have the same tangent on *exactly* the discrete points of the coil..
        #  the forward derivative is different on those points.
        # so we use the parametrisation without being and endpoint
        s         = jnp.linspace(0, 1.0,211, endpoint=False)[1:]
        s_reverse = jnp.linspace(0,-1.0,211, endpoint=False)[1:]
    else:
        s = jnp.linspace(0,1,100)
        s_reverse = jnp.linspace(0,-1,100)
    

    reversed_finitesize_coil = finitesize_coil.reverse_parametrisation()
    position_original = finitesize_coil.position(s)
    position_reversed = reversed_finitesize_coil.position(s_reverse)
    
    radial_vector_original = finitesize_coil.radial_vector(s)
    radial_vector_reversed = reversed_finitesize_coil.radial_vector(s_reverse)

    tangent_original = finitesize_coil.tangent(s)
    tangent_reversed = reversed_finitesize_coil.tangent(s_reverse)
    
    
    onp.testing.assert_allclose(position_original,      position_reversed)        
    onp.testing.assert_allclose(tangent_original,      -tangent_reversed)    
    onp.testing.assert_allclose(radial_vector_original, radial_vector_reversed)

    finitesize_frame_original = finitesize_coil.finite_size_frame(s)
    finitesize_frame_reversed = reversed_finitesize_coil.finite_size_frame(s_reverse)
    # the tangent vector will be reversed, but the radial vector the same, so
    # we need to reverse the tangent vector in the reversed frame to compare
    finitesize_frame_reversed_corrected = finitesize_frame_reversed.at[...,1,:].set(-finitesize_frame_reversed[...,1,:])

    onp.testing.assert_allclose(finitesize_frame_original, finitesize_frame_reversed_corrected)

    width_phi = 0.245
    width_radial = 0.123
    finitesize_original = finitesize_coil.finite_size(s, width_phi, width_radial)
    finitesize_reversed = reversed_finitesize_coil.finite_size(s_reverse, width_phi, width_radial)

    # finite size has    
    #  The finite size is in the following order:
    # v_0 : + radial, + phi
    # v_1 : - radial, + phi
    # v_2 : - radial, - phi
    # v_3 : + radial, - phi
    # since phi -> -phi in the reversed coil
    # we need to swap v_0 with v_3 and v_1 with v_2 to compare
    # i can just flip the arrays around the first axis.    
    finitesize_reversed_corrected = finitesize_reversed[:, ::-1, :]
    onp.testing.assert_allclose(finitesize_original, finitesize_reversed_corrected)

def _get_finitesize_coilset(coils_jax, frame_class : Type[jsb.coils.base_coil.FiniteSizeMethod]):
    #additional_args = additional_arguments_per_coil(frame_class, coils_jax, len(coils_jax))
    finitesize_coilset = [jsb.coils.base_coil.FiniteSizeCoil(coil_jax, frame_class.from_coil(coil_jax, *additional_arguments_per_coil(frame_class, i, len(coils_jax)))) for i, coil_jax in enumerate(coils_jax)]
    return finitesize_coilset

@pytest.mark.parametrize("data_file", list(data_input.glob("*_input.npy")))
@pytest.mark.parametrize("frame_class", classes_discrete)
def test_finitesize_coilset_reverse_discrete(data_file, frame_class):
    coils_jaxsbgeom  = _get_all_discrete_coils(data_file)
    coils_finitesize = _get_finitesize_coilset(coils_jaxsbgeom, frame_class)
    [check_reverse_finite_size(coils_finitesize[i]) for i in range(len(coils_jaxsbgeom))]


@pytest.mark.parametrize("data_file", list(data_input.glob("*_input.npy")))
@pytest.mark.parametrize("frame_class", classes)
def test_finitesize_coilset_reverse_fourier(data_file, frame_class):
    coils_jaxsbgeom = _get_all_fourier_coils(data_file)
    coils_finitesize = _get_finitesize_coilset(coils_jaxsbgeom, frame_class)
    [check_reverse_finite_size(coils_finitesize[i]) for i in range(len(coils_jaxsbgeom))]