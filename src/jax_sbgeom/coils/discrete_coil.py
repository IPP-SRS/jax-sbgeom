from abc import ABC, abstractmethod
from dataclasses import dataclass
import jax 
import jax.numpy as jnp
import numpy as onp
from typing import Literal
from .base_coil import Coil, _finite_size_from_data,_radial_vector_centroid_from_data, _frame_from_radial_vector
from jax_sbgeom.jax_utils import interpolate_array_modulo_broadcasted, interpolate_fractions_modulo, robust_grid_index_modulo
from jax_sbgeom.jax_utils.numerical import reverse_except_begin

import warnings
from functools import partial
from .base_coil import _rmf_radial_vector_from_data

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class DiscreteCoil(Coil):
    '''
    Class representing a coil defined by discrete positions and a centre.
    This centre is precomputed from the given positions. It does not feature in the computation of the coil positions; only the centre.
    '''
    positions    : jnp.ndarray  # [..., 3] Cartesian positions of discrete coil points
    _centre_i    : jnp.ndarray  # Centre of the coil: is simply the mean of the positions. This could be a cached property, but this does not play well with JAX.
        
    @classmethod 
    def from_positions(cls, positions: jnp.ndarray):
        '''
        Create a DiscreteCoil from discrete positions

        Positions are assumed to be non-periodic (in other words, the first and last point are not equal).

        Parameters
        ----------
        positions : jnp.ndarray
            Cartesian positions of discrete coil points [..., 3]
        Returns
        -------
        DiscreteCoil
            DiscreteCoil object
        '''
        return cls(positions=positions, _centre_i = jnp.mean(positions, axis=-2))

    def position(self, s):
        '''
        Position along the coil as a function of arc length

        Parameters
        ----------
        s : jnp.ndarray
            Arc length(s) along the coil
        Returns
        -------
        jnp.ndarray
            Cartesian position(s) along the coil
        '''
        return _discrete_coil_position(self, s)

    
    def tangent(self, s):
        '''
        Tangent vector along the coil as a function of arc length

        Parameters
        ----------
        s : jnp.ndarray
            Arc length(s) along the coil
        Returns
        -------
        jnp.ndarray
            Tangent vector(s) along the coil
        '''
        return _discrete_coil_tangent(self, s)
    
    def centre(self):
        '''
        Centre of the coil

        Returns
        -------
        jnp.ndarray
            Centre of the coil
        '''
        return self._centre_i
    
    def normal(self, s):
        '''
        Normal vector along the coil as a function of arc length
        Not defined for DiscreteCoil
        
        Parameters
        ----------
        s : jnp.ndarray
            Arc length(s) along the coil
        Returns
        -------
        jnp.ndarray
            Normal vector(s) along the coil (jnp.nan)
        '''
        
        return jnp.full(jnp.array(s).shape + (3,), jnp.nan)
    
    def reverse_parametrisation(self):
        '''
        Reverse the parametrisation of the discrete coil. Reverses all points except the first (s=0 remains s=0).
        '''
        return _discrete_coil_reverse_parametrisation(self)
        
    
# ===================================================================================================================================================================================
#                                                                           Implementation
# ===================================================================================================================================================================================
@jax.jit
def _discrete_coil_discrete_position(discrete_coil : DiscreteCoil,  index):
    return discrete_coil.positions[index % discrete_coil.positions.shape[0]]

@jax.jit
def _discrete_coil_discrete_tangent(discrete_coil : DiscreteCoil, index):    
    i1      = index + 1
    pos_i0  = discrete_coil.positions[index % discrete_coil.positions.shape[0]]
    pos_i1  = discrete_coil.positions[i1 %    discrete_coil.positions.shape[0]]
    tangent = pos_i1 - pos_i0
    tangent = tangent / jnp.linalg.norm(tangent, axis=-1, keepdims=True)
    return tangent

@jax.jit
def _discrete_coil_position(discrete_coil : DiscreteCoil,  s):
    return interpolate_array_modulo_broadcasted(discrete_coil.positions, s)

@jax.jit
def _discrete_coil_tangent(discrete_coil : DiscreteCoil, s):
    '''
    Tangent is discontinuous at the discrete data points (position is piecewise-linear
    between them); the forward segment is used by convention.
    '''
    i0 = robust_grid_index_modulo(s, discrete_coil.positions.shape[0])
    return _discrete_coil_discrete_tangent(discrete_coil, i0)



@jax.jit 
def _discrete_coil_reverse_parametrisation(discrete_coil : DiscreteCoil):    
    return DiscreteCoil.from_positions(reverse_except_begin(discrete_coil.positions))
   
# ===================================================================================================================================================================================
#                                                                           Finite Sizes
# ===================================================================================================================================================================================

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#                           Centroids
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------


@jax.jit 
def _discrete_coil_radial_vector_centroid_index(discrete_coil : DiscreteCoil,  index):    
    '''
    
    Internal function to find the centroid radial vector at a discrete coil index

    Parameters
    ----------
    discrete_coil : DiscreteCoil
        Discrete coil object
    index : jnp.ndarray
        Discrete coil indexs
    Returns
    -------
    jnp.ndarray [..., 3]
        Radial vector(s) at the discrete coil indexes
    '''
    pos_i      = _discrete_coil_discrete_position(discrete_coil, index)
    tangent_i  = _discrete_coil_discrete_tangent(discrete_coil, index)
    return _radial_vector_centroid_from_data(discrete_coil._centre_i, pos_i, tangent_i)

@jax.jit
def _discrete_coil_radial_vector_centroid(discrete_coil : DiscreteCoil, s):
    ''' 
    Internal function to find the centroid radial vector at arc length s

    Interpolates radial vectors at surrounding data points

    Parameters
    ----------
    discrete_coil : DiscreteCoil
        Discrete coil object
    s : jnp.ndarray
        Arc length(s) along the coil
    Returns
    -------
    jnp.ndarray [..., 3]
        Radial vector(s) along the coil
    '''
    i0, i1, ds = interpolate_fractions_modulo(s,  discrete_coil.positions.shape[0])      
    radial_i0   = _discrete_coil_radial_vector_centroid_index(discrete_coil, i0)
    radial_i1   = _discrete_coil_radial_vector_centroid_index(discrete_coil, i1)
    return radial_i0 * (1.0 - ds)[..., jnp.newaxis] + radial_i1 * ds[..., jnp.newaxis]

@jax.jit
def _discrete_coil_finite_size_frame_centroid(discrete_coil : DiscreteCoil, s):
    '''
    Compute finite size frame at a location s along the discrete coil, using centroid method

    Centroid frames at the discrete points are interpolated.

    Parameters
    ----------
    discrete_coil : DiscreteCoil
        Discrete coil object
    s : jnp.ndarray 
        Arc length(s) along the coil
    
    Returns
    -------
    jnp.ndarray [..., 2, 3]
        Finite size frame(s) along the coil
    '''
    i0, i1, ds      = interpolate_fractions_modulo(s, discrete_coil.positions.shape[0])  
    radial_vector_i0 = _discrete_coil_radial_vector_centroid_index(discrete_coil, i0)
    radial_vector_i1 = _discrete_coil_radial_vector_centroid_index(discrete_coil, i1)

    tangent_i0      = _discrete_coil_discrete_tangent(discrete_coil, i0)
    tangent_i1      = _discrete_coil_discrete_tangent(discrete_coil, i1)    

    frame_0         = _frame_from_radial_vector(tangent_i0, radial_vector_i0)
    frame_1         = _frame_from_radial_vector(tangent_i1, radial_vector_i1)    
    
    return (1 - ds)[..., jnp.newaxis, jnp.newaxis] * frame_0 + ds[..., jnp.newaxis, jnp.newaxis] * frame_1


# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#                           Frenet-Serret
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
@jax.jit
def _discrete_coil_radial_vector_frenet_serret(discrete_coil : DiscreteCoil, s):
    '''
    Internal function to find the frenet-serret radial vector at arc length s

    Not valid for discrete coils due to vanishing curvature between the data points

    Parameters
    ----------
    discrete_coil : DiscreteCoil
        Discrete coil object
    s : jnp.ndarray
        Arc length(s) along the coil
    Returns
    -------
    jnp.ndarray [..., 3]
        Radial vector(s) along the coil (jnp.nan)


    '''
    warnings.warn("Frenet-Serret frame is ill-defined for DiscreteCoil due to zero curvature. Returning NaN. ", RuntimeWarning)            
    return jnp.full(s.shape + (3,), jnp.nan)

# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#                           RMF
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------