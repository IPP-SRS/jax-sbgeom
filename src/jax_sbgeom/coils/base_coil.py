from abc import ABC, abstractmethod
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import numpy as onp
from typing import Literal
from jax_sbgeom.jax_utils import interpolate_array_modulo_broadcasted, reverse_except_begin
from functools import partial
from typing import Type
import equinox as eqx

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Coil(ABC):
    '''
    Abstract base class for coils. All coil classes should inherit from this class and implement the abstract methods.
    '''
    @abstractmethod 
    def position(self, s):     
        '''Position as a function of normalised arc length s (0 <= s < 1)'''
        ...

    @abstractmethod
    def tangent(self, s):       
        '''Tangent vector as a function of normalised arc length s (0 <= s < 1)'''
        ...

    @abstractmethod
    def centre(self):      
        '''Centre of the coil'''
        ...

    @abstractmethod
    def normal(self, s):
        '''Normal vector as a function of normalised arc length s (0 <= s < 1)'''
        ...
        
    @abstractmethod
    def reverse_parametrisation(self):
        ...
        

# Functional versions for vmapping in coilsets
def coil_position(coil: Coil, s):
    return coil.position(s)

def coil_tangent(coil: Coil, s):
    return coil.tangent(s)

def coil_centre(coil: Coil):
    return coil.centre()

def coil_normal(coil: Coil, s):
    return coil.normal(s)

def coil_reverse_parametrisation(coil: Coil):
    return coil.reverse_parametrisation()
# ===================================================================================================================================================================================
#                                                                           Finite Size Utility Methods
# ===================================================================================================================================================================================
    
# The reason this is not just a function is to allow for finite size methods that need to precompute data from the coil
# such as a RMF method
class FiniteSizeMethod(eqx.Module, ABC):
    '''
    Abstract base class for finite size methods. All finite size method classes should inherit from this class and implement the abstract methods.

    '''

    @classmethod    
    def setup_from_coil(cls, coil : Coil, *args):
        '''
        Function to setup the FiniteSizeMethod from the coil object. This returns 
        the data needed to instantiate the FiniteSizeMethod (in a dictionary)

        This is useful to vmap over the coil object.
        If one would only have a from_coil, this instantiates the FiniteSizeMethod inside the vmap, which is not JIT compatible.
        Therefore, the idea is to create the data needed to instantiate the FiniteSizeMethod inside the vmap, and then instantiate only once.

        Parameters
        ----------
        coil : Coil
            Coil object
        args : tuple
            Additional arguments for the setup (specific to the FiniteSizeMethod)

        
        '''
        return ()

    @classmethod 
    def setup_from_coils(cls, coil : Coil, *args):
        '''
        Vmap version of setup_from_coil
        This is required because there needs to be a choice made whether the extra arguments are vmapped over or not.
        Some coils, like RotationMinimizedFrame, require a static number, and this cannot be vmapped using the base vmap.
        This base version results in all arguments being vmapped over.

        Parameters
        ----------
        coil : Coil
            Coil object (batched over)
        args : tuple
            Additional arguments for the setup (specific to the FiniteSizeMethod). Possibly batched over, depending on the FiniteSizeMethod.
        Returns
        -------
        tuple
            Data needed to instantiate the FiniteSizeMethod (batched over)
        '''
        return _vmap_setup_from_coil_base(cls, coil, *args)

    @classmethod
    def from_coil(cls, coil : Coil, *args):
        data = _setup_finitesizemethod_from_coil(cls, coil, *args)
        return cls(*data)
            
    @abstractmethod
    def compute_radial_vector(self, coil : Coil, s : jnp.ndarray):        
        ...
    
    def reverse_parametrisation(self):
        ''''
        Reverse the finite size method. By default, does nothing.
        Some finite size methods may need to do something different (i.e. interpolated radial frames need to reverse their data)

        Returns
        -------
        FiniteSizeMethod
        '''
        return self

    def compute_finite_size_frame(self, coil : Coil, s : jnp.ndarray):
        return _compute_finite_size_frame(coil, self, s)
    
    

@eqx.filter_jit
def _compute_radial_vector(coil : Coil, finitesizemethod : FiniteSizeMethod, s : jnp.ndarray):
    return finitesizemethod.compute_radial_vector(coil, s)

@eqx.filter_jit
def _compute_finite_size_frame(coil : Coil, finitesizemethod : FiniteSizeMethod, s : jnp.ndarray):
    radial_vectors = _compute_radial_vector(coil, finitesizemethod, s)
    return _frame_from_radial_vector(coil.tangent(s), radial_vectors)

def _setup_finitesizemethod_from_coil(finitesizemethod_cls : Type[FiniteSizeMethod], coil : Coil, *args):
    return finitesizemethod_cls.setup_from_coil(coil, *args)

_vmap_setup_from_coil_base = jax.jit(jax.vmap(_setup_finitesizemethod_from_coil, in_axes=(None, 0)), static_argnums=(0,)) 
# This function vmaps over all the arguments as well. 



@jax.jit
def _frame_from_radial_vector(tangents, radial_vectors):
    '''
    Compute finite size frame from tangent and radial vector.

    First finite size direction is radial direction, second is phi direction (perpendicular to both tangent and radial)
    
    Parameters
    ----------
    tangents : jnp.ndarray [..., 3]
        Tangent vector(s) along the coil
    radial_vectors : jnp.ndarray [..., 3]
        Radial vector(s) along the coil 

    Returns
    -------
    jnp.ndarray [..., 2, 3]
        Finite size frame(s) along the coil
    '''
    e_R_n   = radial_vectors / jnp.linalg.norm(radial_vectors, axis=-1, keepdims=True) # ..., 3
    e_phi_n = jnp.cross(tangents, e_R_n) # ..., 3
    e_phi_n = e_phi_n / jnp.linalg.norm(e_phi_n, axis=-1, keepdims=True) # ..., 3
    return jnp.stack([e_R_n, e_phi_n], axis=-2) # ..., 2, 3

#===================================================================================================================================================================================
#                                                                           Finite Size Coil
# ===================================================================================================================================================================================
@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FiniteSizeCoil(Coil):
    '''
    Class representing a coil with finite size defined by a FiniteSizeMethod.

    Internally, is a coil but also has a coil as member. The coil member is used to compute the position, tangent, centre and normal.
    The FiniteSizeMethod member is used to compute the radial vector and finite size frame.

    '''
    coil : Coil
    finite_size_method : FiniteSizeMethod

    @classmethod    
    def from_coil(cls, coil : Coil, finite_size_method : Type[FiniteSizeMethod], *args):
        return cls(coil = coil, finite_size_method = finite_size_method.from_coil(coil, *args))
      
    def position(self, s):       
        return self.coil.position(s)
    
    def tangent(self, s):        
        return self.coil.tangent(s)
    
    def centre(self):        
        return self.coil.centre()    
        
    def normal(self):        
        return self.coil.centre()    
    
    def reverse_parametrisation(self):
        return FiniteSizeCoil(self.coil.reverse_parametrisation(), self.finite_size_method.reverse_parametrisation())

    def radial_vector(self, s):
        '''
        Radial vector along the coil as a function of arc length

        Uses the FiniteSizeMethod to compute the radial vector
        Uses the finite size method to compute the radial vector

        Parameters
        ----------
        s : jnp.ndarray
            Arc length(s) along the coil
        Returns
        -------
        jnp.ndarray [..., 3]
            Radial vector(s) along the coil
        '''
        return _compute_radial_vector(self.coil, self.finite_size_method, s)

    def finite_size_frame(self, s):
        '''
        Finite size frame along the coil as a function of arc length
        Uses the finite size method to compute the finite size frame.

        Parameters
        ----------
        s : jnp.ndarray
            Arc length(s) along the coil
        Returns
        -------
        jnp.ndarray [..., 2, 3]
            Finite size vector(s) along the coil
        '''
        return _compute_finite_size_frame(self.coil, self.finite_size_method, s)
    
    
    def finite_size(self, s, width_radial : float, width_phi : float):
        '''
        Finite size along the coil as a function of arc length
        Uses the finite size method to compute the finite size.

        Parameters
        ----------
        s : jnp.ndarray
            Arc length(s) along the coil
        width_radial : float
            Width in the first finite size direction
        width_phi : float
            Width in the second finite size direction
            
        Returns
        -------
        jnp.ndarray [..., 4, 3]
            Finite size vector(s) along the coil
        '''
        return _compute_finite_size(self.coil, self.finite_size_method, s, width_radial, width_phi)


@eqx.filter_jit
def _compute_finite_size(coil : Coil, finitesizemethod : FiniteSizeMethod, s : jnp.ndarray, width_radial : float, width_phi : float):
    '''
    Compute finite size along the coil as a function of arc length
    Uses the finite size method to compute the finite size.

    Parameters
    ----------
    coil : Coil
        Coil object
    finitesizemethod : FiniteSizeMethod
        Finite size method object
    s : jnp.ndarray
        Arc length(s) along the coil
    width_radial : float
        Width in the first finite size direction
    width_phi : float
        Width in the second finite size direction   
    Returns
    -------
    jnp.ndarray [..., 4, 3]
        Finite size vector(s) along the coil
    '''
    location = coil.position(s)
    frame    = finitesizemethod.compute_finite_size_frame(coil, s)
    return _finite_size_from_data(location, frame, width_radial, width_phi)


@jax.jit
def _finite_size_from_data(location, frame, width_radial : float, width_phi : float):    
    '''
    Compute the finite size vertices from location, frame and widths. 
    The frame is assumed to be orthonormal and its first index corresponds to the radial direction, second to the phi direction.

    The finite size is in the following order:
    v_0 : + radial, + phi
    v_1 : - radial, + phi
    v_2 : - radial, - phi
    v_3 : + radial, - phi
    

    Parameters
    ----------
    location : jnp.ndarray [..., 3]
        Location(s) along the coil
    frame : jnp.ndarray [..., 2, 3]
        Finite size frame(s) along the coil
    width_radial : float
        Width in the first finite size direction
    width_phi : float
        Width in the second finite size direction
    Returns
    -------
    jnp.ndarray [..., 4, 3]
        Finite size vertex locations along the coil
    '''
    v_0  = location + width_radial * frame[..., 0, :] + width_phi * frame[..., 1, :] # ..., 3
    v_1  = location - width_radial * frame[..., 0, :] + width_phi * frame[..., 1, :] # ..., 3
    v_2  = location - width_radial * frame[..., 0, :] - width_phi * frame[..., 1, :] # ..., 3
    v_3  = location + width_radial * frame[..., 0, :] - width_phi * frame[..., 1, :] # ..., 3 
    return jnp.stack([v_0, v_1, v_2, v_3], axis=-2) # ..., 4, 3
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#                                                                           Centroid
# -----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class CentroidFrame(FiniteSizeMethod):
    '''
    Finite size method using centroid frame

    Centroid frame is defined by the radial vector being the vector that is pointing from the coil centre to the coil position, projected onto the plane normal to the tangent        
    '''
    def compute_radial_vector(self, coil : Coil, s : jnp.ndarray):
        return _compute_radial_vector_centroid(coil, s)

@jax.jit
def _compute_radial_vector_centroid(coil : Coil, s : jnp.ndarray):
    '''
    Compute radial vector along the coil as a function of arc length using centroid method
    Uses the coils internal method to compute the radial vector (as it may be coil-type specific)
    '''
    coil_position = coil.position(s)
    coil_tangent  = coil.tangent(s)
    coil_centre   = coil.centre()
    return _radial_vector_centroid_from_data(coil_centre, coil_position, coil_tangent)

@jax.jit
def _radial_vector_centroid_from_data(coil_centre, positions, tangents):
    '''
    Compute the centroid radial vector from coil centre, position and tangent data
    This is not always desired: discrete coils have discontinuous tangents and thus discontinuous radial vectors
    Therefore, such a discrete coild should interpolate the radial vectors to have a smooth coil.
    '''
    d_i       =  positions - coil_centre # ..., 3 - 3 = ..., 3
    e_R       = d_i - jnp.einsum("...j,...i,...i->...j", tangents, d_i, tangents) # ..., 3
    return e_R / jnp.linalg.norm(e_R, axis=-1, keepdims=True) # ..., 3

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#                                                                           Frenet-Serret
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
class FrenetSerretFrame(FiniteSizeMethod):
    '''
    Finite size method using Frenet-Serret frame.
    '''
    def compute_radial_vector(self, coil : Coil, s : jnp.ndarray):
        return _compute_radial_vector_frenet_serret(coil, s)
 
    
@jax.jit
def _compute_radial_vector_frenet_serret(coil : Coil, s : jnp.ndarray):
    '''
    Compute radial vector along the coil as a function of arc length using Frenet-Serret method
    Uses the coils internal method to compute the radial vector (as it may be coil-type specific or not even exist)
    '''
    return coil.normal(s)
    
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#                                                                           Radial Vector Frame
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class RadialVectorFrame(FiniteSizeMethod):
    '''
    Finite size method using precomputed radial vectors
    It interpolates in between the radial vectors to compute the radial vector and finite size frame at arbitrary locations along the coil.
    The radial vectors are assumed to be given at uniform arc length intervals (endpoint not included).
    '''
    radial_vectors_i : jnp.ndarray

    @classmethod
    def setup_from_coil(cls, coil : Coil, *args):
        return (args[0],)
    
    @classmethod
    def setup_from_coils(cls, coil : Coil, *args):
        '''
        Since the radial vectors are precomputed, this just returns the same as setup_from_coil.
        There is no function called that does any computation so there is no need to call a vmapped function.
        '''
        return (args[0],)

    @classmethod 
    def from_radial_vectors(cls, radial_vectors : jnp.ndarray):        
        return cls(radial_vectors_i = radial_vectors)
    
    def reverse_parametrisation(self):
        return type(self)(reverse_except_begin(self.radial_vectors_i))
        
    def compute_radial_vector(self, coil : Coil, s : jnp.ndarray):
        # Coil was already used to compute radial_vectors_i
        # This assumes that this class was instantiated using from_coil with the same coil
        return _interpolate_radial_vectors(self.radial_vectors_i, s)
    
@jax.jit
def _interpolate_radial_vectors(radial_vectors_rmf, s):
    '''
    Interpolate radial vectors at arc length s.
    Uses modulo arithmic to define s > 1.0.
    '''
    return interpolate_array_modulo_broadcasted(radial_vectors_rmf, s)

#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
#                                                                           Rotation Minimized Frame
#------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

class RotationMinimizedFrame(RadialVectorFrame):
    '''
    Finite size method using rotation minimized frame. This is a subclass of RadialVectorFrame.
    The radial vectors are computed using the rotation minimizing frame algorithm.
    '''
    @classmethod 
    def setup_from_coil(cls, coil : Coil, number_of_rmf_samples : int):
        return (_compute_full_rmf(coil, number_of_rmf_samples),)
    
    @classmethod 
    def setup_from_coils(cls, coil : Coil, *args):
        '''
        Vmap version of setup_from_coil
        This is required because there needs to be a choice made whether the extra arguments are vmapped over or not.
        Some coils, like RotationMinimizedFrame, require a static number, and this cannot be vmapped using the base vmap.
        '''
        return (_setup_from_coils_rotation_minimized(coil, *args),)
    

@partial(jax.jit, static_argnums=(1,))
def _compute_full_rmf(coil : Coil, number_of_rmf_samples : int):
    '''
    Compute the full rotation minimizing frame along the coil from the coil object.
    The radial vectors are computed at uniform arc length intervals (endpoint not included).
    See [1] for computation details.

    Parameters
    ----------
    coil : Coil
        Coil object
    number_of_rmf_samples : int
        Number of samples to compute the RMF at (uniform arc length intervals, endpoint not included
    Returns
    -------
    jnp.ndarray (number_of_rmf_samples, 3)
        Rotation minimizing frame along the coil
    
    References
    ----------
    [1] Wang, Wenping, et al. "Computation of rotation minimizing frames." ACM Transactions on Graphics (TOG) 27.1 (2008): 1-18.


    '''
    s_rmf        = jnp.linspace(0.0,1.0, number_of_rmf_samples, endpoint=False)
    positions_rmf = coil.position(s_rmf)
    tangents_rmf  = coil.tangent(s_rmf)
    coil_centre   = coil.centre()
    return _rmf_radial_vector_from_data(coil_centre, positions_rmf, tangents_rmf)

_setup_from_coils_rotation_minimized = jax.jit(jax.vmap(_compute_full_rmf, in_axes=(0,None)), static_argnums=(1))       

@jax.jit
def _rmf_radial_vector_from_data(coil_centre, positions, tangents):
    '''
    Compute the rotation minimizing frame along the coil from position and tangent data
    
    We assume that the positions and tangents are given at uniform arc length intervals (endpoint not included).
    In other words, sampled at jnp.linspace(0.0, 1.0, n_samples, endpoint=False).

    See [1] for computation details.

    References
    ----------
    [1] Wang, Wenping, et al. "Computation of rotation minimizing frames." ACM Transactions on Graphics (TOG) 27.1 (2008): 1-18.

    Parameters
    ----------
    coil_centre : jnp.ndarray (3,)
        Centre of the coil
    positions : jnp.ndarray (n_samples, 3)
        Positions along the coil
    tangents : jnp.ndarray (n_samples, 3)
        Tangents along the coil

    Returns
    -------
    jnp.ndarray (n_samples, 2, 3)
        Rotation minimizing frame along the coil
    '''
    n_samples = positions.shape[0]    
    
    initial_vec = jnp.cross(tangents[0], positions[0] - coil_centre)

    initial_result = initial_vec / jnp.linalg.norm(initial_vec)

    def rmf_step(carry, x):
        result_prev   = carry # need this at the end
        pos_i, pos_i_p1, tan_i, tan_i_p1 = x

        v1 = pos_i_p1 - pos_i
        c1 = jnp.dot(v1, v1)

        rL_i = result_prev - 2.0 / c1 * jnp.dot(result_prev, v1)  * v1
        tL_i = tan_i       - 2.0 / c1 * jnp.dot(tan_i, v1) * v1

        v2 = tan_i_p1 - tL_i
        c2 = jnp.dot(v2, v2)
        result_i = rL_i - 2.0 / c2 * jnp.dot(rL_i, v2) * v2
        return result_i, result_i

    final_result, result_arr = jax.lax.scan(
        rmf_step, initial_result,
        (positions[:-1], positions[1:], tangents[:-1], tangents[1:]),
    )

    total_ri = jnp.concatenate([initial_result[None, :], result_arr], axis=0)  # (n_samples, 3)

    # Add a simple periodic correction to ensure that the start and end vectors are aligned    
    angle = jnp.arccos(jnp.dot(final_result, initial_result))

    tanv0 = _angle_axis_to_matrix(tangents[-1], angle)

    angle_corr       = jax.lax.cond(jnp.arccos(jnp.dot(initial_result, tanv0 @ final_result)) > angle, lambda _ : -angle, lambda _ : angle, operand = None)

    uniform_rotation = jnp.linspace(0.0, angle_corr, n_samples)
    
    angle_axis_matrices = _angle_axis_to_matrix_vmap(tangents, uniform_rotation)

    result_new = jnp.einsum("...ij,...j->...i", angle_axis_matrices, total_ri)

    return result_new


def _angle_axis_to_matrix(axis, angle):
    """
    Convert an angle-axis representation to a 3x3 rotation matrix.

    Parameters
    ----------
    axis : jnp.ndarray (3,)
        The axis of rotation (should be a unit vector).
    angle : float
        The angle of rotation in radians.   
    Returns
    -------
    jnp.ndarray (3, 3)
        The corresponding rotation matrix.
    """
    axis = axis / jnp.linalg.norm(axis)
    ux, uy, uz = axis

    K = jnp.array([
        [0,    -uz,   uy],
        [uz,    0,   -ux],
        [-uy,  ux,    0]
    ])

    I = jnp.eye(3)
    R = I + jnp.sin(angle) * K + (1 - jnp.cos(angle)) * (K @ K)
    return R

_angle_axis_to_matrix_vmap = jax.vmap(_angle_axis_to_matrix, in_axes=(0, 0))

# ===================================================================================================================================================================================
#                                                                           Convenience functions
# ===================================================================================================================================================================================

@jax.jit
def _coil_rotation_positive(coil):
    '''
    Check whether the coil rotation is positive (increasing angle when looking at R-Z cross section)

    Parameters
    ----------
        coil: Coil
            Coil object to check
    Returns
    -------
        bool
            Whether the coil has positive rotation
    '''

    s = jnp.linspace(0,1,30, endpoint=False)
    pos = coil.position(s) 
    centre = jnp.mean(pos, axis=0)    
    pos_R = jnp.sqrt(pos[:,0]**2 + pos[:,1]**2)
    pos_Z = pos[:,2]
    pos_R_c = jnp.mean(pos_R)
    pos_Z_c = jnp.mean(pos_Z)
    r_offset = pos_R - pos_R_c
    z_offset = pos_Z - pos_Z_c        
    angles = jnp.arctan2(z_offset, r_offset)    
    return jnp.sum(jnp.diff(jnp.unwrap(angles))) > 0    

@partial(jax.jit)
def ensure_coil_rotation(coil : Coil, positive_rotation : bool):    
    '''
    Ensure that the coils in the coilset are ordered in the same rotation (positive is increasing angle when looking at R-Z cross section)

    Parameters
    ----------
        coil: Coil
            Coil object to check
        positive_rotation: bool
            Whether the coil should have positive rotation (if False, negative rotation is enforced)
    Returns
        Coil       : Coil with ensured rotation
    '''    
    return jax.lax.cond(_coil_rotation_positive(coil) == positive_rotation , lambda _ : coil, lambda _ : coil.reverse_parametrisation(), operand=None)
    