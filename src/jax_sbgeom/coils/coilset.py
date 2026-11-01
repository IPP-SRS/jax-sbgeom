from .base_coil import Coil
from jax_sbgeom.jax_utils.utils import tree_take
from typing import List
import jax
import jax.numpy as jnp
from dataclasses import dataclass
from typing import Type
from .base_coil import coil_position, coil_tangent, coil_normal, coil_centre, ensure_coil_rotation

from .base_coil import FiniteSizeMethod, FiniteSizeCoil,  _compute_radial_vector, _compute_finite_size_frame, _compute_finite_size, _setup_finitesizemethod_from_coil

def _stack(*xs):
    return jnp.stack(xs)

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class CoilSet:
    '''
    Class representing a set of coils. Includes methods for batch evaluation of coil properties.
    Including with the same coordinate or different coordinates for each coil.

    Internally, the coils are stored as a batched Coil object. Therefore, no mixed representations are supported.
    
    Example
    -------
    >>> coil1 = DiscreteCoil.from_positions(jnp.stack([ jnp.array([1.0, 0.0, 0.0]), jnp.array( [0.0, 1.0, 0.0]), jnp.array([-1.0, 0.0, 0.0]), jnp.array([0.0, -1.0, 0.0]) ]))
    >>> coil2 = DiscreteCoil.from_positions(jnp.stack([ jnp.array([2.0, 0.0, 0.0]), jnp.array( [0.0, 2.0, 0.0]), jnp.array([-2.0, 0.0, 0.0]), jnp.array([0.0, -2.0, 0.0]) ]))    
    >>> coilset = CoilSet.from_list([coil1, coil2])
    >>> s = jnp.linspace(0, 1, 100)
    >>> positions = coilset.position(s)  # shape (2, 100, 3)
    >>> tangents = coilset.tangent(s)    # shape (2, 100, 3)
    >>> normals = coilset.normal(s)      # shape (2, 100, 3)
    >>> positions_diff = coilset.position_different_s(s[None, :].repeat(coilset.n_coils, axis=0))  # shape (2, 100, 3)
    >>> coil1_copy = coilset[0]  # Get first coil in the coilset
        
    '''
    coils : Coil
    '''Coils in the coilset, stored as a batched Coil object. The batch dimension is the first dimension of all arrays in the Coil dataclass.'''
    
    @classmethod
    def from_list(cls, coils : List[Coil]):
        coils_v = jax.tree.map(_stack, *coils)
        return cls(coils = coils_v)            
    def centre(self):
        return _coil_centre_vmap(self.coils)
    
    def position(self, s):
        return _coil_position_vmap_same_s(self.coils, s)
    def tangent(self, s):
        return _coil_tangent_vmap_same_s(self.coils, s)
    def normal(self, s):
        return _coil_normal_vmap_same_s(self.coils, s)
    
    def position_different_s(self, s):
        return _coil_position_vmap_different_s(self.coils, s)
    def tangent_different_s(self, s):
        return _coil_tangent_vmap_different_s(self.coils, s)
    def normal_different_s(self, s):
        return _coil_normal_vmap_different_s(self.coils, s)
    
    def __getitem__(self, idx):
        return tree_take(self.coils, idx)
    
    @property
    def n_coils(self):
        '''
        Number of coils in the coilset. Uses the batched data shape and therefore is static information
        (can be used in jax.jit compiled functions as static shape).
        '''
        return jax.tree_util.tree_flatten(self.coils)[0][0].shape[0]

_coil_centre_vmap               = jax.jit(jax.vmap(coil_centre, in_axes=(0,)))
_coil_position_vmap_same_s      = jax.jit(jax.vmap(coil_position, in_axes=(0, None)))
_coil_tangent_vmap_same_s       = jax.jit(jax.vmap(coil_tangent, in_axes=(0, None)))
_coil_normal_vmap_same_s        = jax.jit(jax.vmap(coil_normal, in_axes=(0, None)))

_coil_position_vmap_different_s = jax.jit(jax.vmap(coil_position, in_axes=(0, 0)))
_coil_tangent_vmap_different_s  = jax.jit(jax.vmap(coil_tangent, in_axes=(0, 0)))
_coil_normal_vmap_different_s   = jax.jit(jax.vmap(coil_normal, in_axes=(0, 0)))

_ensure_coilset_rotation_vmap   = jax.jit(jax.vmap(ensure_coil_rotation, in_axes=(0, None)))

_radial_vector_same_s           = jax.jit(jax.vmap(_compute_radial_vector, in_axes=(0, 0, None)))
_finite_size_frame_same_s       = jax.jit(jax.vmap(_compute_finite_size_frame, in_axes=(0, 0, None)))
_finite_size_same_s             = jax.jit(jax.vmap(_compute_finite_size, in_axes=(0, 0, None, None, None)))

_radial_vector_different_s      = jax.jit(jax.vmap(_compute_radial_vector, in_axes=(0, 0, 0)))
_finite_size_frame_different_s  = jax.jit(jax.vmap(_compute_finite_size_frame, in_axes=(0, 0, 0)))
_finite_size_different_s        = jax.jit(jax.vmap(_compute_finite_size, in_axes=(0, 0, 0, None, None)))

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FiniteSizeCoilSet(CoilSet):
    '''
    Class representing a set of finite size coils. Includes methods for batch evaluation of coil properties.
    Including with the same coordinate or different coordinates for each coil.
    Internally, the coils are stored as a batched FiniteSizeCoil object. Therefore, no mixed representations are supported.
    Is a subclass of CoilSet, so all methods from CoilSet are also available.

    Example
    -------
    >>> coil1 = DiscreteCoil.from_positions(jnp.stack([ jnp.array([1.0, 0.0, 0.0]), jnp.array( [0.0, 1.0, 0.0]), jnp.array([-1.0, 0.0, 0.0]), jnp.array([0.0, -1.0, 0.0]) ]))
    >>> coil2 = DiscreteCoil.from_positions(jnp.stack([ jnp.array([2.0, 0.0, 0.0]), jnp.array( [0.0, 2.0, 0.0]), jnp.array([-2.0, 0.0, 0.0]), jnp.array([0.0, -2.0, 0.0]) ]))    
    >>> coilset = FiniteSizeCoilSet.from_coils([coil1, coil2], CentroidFrame)
    >>> s = jnp.linspace(0, 1, 100)
    >>> radial_vectors = coilset.radial_vector(s)  # shape (2, 100, 3)
    >>> frames = coilset.finite_size_frame(s)      # shape (2, 100, 3, 3)
    >>> finite_sizes = coilset.finite_size(s, 0.1, 0.1)  # shape (2, 100, 4, 3)
    >>> radial_vectors_diff = coilset.radial_vector_different_s(s[None, :].repeat(coilset.n_coils, axis=0))  # shape (2, 100, 3)
    >>> coil1_copy = coilset[0]  # Get first coil in the coilset    
    '''
    
    @classmethod    
    def from_list(cls, coils : List[FiniteSizeCoil]):
        '''
        Create a FiniteSizeCoilSet from a list of FiniteSizeCoil objects.

        Parameters
        ----------
        coils : List[FiniteSizeCoil]
            List of FiniteSizeCoil objects
        Returns
        -------
        FiniteSizeCoilSet
            FiniteSizeCoilSet object
        '''
        coils_v = jax.tree.map(_stack, *coils)        
        return cls(coils = coils_v)
    
    @classmethod
    def from_coils(cls, coils : List[Coil], method : Type[FiniteSizeMethod], *args):
        '''
        Create a FiniteSizeCoilSet from a list of Coil objects and a FiniteSizeMethod.
        This method is applied to all coils in the list.

        Parameters
        ----------
        coils : List[Coil]
            List of Coil objects
        method : Type[FiniteSizeMethod]
            FiniteSizeMethod to use for meshing
        args : tuple
            Additional arguments for the FiniteSizeMethod setup
        Returns
        -------
        FiniteSizeCoilSet
            FiniteSizeCoilSet object
        '''
        coils_v = jax.tree.map(_stack, *coils)
        finitesizemethod = method.setup_from_coils(coils_v, *args)
        return cls(FiniteSizeCoil(coils_v, method(*finitesizemethod)))
    
    @classmethod
    def from_coilset(cls, coilset : CoilSet, method : Type[FiniteSizeMethod], *args):
        finitesizemethod = method.setup_from_coils(coilset.coils, *args)
        return cls(FiniteSizeCoil(coilset.coils, method(*finitesizemethod)))

    def radial_vector(self, s):
        return _radial_vector_same_s(self.coils.coil, self.coils.finite_size_method, s)
    
    def finite_size_frame(self, s):
        return _finite_size_frame_same_s(self.coils.coil, self.coils.finite_size_method, s)
    
    def finite_size(self, s, width_radial : float, width_phi : float):
        return _finite_size_same_s(self.coils.coil, self.coils.finite_size_method, s, width_radial, width_phi)
    
    def radial_vector_different_s(self, s):
        return _radial_vector_different_s(self.coils.coil, self.coils.finite_size_method, s)
    
    def finite_size_frame_different_s(self, s):
        return _finite_size_frame_different_s(self.coils.coil, self.coils.finite_size_method, s)
    
    def finite_size_different_s(self, s, width_radial : float, width_phi : float):
        return _finite_size_different_s(self.coils.coil, self.coils.finite_size_method, s, width_radial, width_phi)   
    



@jax.jit
def order_coilset_phi(coilset : CoilSet):
    '''
    Orders a CoilSet in increasing toroidal angle (phi). Works with both CoilSet and FiniteSizeCoilSet.

    Parameters
        coilset (CoilSet) : CoilSet to order
    Returns
        CoilSet           : ordered Coil_Set       
    '''

    phis = jnp.arctan2(coilset.centre()[:,1], coilset.centre()[:,0])
    
    permutation = jnp.argsort(phis)
    new_coils = tree_take(coilset.coils, permutation)

    return type(coilset)(coils=new_coils)
    

def ensure_coilset_rotation(coilset : CoilSet, positive_rotation : bool):
    '''
    Ensures that all coils in a CoilSet are defined in the same direction.

    Parameters
        coilset (CoilSet) : CoilSet to ensure rotation
    Returns
        CoilSet           : CoilSet with all coils rotation
    '''
    return type(coilset)(_ensure_coilset_rotation_vmap(coilset.coils, positive_rotation))

def filter_coilset(coilset : CoilSet, mask):
    '''
    Filters a CoilSet to only include coils where mask is True.
    Parameters
        coilset (CoilSet) : CoilSet to filter
        mask (jnp.ndarray): Boolean mask to filter coils
    Returns
        CoilSet           : filtered CoilSet    
    '''
    return type(coilset)(tree_take(coilset.coils, mask))

def filter_coilset_phi(coilset : CoilSet, phi_min : float, phi_max : float):
    '''
    Filters a CoilSet to only include coils with centre phi between phi_min and phi_max.

    Parameters
        coilset (CoilSet) : CoilSet to filter
        phi_min (float)   : minimum phi
        phi_max (float)   : maximum phi
    Returns
        CoilSet           : filtered CoilSet       
    '''
    phis = jnp.arctan2(coilset.centre()[:,1], coilset.centre()[:,0])
    mask = (phis >= phi_min) & (phis <= phi_max)
    return filter_coilset(coilset, mask)