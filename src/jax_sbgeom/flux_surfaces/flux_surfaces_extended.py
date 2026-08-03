import jax.numpy as jnp
import h5py 
import jax
import numpy as onp
from dataclasses import dataclass
from jax_sbgeom.jax_utils import stack_jacfwd, interpolate_fractions, bilinear_interp
from functools import partial
from .flux_surfaces_base import FluxSurface, FluxSurfaceBase, ToroidalExtent, FluxSurfaceSettings, FluxSurfaceData, _data_modes_settings_from_hdf5, _cartesian_to_cylindrical, _principal_curvatures_interpolated, _cylindrical_position_interpolated, _cylindrical_to_cartesian, ParametrisedSurface, _normalize_theta_phi_full_mod


from .flux_surfaces_base import _cartesian_position_interpolated, _normal_interpolated
import equinox as eqx
from typing import Tuple



class FluxSurfaceNormalExtended(FluxSurfaceBase):
    '''
    Class representing a flux surface that is extended along the normal direction.

    The extension is done such that:
    - For s <= 1.0, the original flux surface is used
    - For s > 1.0, the position is given by moving along the normal direction of the flux surface at s = 1.0
    '''
    def cartesian_position(self, s, theta, phi):
        return _normal_extended_cartesian_position(self, s, theta, phi)
    
    def cylindrical_position(self, s, theta, phi):
        return _normal_extended_cylindrical_position(self, s, theta, phi)
    
    def normal(self, s, theta, phi):        
        return _normal_extended_normal(self, s, theta, phi)
    
    
    def principal_curvatures(self, s, theta, phi):        
        return _normal_extended_principal_curvatures(self, s, theta, phi)



class FluxSurfaceNormalExtendedNoPhi(FluxSurfaceBase):
    '''
    Class representing a flux surface that is extended along the normal direction, but with no toroidal (phi) component in the extension.

    The extension is done such that:
    - For s <= 1.0, the original flux surface is used
    - For s > 1.0, the position is given by moving along the normal direction of the flux surface at s = 1.0, but with the toroidal component removed

    This is useful for creating an extension label that preserves phi_in = phi_out but still extends in a straight line. However, the label 
    does not have the meaning of 'distance to the lcfs' anymore, as the extension is not along the actual normal direction.
    '''

    def cartesian_position(self, s, theta, phi):
        return _normal_extended_no_phi_cartesian_position(self, s, theta, phi)
    
    def cylindrical_position(self, s, theta, phi):
        return _normal_extended_no_phi_cylindrical_position(self, s, theta, phi)
    
    def normal(self, s, theta, phi):
        return _normal_extended_no_phi_normal(self, s, theta, phi)

    def principal_curvatures(self, s, theta, phi):
        return _normal_extended_no_phi_principal_curvatures(self, s, theta, phi)


class FluxSurfaceNormalExtendedConstantPhi(FluxSurfaceBase):
    '''
    Class representing a flux surface that is extended along the normal direction, but keeping the toroidal angle (phi) constant during the extension.
    The extension is done such that:
    - For s <= 1.0, the original flux surface is used
    - For s > 1.0, the position is given by moving along the normal direction of the flux surface at s = 1.0, but adjusting the toroidal angle to keep it constant

    This is useful for creating an extension label that preserves phi_in = phi_out while retaining the meaning of 'distance to the lcfs', as the extension is still along the normal direction.
    However, the extension is no longer a straight line in 3D space.
    '''

    def cartesian_position(self, s, theta, phi):
        return _normal_extended_constant_phi_cartesian_position(self, s, theta, phi)
    
    def cylindrical_position(self, s, theta, phi):        
        return _normal_extended_constant_phi_cylindrical_position(self, s, theta, phi)
    
    # For a normal extended flux surface, the normal *remains the same* in the extended region
    def normal(self, s, theta, phi):        
        return _normal_extended_constant_phi_normal(self, s, theta, phi)
    
    # Principal curvatures could be implemented in the extension region using 
    def principal_curvatures(self, s, theta, phi):        
        return _normal_extended_constant_phi_principal_curvatures(self, s, theta, phi)


class FluxSurfaceFourierExtended(FluxSurfaceBase):
    '''
    A flux surface that is extended using another flux surface defined in Fourier space.
    This does not necessarily have to have the same mpol & ntor as the inner flux surface.
    
    The inner flux surface is used for s <= 1.0, and the extension flux surface is used for s > 1.0.

    s = 1.0 corresponds to the LCFS of the inner surface
    s = 2.0 corresponds to the first surface of the extension surface
    etc.. 
    s = n_extension + 1.0 corresponds to the last surface of the extension surface

    Beyond that, the additional s is ignored.
    '''
    extension_flux_surface : FluxSurfaceBase = None

    @classmethod
    def from_flux_surface_and_extension(cls, flux_surface : FluxSurfaceBase, extension_flux_surface : FluxSurfaceBase):
        '''
        Create a FluxSurfaceFourierExtended from a base flux surface and an extension flux surface.

        Parameters
        -----------
        flux_surface : FluxSurfaceBase
            Base flux surface to extend.
        extension_flux_surface : FluxSurfaceBase
            Extension flux surface to use for s > 1.0.
        Returns
        -------
        FluxSurfaceFourierExtended        
        '''
        return cls(data = flux_surface.data, modes = flux_surface.modes, settings = flux_surface.settings, extension_flux_surface = extension_flux_surface)
    
    def cartesian_position(self, s, theta, phi):
        return _fourier_extended_cartesian_position(self, self.extension_flux_surface, s, theta, phi)
    
    def cylindrical_position(self, s, theta, phi):
        return _fourier_extended_cylindrical_position(self, self.extension_flux_surface, s, theta, phi)
    
    def normal(self, s, theta, phi):
        return _fourier_extended_normal(self, self.extension_flux_surface, s, theta, phi)
    
    def principal_curvatures(self, s, theta, phi):
        return _fourier_extended_principal_curvatures(self, self.extension_flux_surface, s, theta, phi)
    


class FluxSurfaceExtendedDistanceMatrix(ParametrisedSurface):    

    flux_surface_extended : FluxSurfaceBase
    '''Flux surface extenion used. Cannot be of type FluxSurface.'''
    

    d_layers : jnp.ndarray
    '''The distance matrices of each layer: assumed to be shaped like [n_theta_sampled, n_phi_sampled]. s[0,0] is (0,0), s[-1-1] corresponds to (2pi, 2pi/nfp)'''


    def __check_init__(self):
        assert not isinstance(self.flux_surface_extended, FluxSurface), "flux_surface_extended cannot be of type FluxSurface; this means no extension exists and class is useless"

    @eqx.filter_jit
    def cylindrical_position(self, s, theta, phi):
        return _cartesian_to_cylindrical(self.cartesian_position(s, theta, phi))
    

    def s_interp(self, s, theta, phi ):
        s_bc, theta_bc, phi_bc = jnp.broadcast_arrays(s, theta, phi)
        d_total_interp = _d_interp_vectorized(self.d_layers, self.flux_surface_extended.nfp, s, theta, phi)

        ds_internal = jnp.where(s_bc <= 1.0, 0.0, jnp.where(s_bc >= 2.0, 1.0, s_bc - 1.0))
        s_internal = jnp.where(s_bc <= 1.0, s_bc, jnp.where(s_bc >= 2.0, 1.0 + d_total_interp, 1.0 + ds_internal * d_total_interp))
        return s_internal


    # @eqx.filter_jit
    # def d_interp(self, s, theta, phi):
    #     return _d_interp_vectorized(self.d_layers, self.flux_surface_extended.nfp, s, theta, phi)

    @eqx.filter_jit
    def cartesian_position(self, s, theta, phi):
        return self.flux_surface_extended.cartesian_position(self.s_interp(s,theta,phi), theta, phi)
                        
        

    @eqx.filter_jit
    def normal(self, s, theta, phi):

        s = jnp.clip(s, max = len(self.d_layers)  + 1 - 1e-6) # to avoid out of bounds access in d_interp
        grad_pos = _cartesian_position_normal_d_matrix(self, s, theta, phi)
        return grad_pos / jnp.linalg.norm(grad_pos, axis=-1, keepdims=True) 
    
    @eqx.filter_jit
    def principal_curvatures(self, s, theta, phi):
        raise NotImplementedError


_cartesian_position_normal_d_matrix = jnp.vectorize(stack_jacfwd(lambda flux_surface, s, theta, phi : flux_surface.cartesian_position(s, theta, phi), argnums=1), excluded = (0,), signature = "(),(),()->(3)")
# ===================================================================================================================================================================================
#                                                                           Normal Extended
# ===================================================================================================================================================================================

@eqx.filter_jit
def _normal_extended_cartesian_position(flux_surface : FluxSurfaceBase,  s,  theta, phi):    
    '''
    Extend the cartesian position of a flux surface along the normal direction.
    For s <= 1.0, the original flux surface is used, while for s > 1.0, the position is given by moving along the normal direction of the flux surface at s = 1.0.

    Parameters
    -----------
    flux_surface : FluxSurfaceBase
        Flux surface to evaluate
    s : jnp.ndarray
        Radial coordinate(s) at which to evaluate the position.
    theta : jnp.ndarray
        Poloidal angle(s) at which to evaluate the position.
    phi : jnp.ndarray
        Toroidal angle(s) at which to evaluate the position.
    Returns
    -------
    jnp.ndarray
        Cartesian position(s) of the extended flux surface.
    '''
    positions = _cartesian_position_interpolated(flux_surface, jnp.minimum(s, 1.0), theta, phi) 
    normals   = _normal_interpolated(flux_surface, 1.0, theta, phi) # this will not give NaNs, as s=1.0 is always on the surface (non axis)    
    distance_1d = jnp.maximum(s - 1.0, 0.0)
    # We have to ensure that both do not produce nan values. 
    # This is the case, as the positions are evaluated at s <= 1.0 and normals at s = 1.0    
    return positions + normals * distance_1d[..., None]

@eqx.filter_jit
def _normal_extended_cylindrical_position(flux_surface : FluxSurfaceBase,  s,  theta, phi):
    '''
    Extend the cartesian position of a flux surface along the normal direction and afterwards convert to cylindrical coordinates.
    For s <= 1.0, the original flux surface is used, while for s > 1.0, the position is given by moving along the normal direction of the flux surface at s = 1.0.

    Parameters
    -----------
    flux_surface : FluxSurfaceBase
        Flux surface to evaluate
    s : jnp.ndarray
        Radial coordinate(s) at which to evaluate the position.
    theta : jnp.ndarray
        Poloidal angle(s) at which to evaluate the position.
    phi : jnp.ndarray
        Toroidal angle(s) at which to evaluate the position.
    Returns
    -------
    jnp.ndarray
        Cartesian position(s) of the extended flux surface.
    '''
    return _cartesian_to_cylindrical(_normal_extended_cartesian_position(flux_surface, s, theta, phi))

@eqx.filter_jit
def _normal_extended_normal(flux_surface : FluxSurfaceBase,  s,  theta, phi):
    '''
    Extend the normal of a flux surface along the normal direction.

    Same as normal itself.

    Parameters
    -----------
    flux_surface : FluxSurfaceBase  
        Flux surface to evaluate
    s : jnp.ndarray
        Radial coordinate(s) at which to evaluate the normal.
    theta : jnp.ndarray
        Poloidal angle(s) at which to evaluate the normal.
    phi : jnp.ndarray
        Toroidal angle(s) at which to evaluate the normal.
    Returns
    -------
    jnp.ndarray
        Normal
    '''
    return _normal_interpolated(flux_surface, jnp.minimum(s, 1.0), theta, phi)

@eqx.filter_jit
def _normal_extended_principal_curvatures(flux_surface : FluxSurfaceBase,  s,  theta, phi):
    '''
    Extend the principal curvatures of a flux surface along the normal direction. Uses the principal curvatures
    formulas in [1].

    [1]: Farouki, R. T. (1986). The approximation of non-degenerate offset surfaces. Computer Aided Geometric Design, 3(1), 15-43.

    Parameters
    -----------
    flux_surface : FluxSurfaceBase
        Flux surface to extend.
    s : jnp.ndarray
        Radial coordinate(s) at which to evaluate the principal curvatures.
    theta : jnp.ndarray
        Poloidal angle(s) at which to evaluate the principal curvatures.
    phi : jnp.ndarray
        Toroidal angle(s) at which to evaluate the principal curvatures.
    Returns
    -------
    jnp.ndarray
        Principal curvatures

    '''
    curvatures = _principal_curvatures_interpolated(flux_surface, jnp.minimum(s, 1.0), theta, phi)
    d = jnp.maximum(s - 1.0, 0.0)        
    gamma_0 = jnp.where(1 + curvatures[...,0] * d >= 0.0, jnp.ones_like(d), jnp.ones_like(d) * -1.0) # the >= or > is arbitrary
    gamma_1 = jnp.where(1 + curvatures[...,1] * d >= 0.0, jnp.ones_like(d), jnp.ones_like(d) * -1.0) # the >= or > is arbitrary        
    kappa_0 = curvatures[...,0] / jnp.abs(1.0 + curvatures[...,0] * d) * gamma_0
    kappa_1 = curvatures[...,1] / jnp.abs(1.0 + curvatures[...,1] * d) * gamma_1
    return jnp.stack([kappa_0, kappa_1], axis=-1)


# ===================================================================================================================================================================================
#                                                                           No Phi
# ===================================================================================================================================================================================
def _hat_phi(positions):
    '''
    Compute the unit vector in the toroidal (phi) direction for given cartesian positions.

    Parameters
    -----------
    positions : jnp.ndarray
        Cartesian positions at which to compute the hat phi vector.
    Returns
    -------
    jnp.ndarray
        Unit vectors in the toroidal direction at the given positions.
    '''
    x = positions[...,0]
    y = positions[...,1]
    z = positions[...,2]
    r = jnp.sqrt(x**2 + y**2)        
    safe_r  = jnp.clip(r, min = 1e-12)
    hat_phi = jnp.stack([-y / safe_r, x / safe_r, jnp.zeros_like(z)], axis=-1)
    return hat_phi

@eqx.filter_jit
def _normal_extended_no_phi_cartesian_position(flux_surface : FluxSurfaceBase,  s,  theta, phi):
    '''
    Extend the cartesian position of a flux surface along the normal direction with no toroidal (phi) component.
    For s <= 1.0, the original flux surface is used, while for s > 1.0, the position is given by moving along the normal direction 
    of the flux surface at s = 1.0, but with the toroidal component removed.

    Parameters
    -----------
    flux_surface : FluxSurfaceBase
        Flux surface to evaluate
    s : jnp.ndarray
        Radial coordinate(s) at which to evaluate the position.
    theta : jnp.ndarray
        Poloidal angle(s) at which to evaluate the position.
    phi : jnp.ndarray
        Toroidal angle(s) at which to evaluate the position.
    Returns
    -------
    jnp.ndarray
        Cartesian position(s) of the extended flux surface.
    '''
    positions                = _cartesian_position_interpolated(flux_surface, jnp.minimum(s, 1.0), theta, phi)
    normals                  = _normal_interpolated(flux_surface, 1.0, theta, phi)
    hat_phi                  = _hat_phi(positions)
    
    phi_component            = jnp.einsum("...i,...i->...", normals, hat_phi)
    
    normal_no_phi            = normals - phi_component[..., None] * hat_phi
    normal_no_phi_normalised = normal_no_phi / jnp.linalg.norm(normal_no_phi, axis=-1, keepdims=True)    
    
    distance_1d              = jnp.maximum(s - 1.0, 0.0)

    return positions + normal_no_phi_normalised * distance_1d[..., None]

@eqx.filter_jit
def _normal_extended_no_phi_cylindrical_position(flux_surface : FluxSurfaceBase,  s,  theta, phi):
    '''
    Extend the cartesian position of a flux surface along the normal direction with no toroidal component and convert to cylindrical coordinates.
    For s <= 1.0, the original flux surface is used, while for s > 1.0, the position is given by moving along the normal direction 
    of the flux surface at s = 1.0, but with the toroidal component removed.

    Parameters
    -----------
    flux_surface : FluxSurfaceBase  
        Flux surface to evaluate
    s : jnp.ndarray
        Radial coordinate(s) at which to evaluate the position.
    theta : jnp.ndarray
        Poloidal angle(s) at which to evaluate the position.
    phi : jnp.ndarray
        Toroidal angle(s) at which to evaluate the position.
    Returns
    -------
    jnp.ndarray
        Cylindrical position(s) of the extended flux surface.
    '''
    return _cartesian_to_cylindrical(_normal_extended_no_phi_cartesian_position(flux_surface, s, theta, phi))

@eqx.filter_jit
def _normal_extended_no_phi_normal(flux_surface : FluxSurfaceBase,  s,  theta, phi):
    '''
    Compute the normal for a flux surface extended with no toroidal component.
    Returns NaN values as the normal is not well-defined in the extended region.
    If one desires the normal as desired from the original surface, use flux_surface.normal(jnp.minimum(s, 1.0), theta, phi) instead.

    Parameters
    -----------
    flux_surface : FluxSurfaceBase
        Flux surface to evaluate
    s : jnp.ndarray
        Radial coordinate(s) at which to evaluate the normal.
    theta : jnp.ndarray
        Poloidal angle(s) at which to evaluate the normal.
    phi : jnp.ndarray
        Toroidal angle(s) at which to evaluate the normal.
    Returns
    -------
    jnp.ndarray
        Array of NaN values with shape matching the input and dimension 3.
    '''
    s_bc, theta_bc, phi_bc = jnp.broadcast_arrays(s, theta, phi)
    return jnp.full(s_bc.shape + (3,), jnp.nan)

@eqx.filter_jit
def _normal_extended_no_phi_principal_curvatures(flux_surface : FluxSurfaceBase,  s,  theta, phi):
    '''
    Compute the principal curvatures for a flux surface extended with no toroidal component.
    Returns NaN values as the principal curvatures in not implemented in the extended region.

    Parameters
    -----------
    flux_surface : FluxSurfaceBase
        Flux surface to evaluate
    s : jnp.ndarray
        Radial coordinate(s) at which to evaluate the principal curvatures.
    theta : jnp.ndarray
        Poloidal angle(s) at which to evaluate the principal curvatures.
    phi : jnp.ndarray
        Toroidal angle(s) at which to evaluate the principal curvatures.
    Returns
    -------
    jnp.ndarray
        Array of NaN values with shape matching the input and dimension 2.
    '''
    s_bc, theta_bc, phi_bc = jnp.broadcast_arrays(s, theta, phi)
    return jnp.full(s_bc.shape + (2,), jnp.nan)

_normal_extended_no_phi_dx_dtheta = jax.jit(jnp.vectorize(jax.jacfwd(_normal_extended_no_phi_cartesian_position, argnums=3), excluded=(0,1), signature='(),(),()->(3)'))

@eqx.filter_jit
def __normal_extended_no_phi_arc_length_theta(flux_surface : FluxSurfaceBase, s, theta, phi):
    '''
    Compute the arc length derivative with respect to theta for a flux surface extended with no toroidal component.

    Uses autodiff to compute the derivative.

    Parameters
    -----------
    flux_surface : FluxSurfaceBase
        Flux surface to evaluate
    s : jnp.ndarray
        Radial coordinate(s) at which to evaluate the arc length.
    theta : jnp.ndarray
        Poloidal angle(s) at which to evaluate the arc length.
    phi : jnp.ndarray
        Toroidal angle(s) at which to evaluate the arc length.
    Returns
    -------
    jnp.ndarray
        Arc length derivative with respect to theta.
    '''
    return jnp.linalg.norm(_normal_extended_no_phi_dx_dtheta(flux_surface, s, theta, phi), axis=-1)




# ===================================================================================================================================================================================
#                                                                          Constant Phi
# ===================================================================================================================================================================================
def _distance_between_angles(angle1, angle2):
    '''
    Compute the signed angular distance between two angles, taking into account periodicity.

    Parameters
    -----------
    angle1 : jnp.ndarray
        First angle(s) in radians.
    angle2 : jnp.ndarray
        Second angle(s) in radians.
    Returns
    -------
    jnp.ndarray
        Signed angular distance from angle2 to angle1, in the range [-pi, pi].
    '''
    return jnp.arctan2(jnp.sin(angle1 - angle2), jnp.cos(angle1 - angle2))

def _distance_between_phi_phi_desired(flux_surface : FluxSurfaceBase, s, theta, phi, x):
    '''
    Compute the angular distance between the toroidal angle of a position and a desired toroidal angle.
    Used as the objective function for finding the correct phi in constant-phi extensions.

    Parameters
    -----------
    flux_surface : FluxSurfaceBase
        Flux surface to evaluate
    s : jnp.ndarray
        Radial coordinate(s).
    theta : jnp.ndarray
        Poloidal angle(s).
    phi : jnp.ndarray
        Desired toroidal angle(s).
    x : jnp.ndarray
        Trial toroidal angle(s) for the extended position.
    Returns
    -------
    jnp.ndarray
        Angular distance between the computed phi and the desired phi.
    '''
    positions =  _normal_extended_cartesian_position(flux_surface, s, theta, x)
    return _distance_between_angles(jnp.arctan2(positions[...,1], positions[...,0]), phi)

@eqx.filter_jit
def _normal_extended_constant_phi_find_phi(flux_surface : FluxSurfaceBase, s , theta, phi, n_iter : int = 5):
    '''
    Find the toroidal angle needed in the extended region to maintain a constant output toroidal angle.
    Uses the secant method to solve for the angle that produces the desired phi after normal extension.

    Parameters
    -----------
    flux_surface : FluxSurfaceBase
        Flux surface to evaluate
    s : jnp.ndarray
        Radial coordinate(s).
    theta : jnp.ndarray
        Poloidal angle(s).
    phi : jnp.ndarray
        Desired output toroidal angle(s).
    n_iter : int, optional
        Number of secant iterations to perform (default: 5).
    Returns
    -------
    jnp.ndarray
        Adjusted toroidal angle(s) to use for the extended position.
    '''
    assert n_iter >= 1, "n_iter must be at least 1"

    _, _, phi_bc = jnp.broadcast_arrays(s, theta, phi)
        
    x_minus_two = phi_bc + 1e-3
    x_minus_one = phi_bc
    
    f_minus_two = _distance_between_phi_phi_desired(flux_surface, s, theta, phi, x_minus_two)

    def secant_iteration(i, vals):
        x_minus_two, x_minus_one, f_minus_two = vals
        f_minus_one = _distance_between_phi_phi_desired(flux_surface, s, theta, phi, x_minus_one)

        x_new = x_minus_one - f_minus_one * (x_minus_one - x_minus_two) / (f_minus_one - f_minus_two + 1e-16)
        return (x_minus_one, x_new, f_minus_one)

    x_final = jax.lax.fori_loop(0, n_iter, secant_iteration, (x_minus_two, x_minus_one, f_minus_two))[1]
    return x_final

@eqx.filter_jit
def _normal_extended_constant_phi_cartesian_position(flux_surface : FluxSurfaceBase,  s,  theta, phi, n_iter : int = 5):
    '''
    Extend the cartesian position of a flux surface along the normal direction while keeping the toroidal angle constant.
    For s <= 1.0, the original flux surface is used, while for s > 1.0, the position is given by moving along the normal 
    direction while adjusting the internal phi to maintain the output phi constant.

    Parameters
    -----------
    flux_surface : FluxSurfaceBase
        Flux surface to evaluate
    s : jnp.ndarray
        Radial coordinate(s) at which to evaluate the position.
    theta : jnp.ndarray
        Poloidal angle(s) at which to evaluate the position.
    phi : jnp.ndarray
        Toroidal angle(s) at which to evaluate the position.
    n_iter : int, optional
        Number of iterations for the phi solver (default: 5).
    Returns
    -------
    jnp.ndarray
        Cartesian position(s) of the extended flux surface.
    '''
    phi_c = _normal_extended_constant_phi_find_phi(flux_surface, s, theta, phi, n_iter)    
    return _normal_extended_cartesian_position(flux_surface, s, theta, phi_c)

@eqx.filter_jit
def _normal_extended_constant_phi_cylindrical_position(flux_surface : FluxSurfaceBase,  s,  theta, phi, n_iter : int = 5):
    '''
    Extend the position of a flux surface along the normal direction while keeping the toroidal angle constant and convert to cylindrical coordinates.
    For s <= 1.0, the original flux surface is used, while for s > 1.0, the position is given by moving along the normal 
    direction while adjusting the internal phi to maintain the output phi constant.

    Parameters
    -----------
    flux_surface : FluxSurfaceBase
        Flux surface to evaluate
    s : jnp.ndarray
        Radial coordinate(s) at which to evaluate the position.
    theta : jnp.ndarray
        Poloidal angle(s) at which to evaluate the position.
    phi : jnp.ndarray
        Toroidal angle(s) at which to evaluate the position.
    n_iter : int, optional
        Number of iterations for the phi solver (default: 5).
    Returns
    -------
    jnp.ndarray
        Cylindrical position(s) of the extended flux surface.
    '''
    return _cartesian_to_cylindrical(_normal_extended_constant_phi_cartesian_position(flux_surface, s, theta, phi, n_iter))

@eqx.filter_jit
def _normal_extended_constant_phi_normal(flux_surface : FluxSurfaceBase,  s,  theta, phi, n_iter : int = 5):
    '''
    Compute the normal for a flux surface extended along the normal direction with constant toroidal angle.

    Parameters
    -----------
    flux_surface : FluxSurfaceBase
        Flux surface to evaluate
    s : jnp.ndarray
        Radial coordinate(s) at which to evaluate the normal.
    theta : jnp.ndarray
        Poloidal angle(s) at which to evaluate the normal.
    phi : jnp.ndarray
        Toroidal angle(s) at which to evaluate the normal.
    n_iter : int, optional
        Number of iterations for the phi solver (default: 5).
    Returns
    -------
    jnp.ndarray
        Normal vector(s) of the extended flux surface.
    '''
    phi_c = _normal_extended_constant_phi_find_phi(flux_surface, s, theta, phi, n_iter)    
    return _normal_extended_normal(flux_surface, s, theta, phi_c)

@eqx.filter_jit
def _normal_extended_constant_phi_principal_curvatures(flux_surface : FluxSurfaceBase,  s,  theta, phi, n_iter : int = 5):
    '''
    Compute the principal curvatures for a flux surface extended along the normal direction with constant toroidal angle.

    Parameters
    -----------
    flux_surface : FluxSurfaceBase
        Flux surface to evaluate
    s : jnp.ndarray
        Radial coordinate(s) at which to evaluate the principal curvatures.
    theta : jnp.ndarray
        Poloidal angle(s) at which to evaluate the principal curvatures.
    phi : jnp.ndarray
        Toroidal angle(s) at which to evaluate the principal curvatures.
    n_iter : int, optional
        Number of iterations for the phi solver (default: 5).
    Returns
    -------
    jnp.ndarray
        Principal curvatures of the extended flux surface.
    '''
    phi_c = _normal_extended_constant_phi_find_phi(flux_surface, s, theta, phi, n_iter)    
    return _normal_extended_principal_curvatures(flux_surface, s, theta, phi_c)


# ===================================================================================================================================================================================
#                                                                          Fourier Extended
# ===================================================================================================================================================================================


@eqx.filter_jit
def _fourier_extended_cylindrical_position(flux_surface : FluxSurfaceBase, extension : FluxSurfaceBase, s, theta, phi):
    '''
    Compute the cylindrical position for a flux surface with Fourier-based extension.
    For s <= 1.0, the original flux surface is used. For 1.0 < s < 2.0, linear interpolation between the LCFS 
    and the first extension surface is used. For s >= 2.0, the extension surface is used directly.

    Parameters
    -----------
    flux_surface : FluxSurfaceBase
        Base flux surface to evaluate (s <= 1.0).
    extension : FluxSurfaceBase
        Extension flux surface for s > 1.0.
    s : jnp.ndarray
        Radial coordinate(s) at which to evaluate the position.
    theta : jnp.ndarray
        Poloidal angle(s) at which to evaluate the position.
    phi : jnp.ndarray
        Toroidal angle(s) at which to evaluate the position.
    Returns
    -------
    jnp.ndarray
        Cylindrical position(s) of the extended flux surface.
    '''
    # This is not necessarily completely efficient: but we cannot avoid evaluating both positions in batched operations. 
    n_surf_extension    = jnp.maximum(extension.data.Rmnc.shape[0], 2) # if there's only one extension surface this ensures we get a valid result (s=0.5 interpolation on the extension is just the surface itself anyway)

    inner_positions     = _cylindrical_position_interpolated(flux_surface, jnp.minimum(s, 1.0), theta, phi) 
    d_value             = jnp.maximum(s - 1.0, 0.0)
    d_value_extension   = jnp.maximum(d_value - 1.0, 0.0)
    normalized_d_value  = d_value_extension / (n_surf_extension - 1.0)

    extension_positions    = _cylindrical_position_interpolated(extension, normalized_d_value , theta, phi)
    extension_positions_d0 = _cylindrical_position_interpolated(extension, jnp.zeros_like(s) , theta, phi)    
    only_extension         = jnp.array(s >=2.0)
    
    return jnp.where(only_extension[..., None], extension_positions,
                     inner_positions + (extension_positions_d0- inner_positions) * d_value[..., None])

@eqx.filter_jit
def _fourier_extended_cartesian_position(flux_surface : FluxSurfaceBase, extension : FluxSurfaceBase, s, theta, phi):
    '''
    Compute the cartesian position for a flux surface with Fourier-based extension.
    For s <= 1.0, the original flux surface is used. For 1.0 < s < 2.0, linear interpolation between the LCFS 
    and the first extension surface is used. For s >= 2.0, the extension surface is used directly.

    Parameters
    -----------
    flux_surface : FluxSurfaceBase
        Base flux surface to evaluate (s <= 1.0).
    extension : FluxSurfaceBase
        Extension flux surface for s > 1.0.
    s : jnp.ndarray
        Radial coordinate(s) at which to evaluate the position.
    theta : jnp.ndarray
        Poloidal angle(s) at which to evaluate the position.
    phi : jnp.ndarray
        Toroidal angle(s) at which to evaluate the position.
    Returns
    -------
    jnp.ndarray
        Cartesian position(s) of the extended flux surface.
    '''    
    return _cylindrical_to_cartesian(_fourier_extended_cylindrical_position(flux_surface, extension, s, theta, phi))
    
@eqx.filter_jit
def _fourier_extended_normal(flux_surface : FluxSurfaceBase, extension : FluxSurfaceBase, s, theta, phi):
    '''
    Compute the normal vector for a flux surface with Fourier-based extension.
    For s <= 1.0, the original flux surface normal is used. For 1.0 < s < 2.0, linear interpolation between 
    the LCFS normal and the first extension surface normal is used. For s >= 2.0, the extension surface normal is used.

    Parameters
    -----------
    flux_surface : FluxSurfaceBase
        Base flux surface to evaluate (s <= 1.0).
    extension : FluxSurfaceBase
        Extension flux surface for s > 1.0.
    s : jnp.ndarray
        Radial coordinate(s) at which to evaluate the normal.
    theta : jnp.ndarray
        Poloidal angle(s) at which to evaluate the normal.
    phi : jnp.ndarray
        Toroidal angle(s) at which to evaluate the normal.
    Returns
    -------
    jnp.ndarray
        Normal vector(s) of the extended flux surface.
    '''
    n_surf_extension    = jnp.maximum(extension.data.Rmnc.shape[0], 2) # if there's only one extension surface this ensures we get a valid result (s=0.5 interpolation on the extension is just the surface itself anyway)

    inner_normals      = _normal_interpolated(flux_surface, jnp.minimum(s, 1.0), theta, phi) 
    d_value            = jnp.maximum(s - 1.0, 0.0)
    d_value_extension  = jnp.maximum(d_value - 1.0, 0.0)
    normalized_d_value = d_value_extension / (n_surf_extension - 1.0)

    extension_normals    = _normal_interpolated(extension, normalized_d_value , theta, phi)
    extension_normals_d0 = _normal_interpolated(extension, jnp.zeros_like(s) , theta, phi)    
    only_extension       = jnp.array(s >=2.0)
    # TODO: This is unnormalized!
    return jnp.where(only_extension[..., None], extension_normals,
                     inner_normals + (extension_normals_d0 - inner_normals) * d_value[..., None])

@eqx.filter_jit
def _fourier_extended_principal_curvatures(flux_surface : FluxSurfaceBase, extension : FluxSurfaceBase, s, theta, phi):
    '''
    Compute the principal curvatures for a flux surface with Fourier-based extension.
    For s <= 1.0, the original flux surface curvatures are used. For 1.0 < s < 2.0, linear interpolation between 
    the LCFS curvatures and the first extension surface curvatures is used. For s >= 2.0, the extension surface curvatures are used.

    Parameters
    -----------
    flux_surface : FluxSurfaceBase
        Base flux surface to evaluate (s <= 1.0).
    extension : FluxSurfaceBase
        Extension flux surface for s > 1.0.
    s : jnp.ndarray
        Radial coordinate(s) at which to evaluate the principal curvatures.
    theta : jnp.ndarray
        Poloidal angle(s) at which to evaluate the principal curvatures.
    phi : jnp.ndarray
        Toroidal angle(s) at which to evaluate the principal curvatures.
    Returns
    -------
    jnp.ndarray
        Principal curvatures of the extended flux surface.
    '''
    n_surf_extension    = jnp.maximum(extension.data.Rmnc.shape[0], 2) # if there's only one extension surface this ensures we get a valid result (s=0.5 interpolation on the extension is just the surface itself anyway)

    inner_curvatures      = _principal_curvatures_interpolated(flux_surface, jnp.minimum(s, 1.0), theta, phi) 
    d_value               = jnp.maximum(s - 1.0, 0.0)
    d_value_extension     = jnp.maximum(d_value - 1.0, 0.0)
    normalized_d_value    = d_value_extension / (n_surf_extension - 1.0)

    extension_curvatures    = _principal_curvatures_interpolated(extension, normalized_d_value , theta, phi)
    extension_curvatures_d0 = _principal_curvatures_interpolated(extension, jnp.zeros_like(s) , theta, phi)    
    only_extension          = jnp.array(s >=2.0)
    
    return jnp.where(only_extension[..., None], extension_curvatures,
                     inner_curvatures + (extension_curvatures_d0 - inner_curvatures) * d_value[..., None])

    




#====================================================================================================================================================================================
# FluxSurfaceExtendedDistanceMatrix methods
#====================================================================================================================================================================================
def _d_interp(d_layers, nfp, s, theta, phi):
    s_norm =  jnp.maximum(0.0, (s - 2.0) / (d_layers.shape[0] - 1))
    s_norm = jnp.minimum(1.0, s_norm)
    i0, i1, ds = interpolate_fractions(s_norm, d_layers.shape[0])
    
    d_lower = d_layers[i0] 
    d_upper = d_layers[i1]
    d_lower_interp = bilinear_interp(*_normalize_theta_phi_full_mod(theta, phi , nfp), d_lower)
    d_upper_interp = bilinear_interp(*_normalize_theta_phi_full_mod(theta, phi , nfp), d_upper)
    d_total_interp  = d_lower_interp * (1-ds) + d_upper_interp * ds
    return d_total_interp


_d_interp_vectorized = eqx.filter_jit(jnp.vectorize(_d_interp,  excluded=(0,1), signature='(),(),()->()'))