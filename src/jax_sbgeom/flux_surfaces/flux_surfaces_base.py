from abc import abstractmethod

import jax.numpy as jnp
import h5py 
import jax
import numpy as onp
from dataclasses import dataclass
from jax_sbgeom.jax_utils import stack_jacfwd, interpolate_array
from functools import partial
import equinox as eqx

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FluxSurfaceSettings:
    mpol : int     # maximum poloidal mode number [inclusive]
    ntor : int     # maximum toroidal mode number [inclusive]
    nfp  : int     # number of field periods    


@eqx.filter_jit
def _create_mpol_vector(settings : FluxSurfaceSettings):    
    '''
    Create the poloidal mode number vector for VMEC representation.

    Uses [0] * ntor + 1, [1] * (2 * ntor + 1), [2] * (2 * ntor + 1), ..., [mpol] * (2 * ntor + 1) 
    First is because for zero poloidal mode, there is no difference between positive and negative toroidal modes
    They can be combined into a single coefficient. Since the zero mode also needs representation, there are ntor + 1 entries for m = 0.
    For m >0, there are 2 * ntor + 1 entries, since both positive and negative toroidal modes need representation and the zero mode.

    Parameters
    ----------
    mpol : int
        Maximum poloidal mode number.
    ntor : int
        Maximum toroidal mode number.    
    Returns
    -------
    jnp.ndarray
        The poloidal mode number vector.
    '''
    return jnp.concatenate([
        jnp.zeros(settings.ntor + 1, dtype=int),
        jnp.repeat(jnp.arange(1, settings.mpol + 1), 2 * settings.ntor + 1)
    ])

@eqx.filter_jit
def _create_ntor_vector(settings : FluxSurfaceSettings):
    '''
    Create the toroidal mode number vector for VMEC representation.

    Uses [0, 1, 2, ..., ntor], [-ntor, ..., -1, 0, 1, ..., ntor], ..., [-ntor, ..., -1, 0, 1, ..., ntor] for m = 0, 1, ..., mpol
    Multiplied by symmetry factor symm.

    Parameters
    ----------
    mpol : int
        Maximum poloidal mode number.
    ntor : int
        Maximum toroidal mode number.    
    symm : int
        The symmetry factor (number of field periods)
    Returns
    -------
    jnp.ndarray
        The toroidal mode number vector.

    '''
    return jnp.concatenate([
        jnp.arange(settings.ntor + 1),
        jnp.tile(jnp.arange(-settings.ntor, settings.ntor + 1), settings.mpol)
    ]) * settings.nfp
    


def _cylindrical_to_cartesian(RZphi : jnp.ndarray):
    R = RZphi[..., 0]
    Z = RZphi[..., 1]
    phi = RZphi[..., 2]
    x = R * jnp.cos(phi)
    y = R * jnp.sin(phi)
    return jnp.stack([x, y, Z], axis=-1)

def _cartesian_to_cylindrical(XYZ : jnp.ndarray):
    x = XYZ[..., 0]
    y = XYZ[..., 1]
    z = XYZ[..., 2]
    R = jnp.sqrt(x**2 + y**2)
    phi = jnp.arctan2(y, x)
    return jnp.stack([R, z, phi], axis=-1)


@jax.jit
def _check_whether_make_normals_point_outwards_required(Rmnc : jnp.ndarray, Zmns : jnp.ndarray, mpol_vector : jnp.ndarray):
    '''
    * Internal * 
    Check whether the Fourier coefficients need to be modified such that the normals point outwards.

    This corresponds to four cases:
    1. theta = 0 is outboard: 
        a. dZ_dtheta > 0 at theta = 0 -> normals point outwards -> no change
        b. dZ_dtheta < 0 at theta = 0 -> normals point inwards -> reverse theta
    2. theta = pi is outboard:
        a. dZ_dtheta > 0 at theta = 0 -> normals point inwards -> reverse theta
        b. dZ_dtheta < 0 at theta = 0 -> normals point outwards -> no change

    Parameters
    -----------
    Rmnc : jnp.ndarray
        Array of radial Fourier coefficients. Shape (nsurf, nmodes)
    Zmns : jnp.ndarray
        Array of vertical Fourier coefficients. Shape (nsurf, nmodes)
    mpol_vector : jnp.ndarray
        Array of poloidal mode numbers. Shape (nmodes,)
    
    Returns
    --------
    flip_theta : bool
        Whether to reverse theta to ensure normals point outwards.
    '''
    # This computes:
    # dZ_dtheta on the lcfs at  theta, phi = 0:
    # dZ_dtheta = jnp.sum(Zmns[-1,:] * mpol_vector * jnp.cos(mpol_vector * 0 - ntor_vector * 0)) = jnp.sum(Zmns[-1,:] * mpol_vector)        
    sum_Zmns = jnp.sum(Zmns[-1,:] * mpol_vector)
    
    # Rmnc at theta, phi = 0
    # R = jnp.sum(Rmnc[-1,:] * jnp.cos(mpol_vector * 0 - ntor_vector * 0)) = jnp.sum(Rmnc[-1,:])
    # Rmnc at theta, phi = pi, 0:
    # R = jnp.sum(Rmnc[-1,:] * jnp.cos(mpol_vector * jnp.pi - ntor_vector * 0)) = jnp.sum(Rmnc[-1,:] * (-1)**mpol_vector)
    # We want to determine whether dZ_dtheta points in the positive or negative Z direction at the outboard midplane.
    # This is accomplished by checking the sign of dZ_dtheta at the outboard midplane.
    # The outboard midplane is at theta = 0 if sum_Rmnc > 0 and at theta = pi if sum_Rmnc < 0
    cond_outboard = jnp.sum(Rmnc[-1,:]) > jnp.sum(Rmnc[-1,:] * (-1)**mpol_vector)

    
    original_u = jnp.where(cond_outboard, 
                       jnp.where(sum_Zmns > 0, True, False),   # cond_outboard == True branch
                       jnp.where(sum_Zmns > 0, False, True)    # cond_outboard == False branch
    )
    return jnp.logical_not(original_u)

def _reverse_theta_single(m_vec, n_vec, coeff_vec, cosine_sign : bool):
    '''
    * Internal * 

    Changes the Fourier coefficients such that theta is replaced by -theta.

    Parameters
    -----------
    m_vec : jnp.ndarray
        Array of poloidal mode numbers.
    n_vec : jnp.ndarray
        Array of toroidal mode numbers.
    coeff_vec : jnp.ndarray
        Array of Fourier coefficients. (Rmnc or Zmns)
    cosine_sign : bool
        If True, the coefficients correspond to cosine terms. If False, they correspond to sine terms.
    Returns
    --------
    new_coeff_vec : jnp.ndarray
        The modified Fourier coefficients after reversing theta.
    '''
    assert coeff_vec.ndim == 1
    assert m_vec.shape == n_vec.shape == coeff_vec.shape
    
    keys = jnp.stack([m_vec, n_vec], axis=1)
    
    reversed_keys = jnp.stack([m_vec, -n_vec], axis=1)  # target keys
    
    reversed_keys_mod = jnp.where(keys[:,0:1] > 0, reversed_keys, keys)
        
    matches = jnp.all(keys[:, None, :] == reversed_keys_mod[None, :, :], axis=-1)

#    assert jnp.all(jnp.any(matches, axis=0))

    idx_map = jnp.argmax(matches, axis=0)
    
    # Build new coefficient map
    # For cosine, we just swap the numbers
    # For sine, we swap and change sign, except for the m=0 terms. These stay the same
    new_coeff_vec = jax.lax.cond(cosine_sign, lambda _ : coeff_vec[idx_map], lambda _ :  jnp.where(keys[:,0] > 0, -coeff_vec[idx_map], coeff_vec[idx_map]), operand=None)
    
    return new_coeff_vec

reverse_theta_total  = jax.jit(jax.vmap(_reverse_theta_single, in_axes=(None, None, 0, None), out_axes=0))

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FluxSurfaceModes:
    mpol_vector : jnp.ndarray
    ntor_vector : jnp.ndarray

    @classmethod
    def from_settings(cls, settings : FluxSurfaceSettings):
        mpol_vector = _create_mpol_vector(settings)
        ntor_vector = _create_ntor_vector(settings)
        return cls(mpol_vector=mpol_vector, ntor_vector=ntor_vector)

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class FluxSurfaceData:
    Rmnc : jnp.ndarray
    Zmns : jnp.ndarray    

    @classmethod
    def from_rmnc_zmns_settings(cls, Rmnc : jnp.ndarray, Zmns : jnp.ndarray, settings : FluxSurfaceSettings, make_normals_point_outwards : bool = True):
        mpol_vector = _create_mpol_vector(settings)
        ntor_vector = _create_ntor_vector(settings)

        if make_normals_point_outwards:
            flip_theta = _check_whether_make_normals_point_outwards_required(Rmnc, Zmns, mpol_vector)
            Rmnc_mod = jnp.where(
                flip_theta, 
                reverse_theta_total(mpol_vector, ntor_vector, Rmnc, True),
                Rmnc

            )
            
            Zmns_mod = jnp.where(
                flip_theta,
                reverse_theta_total(mpol_vector, ntor_vector, Zmns, False),
                Zmns
            )        
        else:
            Rmnc_mod = Rmnc 
            Zmns_mod = Zmns

        assert(Rmnc.shape == Zmns.shape)        
        assert(Rmnc.shape[1] == len(mpol_vector))
        return cls(Rmnc=Rmnc_mod, Zmns=Zmns_mod)
    
    
def _data_modes_settings_from_hdf5(filename : str, make_normals_point_outwards : bool = True):
    """Load a FluxSurface from an VMEC-type HDF5 file.

        Parameters
        ----------
        filename : str
            Path to the HDF5 file.
        Returns
        -------
        FluxSurface
            The loaded FluxSurface object.

    """

    with h5py.File(filename) as f:
        Rmnc = jnp.array(f['rmnc'])            
        Zmns = jnp.array(f['zmns'])            

        mpol = int(f['mpol'][()]) - 1 # vmec uses mpol 1 larger than maximum poloidal mode number
        ntor = int(f['ntor'][()])                        
        nfp  = int(f['nfp'][()])

        settings = FluxSurfaceSettings(
            mpol=mpol,
            ntor=ntor,
            nfp=nfp,                            
        )


        assert( jnp.all( _create_mpol_vector(settings) == jnp.array(f['xm']))) # sanity check
        assert( jnp.all( _create_ntor_vector(settings) == jnp.array(f['xn']))) # sanity check
        
        data = FluxSurfaceData.from_rmnc_zmns_settings(Rmnc, Zmns, settings)        
        modes = FluxSurfaceModes.from_settings(settings)
        
        return data,  modes, settings

class ParametrisedSurface(eqx.Module):
    '''
    * Internal * 

    Class representing a parametrised surface. This is used for the interpolation-based methods that require 2D data. 
    The surface is parametrised by (s, theta, phi) where s is the surface index or normalized flux label, theta is the poloidal angle and phi is the toroidal angle.    
    '''
    @abstractmethod
    def cylindrical_position(self, s, theta, phi):
        raise NotImplementedError
    
    @abstractmethod
    def cartesian_position(self, s, theta, phi):
        raise NotImplementedError
    
    @abstractmethod
    def normal(self, s, theta, phi):
        raise NotImplementedError
    
    @abstractmethod
    def principal_curvatures(self, s, theta, phi):
        raise NotImplementedError
        


class FluxSurfaceBase(ParametrisedSurface):
    ''' 
    Class representing a set of flux surfaces using a VMEC-like representation.
    Base abstract class that provides no implementation. The actual implementation is provided in the FluxSurface class, which inherits from this base class. This allows for different implementations of the same interface, such as interpolation-based or direct evaluation-based methods.

    Attributes:
    -----------
    data     : FluxSurfaceData
        Data object containing the Fourier coefficients Rmnc and Zmns
    modes    : FluxSurfaceModes
        Modes object containing the mode vectors mpol_vector and ntor_vector
    settings : FluxSurfaceSettings
        Settings object containing parameters mpol, ntor, nfp
    '''
    data            : FluxSurfaceData     = None
    modes           : FluxSurfaceModes    = None
    settings        : FluxSurfaceSettings = None
    

    @classmethod
    def from_hdf5(cls, filename : str):        
        '''
        Load a FluxSurface from an VMEC-type HDF5 file.

        Parameters
        ----------
        filename : str
            Path to the HDF5 file.
        Returns
        -------
        FluxSurface
            The loaded FluxSurface object.
        '''
        data, modes, settings = _data_modes_settings_from_hdf5(filename)
        return cls(data=data, modes = modes, settings=settings)
    
    @classmethod
    def from_flux_surface(cls, flux_surface_base : "FluxSurfaceBase"):
        '''
        Create a FluxSurface from another FluxSurface (copy constructor).
        Can be used to convert between subclasses of FluxSurface.

        Parameters
        -----------
        flux_surface_base : FluxSurface
            The input FluxSurface to copy from.
        Returns
        --------
        FluxSurface
            The copied FluxSurface.
        '''
        return cls(data = flux_surface_base.data, modes = flux_surface_base.modes, settings = flux_surface_base.settings)
    
    @classmethod
    def from_rmnc_zmns_settings(cls, Rmnc : jnp.ndarray, Zmns : jnp.ndarray, settings : FluxSurfaceSettings, make_normals_point_outwards : bool = True):                
        '''
        Create a FluxSurface from Fourier coefficients and settings. Optionally, modify the coefficients such that normals point outwards (default=True)

        Parameters
        -----------
        Rmnc : jnp.ndarray
            Array of radial Fourier coefficients. Shape (nsurf, nmodes)
        Zmns : jnp.ndarray
            Array of vertical Fourier coefficients. Shape (nsurf, nmodes)
        settings : FluxSurfaceSettings
            Settings object containing parameters mpol, ntor, nfp   
        make_normals_point_outwards : bool
            Whether to modify the Fourier coefficients such that normals point outwards. Default is True.
        Returns
        --------
        FluxSurface
            The created FluxSurface.
        '''
        data = FluxSurfaceData.from_rmnc_zmns_settings(Rmnc, Zmns, settings, make_normals_point_outwards)
        modes = FluxSurfaceModes.from_settings(settings)
        return cls(data=data, modes = modes, settings=settings)
    
    @classmethod 
    def from_data_settings(cls, data : FluxSurfaceData, settings : FluxSurfaceSettings):        
        '''
        Create a FluxSurface from FluxSurfaceData and FluxSurfaceSettings.

        Parameters
        -----------
        data : FluxSurfaceData
            Data object containing the Fourier coefficients Rmnc and Zmns
        settings : FluxSurfaceSettings
            Settings object containing parameters mpol, ntor, nfp   
        Returns
        --------
        FluxSurface
            The created FluxSurface.
        '''
        modes = FluxSurfaceModes.from_settings(settings)        
        return cls(data=data, modes = modes, settings=settings)    
    @classmethod 
    def from_data_settings_full(cls, data : FluxSurfaceData, settings : FluxSurfaceSettings):        
        '''
        Create a FluxSurface from FluxSurfaceData and FluxSurfaceSettings. In this function, the data is ensured to be 2D. This is 
        required for some functions that interpolate between the different surfaces.

        Parameters
        -----------
        data : FluxSurfaceData
            Data object containing the Fourier coefficients Rmnc and Zmns
        settings : FluxSurfaceSettings
            Settings object containing parameters mpol, ntor, nfp   
        Returns
        --------
        FluxSurface
            The created FluxSurface.
        '''
        data = FluxSurfaceData(Rmnc = jnp.atleast_2d(data.Rmnc), Zmns = jnp.atleast_2d(data.Zmns))
        modes = FluxSurfaceModes.from_settings(settings)        
        return cls(data=data, modes = modes, settings=settings)        
    
    @property 
    def nfp(self):
        return self.settings.nfp



class FluxSurface(FluxSurfaceBase):    


    def cylindrical_position(self, s, theta, phi):
        '''
        Cylindrical position on the flux surface as a function of (s, theta, phi)

        Parameters
        -----------
        s : jnp.ndarray
            Surface index or normalized flux label
        theta : jnp.ndarray
            Poloidal angle(s)
        phi : jnp.ndarray
            Toroidal angle(s)
        Returns
        --------
        jnp.ndarray
            Cylindrical position(s) on the flux surface [R, Z, phi]
        '''
        return _cylindrical_position_interpolated(self, s, theta, phi)
    
    def cartesian_position(self, s, theta, phi):
        '''
        Cartesian position on the flux surface as a function of (s, theta, phi)

        Parameters
        -----------
        s : jnp.ndarray
            Surface index or normalized flux label
        theta : jnp.ndarray
            Poloidal angle(s)
        phi : jnp.ndarray
            Toroidal angle(s)
        Returns
        --------
        jnp.ndarray
            Cartesian position(s) on the flux surface [x, y, z]
        '''
        return _cartesian_position_interpolated(self, s, theta, phi)
    
    def normal(self, s, theta, phi):
        '''
        Normal vector on the flux surface as a function of (s, theta, phi)

        Parameters
        -----------
        s : jnp.ndarray
            Surface index or normalized flux label
        theta : jnp.ndarray
            Poloidal angle(s)
        phi : jnp.ndarray
            Toroidal angle(s)
        Returns
        --------
        jnp.ndarray
            Normal vector(s) on the flux surface
        '''
        return _normal_interpolated(self, s, theta, phi)
    
    def principal_curvatures(self, s, theta, phi):
        '''
        Principal curvatures on the flux surface as a function of (s, theta, phi)

        Parameters
        -----------
        s : jnp.ndarray
            Surface index or normalized flux label
        theta : jnp.ndarray
            Poloidal angle(s)   
        phi : jnp.ndarray
            Toroidal angle(s)
        Returns
        --------
        jnp.ndarray
            Principal curvatures(s) on the flux surface
        '''
        return _principal_curvatures_interpolated(self, s, theta, phi)
    
def make_2d_flux_surface(fs : FluxSurfaceBase) -> FluxSurfaceBase:
    '''
    Convert a FluxSurface with 1D data to a FluxSurface with 2D data by adding a dummy surface dimension.
    This allows you to use the interpolation-based methods that require 2D data. Interpolation on one surface will 
    just return the single surface.

    Parameters
    -----------
    fs : FluxSurfaceBase
        The input FluxSurfaceBase with 1D data.
    Returns
    --------
    FluxSurfaceBase
        The output FluxSurfaceBase with 2D data.
    '''
    return type(fs)(data = FluxSurfaceData(jnp.atleast_2d(fs.data.Rmnc), jnp.atleast_2d(fs.data.Zmns)),
                modes = fs.modes,
                settings = fs.settings)
    
    

class ToroidalExtent(eqx.Module):
    '''
    Class representing a toroidal extent in phi.

    Attributes:
    ----------- 
    start : float
        Starting toroidal angle (in radians)
    end   : float
        Ending toroidal angle (in radians)
    '''
    start : float
    end   : float

    @classmethod
    def half_module(cls, flux_surface : FluxSurfaceBase, dphi = 0.0):
        '''
        Create a ToroidalExtent representing half a field period.

        Parameters
        ----------- 
        flux_surface : FluxSurface
            The flux surface for which to create the toroidal extent.
        dphi : float
            An optional offset to add to both the start and end angles.
        Returns
        --------
        ToroidalExtent
            The created ToroidalExtent.
        '''
        return cls(dphi, 2 * jnp.pi / flux_surface.nfp / 2.0 + dphi)
    
    @classmethod
    def full_module(cls, flux_surface : FluxSurfaceBase, dphi = 0.0):
        '''
        Create a ToroidalExtent representing a full field period.

        Parameters
        ----------- 
        flux_surface : FluxSurface
            The flux surface for which to create the toroidal extent.
        dphi : float
            An optional offset to add to both the start and end angles.
        Returns
        --------
        ToroidalExtent
            The created ToroidalExtent.
        '''
        return cls(dphi, 2 * jnp.pi / flux_surface.nfp + dphi)
    
    @classmethod 
    def full(cls):
        '''
        Create a ToroidalExtent representing the full toroidal angle [0, 2pi].

        Returns
        --------
        ToroidalExtent
            The created ToroidalExtent.
        '''
        return cls(0.0, 2 * jnp.pi)

    def full_angle(self):
        '''
        Whether the toroidal extent represents a full 2pi angle. Used for creating closed meshes.

        Returns
        --------
        bool
            True if the toroidal extent represents a full 2pi angle, False otherwise.
        '''
        return bool(jnp.allclose(self.end - self.start, 2 * jnp.pi))
    
    def __iter__(self):
        return iter((self.start, self.end, self.full_angle()))
    
class ToroidalExtentStatic(ToroidalExtent):
    '''
    Class extending a ToroidalExtent by having a static flag indicating whether it is a full angle.

    For normal ToroidalExtent, the full angle is determined by checking whether end - start is close to 2pi. This 
    can lead to issues when using it under jit for meshing; this will lead to different sized arrays depending on runtime values,
    which is not supported.
    '''
        
    full_angle_bool : bool
    '''
    Whether the toroidal extent represents a full 2 pi angle.
    '''     

    @classmethod
    def half_module(cls, flux_surface : FluxSurfaceBase, dphi = 0.0):    
        return cls(dphi, 2 * jnp.pi / flux_surface.nfp / 2.0 + dphi, False)
    
    @classmethod
    def full_module(cls, flux_surface : FluxSurfaceBase, dphi = 0.0):        
        return cls(dphi, 2 * jnp.pi / flux_surface.nfp + dphi, False)
    
    @classmethod 
    def full(cls):
        '''
        Create a ToroidalExtent representing the full toroidal angle [0, 2pi].

        Returns
        --------
        ToroidalExtent
            The created ToroidalExtent.
        '''
        return cls(0.0, 2 * jnp.pi, True)

    def full_angle(self):
        return self.full_angle_bool



# ===================================================================================================================================================================================
#                                                                           Positions
# ===================================================================================================================================================================================
@partial(jax.jit)
def _cylindrical_position_direct(flux_surface : FluxSurfaceBase, theta, phi):        
    '''
    Cylindrical position on the flux surface as a function of (theta, phi) for a single surface (1D data).

    Parameters
    -----------
    flux_surface : FluxSurface
        The flux surface object.
    theta : jnp.ndarray
        Poloidal angle(s)
    phi : jnp.ndarray
        Toroidal angle(s)
    Returns
    --------
    jnp.ndarray
        Cylindrical position(s) on the flux surface [R, Z, phi]
    '''
    assert flux_surface.data.Rmnc.ndim == 1, "Rmnc must be a 1D array but is of shape {}".format(flux_surface.data.Rmnc.shape)
    
    # This in essence computes:
    # R   = jnp.sum(Rmnc_interp[..., None] * jnp.cos(mpol_vector[..., None] * theta[None, ...] - ntor_vector[..., None] * phi[None, ...]), axis=-1)
    # However, although the above can be more efficient, it creates large intermediate arrays and is thus undesirable.
    # Also, we call interpolate_array once per mode and per point in this setup
    # Instead, we could have vectorized this calculation over all points, but that would also create large intermediate arrays.
    
    # Now, no n_modes x n_points arrays are created.

    # This function is valid for both s,theta,phi all scalars and broadcastable arrays. 
    def fourier_sum(vals, i):
        R, Z = vals
        R = R + flux_surface.data.Rmnc[i] * jnp.cos(flux_surface.modes.mpol_vector[i] * theta - flux_surface.modes.ntor_vector[i] * phi)
        Z = Z + flux_surface.data.Zmns[i] * jnp.sin(flux_surface.modes.mpol_vector[i] * theta - flux_surface.modes.ntor_vector[i] * phi)
        return (R,Z), None
    
    # The fourier_sum function automatically broadcast arrays. However, we need to ensure that 
    # we start the scan with a zero object that has the correct final shape. Thus,
    # we create dummy arrays that have the correct shape.
    # The phi_bc  is required to ensure the final array phi is stackable with R, Z.
    theta_bc, phi_bc = jnp.broadcast_arrays(theta, phi)
    
    n_modes = flux_surface.data.Rmnc.shape[0]
    R,Z = jax.lax.scan(fourier_sum, (jnp.zeros_like(theta_bc), jnp.zeros_like(theta_bc)), jnp.arange(n_modes))[0]    
    return jnp.stack([R, Z, phi_bc],axis=-1)

@partial(jax.jit)
def _cartesian_position_direct(flux_surface : FluxSurfaceBase, theta, phi):
    '''
    Cartesian position on the flux surface as a function of (theta, phi) for a single surface (1D data).

    Parameters
    -----------
    flux_surface : FluxSurfaceBase
        The flux surface object.
    theta : jnp.ndarray
        Poloidal angle(s)
    phi : jnp.ndarray
        Toroidal angle(s)
    Returns
    --------
    jnp.ndarray
        Cartesian position(s) on the flux surface [x, y, z]
    '''
    RZphi = _cylindrical_position_direct(flux_surface, theta, phi)
    return _cylindrical_to_cartesian(RZphi)

_dx_dtheta_direct = jax.jit(jnp.vectorize(jax.jacfwd(_cartesian_position_direct, argnums=1), excluded=(0,), signature='(),()->(3)'))

@partial(jax.jit)
def _arc_length_theta_direct(flux_surface : FluxSurfaceBase, theta, phi):
    '''
    Arc length with respect to theta on the flux surface as a function of (theta, phi) for a single surface (1D data).

    Parameters
    -----------
    flux_surface : FluxSurfaceBase
        The flux surface object.
    theta : jnp.ndarray
        Poloidal angle(s)
    phi : jnp.ndarray
        Toroidal angle(s)
    Returns
    --------
    jnp.ndarray
        Arc length(s) with respect to theta on the flux surface.
    '''
    dx_dtheta = _dx_dtheta_direct(flux_surface, theta, phi)
    dx_dtheta_norm = jnp.linalg.norm(dx_dtheta, axis=-1)
    return dx_dtheta_norm

@eqx.filter_jit
def _cylindrical_position_interpolated(flux_surface : FluxSurfaceBase,  s,  theta, phi):
    '''
    Cylindrical position on the flux surface as a function of (s, theta, phi) for multiple surfaces (2D data).

    Computes:

    R   = sum_mn [ Rmnc(s) * cos(m * theta - n * phi) ]
    Z   = sum_mn [ Zmns(s) * sin(m * theta - n * phi) ]

    where Rmnc(s) and Zmns(s) are obtained via interpolation in the surface index s and all arrays are in the VMEC representation [not 2D arrays but flattened
    with explicit mode vectors].

    Vectorized fully over s, theta, phi.

    This function does not parallelize over modes to avoid creating large intermediate arrays, only over points.

    Parameters
    -----------
    flux_surface : FluxSurfaceBase
        The flux surface object.
    s : jnp.ndarray
        Surface index or normalized flux label
    theta : jnp.ndarray
        Poloidal angle(s)
    phi : jnp.ndarray
        Toroidal angle(s)
    Returns
    --------
    jnp.ndarray
        Cylindrical position(s) on the flux surface [R, Z, phi]
    '''
    assert flux_surface.data.Rmnc.ndim == 2, "Data must be a 2D array but is of shape {} [Did you use a FluxSurfaceBase with only 1D data? Check out make_2d_flux_surface.]".format(flux_surface.data.Rmnc.shape)
    
    # This in essence computes:
    # R   = jnp.sum(Rmnc_interp[..., None] * jnp.cos(mpol_vector[..., None] * theta[None, ...] - ntor_vector[..., None] * phi[None, ...]), axis=-1)
    # However, although the above can be more efficient, it creates large intermediate arrays and is thus undesirable.
    # Also, we call interpolate_array once per mode and per point in this setup
    # Instead, we could have vectorized this calculation over all points, but that would also create large intermediate arrays.
    
    # Now, no n_modes x n_points arrays are created.

    # This function is valid for both s,theta,phi all scalars and broadcastable arrays. 
    @jax.checkpoint 
    def fourier_sum(vals, i):
        R, Z = vals
        R = R + interpolate_array(flux_surface.data.Rmnc[..., i], s) * jnp.cos(flux_surface.modes.mpol_vector[i] * theta - flux_surface.modes.ntor_vector[i] * phi)
        Z = Z + interpolate_array(flux_surface.data.Zmns[..., i], s) * jnp.sin(flux_surface.modes.mpol_vector[i] * theta - flux_surface.modes.ntor_vector[i] * phi)
        return (R,Z), None
    
    # The fourier_sum function automatically broadcast arrays. However, we need to ensure that 
    # we start the scan with a zero object that has the correct final shape. Thus,
    # we create dummy arrays that have the correct shape.
    # The phi_bc  is required to ensure the final array phi is stackable with R, Z.
    s_bc, theta_bc, phi_bc = jnp.broadcast_arrays(s, theta, phi)
    
    n_modes = flux_surface.data.Rmnc.shape[1]
    
    R,Z = jax.lax.scan(fourier_sum, (jnp.zeros_like(theta_bc), jnp.zeros_like(theta_bc)), jnp.arange(n_modes))[0]
    
    return jnp.stack([R, Z, phi_bc],axis=-1)
    

@eqx.filter_jit
def _cartesian_position_interpolated(flux_surface : FluxSurfaceBase, s, theta, phi):
    '''
    Cartesian position on the flux surface as a function of (s, theta, phi) for multiple surfaces (2D data).

    Computes:

    R   = sum_mn [ Rmnc(s) * cos(m * theta - n * phi) ]
    Z   = sum_mn [ Zmns(s) * sin(m * theta - n * phi) ]

    where Rmnc(s) and Zmns(s) are obtained via interpolation in the surface index s and all arrays are in the VMEC representation [not 2D arrays but flattened
    with explicit mode vectors].

    Vectorized fully over s, theta, phi.
    
    This function does not parallelize over modes to avoid creating large intermediate arrays, only over points.

    Parameters
    -----------
    flux_surface : FluxSurfaceBase
        The flux surface object.
    s : jnp.ndarray
        Surface index or normalized flux label
    theta : jnp.ndarray
        Poloidal angle(s)
    phi : jnp.ndarray
        Toroidal angle(s)
    Returns
    --------
    jnp.ndarray
        Cartesian position(s) on the flux surface [R, Z, phi]
    '''
    RZphi = _cylindrical_position_interpolated(flux_surface, s, theta, phi)
    return _cylindrical_to_cartesian(RZphi)

_dx_dtheta = jax.jit(jnp.vectorize(jax.jacfwd(_cartesian_position_interpolated, argnums=2), excluded=(0,), signature='(),(),()->(3)'))

@eqx.filter_jit
def _arc_length_theta(flux_surface : FluxSurfaceBase, s, theta, phi):
    ''''
    Arc length with respect to theta on the flux surface as a function of (s, theta, phi) for multiple surfaces (2D data).

    Parameters
    -----------
    flux_surface : FluxSurfaceBase
        The flux surface object.
    s : jnp.ndarray
        Surface index or normalized flux label
    theta : jnp.ndarray
        Poloidal angle(s)
    phi : jnp.ndarray   
        Toroidal angle(s)
    Returns
    --------
    jnp.ndarray
        Arc length(s) with respect to theta on the flux surface.
    '''
    dx_dtheta = _dx_dtheta(flux_surface, s, theta, phi)
    dx_dtheta_norm = jnp.linalg.norm(dx_dtheta, axis=-1)
    return dx_dtheta_norm
# ===================================================================================================================================================================================
#                                                                           Normals
# ===================================================================================================================================================================================


# this function requires scalars to work since it needs to return a (3,2) array
# vmapping works, but loses the flexibility of either of the inputs being arrays, scalars or multidimensional arrays
# furthermore, the jacobians are stacked to ensure jnp.vectorize can be used (it does not support multiple outputs like given by jacfwd)
_cartesian_position_interpolated_grad = jax.jit(jnp.vectorize(stack_jacfwd(_cartesian_position_interpolated, argnums=(2,3)), excluded=(0,), signature='(),(),()->(3,2)'))

@eqx.filter_jit
def _dx_dphi_cross_dx_dtheta(flux_surface : FluxSurfaceBase, s, theta, phi):
    '''
    Compute the cross product of dr/dphi and dr/dtheta on the flux surface as a function of (s, theta, phi) for multiple surfaces (2D data).

    Parameters
    -----------
    flux_surface : FluxSurfaceBase
        The flux surface object.
    s : jnp.ndarray 
        Surface index or normalized flux label
    theta : jnp.ndarray
        Poloidal angle(s)
    phi : jnp.ndarray
        Toroidal angle(s)
    Returns
    --------
    jnp.ndarray
        The cross product of dr/dphi and dr/dtheta on the flux surface.
    '''
    dX_dtheta_and_dX_dphi = _cartesian_position_interpolated_grad(flux_surface, s, theta, phi)
    # We use dr/dphi x dr/dtheta 
    # Then, we want to have the normal vector outwards to the LCFS and not point into the plasma
    # This is accomplised by using the dphi_x_dtheta member. 
    n = jnp.cross(dX_dtheta_and_dX_dphi[..., 1], dX_dtheta_and_dX_dphi[..., 0])    
    return n

@eqx.filter_jit
def _normal_interpolated(flux_surface : FluxSurfaceBase, s, theta, phi):    
    '''
    Normal vector on the flux surface as a function of (s, theta, phi) for multiple surfaces (2D data).

    Parameters
    -----------
    flux_surface : FluxSurfaceBase
        The flux surface object.
    s : jnp.ndarray
        Surface index or normalized flux label
    theta : jnp.ndarray
        Poloidal angle(s)
    phi : jnp.ndarray
        Toroidal angle(s)
    Returns
    --------
    jnp.ndarray
        Normal vector(s) on the flux surface.
    '''
    # We use dr/dphi x dr/dtheta 
    # Then, we want to have the normal vector outwards to the LCFS and not point into the plasma
    # This is accomplised by using the dphi_x_dtheta member. 
    n = _dx_dphi_cross_dx_dtheta(flux_surface, s, theta, phi)
    n = n / jnp.linalg.norm(n, axis=-1, keepdims=True)
    return n

# ===================================================================================================================================================================================
#                                                                           Principal curvatures
# ===================================================================================================================================================================================
_cartesian_position_interpolated_grad_grad = jax.jit(jnp.vectorize(stack_jacfwd(stack_jacfwd(_cartesian_position_interpolated, argnums=(2,3)), argnums = (2,3)), excluded=(0,), signature='(),(),()->(3,2,2)'))

def _principal_curvatures_from_values(
        dX_dtheta_and_dX_dphi : jnp.ndarray,
        d2X_dtheta2_and_d2X_dthetadphi_and_d2X_dphi2 : jnp.ndarray):
        # dx_dtheta_and_dX_dphi has shape (..., 3, 2), last index 0 is d/dtheta, last index 1 is d/dphi
        # d2X_dtheta2_and_d2X_dthetadphi_and_d2X_dphi2 has shape (..., 3, 2, 2) # 0,0 is d2/dtheta2, 0,1 and 1,0 is d2/dthetadphi, 1,1 is d2/dphi2
        
        E = jnp.einsum("...i, ...i->...", dX_dtheta_and_dX_dphi[..., 0], dX_dtheta_and_dX_dphi[..., 0])
        F = jnp.einsum("...i, ...i->...", dX_dtheta_and_dX_dphi[..., 0], dX_dtheta_and_dX_dphi[..., 1])
        G = jnp.einsum("...i, ...i->...", dX_dtheta_and_dX_dphi[..., 1], dX_dtheta_and_dX_dphi[..., 1])
        
        normal_vector = jnp.cross(dX_dtheta_and_dX_dphi[..., 1], dX_dtheta_and_dX_dphi[..., 0])
        normal_vector = normal_vector / jnp.linalg.norm(normal_vector, axis=-1, keepdims=True)
    
        L = jnp.einsum("...i, ...i->...", normal_vector, d2X_dtheta2_and_d2X_dthetadphi_and_d2X_dphi2[..., 0, 0])
        M = jnp.einsum("...i, ...i->...", normal_vector, d2X_dtheta2_and_d2X_dthetadphi_and_d2X_dphi2[..., 0, 1])
        N = jnp.einsum("...i, ...i->...", normal_vector, d2X_dtheta2_and_d2X_dthetadphi_and_d2X_dphi2[..., 1, 1])
    
        H = (E * N - 2 * F * M + G * L) / (2 * (E * G - F**2))
        K = (L * N - M**2) / (E * G - F**2)
        
        sqrt_discriminant = jnp.sqrt(H**2 - K) 
        k1 = - (H + sqrt_discriminant)
        k2 = - (H - sqrt_discriminant)
    
        return jnp.stack([k1, k2], axis=-1)
    

@eqx.filter_jit
def _principal_curvatures_interpolated(flux_surface : FluxSurfaceBase, s, theta, phi):
    '''
    Principal curvatures on the flux surface as a function of (s, theta, phi) for multiple surfaces (2D data).  

    Parameters
    -----------
    flux_surface : FluxSurfaceBase
        The flux surface object.
    s : jnp.ndarray
        Surface index or normalized flux label
    theta : jnp.ndarray
        Poloidal angle(s)
    phi : jnp.ndarray
        Toroidal angle(s)
    Returns
    --------
    jnp.ndarray
        Principal curvatures(s) on the flux surface, shape (..., 2) where last index 0 is k1 and last index 1 is k2.
    '''
    dX_dtheta_and_dX_dphi                        = _cartesian_position_interpolated_grad(flux_surface, s, theta, phi)
    d2X_dtheta2_and_d2X_dthetadphi_and_d2X_dphi2 = _cartesian_position_interpolated_grad_grad(flux_surface, s, theta, phi)

    return _principal_curvatures_from_values(dX_dtheta_and_dX_dphi, d2X_dtheta2_and_d2X_dthetadphi_and_d2X_dphi2)



# ===================================================================================================================================================================================
#                                                                           Volume and Surface
# ===================================================================================================================================================================================

@eqx.filter_jit
def _volume_from_fourier(flux_surface : FluxSurfaceBase, s : float):
    ''' 
    Compute the volume enclosed by the flux surface at s using a Fourier representation.

    A full module is used for the integration. Using the divergence theorem, the volume is computed as:
        V = (1/3) * ∫∫ (r · n) dA
    where r is the position vector, n is the outward normal vector, and dA is the differential area element on the surface.
    Note that dA = |dx/dtheta x dx/dphi| dtheta dphi and thus r · n dA = r · (dx/dphi x dx/dtheta) dtheta dphi.
    The trapezoidal rule is then used to arrive at the final value.


    Parameters
    -----------
    flux_surface : FluxSurfaceBase
        The flux surface object containing Fourier coefficients and settings.
    s : float
        The normalized flux surface label (0 <= s <= 1).
    Returns
    --------
    volume : float
        The volume enclosed by the flux surface at s.
    '''
    # x:          m,n Fourier modes
    # dx_dtheta:  m,n fourier modes
    # dx_dphi:    m,n fourier modes
    # normal: m + m, m+ fourier modes
    # x.normal -> 3 * m,n fourier modes
    # Nyquist -> 6 times the mode number
    nyquist_sampling = 6

    n_theta = flux_surface.settings.mpol * nyquist_sampling +1
    n_phi   = flux_surface.settings.ntor *  nyquist_sampling  +1
    
    theta = jnp.linspace(0, 2 * jnp.pi, n_theta, endpoint=False)
    phi   = jnp.linspace(0, 2 * jnp.pi / flux_surface.nfp, n_phi, endpoint=False)
    
    dtheta = 2 * jnp.pi / n_theta
    dphi   = 2 * jnp.pi / flux_surface.nfp  /  n_phi 

    tt, pp = jnp.meshgrid(theta, phi, indexing='ij')
    
    surface_normals      = _dx_dphi_cross_dx_dtheta(flux_surface, s, tt, pp)

    r                     = _cartesian_position_interpolated(flux_surface, s, tt, pp)    
    f_ij = jnp.einsum('...i,...i->...', r, surface_normals)

    volume = jnp.sum(f_ij) * dtheta * dphi / 3.0 * flux_surface.nfp 

    return volume

@eqx.filter_jit
def _volume_from_fourier_half_mod(flux_surface : FluxSurfaceBase, s : float):
    ''' 
    Compute the volume enclosed by the flux surface at s using a Fourier representation.

    A half module is used for the integration. Using the divergence theorem, the volume is computed as:
        V = (1/3) * ∫∫ (r · n) dA
    where r is the position vector, n is the outward normal vector, and dA is the differential area element on the surface.
    Note that dA = |dx/dtheta x dx/dphi| dtheta dphi and thus r · n dA = r · (dx/dphi x dx/dtheta) dtheta dphi.
    The trapezoidal rule is then used to arrive at the final value.


    Parameters
    -----------
    flux_surface : FluxSurfaceBase
        The flux surface object containing Fourier coefficients and settings.
    s : float
        The normalized flux surface label (0 <= s <= 1).
    Returns
    --------
    volume : float
        The volume enclosed by the flux surface at s.
    '''

    nyquist_sampling = 6
    n_theta = flux_surface.settings.mpol * nyquist_sampling + 1

    # We add one to always satisfy nyquist.
    n_phi   = int((flux_surface.settings.ntor *  nyquist_sampling + 1) / 2) 
    
    # Now, given that we want to sample half of a module, we have two choices depending on the full module n_phi:

    # - Include the half module boundary

    # - Exclude the half module boundary

    # If we include the half-module boundary, we have to use 
    # phi = jnp.linspace(0, 2 * jnp.pi / settings.nfp , n_phi, endpoint=True)
    # and then subtract half of the initial phi=0 and half of the phi = pi / nfp boundary contributions from the volume integral

    # If we exclude the half-module boundary, we have to use
    # phi = jnp.linspace(0, 2 * jnp.pi / settings.nfp, 2 * n_phi, endpoint=True)[:n_phi]
    # and then double the volume integral and subtracth only half of the initial phi=0 boundary.
    
    # We chose the latter option since it is one less computation. Numerically, they are exactly the same.

    theta = jnp.linspace(0, 2 * jnp.pi, n_theta, endpoint=False)
    phi   = jnp.linspace(0, 2 * jnp.pi / flux_surface.nfp , 2* n_phi, endpoint=True)[:n_phi]
    
    dtheta = 2 * jnp.pi / n_theta
    dphi   = phi[1]- phi[0]
        
    tt, pp = jnp.meshgrid(theta, phi, indexing='ij')
    
    surface_normals        = _dx_dphi_cross_dx_dtheta(flux_surface, s, tt, pp)

    r                      = _cartesian_position_interpolated(flux_surface, s, tt, pp)    
    f_ij                   = jnp.einsum('...i,...i->...', r, surface_normals)

    base_half_mod          = jnp.sum(f_ij) * dtheta * dphi / 3.0 
    boundary_correction_b1 = jnp.sum(f_ij[:,0]) * dtheta * dphi / 3.0 

    return (base_half_mod * 2.0 -  boundary_correction_b1) * flux_surface.nfp
    


#==========================================================================================================================================================================================
#                                                                           d(theta,phi) extensions
#==========================================================================================================================================================================================

#----------------------------------------------------------------------------------------------------------------------------------------------------------
#                                                                          interpolating d(theta,phi) on full module grid
#---------------------------------------------------------------------------------------------------------------------------------------------------------- 
from jax_sbgeom.jax_utils import bilinear_interp
def _normalize_theta_phi_full_mod(theta : jnp.ndarray, phi : jnp.ndarray, nfp : int):
    '''
    Normalize theta and phi to [0, 1] range for full module interpolation

    Computes effectively:
        theta_norm = (theta % (2pi)) / (2pi)    
        phi_norm   = (phi % (2pi/nfp)) / (2pi/nfp)

    Parameters
    -----------
    theta : jnp.ndarray
        Poloidal angles
    phi : jnp.ndarray
        Toroidal angles
    nfp : int
        Number of field periods in the flux surface
    
    Returns
    --------
    theta_norm : jnp.ndarray
        Normalized poloidal angles
    phi_norm : jnp.ndarray  
        Normalized toroidal angles
    '''    
    return(theta % (2 * jnp.pi)) / (2 * jnp.pi), (phi % (2 * jnp.pi / nfp)) / (2 * jnp.pi / nfp)

def _interpolate_s_grid_full_mod(theta : jnp.ndarray, phi : jnp.ndarray, nfp : int, s_grid : jnp.ndarray):
    '''
    Interpolates s values on a full module grid using bilinear interpolation.
    
    The grid of s values is assumed to be a uniformly sampled full module grid: s[0,0] is (0,0). s[-1,-1] is (2pi, 2pi/nfp)

    First normalised theta, phi to the [0, 1] range (within a full module)

    Parameters
    -----------
    theta : jnp.ndarray
        Poloidal angles to interpolate at
    phi : jnp.ndarray
        Toroidal angles to interpolate at
    nfp : int
        Number of field periods in the flux surface
    s_grid : jnp.ndarray [n_theta_sampled, n_phi_sampled]
        Grid of s values to interpolate from. Assumed to be full module: i.e. phi in [0, 2pi/nfp], theta in [0, 2pi] (included endpoints)
    Returns
    --------
    s_interp : jnp.ndarray
        Interpolated s values at (theta, phi)
    '''    
    return bilinear_interp(*_normalize_theta_phi_full_mod(theta, phi, nfp), s_grid)


@partial(jax.jit)
def _cartesian_position_interpolating_s_grid_full_mod(flux_surface : ParametrisedSurface, s_grid : jnp.ndarray, theta : jnp.ndarray, phi : jnp.ndarray):    
    '''
    Compute the Cartesian position of a flux surface at interpolated s values. 
    
    The grid of s values is assumed to be a uniformly sampled full module grid: s[0,0] is (0,0). s[-1,-1] is (2pi, 2pi/nfp)
    
    If the tangent is desired, use dx_dtheta_d_varying instead of the base _dx_dtheta in flux_surface_base.py. This takes into account the ds/dtheta term.
    Parameters
    -----------
    flux_surface : ParametrisedSurface
        Flux surface to compute position on.
    s_grid : jnp.ndarray [n_theta_sampled, n_phi_sampled]
        Grid of s values to interpolate from. Assumed to be full module: i.e. phi in [0, 2pi/nfp], theta in [0, 2pi] (included endpoints)
    theta : jnp.ndarray
        Poloidal angles to compute position at
    phi : jnp.ndarray
        Toroidal angles to compute position at
    Returns
    --------
    positions : jnp.ndarray [..., 3]
        Cartesian positions at (theta, phi) with interpolated s values.        
    '''
    return flux_surface.cartesian_position(_interpolate_s_grid_full_mod(theta,phi, flux_surface.nfp, s_grid), theta, phi)

_dx_dtheta_interpolating_s_grid_full_mod = jax.jit(jnp.vectorize(jax.jacfwd(_cartesian_position_interpolating_s_grid_full_mod, argnums=2), excluded=(0,1), signature = "(),()->(3)"))

@partial(jax.jit)
def _arc_length_theta_interpolating_s_grid_full_mod(flux_surface : ParametrisedSurface, s_grid : jnp.ndarray, theta : jnp.ndarray, phi : jnp.ndarray):
    '''
    Compute the arc length derivative with respect to theta of a flux surface at interpolated s values. 
    
    The grid of s values is assumed to be a uniformly sampled full module grid: s[0,0] is (0,0). s[-1,-1] is (2pi, 2pi/nfp)
    
    Parameters
    -----------
    flux_surface : ParametrisedSurface
        Flux surface to compute position on.
    s_grid : jnp.ndarray [n_theta_sampled, n_phi_sampled]
        Grid of s values to interpolate from. Assumed to be full module: i.e. phi in [0, 2pi/nfp], theta in [0, 2pi] (included endpoints)   
    theta : jnp.ndarray
        Poloidal angles to compute arc length derivative at
    phi : jnp.ndarray
        Toroidal angles to compute arc length derivative at 
    Returns
    --------
    arc_length : jnp.ndarray
        Arc length with respect to theta at (theta, phi) with interpolated s values.        
    '''    
    return jnp.linalg.norm(_dx_dtheta_interpolating_s_grid_full_mod(flux_surface, s_grid, theta, phi), axis=-1)

@partial(jax.jit, static_argnums = (2))
def _arc_length_theta_interpolating_s_grid_full_mod_finite_difference(flux_surface : ParametrisedSurface, s_grid : jnp.ndarray, n_theta : jnp.ndarray, phi : float):
    '''
    Compute the arc length derivative with respect to theta of a flux surface at interpolated s values using finite differences. 

    The _arc_length_theta_interpolating_s_grid_full_mod function uses JAX autodiff to compute the derivative, this can be somewhat slow for especially FluxSurfaceNormalExtendedConstantPhi.
    Instead, this uses the fact that we sample anyway from a grid to compute the derivative using finite differences.
    
    The grid of s values is assumed to be a uniformly sampled full module grid: s[0,0] is (0,0). s[-1,-1] is (2pi, 2pi/nfp)
    
    Parameters
    -----------
    flux_surface : FluxSurface
        Flux surface to compute position on.
    s_grid : jnp.ndarray [n_theta_sampled, n_phi_sampled]
        Grid of s values to interpolate from. Assumed to be full module: i.e. phi in [0, 2pi/nfp], theta in [0, 2pi] (included endpoints)
    n_theta : jnp.ndarray
        Number of theta points to use for finite difference
    phi : float
        Toroidal angle to compute arc length derivative at
    Returns
    --------    
    arc_length : jnp.ndarray        
    '''
    
    theta_grid = jnp.linspace(0, 2 * jnp.pi, n_theta, endpoint=False)
    positions  = flux_surface.cartesian_position(_interpolate_s_grid_full_mod(theta_grid, phi, flux_surface.nfp, s_grid), theta_grid, phi)
    du         = theta_grid[1] - theta_grid[0]
    x          = [jnp.roll(positions, i, axis=0) for i in [0,1,2,3,4,-4,-3,-2,-1]]
    return jnp.linalg.norm(1 /280 * x[-4] + -4 / 105 * x[-3] +  1/5 * x[-2] + -4/5 * x[-1] + 4/5 * x[1] + -1/5 * x[2] + 4/105 * x[3] + -1/280 * x[4], axis=1)  / du    