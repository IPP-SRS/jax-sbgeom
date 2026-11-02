import jax 
import jax_sbgeom
from .coilset import ensure_coilset_rotation, order_coilset_phi
from . import CoilSet, Coil
from functools import partial
from typing import Literal, Union
import jax.numpy as jnp
from jax_sbgeom.jax_utils.optimize import OptimizationSettings
from jax_sbgeom.jax_utils import raytracing as jax_raytracing
@jax.jit
def _s_softplus(d_i : jnp.ndarray, minimum_distance : float = 1e-5):    
    '''
    Compute normalized arc length s in [0, 1] using softplus regularization to ensure positive segment lengths.
    Parameters
    ----------
    d_i : jnp.ndarray [n_coils, n_samples]
        Unregularized segment lengths between consecutive points along each coil.
    Returns
    -------
    s_c : jnp.ndarray [n_coils, n_samples]
        Normalized cumulative arc length along each coil [0,1] endpoints included.
    '''
    soft_plus = jax.nn.softplus(d_i)
    d = soft_plus + minimum_distance

    s_c = jnp.cumsum(d, axis=1)
    dc = s_c[:, -1] - s_c[:, 0]

    return s_c / dc[:, None]

@jax.jit
def _coil_surface_distance_loss(s_arr : jnp.ndarray, coilset : CoilSet):    
        '''
        Computes the distance between adjacent coils, sampled at s_arr.

        \\sum_{ij} (coil_i(s_j) - coil_{i+1}(s_j))^2

        Normalised by the distance between coil centres.

        Parameters
        ----------
        s_arr: jnp.ndarray [n_coils, n_s]
            Sampled arc length positions along each coil
        coilset: CoilSet
            CoilSet containing the coils
        Returns
        -------
        loss: float
            Distance loss, lower is better  

        '''
        positions   = coilset.position_different_s(s_arr[..., :-1])  # [n_coils, n_s -1 , 3] # :- 1 because the last point is the first one by definition.
        obj         = jnp.sum((positions - jnp.roll(positions, 1, axis=0))**2)         
        centre_diff = jnp.sum((coilset.centre() - jnp.roll(coilset.centre(), shift=1, axis=0  ))**2) * (s_arr.shape[1] -1) # multiplied by the number of sample points along the coil
        return obj / centre_diff

@jax.jit
def _uniformity_loss(x : jnp.ndarray):
        '''
        Computes the uniformity loss of points in x:

        \\sum_{ij} (d_{ij} - 1/(N_i -1))^2

        Parameters
        ----------
        x: jnp.ndarray [n_coils, n_samples]
            Points along each coil
        Returns
        -------
        loss: float
            Uniformity loss, lower is better        

        '''
        dx = jnp.diff(x, axis=1)
        ideal = 1.0 / (x.shape[1]-1)
        return jnp.sum(jnp.sum((dx - ideal)**2, axis=1))

def _coil_repulsion(points, p, eps):
    diff = points[:, None] - points[None, :]
    dist = jnp.abs(diff) + jnp.eye(len(points)) * 1e6
    rep = 1.0 / (dist**p + eps)
    return jnp.sum(jnp.triu(rep, k=1))


@jax.jit
def _repulsion_loss(x : jnp.ndarray, p : int = 2, eps : float = 1e-6):
        '''
        Computes a repulsion loss of points in x:

        \\sum_{i<j} 1 / (d_{ij}^p + eps) 

        It is normalised by the repulsion loss of a uniform distribution minus one, so that a uniform distribution gives 0 repulsion loss (ideal).

        Parameters
        ----------
        x: jnp.ndarray [n_coils, n_samples]
            Points along each coil
        p: int
            Power of the repulsion
        eps: float
            Small number to avoid division by zero
        Returns
        -------
        loss: float
            Repulsion loss, lower is better
        '''
        # vmap over coils, then sum
        losses = jax.vmap(_coil_repulsion, in_axes=(0, None, None))(x, p, eps)
        losses_base = _coil_repulsion(jnp.linspace(0.0,1.0, x.shape[1]), p, eps)
        
        return jnp.sum(losses) / (losses.shape[0] * losses_base) - 1.0 


@partial(jax.jit, static_argnums=(1,))
def _create_total_s(d_i : jnp.ndarray, n_coils : int):
    '''
    Create total s array from d_i vector.

    Simply reshapes the d_i vector and computes s using softplus regularization.

    Parameters
    ----------
    d_i : jnp.ndarray [n_coils * n_samples]
        Unregularized segment lengths between consecutive points along each coil.
    n_coils : int
        Number of coils.
    Returns
    -------
    s_c : jnp.ndarray [n_coils, n_samples]
        Normalized cumulative arc length along each coil, ranging from 0 to 1.        
    '''
    return _s_softplus(d_i.reshape((n_coils, -1)))


@partial(jax.jit, static_argnums=(2))
def coil_surface_loss(d_i : jnp.ndarray, coilset : CoilSet, n_coils : int, uniformity_loss_weight : float, repulsive_loss_weight : float):
    '''
    Compute total coil surface loss.

    Parameters
    ----------
    d_i : jnp.ndarray [n_coils * n_samples]
        Unregularized segment lengths between consecutive points along each coil.
    coilset : CoilSet
        CoilSet containing the coils.
    n_coils : int
        Number of coils.
    uniformity_loss_weight : float
        Weight of the uniformity loss.
    repulsive_loss_weight : float
        Weight of the repulsion loss.
    Returns
    -------
    total_loss : float
        Total coil surface loss.


    '''
    total_s_array = _create_total_s(d_i, n_coils)
    return _coil_surface_distance_loss(total_s_array, coilset) + uniformity_loss_weight * _uniformity_loss(total_s_array) + repulsive_loss_weight * _repulsion_loss(total_s_array)

def _coil_surface_loss_from_args(params, args):
    '''Loss as a function of the d_i parameters only; args = (coilset, uniformity_penalty, repulsive_penalty).'''
    coilset, uniformity_penalty, repulsive_penalty = args
    return coil_surface_loss(params, coilset, coilset.n_coils, uniformity_penalty, repulsive_penalty)

def optimize_coil_surface(coilset : CoilSet, uniformity_penalty : float = 1.0, repulsive_penalty : float = 0.1, n_samples_per_coil : int = 100, optimization_settings = OptimizationSettings(100,1e-4)): 
    '''
    Optimize the sampling points of a CoilSet for minimum distance between adjacent coils with penalties for non-uniformity and closeness of points.
    This ensures that the optimizer does not find pathological solutions where points cluster together. The CoilSet is first ordered in phi and ensured to have positive orientation.

    Parameters
    ----------
    coilset : CoilSet
        CoilSet containing the coils to optimize.
    uniformity_penalty : float
        Weight of the uniformity loss.
    repulsive_penalty : float
        Weight of the repulsion loss.
    n_samples_per_coil : int
        Number of sample points per coil.
    optimization_settings : OptimizationSettings
        Settings for the optimization process.
    Returns
    -------
    optimized_params : jnp.ndarray
        Optimized parameters for the coil surface.
    coilset_ordered_and_positive : CoilSet
        CoilSet with ordered and positively oriented coils.

    '''
    coilset_ordered_and_positive = ensure_coilset_rotation(order_coilset_phi(coilset), True)
    args                         = (coilset_ordered_and_positive, uniformity_penalty, repulsive_penalty)
    
    # Initialises with uniform spacing
    x0                           = jnp.ones(coilset.n_coils * n_samples_per_coil)        
    
    return jax_sbgeom.jax_utils.optimize.run_optimization_lbfgs(x0, _coil_surface_loss_from_args, optimization_settings, args), coilset_ordered_and_positive
    
def _cws_fourier(positions_cws : jnp.ndarray, n_points_phi : int):
    '''
    Interpolate coil winding surface positions to Fourier representation.

    Parameters
    ----------
    positions_cws : jnp.ndarray [n_points_per_coil, n_coils, 3]
        Positions of the coil winding surface mesh points.
    n_points_phi : int
        Number of points in the toroidal direction.
    
    Returns
    -------
    positions_cws_fourier : jnp.ndarray [n_points_per_coil, n_points_phi, 3]
        Positions of the coil winding surface mesh points in Fourier representation.
    '''
    fourier_coilset = jax_sbgeom.coils.CoilSet(jax_sbgeom.coils.FourierCoil(*jax_sbgeom.coils.fourier_coil.curve_to_fourier_coefficients(positions_cws))) # coilset is just 1D fourier curves.
    s_sample                    = jnp.linspace(0.0, 1.0, n_points_phi, endpoint=False)
    positions_cws_fourier       = fourier_coilset.position(s_sample)  # n_theta [n_samples_per_coil], n_phi [number_of_coils], 3
    connectivity                = jax_sbgeom.flux_surfaces.flux_surface_meshing._mesh_surface_connectivity(positions_cws_fourier.shape[0], positions_cws_fourier.shape[1], True, True)        
    return positions_cws_fourier.reshape(-1,3), connectivity

def _cws_direct(positions_cws : jnp.ndarray, n_points_phi : int):
    '''
    Create a direct coil winding surface mesh. Uses only the points on the coils themselves.

    Parameters
    ----------
    positions_cws : jnp.ndarray [n_points_per_coil, n_coils, 3]
        Positions of the coil winding surface mesh points.
    n_points_phi : int
        Number of points in the toroidal direction. Not used here.
    
    Returns
    -------
    positions_cws_fourier : jnp.ndarray [n_points_per_coil, n_coils, 3]
        Positions of the coil winding surface mesh points in Fourier representation.
    '''
    connectivity                = jax_sbgeom.flux_surfaces.flux_surface_meshing._mesh_surface_connectivity(positions_cws.shape[0], positions_cws.shape[1], True, True)
    return positions_cws.reshape(-1, 3), connectivity

def _cws_spline(positions_cws : jnp.ndarray, n_points_phi : int):
    '''
    Create a interpolating spline coil winding surface mesh.

    Parameters
    ----------
    positions_cws : jnp.ndarray [n_points_per_coil, n_coils, 3]
        Positions of the coil winding surface mesh points.
    n_points_phi : int
        Number of points in the toroidal direction.
    
    Returns
    -------
    positions_cws_spline : jnp.ndarray [n_points_per_coil, n_points_phi, 3]
        Positions of the coil winding surface mesh points in spline representation.
    '''
    y         = jnp.concatenate([positions_cws, positions_cws[:, :1, :]], axis=1) # add first coil at the end to ensure periodicity
    batched_y = jnp.moveaxis(y, -1, 0)  # 3, n_points_per_coil, n_coils + 1 [we require the last axis to be the spline axis]
    
    # chord length parameterization
    t         = jnp.linalg.norm(y[:,:]-jnp.roll(y,1,axis=1), axis=-1).cumsum(axis=1) # n_points_per_coil, n_coils + 1
    t         = t / t[:,-1:]
    bspline_batch        = jax_sbgeom.jax_utils.splines.periodic_interpolating_spline(t, batched_y, k=3)
    positions_splines    = bspline_batch(jnp.linspace(0.0, 1.0, n_points_phi, endpoint=False))  #3, n_points_per_coil, n_points_phi    
    positions_cws_spline = jnp.moveaxis(positions_splines, 0, -1)  # n_points_per_coil, n_points_phi, 3 [consistent ordering again]
    
    connectivity = jax_sbgeom.flux_surfaces.flux_surface_meshing._mesh_surface_connectivity(positions_cws_spline.shape[0], positions_cws_spline.shape[1], True, True)

    return positions_cws_spline.reshape(-1,3), connectivity
    
def _create_cws_interpolated(coilset : CoilSet, n_points_per_coil : int, d_opt : jnp.ndarray):
    '''
    Sample points on the coilset using optimized d_i parameters.

    Parameters
    ----------
    coilset : CoilSet
        CoilSet containing the coils.
    n_points_per_coil : int
        Number of points per coil in the output mesh.
    d_opt : jnp.ndarray
        Optimized parameters for the coil surface.
    
    Returns
    -------
    positions_cws : jnp.ndarray [n_points_per_coil, n_coils, 3]
        Positions of the coil winding surface mesh points.
    '''

    total_s_array               = _create_total_s(d_opt, coilset.n_coils) # n_coils, n_samples_per_coil_opt

    # Sampled from 0 to 1 endpoint not included. Move axis: n_coils, n_points_per_coil, 3 -> n_points_per_coil, n_coils, 3
    # Ensures we have a consistent ordering: in flux surfaces, the first dimension is theta (similar to points along the coil), the second dimension is phi (similar to number of coils).
    s_sample                    = jnp.linspace(0.0, 1.0, n_points_per_coil, endpoint=False)    
    s_array_interpolated        = jax.vmap(jax_sbgeom.jax_utils.interpolate_array, in_axes=(0,None))(total_s_array, s_sample)
    positions_cws               = jnp.moveaxis(coilset.position_different_s(s_array_interpolated), 0, 1) # ntheta [n_samples_per_coil], nphi [number_of_coils], 3
    return positions_cws


def _create_coil_winding_surface_from_parameters(ordered_coilset : CoilSet, n_points_per_coil : int, n_points_phi : int, d_parameters : jnp.ndarray, surface_type  : Literal['spline','fourier','direct'] = 'spline'):
    '''
    Create a coil winding surface mesh from a CoilSet and optimized d_i parameters. Uses different methods to create the surface.

    Parameters
    ----------
    ordered_coilset : CoilSet
        CoilSet containing the coils to optimize.
    n_points_per_coil : int
        Number of points per coil in the output mesh.
    n_points_phi : int
        Number of points in the toroidal direction if needed.
    d_parameters : jnp.ndarray  
        Optimized parameters for the coil surface.
    surface_type : Literal['spline', 'fourier', 'direct']
        Method to create the surface:
        - "spline" uses a 3D periodic spline on each toroidal line, 
        - "fourier" uses a fourier transformation on each toroidal line
        - "direct" meshes directly between the coils (no intermediate points)
    
    Returns
    -------
    positions : jnp.ndarray [n_points, 3]
        Positions of the coil winding surface mesh points.
    connectivity : jnp.ndarray [n_faces, 3]
        Connectivity of the coil winding surface mesh.
    
    '''
    positions_cws_opt = _create_cws_interpolated(ordered_coilset, n_points_per_coil, d_parameters)    
    if surface_type == 'fourier':
        return _cws_fourier(positions_cws_opt, n_points_phi)
    elif surface_type == 'direct':
        return _cws_direct(positions_cws_opt, n_points_phi)
    elif surface_type == 'spline':
        return _cws_spline(positions_cws_opt, n_points_phi)
    else:
        raise ValueError(f"Unknown surface type: {surface_type} in _create_coil_winding_surface_from_parameters")

     

def create_optimized_coil_winding_surface(coilset : CoilSet, n_points_per_coil : int, n_points_phi : int, surface_type : Literal['spline', 'fourier', 'direct'] = "spline",
                                          uniformity_penalty : float = 1.0, repulsive_penalty : float = 0.1, n_samples_per_coil_opt : int = 100, optimization_settings = OptimizationSettings(100,1e-4)):
    '''
    Create an optimized coil winding surface mesh from a CoilSet. The CoilSet is first ordered in phi and ensured to have positive orientation.
    
    Parameters
    ----------
    coilset : CoilSet
        CoilSet containing the coils to optimize.
    n_points_per_coil : int
        Number of points per coil in the output mesh.
    n_points_phi : int
        Number of points in the toroidal direction if needed.
    surface_type : Literal['spline', 'fourier', 'direct']
        Method to create the surface:
        - "spline" uses a 3D periodic spline on each toroidal line, 
        - "fourier" uses a fourier transformation on each toroidal line
        - "direct" meshes directly between the coils (no intermediate points)
    uniformity_penalty : float
        Weight of the uniformity loss.
    repulsive_penalty : float
        Weight of the repulsion loss.
    n_samples_per_coil_opt : int
        Number of sample points per coil for the optimization.
    optimization_settings : OptimizationSettings
        Settings for the optimization process.
    Returns
    -------
    positions : jnp.ndarray [n_points, 3]
        Positions of the coil winding surface mesh points.
    connectivity : jnp.ndarray [n_faces, 3]
        Connectivity of the coil winding surface mesh.  
    '''

    optimized_params, ordered_coilset = optimize_coil_surface(
        coilset,
        uniformity_penalty,
        repulsive_penalty,
        n_samples_per_coil_opt,
        optimization_settings
    )
    return _create_coil_winding_surface_from_parameters(ordered_coilset, n_points_per_coil, n_points_phi, optimized_params[0], surface_type)

def create_coil_winding_surface(coilset : CoilSet, n_points_per_coil : int, n_points_phi : int, surface_type : Literal['spline', 'fourier', 'direct'] = 'spline'):
    '''
    Create a coil winding surface from a CoilSet. The CoilSet is first ordered in phi and ensured to have positive orientation.

        
    Parameters
    ----------
    coilset : CoilSet
        CoilSet containing the coils to optimize.
    n_points_per_coil : int
        Number of points per coil in the output mesh.
    n_points_phi : int
        Number of points in the toroidal direction if needed.
    surface_type : Literal['spline', 'fourier', 'direct']
        Method to create the surface:
        - "spline" uses a 3D periodic spline on each toroidal line, 
        - "fourier" uses a fourier transformation on each toroidal line
        - "direct" meshes directly between the coils (no intermediate points)    
    Returns
    -------
    positions : jnp.ndarray [n_points, 3]
        Positions of the coil winding surface mesh points.
    connectivity : jnp.ndarray [n_faces, 3]
        Connectivity of the coil winding surface mesh.  
    '''
    ordered_coilset = ensure_coilset_rotation(order_coilset_phi(coilset), True)         
    return _create_coil_winding_surface_from_parameters(ordered_coilset, n_points_per_coil, n_points_phi, jnp.ones(coilset.n_coils * n_points_per_coil), surface_type)

@partial(jax.jit, static_argnums=(2,))
def calculate_normals_from_closest_point_on_mesh(coil : Union[Coil, CoilSet], external_mesh, n_coil_samples : int):
    '''
    Calculate normals at coil positions by finding the closest points on an external mesh and using that normal.

    Parameters
    ----------
    coil : Union[Coil, CoilSet]
        Coil or CoilSet containing the coil(s).
    external_mesh : jax_sbgeom.jax_utils.mesh.Mesh
        External mesh to find closest points on.
    n_coil_samples : int
        Number of samples along the coil(s).
    Returns
    -------
    positions : jnp.ndarray [n_coils, n_coil_samples, 3]
        Positions along the coil(s).
    normals : jnp.ndarray [n_coils, n_coil_samples, 3]
        Normals at the coil positions.
    '''
    positions = coil.position(jnp.linspace(0,1,n_coil_samples, endpoint=False)) #[n_coils, n_coil_samples, 3]        
    closest_points, dmin, d_idx = jax_sbgeom.jax_utils.raytracing.get_closest_points_on_mesh(positions, external_mesh)    
    return positions, jax_sbgeom.jax_utils.surface_normals_from_mesh(external_mesh)[d_idx]
