import jax.numpy as jnp
import jax
# ===================================================================================================================================================================================
#                                                                           Interpolation of arrays
# ===================================================================================================================================================================================

#------------------------------------------------------------------------------------------------------------------------------------------------------------------
# uniform interpolation
#------------------------------------------------------------------------------------------------------------------------------------------------------------------
def interpolate_fractions(s, nsurf):
    '''
    Interpolate fractions for uniform sampling

    Parameters
    ----------
    s : jnp.ndarray
        Normalized parameter(s) between 0 and 1
    nsurf : int
        Number of samples
    Returns
    -------
    i0 : jnp.ndarray
        Lower indices
    i1 : jnp.ndarray
        Upper indices
    ds : jnp.ndarray
        Fraction between i0 and i1
    '''
    s_start =  s * (nsurf-1)
    # Guard against floating-point noise pushing s_start just below an intended
    # integer boundary (e.g. 57.999996 instead of 58.0) -- this happens whenever s
    # is constructed to land exactly on a grid point, e.g. via
    # jnp.linspace(0, 1, nsurf, endpoint=False), and floor() would otherwise
    # silently select the wrong (previous) index. The tolerance scales with the
    # dtype's precision and nsurf, since the absolute error in s*nsurf grows with
    # the magnitude of the result.
    tol     = 8 * jnp.finfo(s_start.dtype).eps * nsurf
    i0      = jnp.floor(s_start + tol).astype(int)
    i1      = jnp.minimum(i0 + 1, nsurf - 1)
    ds      = jnp.clip(s_start - i0, 0.0, 1.0)
    return i0, i1, ds

def interpolate_array(x_interp, s):
    '''
    Interpolate array for uniform sampling

    Parameters
    ----------
    x_interp : jnp.ndarray [1D]
        Array to interpolate
    s : jnp.ndarray [1D]
        Normalized parameter(s) between 0 and 1
    Returns
    -------
    jnp.ndarray
        Interpolated array
    '''
    nsurf = x_interp.shape[0]
    i0, i1, ds   = interpolate_fractions(s, nsurf)
    x0 = x_interp[i0]
    x1 = x_interp[i1]
    return (1 - ds) * x0 + ds * x1

def interpolate_fractions_modulo(s, nsurf):
    '''
    Interpolate fractions for uniform sampling with modulo wrapping
    I.e., s=1 maps to index 0 again.

    Parameters
    ----------
    s : jnp.ndarray
        Normalized parameter(s) between 0 and 1
    nsurf : int
        Number of samples
    Returns
    -------
    i0 : jnp.ndarray
        Lower indices
    i1 : jnp.ndarray
        Upper indices
    ds : jnp.ndarray
        Fraction between i0 and i1
    '''
    s_start =  s * nsurf
    # See interpolate_fractions for why the tolerance is needed: s is routinely
    # constructed to land exactly on a grid point (e.g. jnp.linspace(0, 1, nsurf,
    # endpoint=False)), and without it floor() can silently select the wrong
    # (previous) index due to floating-point rounding of s*nsurf just below the
    # intended integer.
    tol     = 8 * jnp.finfo(s_start.dtype).eps * nsurf
    i0      = jnp.floor(s_start + tol).astype(int)
    i1      = i0 + 1
    ds      = jnp.clip(s_start - i0, 0.0, 1.0)
    return i0, i1, ds

def interpolate_array_modulo(x_interp, s):
    '''
    Interpolate array for uniform sampling with modulo wrapping
    I.e., s=1 maps to index 0 again.    
    Parameters
    ----------
    x_interp : jnp.ndarray [1D]
        Array to interpolate
    s : jnp.ndarray [1D]
        Normalized parameter(s) between 0 and 1
    Returns
    -------
    jnp.ndarray
        Interpolated array
    '''
    nsurf = x_interp.shape[0]
    i0, i1, ds   = interpolate_fractions_modulo(s, nsurf)
    x0 = x_interp[i0 % nsurf]
    x1 = x_interp[i1 % nsurf]
    return (1 - ds) * x0 + ds * x1


def interpolate_array_modulo_broadcasted(x_interp, s):
    '''
    Interpolate array for uniform sampling with modulo wrapping
    I.e., s=1 maps to index 0 again.
    This version supports broadcasting of s to higher dimensions.   
    Parameters
    ----------
    x_interp : jnp.ndarray [s.shape, interpolation_dimension, :]
        Array to interpolate
    s : jnp.ndarray [s.shape]
        Normalized parameter(s) between 0 and 1
    Returns
    -------
    jnp.ndarray[s.shape, :]
        Interpolated array
    '''
    nsurf = x_interp.shape[-2]
    i0, i1, ds   = interpolate_fractions_modulo(s, nsurf)
    x0 = x_interp[..., i0 % nsurf, :] # s_shape, something
    x1 = x_interp[..., i1 % nsurf, :] # s_shape, something

    return (1 - ds[..., None]) * x0 + ds[..., None] * x1


def reverse_except_begin(array : jnp.ndarray):
    return jnp.concatenate([array[0:1], array[1:][::-1,]], axis=0)

@jax.jit
def bilinear_interp(norm_array_0, norm_array_1, interpolate_array):
    '''
    Bilinear interpolation for uniform sampling in 2D
    It is assumed that interpolate_array is defined on a uniform grid normalised to 1 in both dimensions.

    Parameters
    ----------
    norm_array_0 : jnp.ndarray [shape]
        Normalized parameter(s) between 0 and 1 in first dimension
    norm_array_1 : jnp.ndarray [shape]
        Normalized parameter(s) between 0 and 1 in second dimension
    interpolate_array : jnp.ndarray [N0, N1]
        Array to interpolate
    Returns
    -------
    jnp.ndarray [shape]
        Interpolated array
    '''
    i0, i1, ds = interpolate_fractions(norm_array_0, interpolate_array.shape[0]) 
    j0, j1, dp = interpolate_fractions(norm_array_1, interpolate_array.shape[1])

    f_00 = interpolate_array[i0,j0]
    f_10 = interpolate_array[i1,j0]
    f_01 = interpolate_array[i0,j1]
    f_11 = interpolate_array[i1,j1]

    interp_vals = (1.0 - ds) * (1.0 - dp) * f_00 + \
                  ds         * (1.0 - dp) * f_10 + \
                  (1.0 - ds) * dp         * f_01 + \
                  ds         * dp         * f_11
    return interp_vals


#------------------------------------------------------------------------------------------------------------------------------------------------------------------
# non-uniform interpolation
#------------------------------------------------------------------------------------------------------------------------------------------------------------------

def interp1d_jax(x, y, x_new):    
    '''
    Simple 1D linear interpolation function in JAX

    Extrapolates flat with the boundary values.

    Parameters
    ----------
    x : jnp.ndarray [N]
        x-coordinates of the data points
    y : jnp.ndarray [N]
        y-coordinates of the data points
    x_new : jnp.ndarray [M]
        x-coordinates where to interpolate
    Returns
    -------
    jnp.ndarray [M]
        Interpolated y-coordinates at x_new
    '''
    i = jnp.clip(jnp.searchsorted(x, x_new, side='left') - 1, 0, x.size - 2)    
    x0, x1 = x[i], x[i + 1]
    y0, y1 = y[i], y[i + 1]    
    t = (x_new - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)

def _pchip_derivatives(x, y):
    '''
    Piecewise cubic Hermite interpolating polynomial (PCHIP) derivatives in JAX
    Computes the derivatives at the data points needed for evaluation of PCHIP.

    Parameters
    ----------
    x : jnp.ndarray [N]
        x-coordinates of the data points
    y : jnp.ndarray [N]
        y-coordinates of the data points
    Returns
    -------
    jnp.ndarray [N]
        PCHIP Derivatives at the data points
    '''
    # See scipy.interpolate._cubic.PchipInterpolator._find_derivatives and interpax.fd_derivs.py [monotonic]
    # Adapted to ensure no NaNs are propagating in reverse mode (using safe 1/x where x can be zero by setting x to one and jnp where later)
    
    h     = jnp.diff(x)    
    mk    = (y[1:] - y[:-1]) / h
        
    sign_mk  = jnp.sign(mk)
    condition = (sign_mk[1:] != sign_mk[:-1]) | (mk[1:] == 0.0) | (mk[:-1] == 0.0)
    w1 = 2 * h[1:] +     h[:-1] # these are nonzero and positive
    w2 =     h[1:] + 2 * h[:-1] # these are nonzero and positive
    
    mk_condition            = jnp.where(mk == 0.0, jnp.ones_like(mk), mk)
    weighted_mean_condition = jnp.where(condition, jnp.ones_like(w1), (w1 / mk_condition[:-1] + w2 / mk_condition[1:])/(w1 + w2))        
    dk                      = jnp.where(condition, 0.0, 1.0 / weighted_mean_condition)

    def edge_case(h0, h1, m0, m1):
        d = ((2 * h0 + h1) * m0 - h0 * m1) / (h0 + h1)
        # d is a scalar here!
        d = jax.lax.cond(jnp.sign(d) != jnp.sign(m0), lambda d: 0.0, lambda d: d, d)
        d = jax.lax.cond((jnp.sign(m0) != jnp.sign(m1)) & (jnp.abs(d) > jnp.abs(3 * m0)), lambda d: 3 * m0, lambda d: d, d)
        return jnp.array([d])
    
    d0 = edge_case(h[0], h[1], mk[0], mk[1])
    dn = edge_case(h[-1], h[-2], mk[-1], mk[-2])    
    return jnp.concatenate([d0, dk, dn])
    

def _pchip_evaluation(x, y, dxdy, x_eval):
    # Find interval indices for each x_eval
    idx = jnp.clip(jnp.searchsorted(x, x_eval) - 1, 0, len(x)-2)
    h   = jnp.diff(x)    
    xi  =    x[idx]    
    yi  =    y[idx]
    yi1 =    y[idx+1]
    mi  = dxdy[idx]
    mi1 = dxdy[idx+1]
    hi  =    h[idx]
    
    t   = (x_eval - xi) / hi
    t2  = t*t
    t3  = t2*t
    
    h00 =  2 * t3 - 3 * t2 + 1
    h10 =      t3 - 2 * t2 + t
    h01 = -2 * t3 + 3 * t2
    h11 =      t3 -     t2
    
    return h00*yi + h10*hi*mi + h01*yi1 + h11*hi*mi1

def pchip_interpolation(x,y, x_new):
    '''
    Piecewise cubic Hermite interpolating polynomial (PCHIP) interpolation in JAX

    Convenience function: simply calls _pchip_derivatives and _pchip_evaluation
    If you need derivatives or evaluation, you can jax.grad and vectorize the _pchip_evalution function.
    If you need multiple calls with the same x,y data, compute the derivatives once with _pchip_derivatives and pass them to _pchip_evaluation.
    If you need to do this for many different x,y datasets, consider vmap'ing the _pchip_derivatives function.

    Parameters
    ----------
    x : jnp.ndarray [N]
        x-coordinates of the data points
    y : jnp.ndarray [N]
        y-coordinates of the data points
    x_new : jnp.ndarray [M]
        x-coordinates where to interpolate
    Returns
    -------
    jnp.ndarray [M]
        Interpolated y-coordinates at x_new
    '''
    dxdy = _pchip_derivatives(x, y)
    return _pchip_evaluation(x, y, dxdy, x_new)

# ===================================================================================================================================================================================
#                                                                           Integration
# ===================================================================================================================================================================================


def cumulative_trapezoid_uniform_periodic(y : jnp.ndarray, dx : float, initial : float =0.0):

    """
    Cumulative trapezoidal integration of y with respect to x, assuming uniform spacing and periodicity.

    The y is to be sampled at uniform intervals in x, with spacing dx and not including the endpoint.
    i.e., jnp.linspace(0, period, n_samples, endpoint=False)

    Parameters
    ----------
    y : jnp.ndarray [N]
        Values to integrate
    dx : float
        Spacing between samples in x
    initial : float
        Initial value of the integral
    Returns
    -------
    jnp.ndarray [N+1]
        Cumulative integral of y with respect to x
    """    
    integrand = (y + jnp.roll(y, -1)) / 2
    integral = jnp.cumsum(integrand * dx)
    return jnp.concatenate([jnp.array([initial]), integral])


def _get_normalized_cumlative_values(non_uniform_values : jnp.ndarray):
    '''
    Get normalized cumulative values. Assumes non_uniform_values are periodic and sampled on a uniform grid.

    Parameters
    ----------
    non_uniform_values : jnp.ndarray [N]
        Non-uniform values to compute cumulative values for
    Returns
    -------
    jnp.ndarray [N + 1]
        Normalized cumulative values
    jnp.ndarray [N + 1]
        Uniformly sampled s values

    '''
    s_sampled_uniform                = jnp.linspace(0.0, 1.0, non_uniform_values.shape[0] + 1, endpoint=True)    
    cumulative_values                = cumulative_trapezoid_uniform_periodic(non_uniform_values, s_sampled_uniform[1] - s_sampled_uniform[0], initial=0.0)    
    normalized_cumulative_values     = cumulative_values / cumulative_values[-1]    
    return normalized_cumulative_values, s_sampled_uniform

def resample_uniform_periodic_linear(non_uniform_values : jnp.ndarray, n_points_desired : int):
    '''
    Resample non-uniform values to uniform values using linear interpolation. The input values are assumed periodic and sampled on a uniform grid.

    Parameters
    ----------
    non_uniform_values : jnp.ndarray [N]
        Non-uniform values to resample
    n_points_desired : int
        Number of desired uniformly sampled points
    Returns
    -------
    jnp.ndarray [n_points_desired]
        Resampled uniform values
    '''
    normalized_cumulative_values, s_sampled_uniform = _get_normalized_cumlative_values(non_uniform_values)            
    return interp1d_jax(normalized_cumulative_values, s_sampled_uniform, jnp.linspace(0.0, 1.0, n_points_desired, endpoint=False))    

def resample_uniform_periodic_pchip(non_uniform_values : jnp.ndarray, n_points_desired : int):
    '''
    Resample non-uniform values to uniform values using PCHIP interpolation. The input values are assumed periodic and sampled on a uniform grid.

    Parameters
    ----------
    non_uniform_values : jnp.ndarray [N]
        Non-uniform values to resample
    n_points_desired : int
        Number of desired uniformly sampled points
    Returns
    -------
    jnp.ndarray [n_points_desired]
        Resampled uniform values
    '''
    normalized_cumulative_values, s_sampled_uniform = _get_normalized_cumlative_values(non_uniform_values)                
    return pchip_interpolation(normalized_cumulative_values, s_sampled_uniform, jnp.linspace(0.0, 1.0, n_points_desired, endpoint=False))