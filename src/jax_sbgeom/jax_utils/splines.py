import jax
import jax.numpy as jnp
from functools import partial
from dataclasses import dataclass
import equinox as eqx

@partial(jax.jit, static_argnums = (3,))
def _cox_de_boor(x : jnp.ndarray, t : jnp.ndarray, i : int, k : int) -> jnp.ndarray:
    """
    Cox-de Boor recursive definition of B-splines. 
    Since k is fixed, the recursion depth is fixed and can be jitted.

    Parameters
    ----------
    x : jnp.ndarray [M]
        Points at which to evaluate the basis function.
    t : jnp.ndarray [N]
        Knot vector
    i : int  [0,..,N-k-2]
        Knot index
    k : int
        Degree of the basis function.
    Returns 
    -------
    jnp.ndarray [M]
        Values of the basis function at points x

    """


@partial(jax.jit, static_argnums = (3,4))
def _cox_de_boor_gradients(x, t, i, k, derivative):
    """
    Cox-de Boor recursive definition of B-splines. 
    To jit this function, the recursion depth must be fixed. Therefore, the degree k and derivative order are static arguments
    and standard python control flow is used.

    Parameters
    ----------
    x : jnp.ndarray [M]
        Points at which to evaluate the basis function.
    t : jnp.ndarray [N]
        Knot vector
    i : int  [0,..,N-k-2]
        Knot index
    k : int
        Degree of the basis function.
    derivative : int
        Order of the derivative to compute
    Returns 
    -------
    jnp.ndarray [M]
        Values of the basis function at points x

    """

    if k == 0:
        return jnp.where((t[i] <= x) & (x < t[i+1]), 1.0, 0.0) if derivative == 0 else jnp.zeros_like(x)
    else:
        denom1 = t[i+k] - t[i]
        denom2 = t[i+k+1] - t[i+1]

        if derivative == 0:
            term1 = jnp.where(denom1 != 0, (x - t[i]) / denom1 * _cox_de_boor_gradients(x, t, i, k-1, 0), 0.0)
            term2 = jnp.where(denom2 != 0, (t[i+k+1] - x) / denom2 * _cox_de_boor_gradients(x, t, i+1, k-1, 0), 0.0)
            return term1 + term2
        else:
            term1 = jnp.where(denom1 != 0, k / denom1 * _cox_de_boor_gradients(x, t, i, k-1, derivative-1), 0.0)
            term2 = jnp.where(denom2 != 0, -k / denom2 * _cox_de_boor_gradients(x, t, i+1, k-1, derivative-1), 0.0)
            return term1 + term2

@partial(jax.jit, static_argnums = (3, 4))    
def bspline(x : jnp.ndarray, t : jnp.ndarray, c : jnp.ndarray, k : int, derivative : int):    
    '''
    Evaluate a B-spline at points x.

    Note that it is required that c.shape[0] == t.shape[0] - k - 1

    Parameters
    ----------
    x : jnp.ndarray [M]
        Points at which to evaluate the B-spline.
    t : jnp.ndarray [N]
        Knot vector
    c : jnp.ndarray [N - k - 1]
        Coefficients of the B-spline basis functions.
    k : int
        Degree of the B-spline.
    derivative : int
        Order of the derivative to compute
    Returns 
    -------
    jnp.ndarray [M]
        Values of the B-spline at points x
    '''
    return jnp.dot(c, jax.vmap(_cox_de_boor_gradients, in_axes=(None, None, 0, None, None))(x,t, jnp.arange(t.shape[0] - k - 1), k, derivative))
    

def _periodic_knots(x : jnp.ndarray, k :int):
    '''
    Creates knot vector for periodic B-spline interpolation.
    See scipy.interpolate.BSpline._periodic_knots for reference.

    Parameters
    ----------
    x : jnp.ndarray [n]
        Data points to be interpolated.
    k : int
        Degree of the B-spline.
    Returns 
    -------
    jnp.ndarray [n + 2*k]
        Knot vector for periodic B-spline interpolation.
    '''

    xc = x
    n = xc.shape[0]
    if k%2 == 0:
        dx = jnp.diff(xc)
        xc = xc.at[1:-1].set(xc[1:-1] - 0.5 * (dx[:-1] ))
    dx = jnp.diff(xc)
    t = jnp.zeros(n + 2  * k)
    t = t.at[k:-k].set(xc)
    for i in range(k):
        t = t.at[k - i - 1].set(t[k - i] - dx[-(i % (n - 1)) - 1])
        t = t.at[-k+i].set(t[-k + i - 1] + dx[i % (n - 1)])        
    return t

bspline_vectorized = jax.jit(jnp.vectorize(bspline, excluded = (3,4), signature = '(k),(n),(m)->(k)'), static_argnums=(3,4))


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class BSpline:
    '''
    Convenience class for representing an arbitrarily batched B-spline.

    Data
    ----------
    t : jnp.ndarray [..., N]
        Knot vector
    c : jnp.ndarray [..., N - k - 1]
        Coefficients of the B-spline basis functions.
    k : int
        Degree of the B-spline.    
    '''
    t : jnp.ndarray
    c : jnp.ndarray
    k : int
    def __call__(self, x : jnp.ndarray, derivative : int = 0) -> jnp.ndarray:
        '''
        Evaluate the B-spline at points x with derivative order.
        Can be arbitrarily batched, but x is not batched and evaluated for the same x points for all splines.
        '''
        return bspline_vectorized(x, self.t, self.c, int(self.k), derivative)

@partial(jax.jit, static_argnums = (2,))
def _fit_periodic_interpolating_spline(x,y,k:int):
    '''
    Fit a periodic interpolating B-spline to data points (x,y) of degree k.
    The resulting spline satisfies S(x[i]) = y[i] for all i and is periodic (including derivatives up to < k).

    Currently construct a fully dense matrix and directly solves the linear system. See scipy.interpolate._bsplines._make_periodic_spline 
    for a more advanced implementation.

    This matrix is:
    A_{ij} = B_{j}^0(x[i]) for i < n-1 (periodic last point not used)
    A_{n-1 + m, j} = B_{j}^{(m)}(x[0]) - B_{j}^{(m)}(x[n-1]) for m in 0,..,k-1 (periodic derivative conditions)
    
    where m is the derivative order and B_{j}^{(m)} is the m-th derivative of the j-th B-spline basis function (see _cox_de_boor_gradients).


    Parameters
    ----------
    x : jnp.ndarray [n]
        Data points to be interpolated.
    y : jnp.ndarray [n]
        Data values to be interpolated.
    k : int
        Degree of the B-spline.
    Returns 
    -------
    t_knots : jnp.ndarray [n + 2*k]
        Knot vector for periodic B-spline interpolation.
    c_coeff : jnp.ndarray [n + k - 1]
        Coefficients of the B-spline basis functions.

    '''
    t_knots  = _periodic_knots(x, k)    
    nt       = len(t_knots) - k - 1        
    total_AA = jax.vmap(_cox_de_boor_gradients, in_axes = (None, None, 0, None, None))(x[:-1], t_knots, jnp.arange(nt), k, 0)

    rhs = jnp.concatenate([y[:-1], jnp.zeros(k)])
    derivs = []
    for m in range(k):
        derivs.append(jax.vmap(_cox_de_boor_gradients, in_axes=(None, None, 0, None, None))(jnp.array([x[0], x[-1]]), t_knots, jnp.arange(nt), k, m))    
    A_total = jnp.zeros((nt,nt))
    A_total = A_total.at[:-k, :].set(total_AA.T)
    for i in range(k):
        A_total = A_total.at[-i-1, :].set(derivs[i][:, -1] - derivs[i][:, 0])                
    c_coeff =  jnp.linalg.solve(A_total, rhs)
    return t_knots, c_coeff

_fit_periodic_interpolating_spline_vec = jax.jit(jnp.vectorize(_fit_periodic_interpolating_spline, signature = '(n),(n)->(m),(k)', excluded=(2,)), static_argnums=(2,))

    
@partial(jax.jit, static_argnums = (2,))
def periodic_interpolating_spline(x,y, k : int):   
    '''
    Fit a periodic interpolating B-spline to batched data points (x,y) of degree k.
    The resulting spline satisfies S(x[i]) = y[i] for all i and is periodic (including derivatives up to < k).

    Parameters
    ----------
    x : jnp.ndarray [..., n]
        Data points to be interpolated.
    y : jnp.ndarray [..., n]
        Data values to be interpolated.
    k : int
        Degree of the B-spline.
    Returns 
    -------
    BSpline
        Fitted periodic interpolating B-spline.

    '''
    return BSpline(*_fit_periodic_interpolating_spline_vec(x,y, k), k)

    
#=======================================================================
# BSpline for paramatrization
#=======================================================================
def make_periodic_knots(n: int, k: int, period: float = 2 * jnp.pi) -> jnp.ndarray:
    """
    Build a periodic knot vector for n control points, degree k,
    on the domain [0, period).
    This is different than _periodic_knots, which builds a knot vector for periodic interpolation of given data points.
    This function is for building a knot vector for periodic B-spline parametrization.


    Parameters
    ----------
    n : int
        Number of control points.
    k : int
        Degree of the B-spline.
    period : float, optional
        Period of the B-spline. Default is 2*pi.

    Returns
    -------
    jnp.ndarray [n + 2*k + 1]
        Knot vector for periodic B-spline interpolation.
    """
    base = jnp.linspace(0, period, n + 1)
    left  = base[-k-1:-1] - period
    right = base[1:k+1]   + period
    return jnp.concatenate([left, base, right])

def make_periodic_coeffs(c: jnp.ndarray, k: int) -> jnp.ndarray:
    """
    Extend n coefficients to n+k by wrapping the first k,
    enforcing periodicity.

    Parameters
    ----------
    c : jnp.ndarray [n]
        Coefficients of the B-spline basis functions.
    k : int
        Degree of the B-spline.
    
    Returns
    -------
    jnp.ndarray [n + k]
        Extended coefficients for periodic B-spline interpolation.
    """
    return jnp.concatenate([c, c[:k]])

@eqx.filter_jit
def periodic_bspline(theta: jnp.ndarray, c: jnp.ndarray, k: int, period: float = 2 * jnp.pi, derivative: int = 0):
    """
    Function to evaluate a periodic B-spline at angles theta given coefficients c and degree k.

    Note that the coefficients c are the free parameters, which do not map directly to a control point
    given by their index. Use :func`greville_abscissa_periodic_bspline` to compute the control points location.

    Parameters
    ----------
    theta : jnp.ndarray [M]
        Angles at which to evaluate the B-spline, in radians.
    c : jnp.ndarray [n]
        Coefficients of the B-spline basis functions. Must have length n, which is the number of free parameters. Internally extended to n+k for periodicity.
    k : int
        Degree of the B-spline.
    period : float, optional
        Period of the B-spline. Default is 2*pi.
    derivative : int, optional
        Order of the derivative to compute. Default is 0 (the function itself).

    Returns
    -------
    jnp.ndarray [M]
        Values of the periodic B-spline (or its derivative) at the angles theta.

    """
    n = c.shape[0]
    t = make_periodic_knots(n, k, period)
    c_ext = make_periodic_coeffs(c, k)
    return bspline(jnp.mod(theta, period), t, c_ext, k, derivative)

@eqx.filter_jit
def greville_abscissa_periodic_bspline(n: int, k: int, period: float = 2 * jnp.pi) -> jnp.ndarray:
    t = make_periodic_knots(n, k, period)
    theta_ctrl = jnp.array([jnp.mean(t[i:i+k+2]) for i in range(n)])
    return jnp.mod(theta_ctrl, period)

@eqx.filter_jit
def periodic_bspline_2d(theta, phi, c: jnp.ndarray, k_theta: int, k_phi: int,
                        period_theta: float = 2 * jnp.pi, period_phi: float = 2 * jnp.pi):
    """
    Evaluate a 2D tensor-product periodic B-spline at a single point (theta, phi).

    Parameters
    ----------
    theta : scalar
        Poloidal angle, in radians.
    phi : scalar
        Toroidal angle, in radians.
    c : jnp.ndarray [n_theta, n_phi]
        Coefficient matrix.
    k_theta : int
        Degree in the theta direction.
    k_phi : int
        Degree in the phi direction.
    period_theta : float, optional
        Period in the theta direction. Default is 2*pi.
    period_phi : float, optional
        Period in the phi direction. Default is 2*pi.

    Returns
    -------
    scalar
        Value of the spline at (theta, phi).
    """
    h = eqx.filter_vmap(periodic_bspline, in_axes=(None, 0, None, None))(phi, c, k_phi, period_phi)
    return periodic_bspline(theta, h, k_theta, period_theta)

@eqx.filter_jit
def periodic_bspline_2d_vec(theta, phi, c: jnp.ndarray, k_theta: int, k_phi: int,
                            period_theta: float = 2 * jnp.pi, period_phi: float = 2 * jnp.pi):
    """
    Vectorized version of periodic_bspline_2d for evaluating at multiple (theta, phi) points.

    Just uses jnp.vectorize under the hood; this function wraps over it to ensure the correct number of arguments 
    are always passed.

    Parameters
    ----------
    theta : jnp.ndarray [...]
        Poloidal angles, in radians.
    phi : jnp.ndarray [...]
        Toroidal angles, in radians.
    c : jnp.ndarray [n_theta, n_phi]
        Coefficient matrix.
    k_theta : int
        Degree in the theta direction.
    k_phi : int
        Degree in the phi direction.
    period_theta : float, optional
        Period in the theta direction. Default is 2*pi.
    period_phi : float, optional
        Period in the phi direction. Default is 2*pi.

    Returns
    -------
    jnp.ndarray [...]
        Values of the spline at each (theta[i], phi[j]).
    """
    return jnp.vectorize(periodic_bspline_2d, signature='(),()->()', excluded=(2,3,4,5,6))(theta, phi, c , k_theta, k_phi, period_theta, period_phi)
