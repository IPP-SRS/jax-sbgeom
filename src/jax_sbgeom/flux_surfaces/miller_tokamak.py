from .flux_surfaces_base import ParametrisedSurface, _cylindrical_to_cartesian, _principal_curvatures_from_values
import jax.numpy as jnp
import equinox as eqx

class MillerTokamak(ParametrisedSurface):

    rminor : jnp.ndarray
    rmajor : jnp.ndarray
    elongation : jnp.ndarray
    triangularity : jnp.ndarray

    def cylindrical_position(self, s, theta, phi):

        s,theta,phi = jnp.broadcast_arrays(s,theta,phi)
        
        return jnp.stack([
            self.rmajor + self.rminor * jnp.cos(theta + jnp.arcsin(self.triangularity) * jnp.sin(theta)) * s,
            self.rminor * self.elongation * jnp.sin(theta) * s,
            phi
        ], axis=-1)
    
    def cartesian_position(self, s, theta, phi):
        return _cylindrical_to_cartesian(self.cylindrical_position(s, theta, phi))

    def normal(self, s, theta, phi):
        return normal_miller(self, s, theta, phi)

    def principal_curvatures(self, s, theta, phi):
        return principal_curvatures_miller(self, s, theta, phi)


import jax
from jax_sbgeom.jax_utils import stack_jacfwd

@eqx.filter_jit
def _dx_dtheta_dphi_miller(self, s, theta, phi):
    return  jnp.vectorize(stack_jacfwd(self.cartesian_position, argnums=(1,2)), excluded=(), signature='(),(),()->(3,2)')(s, theta, phi)

@eqx.filter_jit
def _second_derivatives_miller(self, s, theta, phi):
    return jax.jit(jnp.vectorize(stack_jacfwd(stack_jacfwd(self.cartesian_position, argnums=(1,2)), argnums=(1,2)), excluded=(), signature='(),(),()->(3,2,2)'))(s, theta, phi)

@eqx.filter_jit
def normal_miller(self, s, theta, phi):

    dx_dtheta_dphi = _dx_dtheta_dphi_miller(self, s, theta, phi)
    normals = jnp.cross(dx_dtheta_dphi[...,1], dx_dtheta_dphi[...,0])
    return normals / jnp.linalg.norm(normals, axis=-1, keepdims=True)


@eqx.filter_jit
def principal_curvatures_miller(self, s, theta, phi):
    dX_dtheta_and_dX_dphi                        = _dx_dtheta_dphi_miller(self, s, theta, phi)
    d2X_dtheta2_and_d2X_dthetadphi_and_d2X_dphi2 = _second_derivatives_miller(self, s, theta, phi)

    return _principal_curvatures_from_values(dX_dtheta_and_dX_dphi, d2X_dtheta2_and_d2X_dthetadphi_and_d2X_dphi2)


