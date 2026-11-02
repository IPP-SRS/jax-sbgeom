# %%
import os
#os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax_sbgeom as jsb

import jax.numpy as jnp
import jax
jax.config.update("jax_enable_x64", True)
import numpy as onp
import sys 
import os

project_root = os.path.abspath(os.path.join(os.getcwd(), ".."))
sys.path.append(project_root)


from functools import partial
import jax_sbgeom.coils as jsc
import jax_sbgeom.flux_surfaces as jsf

from dataclasses import dataclass
import functools
from jax_sbgeom.jax_utils import mesh_to_pyvista_mesh, cumulative_trapezoid_uniform_periodic, interp1d_jax
import StellBlanket.SBGeom as sbg
import StellBlanket
from StellBlanket.SBGeom import Coils_jax as CJ
import pyvista as pv

import matplotlib.pyplot as plt
import jax_sbgeom.coils.coil_winding_surface as cws
vmec_files = ["examples/data/vmecs/helias3_vmec.nc4", "examples/data/vmecs/helias5_vmec.nc4", "examples/data/vmecs/squid_vmec.nc4"]
vmec_file = vmec_files[2]

coil_files = ["examples/data/vmecs/HELIAS3_coils_all.h5", "examples/data/vmecs/HELIAS5_coils_all.h5", "examples/data/vmecs/squid_coilset.h5"]

# %%
coilsd = onp.loadtxt("filament_s4uu_5ci_23")[:,:]
amax = 101
i = 0
coils = []
currents = []
for i in range(40):
#    if i not in [0, 4, 8, 12, 16, 20,24, 28, 32,35, 36,39] :
    coils.append(coilsd[i * amax : i * amax + amax -1,:-1])
    currents.append(coilsd[i * amax, -1])



jax_coils = [jsc.DiscreteCoil.from_positions(jnp.array(coil)) for coil in coils]
jax_coilset = jsc.CoilSet.from_list(jax_coils)
jax_currents = jnp.array(currents)
print(jax_currents.shape)


# %%
lcfs = jsb.flux_surfaces.FluxSurface.from_hdf5(vmec_file)

n_theta = 51
n_phi   = 61

theta = jnp.linspace(0, 2 * jnp.pi, n_theta, endpoint=False)
phi   = jnp.linspace(0, 2 * jnp.pi, n_phi, endpoint=False)

theta_grid, phi_grid = jnp.meshgrid(theta, phi, indexing="ij")  
positions_lcfs = lcfs.cartesian_position(1.0, theta_grid, phi_grid)

# %%
nsamples_per_coil = 112
s_samples = jnp.linspace(0, 1, nsamples_per_coil, endpoint=False)
positions_coils    = jax_coilset.position(s_samples)
jax_currents_tot   = jnp.broadcast_to(jax_currents[:, None], (jax_coilset.n_coils, nsamples_per_coil))

segments = jnp.roll(positions_coils, -1, axis=1) - positions_coils



# %%
def Bfield(r):    
    return jsc.biot_savart.biot_savart(jax_currents_tot.reshape(-1), positions_coils.reshape(-1,3), segments.reshape(-1,3), r.reshape(-1,3), 64)


bfield_computed = Bfield(positions_lcfs)

# %%
plotter = pv.Plotter()
plotter.add_arrows(positions_lcfs.reshape(-1,3), bfield_computed.reshape(-1,3), mag=0.1, color="red")
plotter.show()

# %%
from diffrax import ODETerm, Dopri5, PIDController, SaveAt, diffeqsolve, Tsit5

# %%
def Bfield_single(r):    
    return jsc.biot_savart.biot_savart_single(jax_currents_tot.reshape(-1), positions_coils.reshape(-1,3), segments.reshape(-1,3), r)



# %%
def B_field_diffrax(t, r, args= None):
    B = Bfield_single(r)
    drdt = B / jnp.linalg.norm(B)
    return drdt


term = ODETerm(B_field_diffrax)

# %%
r0 = positions_lcfs[0,0]
solver = Dopri5()
t0 = 0.0
t1 = 5000.0 
dt0 = 1.0

# %%
saveat = SaveAt(ts=jnp.linspace(t0, t1, 1000))

# %%
initial_position = lcfs.cartesian_position(0.6467, 0.0,0.0)

# %%
sol = diffeqsolve(term, solver, t0= t0, t1=t1, dt0=dt0, y0=initial_position, saveat=saveat, max_steps=12000)

# %%
print(sol.ys)

# %%
plt.plot(jnp.arctan2(sol.ys[:,1], sol.ys[:,0]))

# %%
mesh_thing = jsb.flux_surfaces.flux_surface_meshing.mesh_surface(lcfs, 0.6467, jsb.flux_surfaces.ToroidalExtent.full(), 100, 200)


# %%
plotter = pv.Plotter()

finitesize_coilset = jsc.FiniteSizeCoilSet.from_coilset(jax_coilset, jsc.RotationMinimizedFrame, 100)


base_mesh = mesh_to_pyvista_mesh(*jsc.mesh_coilset_surface(finitesize_coilset, 100, 0.2, 0.2))
plotter.add_mesh(pv.Spline(sol.ys))
plotter.add_mesh(base_mesh)
plotter.add_mesh(mesh_to_pyvista_mesh(*mesh_thing))
plotter.show()


