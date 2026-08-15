import jax
import jax_sbgeom
import jax.numpy as jnp 
from jax_sbgeom.flux_surfaces import FluxSurface
from jax_sbgeom.coils import FiniteSizeCoilSet
from .blanket_creation import LayeredDiscreteBlanket, DiscreteFiniteSizeCoilSet
import numpy as onp
from typing import List

def create_dagmc_surface_mesh(discrete_blanket : LayeredDiscreteBlanket, flux_surface : FluxSurface, material_names  : List[str], finitesize_coilset : FiniteSizeCoilSet = None, discrete_coilset : DiscreteFiniteSizeCoilSet = None, coil_material_name : str = None):
    
    '''
    Create a DAGMC blanket geometry from a layered discrete blanket and a flux surface.

    Requires PyDAGMC and PyMOAB to be installed. These are not required dependencies for jax-sbgeom, so this function is only available when those packages are installed.    

    Note for writing the result to a .h5m file MOAB should have been compiled with HDF5 support.

    Uses a conversion factor from meter to centimeter (all positions multiplied by 100)

    Parameters
    ----------
    discrete_blanket : LayeredDiscreteBlanketet
        The layered discrete blanket to create the DAGMC geometry for.
    flux_surface : FluxSurface
        The flux surface from which to create the blanket. Expected to be a FluxSurfaceFourierExtended instance (where d = 2.0 is the first external surce, and then etc.)
    material_names : List[str]
        The names of the materials for each blanket layer.
    finitesize_coilset : FiniteSizeCoilSet, optional
        The finite size coilset to include in the DAGMC geometry. If None, no coils are included. Default is None.
    discrete_coilset : DiscreteFiniteSizeCoilSet, optional
        The discrete coilset parameters for the finite size coilset. Required if finitesize_coilset is provided. Default is None.
    coil_material_name : str or List[str], optional
        The name of the material for the coils. If a list is provided, it must have the same length as the number of coils in the finitesize_coilset after truncating to the phi range in discrete_coilset. Required if finitesize_coilset is provided. Default is None.        

    Returns
    -------
    dagmc_blanket : DAGMCModel
        The created DAGMC blanket geometry.
    '''
    try:
        import pymoab
        from pymoab import core, types        
    except ImportError as e:                
        raise ImportError(str(e) + "\n\nPyMOAB is required to create a DAGMC blanket geometry. Please install PyMOAB to use this function.")
    try:
        import pydagmc
    except ImportError as e:        
        raise ImportError(str(e) + "\n\nPyDAGMC is required to create a DAGMC blanket geometry. Please install PyDAGMC to use this function.")
    
    
    def create_surface(moab_core, moab_vertices, surface_connectivity):
        moab_connectivity = moab_vertices[surface_connectivity]
        return moab_core.create_elements(types.MBTRI, moab_connectivity)
    
    assert len(material_names) == discrete_blanket.n_physical_layers - 1, "Number of material names must be equal to number of blanket layers but got {} names for {} layers".format(len(material_names), discrete_blanket.n_physical_layers - 1)
    
    surfaces = discrete_blanket.surface_mesh(flux_surface)
    # surfaces = [array[n_points, 3] , [[surf_0, 3], [surf_1, 3], ... ] ]
    
    moab_core   = core.Core()
    dagmc_model = pydagmc.dagnav.Model(moab_core)

    m_to_cm = 100
        
    moab_vertices = moab_core.create_vertices(onp.array(surfaces[0]) * m_to_cm).to_array()

    surface_ids = []
    for surface in surfaces[1]:        
        
        moab_surface = create_surface(moab_core, moab_vertices, onp.array(surface, dtype=onp.int32))
        dagmc_surface = dagmc_model.create_surface()

        dagmc_model.mb.add_entities(dagmc_surface.handle, moab_surface)
        surface_ids.append(dagmc_surface.id)
    
    no_volumes = len(material_names)

    for volume in range(no_volumes):
        dagmc_model.create_volume()
    
    is_full_angle = discrete_blanket.toroidal_extent.full_angle()
    for surface_i, surface in enumerate(dagmc_model.surfaces):
        if is_full_angle:
            # Full-torus shells are closed surfaces. The first and last shell face the
            # implicit complement, while all interior shell surfaces bound two adjacent
            # material volumes.
            if surface_i == 0:
                surface.senses = [dagmc_model.volumes[0], None]
            elif surface_i == len(dagmc_model.surfaces) - 1:
                surface.senses = [dagmc_model.volumes[-1], None]
            else:
                surface.senses = [dagmc_model.volumes[surface_i - 1], dagmc_model.volumes[surface_i]]
        else:
            # Half-module / open-torus geometry adds constant-phi caps between the curved
            # radial surfaces, so the original layering logic remains appropriate there.
            if surface_i == 0:                                # first curved surface
                surface.senses  = [dagmc_model.volumes[-0]        ,  None, ]
            elif surface_i == len(dagmc_model.surfaces) - 1:  # last curved surface
                surface.senses  = [dagmc_model.volumes[-1]        ,  None, ]
            elif surface_i % 2 == 1:                          # constant phi surfaces
                volume_id =  surface_i // 2
                surface.senses  = [dagmc_model.volumes[ volume_id] , None, ]
            else:                                             # intermediate curved surfaces
                volume_id_start = surface_i // 2 - 1
                volume_id_end   = surface_i // 2 
                surface.senses  = [dagmc_model.volumes[volume_id_start], dagmc_model.volumes[volume_id_end]]
    
    # Tag volumes with appropriate materials
    for volume, material in zip(dagmc_model.volumes, material_names):
        material_group = pydagmc.Group.create(dagmc_model, name = "mat:" + material)
        material_group.add_set(volume)    

    # Optionally add finite size coils to the DAGMC model
    if finitesize_coilset is not None:
        assert coil_material_name is not None, "coil_material_name must be provided when finitesize_coil is provided"
        assert discrete_coilset is not None, "discrete_coilset must be provided when finitesize_coil is provided"
        coilset_trunc                    = jax_sbgeom.coils.coilset.filter_coilset_phi(finitesize_coilset, discrete_coilset.toroidal_extent.start, discrete_coilset.toroidal_extent.end)        
        coil_vertices, coil_connectivity = jax_sbgeom.coils.mesh_coilset_surface(coilset_trunc, discrete_coilset.n_points_per_coil, discrete_coilset.width_R, discrete_coilset.width_phi)        
        
        moab_coil_vertices = moab_core.create_vertices(onp.array(coil_vertices) * m_to_cm).to_array()        
        coil_connectivity_rs = onp.array(coil_connectivity, dtype=onp.int32).reshape(coilset_trunc.n_coils, discrete_coilset.n_points_per_coil, 4, 2, 3)
        coil_connectivity_rs = onp.moveaxis(coil_connectivity_rs, 2, 1)
        coil_connectivity_rs = coil_connectivity_rs.reshape(coilset_trunc.n_coils, 4, -1, 3)        

        no_coil_volumes = coilset_trunc.n_coils

        offset_fs_volumes = len(dagmc_model.volumes)

        for coil_i in range(no_coil_volumes):
            dagmc_model.create_volume()

            for surface_i  in range(4): # finite lines are 4 lines around the coils
                moab_surface = create_surface(moab_core, moab_coil_vertices, coil_connectivity_rs[coil_i, surface_i])
                dagmc_surface = dagmc_model.create_surface()
                dagmc_model.mb.add_entities(dagmc_surface.handle, moab_surface)

                dagmc_surface.senses = [dagmc_model.volumes[offset_fs_volumes + coil_i], None]
        if isinstance(coil_material_name, str):
            coil_material_names = [coil_material_name] * no_coil_volumes
        elif len(coil_material_name) != no_coil_volumes:
            raise ValueError(f"Length of coil_material_name {len(coil_material_name)} does not match number of coil volumes {no_coil_volumes}.")
        else:
            coil_material_names = coil_material_name
    
        for volume, material in zip(dagmc_model.volumes[-no_coil_volumes:], coil_material_names):            
            coil_material_group = pydagmc.Group.create(dagmc_model, name = "mat:" + material)
            coil_material_group.add_set(volume)

    return dagmc_model


def tetrahedral_mesh_to_moab_mesh(vertices : jnp.ndarray, connectivity : jnp.ndarray, conversion_factor : float  = 100.0):
    '''
    Convert a tetrahedral mesh to a MOAB mesh.

    PyMOAB is required to use this function. 
    
    Note for writing the result to a .h5m file MOAB should have been compiled with HDF5 support.
    
    Parameters
    ----------
    vertices : jnp.ndarray
        Vertices of the tetrahedral mesh.
    connectivity : jnp.ndarray
        Connectivity of the tetrahedral mesh.
    conversion_factor : float 
        Conversion factor (scaling from m->cm ), default 100
    Returns
    -------
    moab_core : pymoab.core.Core
        The MOAB core containing the tetrahedral mesh.      
    '''
    try:
        import pymoab
        from pymoab import core, types        
    except ImportError as e:                
        raise ImportError(str(e) + "\n\nPyMOAB is required to create a DAGMC mesh. Please install PyMOAB to use this function.")
    
    assert connectivity.shape[1] == 4, "Connectivity must be for tetrahedral elements."
    
    moab_core   = core.Core()
    
    
            
    moab_vertices = moab_core.create_vertices(onp.array(vertices) * conversion_factor).to_array()    
    mesh_set      = moab_core.create_meshset()
    moab_core.add_entity(mesh_set, moab_vertices)
    connectivity  = onp.array(connectivity)    

    connectivity_moab = moab_vertices[connectivity]
    tets_i            = moab_core.create_elements(types.MBTET, connectivity_moab)        
    moab_core.add_entities(mesh_set, tets_i)    
    return moab_core