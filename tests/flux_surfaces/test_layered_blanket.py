import os

import jax_sbgeom as jsb
import jax.numpy as jnp
import numpy as onp

import jax
jax.config.update("jax_enable_x64", True)


import time
from functools import partial
from jax_sbgeom.flux_surfaces.flux_surfaces_base import _check_whether_make_normals_point_outwards_required, ToroidalExtent
import pytest
from functools import lru_cache

from jax_sbgeom.jax_utils import mesh_to_pyvista_mesh
from pathlib import Path
from .test_flux_surface_base_data import DATA_INPUT_FLUX_SURFACES
import pickle as pkl
import h5py


def _assert_hdf5_equal(a, b, skip_keys=("history",)):
    """Recursively compare two HDF5 groups/files, skipping metadata keys."""
    assert set(a.keys()) - set(skip_keys) == set(b.keys()) - set(skip_keys), \
        f"HDF5 keys differ: {set(a.keys())} vs {set(b.keys())}"
    for key in a.keys():
        if key in skip_keys:
            continue
        if isinstance(a[key], h5py.Datatype):
            continue
        elif isinstance(a[key], h5py.Group):
            _assert_hdf5_equal(a[key], b[key], skip_keys)
        else:
            arr_a = onp.array(a[key][:])
            arr_b = onp.array(b[key][:])
            if onp.issubdtype(arr_a.dtype, onp.number):
                onp.testing.assert_allclose(arr_a, arr_b, rtol=1e-10, atol=1e-10, err_msg=f"Dataset '{key}' differs")
            else:
                onp.testing.assert_array_equal(arr_a, arr_b, err_msg=f"Dataset '{key}' differs")


@pytest.fixture(scope="class", params=list(DATA_INPUT_FLUX_SURFACES.glob("*_input.h5")))
def data_file(request):
    return request.param


class TestLayeredBlanket:

    @pytest.fixture(autouse=True, scope="class")
    def setup(self, request, data_file):
        fs_jax = jsb.flux_surfaces.FluxSurfaceNormalExtendedNoPhi.from_hdf5(data_file)

        d_layers = jnp.array([0.2, 0.3, 0.7])
        layered_blanket = jsb.interfaces.blanket_creation.LayeredDiscreteBlanketPlasmaTransformed(
            n_theta=10,
            n_phi=20,
            resolution_layers=(5, 6, 3),
            toroidal_extent=ToroidalExtent.half_module(fs_jax)
        )
        equal_arclength_flux_surface = jsb.flux_surfaces.create_extended_flux_surface_d_interp_equal_arclength(
            fs_jax, d_layers, 50, 60, 100
        )

        request.cls.layered_blanket = layered_blanket
        request.cls.equal_arclength_flux_surface = equal_arclength_flux_surface
        request.cls.base_file_loc = str(data_file)[:-len("_input.h5")] + "_layered_blanket_mesh"

    def test_n_discrete_layers(self):
        assert self.layered_blanket.n_discrete_layers == sum(self.layered_blanket.resolution_layers)

    def test_volume_mesh(self):
        mesh = jsb.interfaces.blanket_creation.mesh_tetrahedral_blanket_transformed_axis(
            self.equal_arclength_flux_surface, self.layered_blanket, 2
        )
        onp.testing.assert_allclose(onp.load(self.base_file_loc + "_points.npy"), mesh[0],  rtol = 1e-10, atol = 1e-10)
        onp.testing.assert_allclose(onp.load(self.base_file_loc + "_connectivity.npy"), mesh[1], rtol = 1e-10, atol = 1e-10)
        assert self.layered_blanket.volume_mesh_structure.n_elements == mesh[1].shape[0]
        assert self.layered_blanket.volume_mesh_structure.n_points == mesh[0].shape[0]

    def test_surface_mesh(self):
        surface_mesh = self.layered_blanket.surface_mesh(self.equal_arclength_flux_surface)
        with open(self.base_file_loc + "_surface_mesh.pkl", "rb") as f:
            surface_mesh_loaded = pkl.load(f)
        onp.testing.assert_allclose(surface_mesh[0], surface_mesh_loaded[0], rtol = 1e-10, atol =1e-10)
        for i in range(len(surface_mesh[1])):
            onp.testing.assert_allclose(surface_mesh[1][i], surface_mesh_loaded[1][i], rtol = 1e-10, atol =1e-10)
    
    @pytest.mark.dagmc
    def test_dagmc(self, tmp_path):
        from jax_sbgeom.interfaces.dagmc_interface import create_dagmc_surface_mesh

        dagmc_blanket = create_dagmc_surface_mesh(self.layered_blanket, self.equal_arclength_flux_surface, material_names=["mat_1", "mat_2"])
    
        tmp_file = tmp_path / "dagmc.h5m"
        dagmc_blanket.write_file(str(tmp_file))        

        ref_file = self.base_file_loc + "_dagmc_blanket.h5m"
        with h5py.File(tmp_file, "r") as out, h5py.File(ref_file, "r") as ref:
            _assert_hdf5_equal(out, ref)

    @pytest.mark.dagmc
    def test_dagmc_full_torus_sense_assignment(self):
        from jax_sbgeom.interfaces.dagmc_interface import create_dagmc_surface_mesh

        full_blanket = jsb.interfaces.blanket_creation.LayeredDiscreteBlanketPlasmaTransformed(
            n_theta=self.layered_blanket.n_theta,
            n_phi=self.layered_blanket.n_phi,
            resolution_layers=self.layered_blanket.resolution_layers,
            toroidal_extent=ToroidalExtent.full(),
            s_power_sampling=2.0,
        )

        dagmc_blanket = create_dagmc_surface_mesh(
            full_blanket,
            self.equal_arclength_flux_surface,
            material_names=["mat_1", "mat_2"],
        )

        assert len(dagmc_blanket.volumes) == full_blanket.n_physical_layers - 1
        assert len(dagmc_blanket.surfaces) == full_blanket.n_physical_layers

        for i, surface in enumerate(dagmc_blanket.surfaces):
            if i == 0 or i == len(dagmc_blanket.surfaces) - 1:
                assert surface.senses[1] is None
            else:
                assert surface.senses[1] is not None