"""area_of_mesh: triangle meshes."""
import jax
import jax.numpy as jnp
import numpy as onp
import pytest

from jax_sbgeom.jax_utils.mesh import area_of_mesh

jax.config.update("jax_enable_x64", True)


def _unit_cube():
    pos = onp.array([[x, y, z] for x in (0, 1) for y in (0, 1) for z in (0, 1)], dtype=float)
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    tri = [t for q in quads for t in ((q[0], q[1], q[2]), (q[0], q[2], q[3]))]
    return jnp.asarray(pos), jnp.asarray(tri)


def _uv_sphere(r, n_th, n_ph):
    th = onp.linspace(0, onp.pi, n_th + 1)[1:-1]
    ph = onp.linspace(0, 2 * onp.pi, n_ph, endpoint=False)
    pts = [[0, 0, r], [0, 0, -r]]
    pts += [[r * onp.sin(t) * onp.cos(p), r * onp.sin(t) * onp.sin(p), r * onp.cos(t)] for t in th for p in ph]
    idx = lambda i, j: 2 + i * n_ph + (j % n_ph)
    tri = []
    for j in range(n_ph):
        tri.append((0, idx(0, j), idx(0, j + 1)))
        tri.append((1, idx(n_th - 2, j + 1), idx(n_th - 2, j)))
    for i in range(n_th - 2):
        for j in range(n_ph):
            tri.append((idx(i, j), idx(i + 1, j), idx(i, j + 1)))
            tri.append((idx(i, j + 1), idx(i + 1, j), idx(i + 1, j + 1)))
    return jnp.asarray(onp.array(pts)), jnp.asarray(tri)


def test_unit_cube():
    pos, tri = _unit_cube()
    assert jnp.allclose(area_of_mesh(pos, tri), 6.0, rtol=1e-14)


def test_sphere_approximation():
    pos, tri = _uv_sphere(2.0, 120, 240)
    assert abs(float(area_of_mesh(pos, tri)) / (4 * jnp.pi * 4.0) - 1) < 2e-4


def test_tetrahedra_rejected():
    pos, _ = _unit_cube()
    with pytest.raises(ValueError):
        area_of_mesh(pos, jnp.array([[0, 1, 2, 4]]))
