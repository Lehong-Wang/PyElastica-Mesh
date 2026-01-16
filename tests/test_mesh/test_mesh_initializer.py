import warnings
import numpy as np
import open3d as o3d
from numpy.testing import assert_allclose

from elastica.mesh import Mesh


def _make_centered_box(
    width: float, height: float, depth: float
) -> o3d.geometry.TriangleMesh:
    box = o3d.geometry.TriangleMesh.create_box(width=width, height=height, depth=depth)
    box.compute_vertex_normals()
    box.compute_triangle_normals()
    box.translate(np.array([-0.5 * width, -0.5 * height, -0.5 * depth]))
    return box


def test_mesh_load_stl():
    mesh = Mesh("tests/cube.stl")
    assert mesh.n_triangles > 0
    assert mesh.vertices.shape[0] > 0


def test_mesh_compute_volume_com_inertia_watertight():
    box = _make_centered_box(2.0, 2.0, 2.0)
    mesh = Mesh(box)

    volume = mesh.compute_volume()
    com = mesh.compute_center_of_mass()
    inertia = mesh.compute_inertia_tensor()

    expected_volume = 8.0
    expected_com = np.array([0.0, 0.0, 0.0])
    mass = expected_volume * 1.0
    expected_inertia_diag = np.array([mass * (2.0**2 + 2.0**2) / 12.0] * 3)

    assert_allclose(volume, expected_volume, rtol=1e-6, atol=1e-12)
    assert_allclose(com, expected_com, atol=1e-12)
    assert_allclose(np.diag(inertia), expected_inertia_diag, rtol=1e-6, atol=1e-12)


def test_mesh_triangle_normals_unit():
    box = _make_centered_box(1.0, 1.0, 1.0)
    mesh = Mesh(box)
    norms = np.linalg.norm(mesh.triangle_normals, axis=1)
    assert_allclose(norms, np.ones_like(norms), atol=1e-12)


def test_mesh_non_watertight_warning_and_fallback():
    with warnings.catch_warnings(record=True) as caught:
        mesh = Mesh("tests/cube.stl", warn_if_not_watertight=True)
    assert any("not watertight" in str(w.message) for w in caught)

    volume = mesh.compute_volume()
    inertia = mesh.compute_inertia_tensor()
    expected_volume = 8.0
    expected_inertia_diag = np.array([8.0 * (2.0**2 + 2.0**2) / 12.0] * 3)

    assert_allclose(volume, expected_volume, rtol=1e-6, atol=1e-12)
    assert_allclose(np.diag(inertia), expected_inertia_diag, rtol=1e-6, atol=1e-12)


def test_mesh_recenter_true_centers_geometry():
    box = _make_centered_box(1.0, 1.0, 1.0)
    box.translate(np.array([1.0, -2.0, 0.5]))
    mesh = Mesh(box, recenter_to_com=True)

    com_after = mesh.compute_center_of_mass()
    assert_allclose(com_after, np.zeros(3), atol=1e-12)
    assert_allclose(mesh.vertices.mean(axis=0), np.zeros(3), atol=1e-12)


def test_mesh_recenter_false_leaves_geometry_untouched():
    box = _make_centered_box(1.0, 1.0, 1.0)
    box.translate(np.array([1.0, -2.0, 0.5]))
    mesh = Mesh(box, recenter_to_com=False)

    com_after = mesh.compute_center_of_mass()
    assert_allclose(com_after, np.array([1.0, -2.0, 0.5]), atol=1e-12)
    assert_allclose(mesh.vertices.mean(axis=0), np.array([1.0, -2.0, 0.5]), atol=1e-12)
