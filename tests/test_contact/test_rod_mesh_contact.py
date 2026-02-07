import pytest
import numpy as np
from numpy.testing import assert_allclose

from elastica.mesh import Mesh
from elastica.rigidbody.mesh_rigid_body import MeshRigidBody
from elastica.contact_forces import RodMeshContact
from elastica._contact_functions import _calculate_contact_forces_rod_mesh


def _centered_box(width: float, height: float, depth: float):
    import open3d as o3d

    box = o3d.geometry.TriangleMesh.create_box(width=width, height=height, depth=depth)
    box.compute_vertex_normals()
    box.compute_triangle_normals()
    box.translate(np.array([-0.5 * width, -0.5 * height, -0.5 * depth]))
    return box


class _MockRod:
    def __init__(self, node_positions: np.ndarray, radius: float):
        self.position_collection = node_positions
        self.velocity_collection = np.zeros_like(node_positions)
        self.radius = np.full(node_positions.shape[1] - 1, radius, dtype=np.float64)
        self.external_forces = np.zeros_like(node_positions)


def test_rod_mesh_no_contact():
    mesh = Mesh(_centered_box(1.0, 1.0, 1.0))
    mesh_body = MeshRigidBody(mesh=mesh, density=1.0, volume=mesh.compute_volume())

    nodes = np.array([[0.0, 0.0], [0.0, 0.0], [2.0, 2.2]])
    rod = _MockRod(nodes, radius=0.1)

    contact = RodMeshContact(k=1e4, nu=10.0)
    contact.apply_contact(rod, mesh_body)

    assert_allclose(rod.external_forces, np.zeros_like(rod.external_forces), atol=1e-12)
    assert_allclose(
        mesh_body.external_forces, np.zeros_like(mesh_body.external_forces), atol=1e-12
    )


def test_rod_mesh_penetration_force_balance():
    mesh = Mesh(_centered_box(1.0, 1.0, 1.0))
    mesh_body = MeshRigidBody(mesh=mesh, density=1.0, volume=mesh.compute_volume())

    nodes = np.array([[0.0, 0.0], [0.0, 0.0], [0.6, 0.8]])
    rod = _MockRod(nodes, radius=0.4)

    contact = RodMeshContact(k=1e3, nu=0.0)
    contact.apply_contact(rod, mesh_body)

    rod_force_sum = rod.external_forces.sum(axis=1)
    mesh_force = mesh_body.external_forces[:, 0]

    assert np.linalg.norm(rod_force_sum) > 0.0
    assert_allclose(rod_force_sum + mesh_force, np.zeros(3), atol=1e-9)


def test_frozen_mesh_uses_zero_velocity():
    mesh = Mesh(_centered_box(1.0, 1.0, 1.0))

    nodes = np.array([[0.0, 0.0], [0.0, 0.0], [0.4, 0.6]])
    rod_fast = _MockRod(nodes.copy(), radius=0.4)
    rod_static = _MockRod(nodes.copy(), radius=0.4)

    mesh_fast = MeshRigidBody(mesh=mesh, density=1.0, volume=mesh.compute_volume())
    mesh_fast.velocity_collection[:] = np.array([[5.0], [0.0], [0.0]])
    mesh_fast.omega_collection[:] = np.array([[3.0], [0.0], [0.0]])

    mesh_static = MeshRigidBody(mesh=mesh, density=1.0, volume=mesh.compute_volume())

    contact = RodMeshContact(k=1e3, nu=5.0, mesh_frozen=True)
    contact.apply_contact(rod_fast, mesh_fast)

    contact_static = RodMeshContact(k=1e3, nu=5.0, mesh_frozen=True)
    contact_static.apply_contact(rod_static, mesh_static)

    assert_allclose(rod_fast.external_forces, rod_static.external_forces, atol=1e-12)
    assert_allclose(mesh_fast.external_forces, np.zeros_like(mesh_fast.external_forces))
    assert_allclose(
        mesh_fast.external_torques, np.zeros_like(mesh_fast.external_torques)
    )


def test_mesh_contact_uses_world_frame_angular_velocity_for_point_velocity():
    closest_points = np.array([[0.0, 1.0, 0.0]], dtype=np.float64)
    contact_distances = np.array([0.0], dtype=np.float64)
    contact_normals = np.array([[0.0, 0.0, 1.0]], dtype=np.float64)

    rod_element_positions = np.array([[0.0], [1.0], [0.0]], dtype=np.float64)
    rod_velocity_collection = np.zeros((3, 2), dtype=np.float64)
    rod_radii = np.array([1.0], dtype=np.float64)
    rod_external_forces = np.zeros((3, 2), dtype=np.float64)

    mesh_position = np.zeros(3, dtype=np.float64)
    mesh_velocity = np.zeros(3, dtype=np.float64)
    mesh_angular_velocity = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    mesh_external_forces = np.zeros((3, 1), dtype=np.float64)
    mesh_external_torques = np.zeros((3, 1), dtype=np.float64)

    # world->mesh rotation: +90 deg about y
    mesh_director = np.array(
        [[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float64
    )

    _calculate_contact_forces_rod_mesh(
        closest_points,
        contact_distances,
        contact_normals,
        rod_element_positions,
        rod_velocity_collection,
        rod_radii,
        rod_external_forces,
        mesh_position,
        mesh_velocity,
        mesh_angular_velocity,
        mesh_external_forces,
        mesh_external_torques,
        mesh_director,
        False,
        np.float64(0.0),  # k
        np.float64(1.0),  # nu
        np.float64(0.0),
        np.float64(0.0),
    )

    rod_force_sum = rod_external_forces.sum(axis=1)
    assert_allclose(rod_force_sum, np.array([0.0, 0.0, -1.0]), atol=1e-12)
