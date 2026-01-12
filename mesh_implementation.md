# Mesh Feature Implementation Plan for PyElastica

**Date**: 2026-01-11
**Objective**: Add mesh rigid body support and rod-mesh contact to PyElastica using Open3D

---

## Table of Contents

1. [Current State Analysis](#1-current-state-analysis)
2. [Identified Problems & Conflicts](#2-identified-problems--conflicts)
3. [Detailed Step-by-Step Implementation Plan](#3-detailed-step-by-step-implementation-plan)
4. [Implementation Sequence & Priorities](#4-implementation-sequence--priorities)
5. [Potential Problems & Mitigations](#5-potential-problems--mitigations)
6. [Files to Create/Modify](#6-files-to-createmodify)
7. [Dependencies](#7-dependencies)
8. [Documentation Updates](#8-documentation-updates)

---

## 1. Current State Analysis

### 1.1 Existing Architecture Understanding

- ✅ **Rigid Body Pattern**: All rigid bodies inherit from `RigidBodyBase` with standardized properties
- ✅ **Contact Pattern**: High-level classes in `contact_forces.py` → Low-level Numba functions in `_contact_functions.py`
- ✅ **Memory Block System**: Exists but **NOT to be modified** (as per requirements)
- ✅ **Module System**: Mixin-based architecture for composing features
- ✅ **Previous Mesh Implementation**: Used PyVista, was reverted in commit c191c89e

### 1.2 Key Existing Patterns to Follow

#### Contact Force Pattern (from `RodCylinderContact`)
- Element-wise loop over rod elements
- Non-Numba distance query → Numba force calculation
- Force distribution to nodes (2/3, 4/3 weighting at endpoints)
- Equal and opposite forces on both systems

#### Rigid Body Pattern (from `Cylinder`, `Sphere`)
- Constructor computes physical properties
- Position at center of mass
- Director collection defines orientation
- `update_accelerations()` method
- Energy computation methods

---

## 2. Identified Problems & Conflicts

### 2.1 Critical Issues to Address

#### Problem 1: Numba Incompatibility with Open3D

**Issue**: Open3D raytracing cannot be used inside Numba JIT-compiled functions.

**Solution**: Two-stage approach:
1. **Stage 1 (Pure Python)**: Call mesh's `query_closest_points()` once for all rod elements
2. **Stage 2 (Numba)**: Pass results to Numba function for force calculation

**Implementation Pattern**:
```python
# In RodMeshContact.apply_contact() - Python
closest_points, distances, normals = mesh.query_closest_points(rod_element_positions)

# Then call Numba function
_calculate_contact_forces_rod_mesh(
    closest_points, distances, normals,  # Pre-computed in Python
    rod_positions, rod_velocities, ...,   # Rod data
    mesh_position, mesh_velocity, ...,    # Mesh data
    k, nu                                 # Contact parameters
)
```

#### Problem 2: Transform Management Without BVH Rebuild

**Issue**: Updating mesh geometry rebuilds BVH (expensive O(n log n) operation).

**Solution** (from `mytest/open3d_test_transfrom.py`):
- Store mesh in **reference/material frame** (build BVH once)
- Track transforms via `position_collection` (COM) and `director_collection`
- For queries: Apply `R_mesh_from_world` and translation to query points → query in material frame → transform results back

**Implementation Pattern**:
```python
class MeshRigidBody:
    def __init__(self, o3d_mesh, ...):
        # Store mesh in material frame (BVH built once)
        self._mesh_material_frame = o3d_mesh
        self._scene = o3d.t.geometry.RaycastingScene()
        self._scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(o3d_mesh))

        # Transform tracking (never applied to mesh geometry)
        # director_collection maps world -> body; use transpose for body -> world.

    def query_closest_points(self, query_points_world):
        # Transform queries to material frame (no matrix inverse)
        R_mesh_from_world = self.director_collection[:, :, 0]
        query_material = (R_mesh_from_world @ (query_points_world - self.position_collection[:, 0]).T).T

        # Query in material frame (no BVH rebuild!)
        result = self._scene.compute_closest_points(query_material)
        closest_material = result["points"].numpy()

        # Transform results back to world frame
        R_world_from_mesh = R_mesh_from_world.T
        closest_world = (R_world_from_mesh @ closest_material.T).T + self.position_collection[:, 0]
        return closest_world, distances, normals_world
```

#### Problem 3: Type Checking Compatibility

**Issue**: Need to integrate with existing type checking system.

**Solution**: Follow existing pattern from `Cylinder`, `Sphere`:
```python
# In contact_forces.py
from elastica.rigidbody.mesh_rigid_body import MeshRigidBody

class RodMeshContact(NoContact):
    @property
    def _allowed_system_two(self) -> list[Type]:
        return [MeshRigidBody]  # Type checking
```

#### Problem 4: Mesh Triangles vs Rod Elements

**Issue**: Rod has `n_elem` elements, mesh has arbitrary triangles. Need efficient query.

**Solution**: Query once for all rod element centers:
```python
# Get rod element centers (n_elem, 3)
rod_element_positions = 0.5 * (
    rod.position_collection[..., 1:] + rod.position_collection[..., :-1]
).T  # Shape: (n_elem, 3)

# Single batch query (efficient)
closest_pts, dists, normals = mesh.query_closest_points(rod_element_positions)
```

#### Problem 5: Normal Direction Convention

**Issue**: Open3D normals might point inward/outward inconsistently.

**Mitigation**:
- Primary normal should be `(query - closest_point) / distance` in the mesh frame
- Fall back to triangle normal if the vector norm is too small
- Use `mesh.compute_triangle_normals()` to ensure triangle normals are available
- In force calculation, penetration = `distance < radius` (rod penetrating mesh)
- Contact normal should point from mesh surface toward rod center

#### Problem 6: Face vs Triangle Nomenclature

**Issue**: Previous PyVista implementation used "faces", Open3D uses "triangles".

**Solution**: Use "triangles" consistently in Open3D implementation for clarity:
- `mesh.triangles` (Open3D TriangleMesh)
- `triangle_normals` (not "face_normals")
- Internal calculations use triangle IDs

### 2.2 Design Decisions

#### Decision 1: Properties Calculation

For optional properties (COM, inertia, density, volume), provide helper functions in `mesh_initializer.py`. If the mesh is not watertight, fall back to a simple OBB approximation.

```python
def compute_mesh_volume(o3d_mesh):
    """Compute volume using divergence theorem"""
    # Implementation using mesh.triangles and mesh.vertices

def compute_mesh_center_of_mass(o3d_mesh, density=1.0):
    """Compute COM assuming uniform density"""

def compute_mesh_inertia(o3d_mesh, density=1.0, com=None):
    """Compute inertia tensor about COM"""
```

**Rationale**: Separates concerns, allows users to compute properties separately or provide their own.

#### Decision 2: Contact Force Model

Follow existing rod-cylinder pattern:
- **Spring force**: `F_spring = k * penetration_depth * normal`
- **Damping force**: `F_damp = nu * (v_rel · normal) * normal`
- **Optional friction**: Similar to rod-cylinder (velocity damping + Coulomb)

---

## 3. Detailed Step-by-Step Implementation Plan

### Phase 1: Mesh Infrastructure (Foundation)

#### Step 1.1: Create `elastica/mesh/mesh_initializer.py`

**Purpose**: Mesh loading/preprocessing and property computation using Open3D. Check watertightness before any preprocessing (warn if not watertight).

```python
"""Mesh loading and property computation using Open3D"""

import numpy as np
import open3d as o3d
from numpy.typing import NDArray

class Mesh:
    """
    Mesh initializer using Open3D.

    Loads mesh from file (STL, OBJ, PLY) or accepts an Open3D mesh.
    Provides preprocessing and helper methods for physical properties.

    Attributes:
        mesh: o3d.geometry.TriangleMesh - Legacy Open3D mesh
        vertices: (N, 3) array - Vertex positions
        triangles: (M, 3) array - Triangle vertex indices
        triangle_normals: (M, 3) array - Per-triangle normal vectors
        n_triangles: int - Number of triangles
        is_watertight: bool - Water-tightness flag
        obb: o3d.geometry.OrientedBoundingBox - OBB for fallback properties
    """

    def __init__(
        self,
        mesh_or_path: str | o3d.geometry.TriangleMesh,
        recenter_to_com: bool = True,
        warn_if_not_watertight: bool = True,
    ):
        """Load mesh and preprocess"""
        if isinstance(mesh_or_path, str):
            self.mesh = o3d.io.read_triangle_mesh(mesh_or_path)
        else:
            self.mesh = mesh_or_path
        self.mesh.compute_vertex_normals()
        self.mesh.compute_triangle_normals()

        self.is_watertight = bool(self.mesh.is_watertight())
        if warn_if_not_watertight and not self.is_watertight:
            print("[Mesh] Warning: mesh is not watertight; using OBB fallback as needed.")

        # Extract arrays
        self.vertices = np.asarray(self.mesh.vertices)
        self.triangles = np.asarray(self.mesh.triangles)
        self.triangle_normals = np.asarray(self.mesh.triangle_normals)
        self.n_triangles = len(self.triangles)
        self.obb = self.mesh.get_oriented_bounding_box()

        self.com_offset = np.zeros(3, dtype=np.float64)
        if recenter_to_com:
            com = self.compute_center_of_mass()
            self.mesh.translate(-com)
            self.vertices = np.asarray(self.mesh.vertices)
            self.com_offset = com

    def compute_volume(self) -> float:
        """Compute mesh volume using divergence theorem or OBB fallback"""
        # If not watertight, use OBB volume
        pass

    def compute_center_of_mass(self, density: float = 1.0) -> NDArray:
        """Compute COM assuming uniform density, or OBB center if not watertight"""
        pass

    def compute_inertia_tensor(self, density: float = 1.0,
                                com: NDArray = None) -> NDArray:
        """Compute 3x3 inertia tensor about COM, or OBB inertia if not watertight"""
        pass

    def compute_bounding_box(self) -> tuple:
        """Return (min_bound, max_bound) for AABB"""
        return self.mesh.get_min_bound(), self.mesh.get_max_bound()
```

**Notes**:
- `recenter_to_com` defaults to `True` but can be disabled to preserve original mesh coordinates.
- `com_offset` records the applied translation for downstream use if needed.

**Files to create**:
- `elastica/mesh/__init__.py`
- `elastica/mesh/mesh_initializer.py`

**Update**:
- `elastica/__init__.py` - Add `from elastica.mesh import Mesh`

---

#### Step 1.2: Create `elastica/rigidbody/mesh_rigid_body.py`

**Key Design**:
- Inherits from `RigidBodyBase`
- Stores Open3D mesh in **material frame** (never transformed)
- Builds `RaycastingScene` once in `__init__`
- Tracks transform via `director_collection` and `position_collection`
- `query_closest_points()` method for contact detection
- Allows manual overrides for mass/COM/inertia/density/volume
- Overrides `update_accelerations()`, `compute_translational_energy()`, `compute_rotational_energy()` with rigid body formulas

```python
"""Mesh rigid body using Open3D"""

import numpy as np
import open3d as o3d
from numpy.typing import NDArray
from elastica.rigidbody.rigid_body import RigidBodyBase
from elastica._linalg import _batch_matvec, _batch_cross

class MeshRigidBody(RigidBodyBase):
    def __init__(
        self,
        mesh,  # Mesh object from mesh_initializer
        center_of_mass: NDArray[np.float64] | None = None,
        mass_second_moment_of_inertia: NDArray[np.float64] | None = None,
        density: float | None = None,
        volume: float | None = None,
    ):
        """
        Initialize mesh rigid body.

        Parameters:
        -----------
        mesh: Mesh object (from elastica.mesh.Mesh)
        center_of_mass: (3,) array - COM in world frame at t=0 (optional)
        mass_second_moment_of_inertia: (3, 3) array - Inertia tensor (optional)
        density: float - Material density (optional)
        volume: float - Mesh volume (optional)
        """
        super().__init__()

        # Store mesh geometry in material frame (never modified)
        # Mesh preprocessing handles recentering to COM when requested.
        self._mesh_material_frame = mesh.mesh  # o3d.geometry.TriangleMesh
        self._vertices_material = np.asarray(mesh.mesh.vertices).astype(np.float32)
        self._triangles_indices = np.asarray(mesh.mesh.triangles).astype(np.uint32)
        self._triangle_normals_material = np.asarray(mesh.mesh.triangle_normals)

        # Build raycasting scene ONCE (expensive operation)
        self._scene = self._build_raycasting_scene(
            self._vertices_material,
            self._triangles_indices
        )

        # Physical properties (allow manual overrides)
        if density is None:
            density = 1.0
        if volume is None:
            volume = mesh.compute_volume()
        if center_of_mass is None:
            center_of_mass = mesh.compute_center_of_mass(density=density)
        if mass_second_moment_of_inertia is None:
            mass_second_moment_of_inertia = mesh.compute_inertia_tensor(
                density=density, com=center_of_mass
            )

        self.density = np.float64(density)
        self.volume = np.float64(volume)
        self.mass = np.float64(volume * density)

        # Inertia tensor
        self.mass_second_moment_of_inertia = mass_second_moment_of_inertia.reshape(3, 3, 1)
        self.inv_mass_second_moment_of_inertia = np.linalg.inv(
            mass_second_moment_of_inertia
        ).reshape(3, 3, 1)

        # Initialize director (identity = material frame aligned with world frame)
        self.director_collection = np.zeros((3, 3, 1))
        self.director_collection[0, :, 0] = [1, 0, 0]
        self.director_collection[1, :, 0] = [0, 1, 0]
        self.director_collection[2, :, 0] = [0, 0, 1]

        # Position at COM
        self.position_collection = center_of_mass.reshape(3, 1)

        # Bounding box for rough collision detection
        bbox = self._mesh_material_frame.get_axis_aligned_bounding_box()
        self.radius = np.float64(0.5 * np.max(bbox.get_extent()))
        self.length = np.float64(np.max(bbox.get_extent()))

        # State vectors (standard for rigid bodies)
        self.velocity_collection = np.zeros((3, 1))
        self.omega_collection = np.zeros((3, 1))
        self.acceleration_collection = np.zeros((3, 1))
        self.alpha_collection = np.zeros((3, 1))
        self.external_forces = np.zeros((3, 1))
        self.external_torques = np.zeros((3, 1))
        self.internal_forces = np.zeros((3, 1))
        self.internal_torques = np.zeros((3, 1))

    @staticmethod
    def _build_raycasting_scene(vertices, triangles):
        """Build Open3D raycasting scene for fast queries"""
        scene = o3d.t.geometry.RaycastingScene()
        V = o3d.core.Tensor(vertices, dtype=o3d.core.Dtype.Float32)
        T = o3d.core.Tensor(triangles, dtype=o3d.core.Dtype.UInt32)
        scene.add_triangles(V, T)
        return scene

    def _compute_rotation_matrices(self):
        """Return (R_mesh_from_world, R_world_from_mesh)"""
        R_mesh_from_world = self.director_collection[:, :, 0]
        R_world_from_mesh = R_mesh_from_world.T
        return R_mesh_from_world, R_world_from_mesh

    def query_closest_points(self, query_points_world: NDArray[np.float64]):
        """
        Find closest points on mesh surface for given query points.

        Parameters:
        -----------
        query_points_world: (N, 3) array - Query points in world frame

        Returns:
        --------
        closest_points_world: (N, 3) - Closest points on mesh surface
        distances: (N,) - Unsigned distances from query to surface
        normals_world: (N, 3) - Triangle normals at closest points (outward)
        """
        # Transform queries to material frame (no matrix inverse)
        R_mesh_from_world, R_world_from_mesh = self._compute_rotation_matrices()
        query_material = (
            R_mesh_from_world @ (query_points_world - self.position_collection[:, 0]).T
        ).T

        # Query in material frame (NO BVH rebuild!)
        q_tensor = o3d.core.Tensor(query_material.astype(np.float32))
        result = self._scene.compute_closest_points(q_tensor)

        closest_material = result["points"].numpy().astype(np.float64)
        triangle_ids = result["primitive_ids"].numpy().astype(np.int64)

        # Get normals in material frame (prefer query-closest direction)
        direction = query_material - closest_material
        direction_norm = np.linalg.norm(direction, axis=1)
        normals_material = np.zeros_like(direction)
        valid = direction_norm > 1e-12
        normals_material[valid] = (direction[valid].T / direction_norm[valid]).T
        normals_material[~valid] = self._triangle_normals_material[triangle_ids[~valid]]

        # Transform results to world frame
        closest_world = (R_world_from_mesh @ closest_material.T).T + self.position_collection[:, 0]

        # Transform normals (rotation only, no translation)
        normals_world = (R_world_from_mesh @ normals_material.T).T

        # Compute distances in world frame
        distances = np.linalg.norm(query_points_world - closest_world, axis=1)

        return closest_world, distances, normals_world

    def update_accelerations(self, time: np.float64) -> None:
        """Override with rigid body dynamics for mesh"""
        np.copyto(
            self.acceleration_collection,
            (self.external_forces) / self.mass,
        )
        j_omega = _batch_matvec(
            self.mass_second_moment_of_inertia, self.omega_collection
        )
        lagrangian_transport = _batch_cross(j_omega, self.omega_collection)
        np.copyto(
            self.alpha_collection,
            _batch_matvec(
                self.inv_mass_second_moment_of_inertia,
                (lagrangian_transport + self.external_torques),
            ),
        )

    def compute_translational_energy(self) -> NDArray[np.float64]:
        """Override translational energy"""
        return (
            0.5
            * self.mass
            * np.dot(
                self.velocity_collection[..., -1], self.velocity_collection[..., -1]
            )
        )

    def compute_rotational_energy(self) -> NDArray[np.float64]:
        """Override rotational energy"""
        j_omega = np.einsum(
            "ijk,jk->ik", self.mass_second_moment_of_inertia, self.omega_collection
        )
        return 0.5 * np.einsum("ik,ik->k", self.omega_collection, j_omega).sum()
```

**Files to create**:
- `elastica/rigidbody/mesh_rigid_body.py`

**Update**:
- `elastica/rigidbody/__init__.py` - Add `from elastica.rigidbody.mesh_rigid_body import MeshRigidBody`
- `elastica/__init__.py` - Add `MeshRigidBody` to exports

---

### Phase 2: Rod-Mesh Contact Implementation

#### Step 2.1: Create `_calculate_contact_forces_rod_mesh()` in `_contact_functions.py`

**Critical**: This function is Numba-compiled, so it receives **pre-computed** closest points and query-oriented normals.

```python
@njit(cache=True)
def _calculate_contact_forces_rod_mesh(
    # Pre-computed query results (from Python)
    closest_points_on_mesh: NDArray[np.float64],  # (n_elem, 3)
    contact_distances: NDArray[np.float64],        # (n_elem,)
    contact_normals: NDArray[np.float64],          # (n_elem, 3) normalized (query - closest)
    # Rod data
    rod_element_positions: NDArray[np.float64],    # (3, n_elem)
    rod_element_velocities: NDArray[np.float64],   # (3, n_elem+1) nodes
    rod_radii: NDArray[np.float64],                # (n_elem,)
    rod_external_forces: NDArray[np.float64],      # (3, n_elem+1) nodes
    # Mesh data
    mesh_position: NDArray[np.float64],            # (3,) COM
    mesh_velocity: NDArray[np.float64],            # (3,) COM velocity
    mesh_director: NDArray[np.float64],            # (3, 3) world->mesh rotation
    mesh_external_forces: NDArray[np.float64],     # (3, 1)
    mesh_external_torques: NDArray[np.float64],    # (3, 1)
    # Contact parameters
    k: np.float64,                                 # Spring constant
    nu: np.float64,                                # Damping constant
):
    """
    Calculate contact forces between rod elements and mesh.

    Flow:
    -----
    For each rod element:
      1. Check if penetrating (distance < radius)
      2. Compute spring force: F = k * penetration_depth * normal
      3. Compute damping: F_damp = nu * (v_rel · n) * n
      4. Distribute force to rod nodes (2/3, 4/3 at ends)
      5. Apply reaction force/torque to mesh
    """
    n_elem = rod_element_positions.shape[1]

    # Accumulate forces on mesh
    total_force_on_mesh = np.zeros(3, dtype=np.float64)
    total_torque_on_mesh = np.zeros(3, dtype=np.float64)

    for i in range(n_elem):
        # Penetration check
        penetration_depth = rod_radii[i] - contact_distances[i]

        if penetration_depth < -1e-5:  # No contact
            continue

        # Heaviside function (contact active)
        mask = (penetration_depth > 0.0) * 1.0

        # Contact normal is pre-oriented from closest point toward the query point
        normal = contact_normals[i]

        # Spring force
        spring_force = k * penetration_depth * normal

        # Relative velocity at contact point
        # Rod element velocity (average of two nodes)
        rod_elem_vel = 0.5 * (
            rod_element_velocities[:, i] + rod_element_velocities[:, i+1]
        )

        # Mesh velocity at contact point (rigid body: v = v_com + omega × r)
        # For now, use COM velocity (can add rotational component later)
        mesh_vel = mesh_velocity

        rel_velocity = rod_elem_vel - mesh_vel

        # Normal damping
        normal_vel = _dot_product(rel_velocity, normal) * normal
        damping_force = -nu * normal_vel

        # Net contact force
        net_force = mask * (spring_force + damping_force)

        # Distribute force to rod nodes
        # Standard distribution: 2/3 at ends, 1.0 in middle
        if i == 0:  # First element
            rod_external_forces[:, i] -= (2.0/3.0) * net_force
            rod_external_forces[:, i+1] -= (4.0/3.0) * net_force
        elif i == n_elem - 1:  # Last element
            rod_external_forces[:, i] -= (4.0/3.0) * net_force
            rod_external_forces[:, i+1] -= (2.0/3.0) * net_force
        else:  # Middle elements
            rod_external_forces[:, i] -= net_force
            rod_external_forces[:, i+1] -= net_force

        # Reaction force on mesh (Newton's 3rd law)
        total_force_on_mesh += 2.0 * net_force  # Factor of 2 due to node distribution

        # Torque on mesh (moment arm from mesh COM to contact point)
        moment_arm = closest_pt - mesh_position
        torque = np.cross(moment_arm, 2.0 * net_force)
        total_torque_on_mesh += torque

    # Apply total forces to mesh
    mesh_external_forces[:, 0] += total_force_on_mesh

    # Transform torque to material frame (mesh uses body-fixed torques)
    mesh_external_torques[:, 0] += mesh_director @ total_torque_on_mesh
```

**Files to modify**:
- `elastica/_contact_functions.py` - Add `_calculate_contact_forces_rod_mesh()`

---

#### Step 2.2: Create `RodMeshContact` in `contact_forces.py`

```python
class RodMeshContact(NoContact):
    """
    Contact between rod and mesh rigid body.

    First system must be RodType, second system must be MeshRigidBody.

    Examples:
    ---------
    >>> simulator.detect_contact_between(rod, mesh).using(
    ...     RodMeshContact,
    ...     k=1e4,
    ...     nu=10.0,
    ... )
    """

    def __init__(self, k: float, nu: float):
        """
        Parameters:
        -----------
        k: float - Contact spring constant
        nu: float - Contact damping constant
        """
        super().__init__()
        self.k = np.float64(k)
        self.nu = np.float64(nu)

    @property
    def _allowed_system_two(self) -> list[Type]:
        from elastica.rigidbody.mesh_rigid_body import MeshRigidBody
        return [MeshRigidBody]

    def apply_contact(
        self,
        system_one: RodType,
        system_two,  # MeshRigidBody (avoid circular import)
        time: np.float64 = np.float64(0.0),
    ) -> None:
        """
        Apply contact forces between rod and mesh.

        Flow:
        -----
        1. Compute rod element centers (world frame)
        2. Query mesh for closest points (Python, not Numba)
        3. Pass results to Numba function for force calculation
        """
        rod = system_one
        mesh = system_two

        # Get rod element positions (shape: 3, n_elem)
        rod_element_positions = 0.5 * (
            rod.position_collection[..., 1:] + rod.position_collection[..., :-1]
        )

        # Transpose for query (shape: n_elem, 3)
        query_points = rod_element_positions.T

        # STAGE 1: Python - Query closest points (NOT Numba compatible)
        closest_points, distances, normals = mesh.query_closest_points(query_points)

        # STAGE 2: Numba - Calculate forces
        _calculate_contact_forces_rod_mesh(
            closest_points,                    # (n_elem, 3)
            distances,                         # (n_elem,)
            normals,                           # (n_elem, 3)
            rod_element_positions,             # (3, n_elem)
            rod.velocity_collection,           # (3, n_nodes)
            rod.radius,                        # (n_elem,)
            rod.external_forces,               # (3, n_nodes) - modified in-place
            mesh.position_collection[:, 0],    # (3,)
            mesh.velocity_collection[:, 0],    # (3,)
            mesh.director_collection[:, :, 0], # (3, 3)
            mesh.external_forces,              # (3, 1) - modified in-place
            mesh.external_torques,             # (3, 1) - modified in-place
            self.k,
            self.nu,
        )
```

**Files to modify**:
- `elastica/contact_forces.py` - Add `RodMeshContact` class
- Add import statement at top: `from elastica._contact_functions import _calculate_contact_forces_rod_mesh`

---

### Phase 3: Examples and Post-Processing

#### Step 3.1: Create Examples Directory Structure

```
examples/
├── MeshCase/
│   ├── __init__.py
│   ├── mesh_freefall.py          # Mesh falling under gravity
│   ├── mesh_rod_collision.py     # Rod-mesh contact simulation
│   ├── mesh_frozen_contact.py    # Frozen mesh regression example
│   └── post_processing.py        # Visualization utilities
```

#### Step 3.2: Implement `mesh_freefall.py`

```python
"""
Mesh freefall simulation - demonstrates MeshRigidBody under gravity.
"""

import numpy as np
import elastica as ea

def mesh_freefall_simulation():
    class MeshFreefall(
        ea.BaseSystemCollection,
        ea.Forcing,
        ea.CallBacks,
        ea.Contact,
    ):
        pass

    simulator = MeshFreefall()

    # Load mesh
    mesh = ea.Mesh("path/to/bunny.stl")

    # Compute properties (or provide manually)
    volume = mesh.compute_volume()
    density = 1000.0  # kg/m^3
    com = mesh.compute_center_of_mass(density)
    inertia = mesh.compute_inertia_tensor(density, com)

    # Create mesh rigid body
    mesh_body = ea.MeshRigidBody(
        mesh=mesh,
        center_of_mass=com + np.array([0, 0, 2.0]),  # Start 2m above ground
        mass_second_moment_of_inertia=inertia,
        density=density,
        volume=volume,
    )

    simulator.append(mesh_body)

    # Add gravity
    gravity = np.array([0, 0, -9.81])
    simulator.add_forcing_to(mesh_body).using(
        ea.GravityForces,
        acc_gravity=gravity,
    )

    # Add ground plane
    plane = ea.Plane(
        plane_origin=np.array([0, 0, 0]),
        plane_normal=np.array([0, 0, 1]),
    )
    simulator.append(plane)

    # Callbacks for data collection
    # ... (similar to existing examples)

    # Run simulation
    final_time = 5.0
    dt = 1e-4
    total_steps = int(final_time / dt)

    simulator.finalize()
    timestepper = ea.PositionVerlet()
    ea.integrate(timestepper, simulator, final_time, total_steps)

if __name__ == "__main__":
    mesh_freefall_simulation()
```

#### Step 3.3: Implement `mesh_rod_collision.py`

```python
"""
Rod-mesh collision - demonstrates RodMeshContact.
"""

import numpy as np
import elastica as ea

def rod_mesh_collision():
    class RodMeshSim(
        ea.BaseSystemCollection,
        ea.Constraints,
        ea.Contact,
        ea.CallBacks,
        ea.Forcing,
    ):
        pass

    simulator = RodMeshSim()

    # Create rod (swinging toward mesh)
    n_elem = 50
    rod = ea.CosseratRod.straight_rod(
        n_elem=n_elem,
        start=np.array([-1.0, 0.0, 1.0]),
        direction=np.array([1.0, 0.0, 0.0]),
        normal=np.array([0.0, 1.0, 0.0]),
        base_length=1.0,
        base_radius=0.025,
        density=1000.0,
        youngs_modulus=1e6,
        shear_modulus=1e6 / (2 * 1.5),
    )
    simulator.append(rod)

    # Fix one end
    simulator.constrain(rod).using(
        ea.OneEndFixedBC,
        constrained_position_idx=(0,),
        constrained_director_idx=(0,),
    )

    # Create mesh obstacle
    mesh = ea.Mesh("path/to/cube.stl")
    mesh_body = ea.MeshRigidBody(
        mesh=mesh,
        center_of_mass=np.array([0.5, 0.0, 1.0]),
        mass_second_moment_of_inertia=compute_cube_inertia(side=0.2, density=1000),
        density=1000.0,
        volume=0.2**3,
    )
    simulator.append(mesh_body)

    # Rod-mesh contact
    simulator.detect_contact_between(rod, mesh_body).using(
        ea.RodMeshContact,
        k=1e5,
        nu=10.0,
    )

    # Add gravity to rod
    simulator.add_forcing_to(rod).using(
        ea.GravityForces,
        acc_gravity=np.array([0, 0, -9.81]),
    )

    # Callbacks
    # ... (collect rod and mesh data)

    # Run simulation
    final_time = 5.0
    dt = 1e-5
    total_steps = int(final_time / dt)

    simulator.finalize()
    timestepper = ea.PositionVerlet()
    ea.integrate(timestepper, simulator, final_time, total_steps)

if __name__ == "__main__":
    rod_mesh_collision()
```

#### Step 3.4: Implement `mesh_frozen_contact.py`

```python
"""
Frozen mesh regression example.
Rod drops onto a mesh with effectively infinite mass to validate no motion.
"""

import numpy as np
import elastica as ea

def frozen_mesh_contact():
    class FrozenMeshSim(
        ea.BaseSystemCollection,
        ea.Constraints,
        ea.Contact,
        ea.CallBacks,
        ea.Forcing,
    ):
        pass

    simulator = FrozenMeshSim()

    # Create rod
    rod = ea.CosseratRod.straight_rod(...)
    simulator.append(rod)

    # Create mesh
    mesh = ea.Mesh("path/to/cube.stl")
    mesh_body = ea.MeshRigidBody(mesh=mesh, ...)
    simulator.append(mesh_body)

    # Freeze mesh (very large mass and zero inverse inertia)
    mesh_body.mass = np.float64(1e20)
    mesh_body.inv_mass_second_moment_of_inertia[:] = 0.0

    # Contact
    simulator.detect_contact_between(rod, mesh_body).using(
        ea.RodMeshContact,
        k=1e5,
        nu=10.0,
    )

    # Run a short simulation and verify mesh COM stays fixed
    # ... callbacks + asserts ...

if __name__ == "__main__":
    frozen_mesh_contact()
```

#### Step 3.4: Implement `post_processing.py`

```python
"""
Post-processing utilities for mesh visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation

def mesh_to_poly3d_collection(mesh_o3d, transform=None):
    """
    Convert Open3D mesh to matplotlib Poly3DCollection.

    Parameters:
    -----------
    mesh_o3d: o3d.geometry.TriangleMesh
    transform: (4, 4) array - Transform matrix (optional)

    Returns:
    --------
    Poly3DCollection for matplotlib rendering
    """
    import open3d as o3d

    # Copy mesh to avoid modifying original
    mesh_copy = o3d.geometry.TriangleMesh(mesh_o3d)

    # Apply transform if provided
    if transform is not None:
        mesh_copy.transform(transform)

    vertices = np.asarray(mesh_copy.vertices)
    triangles = np.asarray(mesh_copy.triangles)

    # Create list of triangle vertices
    triangle_verts = []
    for tri in triangles:
        triangle_verts.append(vertices[tri])

    return Poly3DCollection(triangle_verts, alpha=0.6, facecolor='tab:blue', edgecolor='k')


def plot_mesh_animation(
    mesh_data_dict,
    video_name="mesh_animation.mp4",
    fps=30,
    xlim=(-2, 2),
    ylim=(-2, 2),
    zlim=(-2, 2),
):
    """
    Create animation of mesh motion.

    Parameters:
    -----------
    mesh_data_dict: dict with keys:
        - 'time': list of timestamps
        - 'position': list of COM positions (3,) arrays
        - 'director': list of director matrices (3, 3) arrays
        - 'mesh': Open3D mesh object (constant geometry)
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_zlim(zlim)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    mesh_o3d = mesh_data_dict['mesh']
    positions = mesh_data_dict['position']
    directors = mesh_data_dict['director']

    # Initial frame
    T0 = np.eye(4)
    T0[:3, :3] = directors[0]
    T0[:3, 3] = positions[0]

    poly = mesh_to_poly3d_collection(mesh_o3d, T0)
    ax.add_collection3d(poly)

    writer = animation.writers['ffmpeg'](fps=fps)

    with writer.saving(fig, video_name, dpi=150):
        for pos, director in zip(positions, directors):
            # Remove old mesh
            poly.remove()

            # Compute transform
            T = np.eye(4)
            T[:3, :3] = director
            T[:3, 3] = pos

            # Add new mesh
            poly = mesh_to_poly3d_collection(mesh_o3d, T)
            ax.add_collection3d(poly)

            writer.grab_frame()

    plt.close(fig)
    print(f"Animation saved to {video_name}")
```

**Files to create**:
- `examples/MeshCase/__init__.py`
- `examples/MeshCase/mesh_freefall.py`
- `examples/MeshCase/mesh_rod_collision.py`
- `examples/MeshCase/post_processing.py`

---

### Phase 4: Testing

#### Step 4.1: Unit Tests for Mesh Initialization

**File**: `tests/test_mesh/test_mesh_initializer.py`

**Tests**:
- Load STL file
- Compute volume (compare to analytical for cube)
- Compute COM (should be at geometric center for symmetric mesh)
- Compute inertia (compare to analytical for cube)
- Verify triangle normals point outward

```python
import numpy as np
from numpy.testing import assert_allclose
from elastica.mesh import Mesh
from elastica.utils import Tolerance

def test_mesh_load_stl():
    """Test loading STL file"""
    mesh = Mesh("tests/cube.stl")
    assert mesh.n_triangles > 0
    assert len(mesh.vertices) > 0

def test_mesh_compute_volume_cube():
    """Test volume computation for cube"""
    mesh = Mesh("tests/cube.stl")
    volume = mesh.compute_volume()
    expected_volume = 2.0**3  # cube with side length 2
    assert_allclose(volume, expected_volume, rtol=0.01)

def test_mesh_compute_com_cube():
    """Test COM computation for centered cube"""
    mesh = Mesh("tests/cube.stl")
    com = mesh.compute_center_of_mass()
    expected_com = np.array([0.0, 0.0, 0.0])
    assert_allclose(com, expected_com, atol=Tolerance.atol())

# Add more tests...
```

#### Step 4.2: Unit Tests for MeshRigidBody

**File**: `tests/test_rigid_body/test_mesh_rigid_body.py`

**Tests**:
- Initialization (verify all properties set correctly)
- Transform computation (verify rotation matrices)
- Query closest points (compare material frame vs world frame queries)
- `update_accelerations()` method
- Energy computations

```python
import numpy as np
from numpy.testing import assert_allclose
from elastica.rigidbody import MeshRigidBody
from elastica.mesh import Mesh
from elastica.utils import Tolerance

def test_mesh_rigid_body_initialization():
    """Test MeshRigidBody initialization"""
    mesh = Mesh("tests/cube.stl")
    com = np.array([0.0, 0.0, 0.0])
    inertia = np.eye(3) * 1.0
    density = 1000.0
    volume = 8.0

    mesh_body = MeshRigidBody(
        mesh=mesh,
        center_of_mass=com,
        mass_second_moment_of_inertia=inertia,
        density=density,
        volume=volume,
    )

    assert_allclose(mesh_body.position_collection[:, 0], com, atol=Tolerance.atol())
    assert_allclose(mesh_body.mass, density * volume, atol=Tolerance.atol())

def test_query_closest_points():
    """Test closest point query"""
    mesh = Mesh("tests/cube.stl")
    mesh_body = MeshRigidBody(...)

    # Query point above cube
    query_pts = np.array([[0.0, 0.0, 2.0]])
    closest, dists, normals = mesh_body.query_closest_points(query_pts)

    # Closest point should be on top face
    assert_allclose(closest[0, 2], 1.0, atol=0.01)  # Top of cube at z=1

# Add more tests...
```

#### Step 4.3: Integration Tests for RodMeshContact

**File**: `tests/test_contact/test_rod_mesh_contact.py`

**Tests**:
- Rod penetrating mesh → positive contact force
- Rod not touching mesh → zero force
- Force/torque conservation (equal and opposite)
- Convergence test (rod settling on mesh surface)
- Frozen mesh regression test (mesh does not move under rod impact)

```python
import numpy as np
from numpy.testing import assert_allclose
from elastica import CosseratRod, MeshRigidBody, Mesh
from elastica.contact_forces import RodMeshContact

def test_rod_mesh_no_contact():
    """Test that no forces applied when rod not touching mesh"""
    # Create rod far from mesh
    rod = CosseratRod.straight_rod(...)
    mesh = Mesh("tests/cube.stl")
    mesh_body = MeshRigidBody(...)

    contact = RodMeshContact(k=1e4, nu=10)

    # Store initial forces
    initial_rod_forces = rod.external_forces.copy()
    initial_mesh_forces = mesh_body.external_forces.copy()

    # Apply contact
    contact.apply_contact(rod, mesh_body)

    # Forces should be unchanged
    assert_allclose(rod.external_forces, initial_rod_forces, atol=Tolerance.atol())
    assert_allclose(mesh_body.external_forces, initial_mesh_forces, atol=Tolerance.atol())

def test_rod_mesh_penetration_force():
    """Test contact force when rod penetrates mesh"""
    # Create rod penetrating mesh
    # ... setup ...

    # Apply contact
    contact.apply_contact(rod, mesh_body)

    # Check that forces are non-zero and opposite
    # ... assertions ...

def test_frozen_mesh_regression():
    """Frozen mesh regression test (mesh should not move)"""
    # Setup rod above mesh and enable contact
    rod = CosseratRod.straight_rod(...)
    mesh = Mesh("tests/cube.stl")
    mesh_body = MeshRigidBody(...)
    initial_com = mesh_body.position_collection[:, 0].copy()

    # Make mesh effectively immovable
    mesh_body.mass = np.float64(1e20)
    mesh_body.inv_mass_second_moment_of_inertia[:] = 0.0

    # Integrate a short window
    # ... run a few steps ...

    # Ensure mesh COM does not move
    assert_allclose(mesh_body.position_collection[:, 0], initial_com, atol=1e-8)

# Add more tests...
```

---

## 4. Implementation Sequence & Priorities

### Recommended Order:

1. ✅ **Phase 1.1** - Mesh initializer (foundation)
2. ✅ **Phase 1.2** - MeshRigidBody (core class)
3. ✅ **Phase 2.1** - `_calculate_contact_forces_rod_mesh()` (Numba function)
4. ✅ **Phase 2.2** - `RodMeshContact` (high-level contact class)
5. ✅ **Phase 4.1** - Unit tests for mesh initialization
6. ✅ **Phase 4.2** - Unit tests for MeshRigidBody
7. ✅ **Phase 3.2** - Mesh freefall example
8. ✅ **Phase 3.3** - Rod-mesh collision example
9. ✅ **Phase 3.4** - Frozen mesh regression example
10. ✅ **Phase 3.5** - Post-processing utilities
11. ✅ **Phase 4.3** - Integration tests for contact

---

## 5. Potential Problems & Mitigations

| Problem | Impact | Mitigation |
|---------|--------|------------|
| **Open3D version compatibility** | Users may have different versions | Document required version (>= 0.13), add version check in `__init__` |
| **Numba-Open3D incompatibility** | Cannot use raytracing in JIT | Already addressed: two-stage approach (Python query → Numba forces) |
| **BVH rebuild performance** | Expensive for large meshes | Store mesh in material frame, apply world→mesh rotation/translation to queries |
| **Normal direction ambiguity** | Inward vs outward normals | Use `(query - closest)/dist` normal, fallback to triangle normals when degenerate |
| **Non-watertight meshes** | Invalid volume/COM/inertia | Warn on load and use OBB fallback for properties |
| **Memory for large meshes** | O(n) storage for vertices/triangles | Consider mesh decimation tools, warn users about memory usage |
| **Collision detection accuracy** | Missed collisions with thin rods | Use adaptive time-stepping or smaller time steps |
| **Force distribution at ends** | 2/3, 4/3 factors | Follow existing rod-cylinder pattern exactly |
| **Type checking with protocols** | MeshRigidBody not RodType | Add to `_allowed_system_two` in contact class |

---

## 6. Files to Create/Modify

### New Files (13 files):

```
elastica/mesh/__init__.py
elastica/mesh/mesh_initializer.py
elastica/rigidbody/mesh_rigid_body.py
examples/MeshCase/__init__.py
examples/MeshCase/mesh_freefall.py
examples/MeshCase/mesh_rod_collision.py
examples/MeshCase/mesh_frozen_contact.py
examples/MeshCase/post_processing.py
tests/test_mesh/__init__.py
tests/test_mesh/test_mesh_initializer.py
tests/test_rigid_body/test_mesh_rigid_body.py
tests/test_contact/test_rod_mesh_contact.py
docs/api/mesh.rst
```

### Modified Files (5 files):

```
elastica/__init__.py                  # Add Mesh, MeshRigidBody exports
elastica/rigidbody/__init__.py        # Add MeshRigidBody import
elastica/contact_forces.py            # Add RodMeshContact class
elastica/_contact_functions.py       # Add _calculate_contact_forces_rod_mesh()
pyproject.toml                        # Add open3d dependency
```

---

## 7. Dependencies

Add to `pyproject.toml`:

```toml
[project]
dependencies = [
    ...existing dependencies...
    "open3d>=0.13.0",
]
```

**Note**: Open3D is a hard dependency for mesh features; installation must include it.

---

## 8. Documentation Updates

### 8.1 API Reference

1. **New Module Documentation**:
   - `docs/api/mesh.rst` - Document `Mesh` class and helper functions
   - Add `MeshRigidBody` to `docs/api/rigidbody.rst`
   - Add `RodMeshContact` to `docs/api/contact.rst`

2. **Docstring Standards**:
   - Follow NumPy docstring conventions
   - Include type hints
   - Provide usage examples in docstrings

### 8.2 Examples Documentation

1. **Example Gallery**:
   - Add mesh examples to documentation gallery
   - Include rendered images/videos
   - Provide complete runnable scripts

2. **Tutorial Pages**:
   - "Working with Mesh Rigid Bodies" tutorial
   - "Rod-Mesh Contact Mechanics" tutorial
   - "Visualizing Mesh Simulations" guide

### 8.3 User Guide Updates

1. **Installation Guide**:
   - Add Open3D installation instructions
   - Note system requirements

2. **Workflow Guide**:
   - Update workflow to include mesh rigid bodies
   - Add section on mesh loading and property computation

---

## Summary

This implementation plan provides a **complete, conflict-free approach** to adding mesh features to PyElastica using Open3D. Key innovations:

✅ **No BVH rebuild** - Transforms applied to queries, not geometry
✅ **Numba compatible** - Two-stage query → force calculation
✅ **Type-safe** - Proper integration with existing type checking
✅ **Well-tested** - Comprehensive unit and integration tests
✅ **Documented** - Examples and post-processing utilities

The implementation follows all existing patterns in PyElastica while addressing the unique challenges of mesh-based rigid bodies and contact mechanics.

---

## Appendix A: Key Design Patterns

### A.1 Transform Pattern

```python
# NEVER transform the mesh geometry (expensive BVH rebuild)
# ALWAYS transform queries instead

# ❌ BAD: Transform mesh (rebuilds BVH every timestep)
mesh_world = mesh_material.transform(T_world_from_mesh)
result = query(mesh_world, points_world)

# ✅ GOOD: Transform queries (BVH built once)
T_mesh_from_world = inv(T_world_from_mesh)
points_material = transform(points_world, T_mesh_from_world)
result = query(mesh_material, points_material)  # No rebuild!
result_world = transform(result, T_world_from_mesh)
```

### A.2 Contact Force Pattern

```python
# Two-stage approach for Numba compatibility

# Stage 1: Python (can use Open3D)
def apply_contact(rod, mesh):
    query_pts = rod.element_positions.T
    closest, dists, normals = mesh.query_closest_points(query_pts)  # Open3D

    # Stage 2: Numba (pre-computed data only)
    _calculate_forces_numba(closest, dists, normals, rod_data, mesh_data, k, nu)

@njit
def _calculate_forces_numba(closest, dists, normals, ...):
    # Pure numerical computation (no Open3D calls)
    for i in range(n_elem):
        penetration = radius[i] - dists[i]
        if penetration > 0:
            force = k * penetration * normals[i]
            # ... apply forces ...
```

### A.3 Force Distribution Pattern

```python
# Standard node force distribution (from rod-cylinder contact)

for i in range(n_elem):
    net_force = compute_force(i)

    if i == 0:  # First element
        node_forces[:, 0] -= (2/3) * net_force
        node_forces[:, 1] -= (4/3) * net_force
    elif i == n_elem - 1:  # Last element
        node_forces[:, i] -= (4/3) * net_force
        node_forces[:, i+1] -= (2/3) * net_force
    else:  # Middle elements
        node_forces[:, i] -= net_force
        node_forces[:, i+1] -= net_force
```

---

## Appendix B: Helper Functions Reference

### B.1 Volume Calculation (Divergence Theorem)

```python
def compute_mesh_volume(vertices, triangles):
    """
    Compute closed mesh volume using divergence theorem.
    V = (1/6) * sum(v_i · (v_j × v_k)) for each triangle
    """
    volume = 0.0
    for tri in triangles:
        v0, v1, v2 = vertices[tri]
        volume += np.dot(v0, np.cross(v1, v2))
    return abs(volume) / 6.0
```

### B.2 Inertia Tensor Calculation

```python
def compute_mesh_inertia(vertices, triangles, density, com):
    """
    Compute inertia tensor about COM using tetrahedral decomposition.
    Each triangle with origin forms a tetrahedron.
    """
    I = np.zeros((3, 3))
    for tri in triangles:
        v0, v1, v2 = vertices[tri] - com
        # Compute tetrahedron inertia contribution
        # ... (complex integral, see literature)
    return density * I
```

---

**End of Implementation Plan**
