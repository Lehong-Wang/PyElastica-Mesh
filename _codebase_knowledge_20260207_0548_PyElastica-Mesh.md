# _codebase_knowledge_20260207_0548_PyElastica-Mesh

## 0) Session and Repo
- session_id: 20260207_0548
- repo root: /Users/lehongwang/Desktop/PyElastica-Mesh
- current branch: master (tracking origin/master)
- current HEAD: 2e0d58cf (packaged cosim code)
- knowledge predecessor: _codebase_knowledge_20260126_2317_PyElastica-Mesh.md
- local worktree notes (pre-existing, not edited by this pass):
  - modified: IMPLEMENTATION_LOG.md
  - modified: docs/index.rst
  - untracked: docs/advanced/CoSimulation.md
- language mix (high signal): Python-heavy simulation code, tests, docs, and rendering scripts.

Evidence:
- elastica/__init__.py :: module exports [L1-L86]
- pyproject.toml :: project/dependencies [L1-L42]

## 1) High-level Purpose (1-5 lines)
PyElastica-Mesh is a fork/extension of PyElastica focused on Cosserat rod dynamics with rigid body coupling and contact, including rod-mesh contact via Open3D ray queries and signed-distance logic. Recent repository work adds a co-simulation package (`co_sim`) that treats one frame as kinematic command input and returns impulse feedback over each external update.

Evidence:
- README.md :: project overview [L1-L24]
- elastica/contact_forces.py :: RodMeshContact [L469-L528]
- co_sim/engine.py :: CoSimEngine [L128-L306]

## 2) Repo Map (directories -> purpose)
- elastica/: core mechanics engine.
  - elastica/rod/: Cosserat rod state, constitutive model, kernels.
  - elastica/rigidbody/: rigid body types (Cylinder, MeshRigidBody).
  - elastica/mesh/: mesh loading + geometric properties.
  - elastica/modules/: simulator mixins and finalize-time operator wiring.
  - elastica/timestepper/: symplectic steppers and integration loop.
  - elastica/memory_block/: block-structured contiguous storage for rods/rigid bodies.
  - elastica/_contact_functions.py + elastica/contact_forces.py: contact kernels + class wrappers.
- co_sim/: packaged co-simulation API and utilities.
- mytest/: experiments and integration scripts (legacy and current co-sim scripts).
- render_scripts/: shared rendering/post-processing utilities.
- examples/MeshCase/: rod-mesh and mesh-only demos.
- tests/: regression and unit coverage for core, mesh, contact, modules.
- docs/: user/API docs, now including CoSimulation.md in advanced guide.

Evidence:
- elastica/modules/base_system.py :: BaseSystemCollection [L34-L284]
- elastica/modules/memory_block.py :: construct_memory_block_structures [L22-L88]
- docs/index.rst :: toctree includes advanced/CoSimulation.md [L75-L82]

## 3) Primary Workflows (Backbone traces)

### 3.1 Workflow A: Simulator Build -> Finalize -> Symplectic Step
Trace:
1. User builds simulator from mixins over BaseSystemCollection.
2. Systems are appended (rod, rigid body, surface).
3. finalize() creates memory blocks and runs module finalizers.
4. stepper performs staged kinematic update -> constrain values -> compute internal forces/torques -> synchronize modules (forces/connections/contact) -> dynamic update -> constrain rates/damping -> callbacks -> zero external loads.

Key state:
- system list, block systems, feature operator groups.
- rod/rbody state arrays and module-generated operators.

Evidence:
- elastica/modules/base_system.py :: BaseSystemCollection.__init__ [L56-L93]
- elastica/modules/base_system.py :: BaseSystemCollection.finalize [L222-L245]
- elastica/modules/memory_block.py :: construct_memory_block_structures [L22-L88]
- elastica/timestepper/symplectic_steppers.py :: SymplecticStepperMixin.do_step [L79-L135]
- elastica/timestepper/__init__.py :: integrate [L32-L72]

### 3.2 Workflow B: Rod-Mesh Contact
Trace:
1. `RodMeshContact.apply_contact` computes rod element centers and queries closest points on mesh.
2. MeshRigidBody returns closest points, signed distances, and normals in world frame.
3. Numba kernel computes penetration, normal spring+damping, optional slip friction, node force distribution, and mesh reaction force/torque (unless mesh_frozen).

Key state:
- rod element centers, radii, velocities.
- mesh COM position, director, linear/angular velocity.
- signed distance + normal from ray/occupancy based query.

Evidence:
- elastica/contact_forces.py :: RodMeshContact.apply_contact [L493-L527]
- elastica/rigidbody/mesh_rigid_body.py :: MeshRigidBody.query_closest_points [L119-L189]
- elastica/_contact_functions.py :: _calculate_contact_forces_rod_mesh [L854-L956]

### 3.3 Workflow C: Co-simulation External Tick -> Internal Substeps -> Impulse Output
Trace:
1. External side provides `FrameState` command.
2. Engine writes frame command into kinematic frame state buffer.
3. Engine advances multiple internal `py_dt` steps up to target external duration.
4. A forcing hook records frame loads as impulse and clears them so frame remains kinematic.
5. Engine returns `ImpulseResult` with elapsed time and impulses.

Key state:
- command frame buffer (position/director/vel/acc/omega/alpha).
- `_ImpulseAccumulator` (linear/angular impulse).
- `_StepSizeBuffer` for variable last step in each update window.

Evidence:
- co_sim/models.py :: FrameState [L24-L43]
- co_sim/engine.py :: FrameStateBuffer [L35-L73]
- co_sim/engine.py :: RecordAndZeroFrameLoads [L106-L126]
- co_sim/engine.py :: CoSimEngine.update_frame_state [L249-L306]
- mytest/cosim_test_isaac_mock.py :: _run_isaac_loop [L77-L143]

### 3.4 Workflow D: Time-based Sampling and 4-view Rendering
Trace:
1. Simulation produces sampled trajectories and optional mean force vectors.
2. Rendering utility computes frame indices from real simulation timestamps.
3. Multiview renderer updates 3D/front/right/top views per sampled frame, optionally overlaying vectors.

Evidence:
- mytest/cosim_test_isaac_mock.py :: _buffer_to_arrays [L62-L75]
- render_scripts/post_processing.py :: _compute_render_indices [L49-L95]
- render_scripts/post_processing.py :: plot_rods_multiview [L253-L442]

## 4) Key Symbols (top 10-30)
- `elastica/modules/base_system.py :: BaseSystemCollection.finalize [L222-L245]` - seals simulator and builds memory blocks.
- `elastica/modules/constraints.py :: Constraints._finalize_constraints [L70-L128]` - materializes constraints; injects periodic boundary constraints for ring rods.
- `elastica/modules/connections.py :: Connections._finalize_connections [L84-L116]` - wires joint force/torque operators into synchronize phase.
- `elastica/modules/contact.py :: Contact._finalize_contact [L77-L105]` - wires contact operators and validates system type pairing.
- `elastica/modules/forcing.py :: Forcing._finalize_forcing [L65-L86]` - wires apply_forces/apply_torques operators.
- `elastica/modules/damping.py :: Damping._finalize_dampers [L67-L87]` - adds dampen_rates into constrain_rates phase.
- `elastica/modules/callbacks.py :: CallBacks._finalize_callback [L63-L79]` - registers callbacks and runs initial callback at t=0.
- `elastica/timestepper/symplectic_steppers.py :: SymplecticStepperMixin.do_step [L79-L135]` - core stage loop.
- `elastica/rod/cosserat_rod.py :: CosseratRod.compute_internal_forces_and_torques [L550-L605]` - rod internal model entrypoint.
- `elastica/rod/cosserat_rod.py :: _compute_geometry_from_state [L716-L741]` - updates lengths/tangents/radius with length floor.
- `elastica/rod/cosserat_rod.py :: _compute_internal_forces [L926-L983]` - constitutive stress -> nodal forces.
- `elastica/rod/cosserat_rod.py :: _compute_internal_torques [L987-L1072]` - bending/shear/transport/unsteady torque terms.
- `elastica/rod/cosserat_rod.py :: _update_accelerations [L1076-L1108]` - translational/angular accelerations.
- `elastica/rod/factory_function.py :: allocate [L11-L333]` - rod geometry/material initialization and array allocation.
- `elastica/memory_block/memory_block_rod.py :: MemoryBlockCosseratRod.__init__ [L33-L204]` - block packing for rods (incl. ring rods/periodic boundaries).
- `elastica/memory_block/memory_block_rod.py :: MemoryBlockCosseratRod._map_system_properties_to_block_memory [L471-L575]` - view mapping between systems and block memory.
- `elastica/mesh/mesh_initializer.py :: Mesh.__init__ [L30-L66]` - mesh load/cleanup/watertight detection without COM recentering.
- `elastica/mesh/mesh_initializer.py :: Mesh.compute_inertia_tensor [L101-L179]` - watertight tetrahedral inertia + OBB fallback.
- `elastica/rigidbody/mesh_rigid_body.py :: MeshRigidBody.query_closest_points [L119-L189]` - closest points + signed distances + normals.
- `elastica/contact_forces.py :: RodMeshContact.apply_contact [L493-L527]` - rod-mesh wrapper.
- `elastica/_contact_functions.py :: _calculate_contact_forces_rod_mesh [L854-L956]` - penalty+damping+friction rod-mesh kernel.
- `elastica/joint.py :: FixedJoint.apply_torques [L292-L340]` - rotational spring-damper enforcement between connected systems.
- `co_sim/models.py :: CoSimConfig [L46-L99]` - co-sim runtime parameter model.
- `co_sim/engine.py :: CoSimEngine.__init__ [L147-L235]` - scene build + joint + kinematic frame setup.
- `co_sim/engine.py :: CoSimEngine.update_frame_state [L249-L306]` - one external update execution.
- `mytest/cosim_test_isaac_mock.py :: run_demo [L191-L253]` - end-to-end packaged demo driver.
- `render_scripts/post_processing.py :: plot_rods_multiview [L253-L442]` - shared 4-view animation with optional vector overlay.

## 5) Data and State Model
Where state lives:
- Rod main state: node and element arrays (`position_collection`, `velocity_collection`, `director_collection`, `omega_collection`, strains/stresses/forces/torques).
- Rigid body state: single-node vectors and 3x3 inertia/director tensors.
- Block state: simulator-level contiguous arrays mapped back to individual systems via views.
- Co-sim command state: `FrameStateBuffer` world-frame command plus derived local angular rates.
- Co-sim reaction state: `_ImpulseAccumulator` updated each internal step.

Mutation points:
- Symplectic kinematic/dynamic operators mutate position/director/rate collections in-place.
- Constraints and damping mutate velocities/omegas (and sometimes values) after each stage.
- Contact and connections mutate `external_forces`/`external_torques` in synchronize phase.
- End-of-step clears external loads.

Ownership rules:
- After finalize, systems and memory blocks share array views; updates are aliased by design.
- Mesh geometry in MeshRigidBody is material-frame static; only pose (position/director) evolves.

Evidence:
- elastica/rod/cosserat_rod.py :: CosseratRod.__init__ [L155-L247]
- elastica/rigidbody/rigid_body.py :: RigidBodyBase.__init__ [L24-L51]
- elastica/memory_block/memory_block_rod.py :: _map_system_properties_to_block_memory [L471-L575]
- elastica/memory_block/memory_block_rigid_body.py :: _map_system_properties_to_block_memory [L181-L232]
- co_sim/engine.py :: FrameStateBuffer.apply_to_system [L55-L63]

## 6) Math/Algorithm Notes (Conservative)

### Supported
- Symplectic splitting pipeline (kinematic/dynamic alternating, stage prefactors) is explicit in stepper implementation.
  - elastica/timestepper/symplectic_steppers.py :: SymplecticStepperMixin.step_methods [L35-L60]
  - elastica/timestepper/symplectic_steppers.py :: PositionVerlet.get_steps [L164-L176]
- Cosserat rod strain/force/torque pipeline is directly encoded in numba kernels.
  - elastica/rod/cosserat_rod.py :: _compute_shear_stretch_strains [L805-L838]
  - elastica/rod/cosserat_rod.py :: _compute_bending_twist_strains [L880-L894]
  - elastica/rod/cosserat_rod.py :: _compute_internal_forces [L926-L983]
  - elastica/rod/cosserat_rod.py :: _compute_internal_torques [L987-L1072]
- Rod-mesh contact uses penalty spring, normal damping, and slip-direction friction clamped by min(velocity damping, Coulomb cap).
  - elastica/_contact_functions.py :: _calculate_contact_forces_rod_mesh [L878-L927]
- Mesh signed-distance semantics are occupancy-based for watertight meshes (negative inside), with normal sign adjusted accordingly.
  - elastica/rigidbody/mesh_rigid_body.py :: MeshRigidBody.query_closest_points [L167-L182]

### Likely
- Contact force node weighting (2/3, 4/3 at ends) is a heuristic conservation/distribution scheme reused across contact kernels, likely for endpoint consistency with nodal discretization.
  - elastica/_contact_functions.py :: _calculate_contact_forces_rod_mesh [L929-L952]
  - elastica/_contact_functions.py :: _calculate_contact_forces_rod_cylinder [L129-L149]
- Co-sim impulse is intended as external-coupling observable (mean force over update window) rather than strict fully coupled implicit interface.
  - co_sim/engine.py :: RecordAndZeroFrameLoads [L106-L126]
  - co_sim/engine.py :: CoSimEngine.update_frame_state [L300-L306]

### Hypothesis
- For dense rod-mesh scenes, lack of explicit broadphase pruning in rod-mesh path may become a cost bottleneck relative to rod-rod/rod-cylinder AABB-pruned paths.
  - elastica/contact_utils.py :: _prune_using_aabbs_rod_rod [L166-L200]
  - elastica/contact_forces.py :: RodMeshContact.apply_contact [L499-L508]
  - confirm by profiling contact-heavy scenarios with large n_elems and high mesh triangle counts.

## 7) Stability, Performance, Pitfalls
- Mesh centering contract is strict now: mesh loader does not recenter; users must provide COM-centered geometry when expected.
  - elastica/mesh/mesh_initializer.py :: Mesh.__init__ notes [L23-L27]
  - elastica/rigidbody/mesh_rigid_body.py :: MeshRigidBody notes [L21-L24]
- Non-watertight meshes fallback to OBB for volume/COM/inertia; this can materially alter effective mass properties.
  - elastica/mesh/mesh_initializer.py :: Mesh.__init__ warning path [L52-L59]
  - elastica/mesh/mesh_initializer.py :: Mesh.compute_volume [L72-L81]
  - elastica/mesh/mesh_initializer.py :: Mesh.compute_inertia_tensor fallback [L157-L179]
- Rod-mesh contact remains timestep sensitive (explicit penalty + damping + friction).
  - elastica/_contact_functions.py :: _calculate_contact_forces_rod_mesh [L878-L927]
  - examples/MeshCase/mesh_rod_collision_stable.py :: rod_mesh_collision params [L15-L22]
- Frame conventions are critical: mesh angular velocity is stored local/material frame and must be converted to world before `omega x r`.
  - elastica/_contact_functions.py :: _calculate_contact_forces_rod_mesh [L890-L895]
- External loads are zeroed after each top-level step; persistent forcing must come from modules, not one-off assignments.
  - elastica/timestepper/symplectic_steppers.py :: SymplecticStepperMixin.do_step [L131-L134]
- Co-sim update loop includes floating-point non-progress guard near target boundary to avoid hangs.
  - co_sim/engine.py :: CoSimEngine.update_frame_state [L281-L294]
- Memory blocks use `np.lib.stride_tricks.as_strided`; correctness depends on disciplined mapping and domain index handling.
  - elastica/memory_block/memory_block_rod.py :: _map_system_properties_to_block_memory [L557-L575]
  - elastica/memory_block/memory_block_rigid_body.py :: _map_system_properties_to_block_memory [L218-L232]

## 8) External Dependencies (uncertain behavior)
- `open3d`: mesh IO/cleanup, raycasting scene, occupancy queries for signed-distance sign.
  - elastica/mesh/mesh_initializer.py :: imports and usage [L7-L8, L35-L50]
  - elastica/rigidbody/mesh_rigid_body.py :: raycasting scene/query [L6-L7, L99-L112, L137-L174]
- `numba`: core computational kernels for rod/contact/constraint synchronization.
  - elastica/rod/cosserat_rod.py :: numba kernels [L715-L1132]
  - elastica/_contact_functions.py :: contact kernels [L31-L956]
  - elastica/_synchronize_periodic_boundary.py :: periodic sync kernels [L13-L83]
- `matplotlib` + `ffmpeg` writer: rendering pipeline; output depends on local ffmpeg availability.
  - render_scripts/post_processing.py :: animation writer usage [L31-L32, L407-L411, L657-L661]
  - examples/MeshCase/post_processing.py :: animation writer usage [L108-L110, L241-L245]

## 9) Git History Update (since prior KB date)
Reference window: commits after the previous knowledge snapshot date (2026-01-26).

### 2026-02-03 to 2026-02-07 key commits
- `1f19cdae` - added deep-reader skill assets and substantial rendering helpers.
- `4719d6e2` - major rod-mesh penetration fixes and removed mesh auto-recentering.
- `3e2b45d3` - additional penetration fix: world-frame angular velocity conversion in rod-mesh point velocity.
- `03294b19` - introduced initial co-simulation scripts under `mytest/`.
- `2e0d58cf` - packaged co-simulation into `co_sim/` and added wire-property comparison script.

Commit evidence:
- git show --stat 1f19cdae (skill + rendering additions)
- git show --stat 4719d6e2 (mesh/contact changes)
- git show --stat 3e2b45d3 (contact frame conversion fix)
- git show --stat 03294b19 (co-sim prototype scripts)
- git show --stat 2e0d58cf (co_sim package)

### Code-level deltas now visible at HEAD
- Mesh loader and rigid body now encode a no-recenter COM assumption.
  - elastica/mesh/mesh_initializer.py :: Mesh.__init__ [L30-L66]
  - elastica/rigidbody/mesh_rigid_body.py :: MeshRigidBody.__init__ [L31-L69]
- Rod-mesh kernel uses world-frame angular velocity for mesh point velocity.
  - elastica/_contact_functions.py :: _calculate_contact_forces_rod_mesh [L890-L895]
- Packaged co-sim API exists with engine/models/plotting split.
  - co_sim/__init__.py :: public exports [L1-L17]
  - co_sim/engine.py :: CoSimEngine [L128-L306]
  - co_sim/models.py :: datamodels [L24-L135]

## 10) Open Questions
- Should rod-mesh contact get a broadphase pruning stage (AABB/BVH element culling) before closest-point queries for long rods?
- Should mesh COM-centering precheck/utility be exposed as an explicit helper to reduce user mistakes with uncentered assets?
- Should co-sim frame constraints enforce acceleration/alpha explicitly in constrain_rates or keep current velocity/omega-only behavior?
- Should impulse recording optionally include per-substep force history directly in engine instead of observer callback plumbing?

