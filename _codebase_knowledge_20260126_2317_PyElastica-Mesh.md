Session
- Repo: PyElastica-Mesh (Python, numba, Open3D)
- Date: 2026-01-26 23:17 local

Purpose
- Cosserat-rod simulation framework with rigid bodies, contact (including triangle-mesh obstacles), symplectic time integration, and example scenes.

Repo Map (high signal)
- elastica/rod: CosseratRod core state, allocation, energy, internal force/torque assembly.
- elastica/rigidbody: RigidBodyBase plus MeshRigidBody using Open3D raycasting.
- elastica/mesh: Mesh loader/geometry properties (volume/COM/inertia) via Open3D.
- elastica/modules: Mixins wiring constraints, forcing, damping, contact, callbacks into stepper pipeline.
- elastica/timestepper: Symplectic steppers (PositionVerlet, PEFRL) and integrate helper.
- elastica/contact_forces + _contact_functions: Numba contact kernels for rod–rod, rod–mesh, etc.
- examples/MeshCase: Demonstrations of rod–mesh impact with rendering/energy logging.

Backbone Workflow (call/data flow)
1) User defines simulator class mixing BaseSystemCollection + desired modules (Constraints, Forcing, Contact, Damping, CallBacks). elastica/modules/base_system.py :: BaseSystemCollection.__init__ [L56–93]
2) Append systems (CosseratRod, MeshRigidBody, etc.); register constraints/forces/contact/damping via module helpers; finalize builds memory blocks and binds operators. elastica/modules/base_system.py :: BaseSystemCollection.finalize [L222–244]; elastica/modules/contact.py :: Contact._finalize_contact [L77–105]; elastica/modules/forcing.py :: Forcing._finalize_forcing [L65–86]; elastica/modules/damping.py :: Damping._finalize_dampers [L67–87]
3) Integrate with symplectic stepper: for each stage, advance kinematics, constrain values, compute internal forces/torques, synchronize (forces/contact), apply dynamic step, constrain rates/damping; after last stage run callbacks and zero externals. elastica/timestepper/symplectic_steppers.py :: SymplecticStepperMixin.do_step [L66–135]
4) Driver `integrate` computes dt, loops n_steps calling stepper.step; prints final time. elastica/timestepper/__init__.py :: integrate [L32–72]

Key Symbols (evidence)
- PositionVerlet step definitions (prefactors and kinematic/dynamic steps). elastica/timestepper/symplectic_steppers.py :: PositionVerlet.get_steps [L164–176]; PositionVerlet._first_kinematic_step [L180–191]; PositionVerlet._first_dynamic_step [L193–199]
- CosseratRod construction for straight rods (allocates geometry/material arrays). elastica/rod/cosserat_rod.py :: CosseratRod.straight_rod [L253–351]
- Internal force/torque assembly and acceleration update. elastica/rod/cosserat_rod.py :: CosseratRod.compute_internal_forces_and_torques [L550–604]; CosseratRod.update_accelerations [L607–627]
- Mesh loader and mass properties (watertight + OBB fallback). elastica/mesh/mesh_initializer.py :: Mesh.__init__ [L32–87]; Mesh.compute_volume [L88–103]; Mesh.compute_center_of_mass [L104–120]; Mesh.compute_inertia_tensor [L122–199]
- Mesh rigid body with raycasting scene and closest-point query. elastica/rigidbody/mesh_rigid_body.py :: MeshRigidBody.__init__ [L27–90]; MeshRigidBody._build_raycasting_scene [L92–105]; MeshRigidBody.query_closest_points [L112–187]
- Rod–mesh contact hookup and kernel. elastica/contact_forces.py :: RodMeshContact.apply_contact [L493–527]; elastica/_contact_functions.py :: _calculate_contact_forces_rod_mesh [L854–946]
- Module execution ordering hooks. elastica/modules/base_system.py :: BaseSystemCollection.synchronize [L248–255]; constrain_values [L257–263]; constrain_rates [L266–274]; apply_callbacks [L277–283]
- Damping option (analytical linear, unconditional stability wrt dt). elastica/dissipation.py :: AnalyticalLinearDamper [L80–142]
- Example pipeline for rod–mesh impact with damping + rendering. examples/MeshCase/mesh_rod_collision_stable.py :: rod_mesh_collision [L15–207]
- Dependencies incl. Open3D requirement. pyproject.toml :: dependencies [L30–42]

Data/State Model
- Rods: position_collection (3, n_nodes), velocity_collection, director_collection (3,3,n_elems), omega_collection; per-element lengths, dilatation, shear/bend strains, internal/external forces & torques; inertia stored per element, mass stored per node (end nodes get half-element mass; ring rods uniform). elastica/rod/cosserat_rod.py :: CosseratRod.__init__ [L220–245]; elastica/rod/factory_function.py :: mass assembly [L248–258]
- Rigid bodies: single node; position_collection (3,1), velocity/omega, director (3x3); mass, inertia (3x3x1), external forces/torques. elastica/rigidbody/rigid_body.py :: RigidBodyBase.__init__ [L24–51]
- MeshRigidBody stores material-frame vertices/triangles, triangle normals, raycasting scene; world transform via director+position each query. elastica/rigidbody/mesh_rigid_body.py :: MeshRigidBody.__init__ [L37–90]
- Simulation collections keep ordered systems list; module operator groups executed in fixed stage order. elastica/modules/base_system.py :: BaseSystemCollection._feature_group_* [L58–75]

Math / Algorithm Notes
- Time integration: symplectic position Verlet (second order, symmetric) with operator splitting between kinematic and dynamic updates; PEFRL also present for higher-order symplectic integration. elastica/timestepper/symplectic_steppers.py :: PositionVerlet.get_steps [L164–176]; PEFRL comments [L202–236]
- Cosserat rod mechanics: shear/stretch and bending/twist strains computed at init; internal forces/torques built each step then accelerations via mass and inertia inverses. elastica/rod/cosserat_rod.py :: CosseratRod.compute_internal_forces_and_torques [L550–604]; CosseratRod.update_accelerations [L607–627]
- Contact rod–mesh: penalty spring k with damping on normal relative speed; friction limited by min(damping*|slip_vel|, mu*|normal_force|); equal/opposite forces/torques applied to mesh unless frozen. elastica/_contact_functions.py :: _calculate_contact_forces_rod_mesh [L854–946]
- Mesh inertia: divergence-theorem integration of tetrahedra if watertight; OBB analytic inertia otherwise, rotated to world with COM shift. elastica/mesh/mesh_initializer.py :: Mesh.compute_inertia_tensor [L122–199]

Stability / Performance / Pitfalls
- Mesh centering: loader no longer recenters; meshes must be provided COM-centered or downstream mass properties/contact will be biased. elastica/mesh/mesh_initializer.py :: Mesh.__init__ [L32–87]
- Watertightness: non-watertight meshes fall back to OBB volume/COM/inertia (can overestimate slender shapes); warnings issued. elastica/mesh/mesh_initializer.py :: Mesh.__init__ [L55–63]; compute_volume [L88–103]
- Contact timestep: penalty contact (k, nu) plus friction can require small dt to avoid chatter; see stable example using dt=3e-5, k=1e4, nu=5e-2 with damping. examples/MeshCase/mesh_rod_collision_stable.py :: rod_mesh_collision [L15–207]
- External loads clearing: after each step, external_forces/torques zeroed; persistent loads must be registered as modules, not one-off assignments. elastica/timestepper/symplectic_steppers.py :: SymplecticStepperMixin.do_step [L128–134]
- Raycasting safety: closest-point query raises if any primitive_id invalid; ensure meshes cleaned (remove degenerates, normals computed). elastica/rigidbody/mesh_rigid_body.py :: MeshRigidBody.query_closest_points [L132–139]
- Damping tuning: AnalyticalLinearDamper is unconditionally stable wrt dt; start large to suppress impact oscillations then reduce for fidelity. elastica/dissipation.py :: AnalyticalLinearDamper [L80–142]

External Dependencies (uncertain behavior noted)
- numba, numpy, scipy, matplotlib, tqdm, mypy toolchain; Open3D>=0.13 for mesh IO + raycasting (API differences across versions may affect TriangleMesh.t vs legacy paths). pyproject.toml :: dependencies [L30–42]

Open Questions / Next Steps
- Validate mesh watertightness and COM alignment for custom assets; add preprocessing script if needed.
- Determine contact parameter scaling (k, nu, friction) for different mesh scales; consider automatic CFL-like suggestion based on rod stiffness and impact speed.
- Consider PEFRL vs PositionVerlet for stiff contact cases; benchmark energy drift with damping off.

Updates 2026-01-27 (deep-read)
- Geometry guard: lengths have a 1e-14 floor to avoid divide-by-zero, which slightly stretches near-collapsed elements and rescales radius from conserved volume each step. elastica/rod/cosserat_rod.py :: _compute_geometry_from_state [L715–741]
- Torque stiffness: internal torque terms scale with ε⁻³ and include Jω/e×ω and (Jω/e²)ė, so near-compression can explode torques; dt must shrink when dilatation→0. elastica/rod/cosserat_rod.py :: _compute_internal_torques [L987–1072]
- Material model defaults: shear_modulus derived from Poisson=0.5 and Kaneko αc=27/28 is baked into shear_matrix; override shear_modulus to change ν. elastica/rod/factory_function.py :: allocate [L196–218]
- Ring rods require Constraints mixin (automatically added to REQUISITE_MODULES). elastica/rod/cosserat_rod.py :: ring_rod [L401–548]
- Mesh contact robustness: closest-point query combines parity ray test, SDF, and plane test to set inside/outside sign and normals. elastica/rigidbody/mesh_rigid_body.py :: MeshRigidBody.query_closest_points [L112–187]
- Damping options: AnalyticalLinearDamper supports uniform and physical protocols (dt-unconditionally stable); LaplaceDissipationFilter provides high-frequency velocity filtering for rods. elastica/dissipation.py :: AnalyticalLinearDamper [L80–173]; elastica/dissipation.py :: LaplaceDissipationFilter.dampen_rates [L328–367]
- Contact friction cap: rod–mesh friction limited by min(velocity_damping_coefficient·|v_slip|, μ|F_n|); penalty contact remains dt-sensitive. elastica/_contact_functions.py :: _calculate_contact_forces_rod_mesh [L912–919]; elastica/_contact_functions.py :: _calculate_contact_forces_rod_mesh [L854–944]

Docs notes 2026-01-27
- Units are not enforced; keep inputs consistent and rescale very small geometries to avoid tiny dt/roundoff. docs/guide/workflow.md :: workflow intro [L3–7]
- Discretization heuristics: start with 30–50 elements per rod; dt heuristic ~0.01·dx (s/m); for wave resolution use dt = dx·sqrt(ρ/G) (shear) or dx·sqrt(ρ/E) (flexural). docs/guide/discretization.md :: Number of elements [L9–10]; Choosing dx and dt [L12–15]
- Runtime scaling: cost ∝ time steps; element count weakly affects runtime (Python overhead); rod collisions/self-contact are O(N²) and expensive. docs/guide/discretization.md :: Run time scaling [L17–23]
- Operator ordering: define contact before friction (friction depends on normal forces); order follows definition sequence. docs/guide/workflow.md :: operation order note [L151–154]
- Data output: no automatic saving—callbacks required to record trajectories; otherwise only final state remains. docs/guide/workflow.md :: callback note [L185–187]
- Rod state layout table (nodes/elements/Voronoi) clarifies where geometry/kinematics/elasticity quantities live; useful for custom modules. docs/api/rods.rst :: Cosserat Rod table [L11–36]
- Environment note: run tests/examples with the repo’s virtualenv interpreter `.venv/bin/python` to ensure correct dependencies/versions.
- Timestep hygiene: a practical stable starting dt is 1e-5 s for most examples; reduce further for stiff contact, increase only after convergence checks.
- Numerical damping: it is generally good practice to include damping (e.g., AnalyticalLinearDamper or Laplace filter) to suppress high-frequency noise/impact chatter.
- Built-in modules catalog for simulation construction:
  - Constraints: FreeBC, OneEndFixedBC, GeneralConstraint, FixedConstraint, HelicalBucklingBC (rod-only); compatibility table given. docs/api/constraints.rst :: Available Constraint [L18–38]
  - External forces: NoForces, EndpointForces, GravityForces, UniformForces/Torques, MuscleTorques, EndpointForcesSinusoidal; compatibility lists rods vs rigid bodies. docs/api/external_forces.rst :: Available Forcing [L19–49]; compatibility [L40–58]
  - Environment interactions: AnisotropicFrictionalPlane, InteractionPlane, SlenderBodyTheory (rods only). docs/api/external_forces.rst :: Available Interaction [L28–36]; compatibility [L52–58]
  - Contact classes: RodRod, RodCylinder, RodSelf, RodSphere, RodPlane (anisotropic), CylinderPlane, RodMesh; all exposed for .detect_contact_between. docs/api/contact.rst :: Available Contact Classes [L13–25]
  - Joints: FreeJoint, FixedJoint, HingeJoint (rods only). docs/api/connections.rst :: Available Connections/Joints [L13–29]
  - Damping: AnalyticalLinearDamper (rods + rigid bodies), LaplaceDissipationFilter (rods only). docs/api/damping.rst :: Available Damping [L13–28]
  - Simulator scaffolding via BaseSystemCollection plus module mixins; API surfaces in simulator.rst. docs/api/simulator.rst :: modules listing [L4–25]
- Callback guidance: saving every ~100 iterations is usually cheap; callbacks store in memory by default—consider writing to disk for long runs/many rods. docs/api/callback.rst :: Description [L11–14]
- Mesh API exposes Mesh initializer and members for volume/COM/inertia; surfaces expose Plane; rigid bodies include Cylinder, Sphere, Mesh. docs/api/mesh.rst :: members [L1–6]; docs/api/rigidbody.rst :: rigid bodies [L1–18]; docs/api/surface.rst :: plane [L1–15]
- Binder notebooks available for quick-start tutorials; additional examples in repo/examples. docs/guide/binder.md :: Binder link [entire file]
- LocalizedForceTorque note: adding torques to neighboring elements with endpoint forces can improve convergence for point loads (see issue #39). docs/advanced/LocalizedForceTorque.md :: Modified Implementation [whole page]

Doc-derived quick simulation checklist (concise)
- Build simulator class with needed mixins (Constraints/Forcing/Connections/Contact/Damping/CallBacks) over BaseSystemCollection. docs/guide/workflow.md :: Setup Simulation [L9–33]
- Create rods (straight_rod or ring_rod); choose ~30–50 elements to start; set shear_modulus if Poisson≠0.5. docs/guide/workflow.md :: Create Rods [L49–88]; docs/guide/discretization.md :: Number of elements [L9–10]
- Apply constraints, then contact, then friction/forces; order follows definition sequence. docs/guide/workflow.md :: operation order note [L151–154]
- Add forces/interactions as needed (gravity, endpoint, uniform, muscle; plane/SBT) and joints (Fixed/Hinge/Free) if connecting rods. docs/api/external_forces.rst :: Available Forcing/Interaction [L19–58]; docs/api/connections.rst :: Available Connections/Joints [L13–29]
- Add damping (AnalyticalLinearDamper default; Laplace filter to suppress high-frequency noise). docs/api/damping.rst :: Available Damping [L13–28]
- Register callbacks to save data; for long runs consider writing to disk. docs/api/callback.rst :: Description [L11–14]
- finalize(); select PositionVerlet; pick dt ≈ 0.01·dx (reduce for stiff/contact cases); integrate. docs/guide/workflow.md :: Finalize Simulator [L223–231]; Set Timestepper [L233–249]; docs/guide/discretization.md :: Choosing dx and dt [L12–15]
Cosserat Rod Deep Dive (math + implementation)
- Geometry & strains: lengths/tangents from node differences; radii resized for volume conservation; dilatation = l/l0 and Voronoi dilatation from averaged lengths; shear/stretch strain σ = e·(Q·t) − e_z. elastica/rod/cosserat_rod.py :: _compute_geometry_from_state [L715–741]; _compute_all_dilatations [L744–771]; _compute_shear_stretch_strains [L804–838]
- Bending/twist: κ obtained by rotating directors back to lab (inv-rotation) then dividing by Voronoi rest length; internal couple τ = B·(κ−κ₀). elastica/rod/cosserat_rod.py :: _compute_bending_twist_strains [L880–894]; _compute_internal_bending_twist_stresses_from_model [L897–923]
- Internal forces: shear matrix maps (σ−σ₀) to internal stress n_L; rotated into material frame, divided by dilatation, differenced (ghost-aware) to nodal internal forces. elastica/rod/cosserat_rod.py :: _compute_internal_shear_stretch_stresses_from_model [L840–877]; _compute_internal_forces [L926–983]
- Internal torques: summed terms = Δ(τ/ε³) + A[(κ×τ)·Ď/ε³] + (Qᵀt×n_L)·l̂ + (Jω/e×ω) + (Jω/e²)·de/dt. Captures curvature gradients, geometric stiffening, shear-induced couple, transport and unsteady dilatation. elastica/rod/cosserat_rod.py :: _compute_internal_torques [L987–1072]
- Accelerations: a_i = (f_int+f_ext)/m; α = J⁻¹(τ_int+τ_ext)·ε per element. elastica/rod/cosserat_rod.py :: _update_accelerations [L1076–1108]
- Energies: bending = ½(κ−κ₀)·B·(κ−κ₀)·L_v; shear = ½(σ−σ₀)·S·(σ−σ₀)·L. elastica/rod/cosserat_rod.py :: compute_bending_energy [L682–696]; compute_shear_energy [L698–708]
- Strain init nuance: at construction, shear/bending strains are initialized only for non-ring rods; ring rods defer to periodic/ghost handling during memory-block setup. elastica/rod/cosserat_rod.py :: CosseratRod.__init__ strain init guard [L229–247]; ring_rod path [L401–547]

Contact & Friction (rod-centric)
- Rod–mesh: penetration = r_i − signed_distance; normal spring k and damping ν on normal relative speed; slip friction limited by min(c_damp·|v_slip|, μ·|F_n|); loads split 2/3–4/3 at ends, equal/opposite to mesh unless frozen. elastica/_contact_functions.py :: _calculate_contact_forces_rod_mesh [L854–944]; elastica/contact_forces.py :: RodMeshContact.apply_contact [L493–527]
- Rod–rod: shortest-segment distance; compressive equilibrium force used to bias normal; penalty+damping; nodal force distribution symmetric. elastica/_contact_functions.py :: _calculate_contact_forces_rod_rod [L160–273]
- Rod–cylinder: similar penalty/damping plus Coulomb-limited slip friction; accumulates force/torque on cylinder frame. elastica/_contact_functions.py :: _calculate_contact_forces_rod_cylinder [L31–156]
- Plane anisotropic friction: splits axial vs rolling kinetic/static friction with separate μ arrays; only in-plane components survive; no-contact indices zeroed. elastica/_contact_functions.py :: _calculate_contact_forces_rod_plane_with_anisotropic_friction [L575–783]

Optimization & Layout
- All heavy kernels numba-jitted (cache=True); batch linalg loops beat einsum 3–4× (see benchmarks in comments). elastica/_linalg.py :: _batch_matvec [L31–61]; _batch_cross [L102–141]
- In-place updates for geometry, strains, forces/torques; ghost index handling keeps periodic/ring rods branchless in kernels. elastica/rod/cosserat_rod.py :: _compute_geometry_from_state [L715–741]; _compute_internal_forces [L926–983]
- State views shared with steppers (no copies): kinematic_states/dynamic_states wrap underlying arrays. elastica/rod/data_structures.py :: _RodSymplecticStepperMixin.__init__ [L53–69]
- External loads zeroed via tight loops (ns-scale). elastica/rod/cosserat_rod.py :: _zeroed_out_external_forces_and_torques [L1110–1132]
