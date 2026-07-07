import os
import numpy as np
import pandas as pd
import trimesh
import matplotlib.pyplot as plt
import time
import ansys.fluent.core as pyfluent


DATA_DIR = "thickness_test"
RESULTS_DIR = "results"

# Setup
CONTACT_ANGLE = 28.75 # degrees
SURFACE_TENSION = 0.035 # N/m
GRAVITY_VEC = np.array([0, -9.81, 0]) # m/s^2
CONSISTENCY_INDEX = 19.211412303007297
POWER_LAW_INDEX = 0.18510640672280543
DENSITY = 1050 # kg/m^3
VELOCITY = 0.2 # m/s

K = 25 # Sim time constant
PLOT_FREQ = 10


def calculate_sim_time(mesh):
    L_area = np.sqrt(mesh.area / np.pi)
    L_feret = np.sqrt(mesh.convex_hull.projected([0, 1, 0]).area / np.pi)

    tau_v = DENSITY * L_area ** (POWER_LAW_INDEX + 1) / CONSISTENCY_INDEX * VELOCITY ** -(POWER_LAW_INDEX - 1) # viscous diffusion time
    tau_c = L_feret / VELOCITY # convective time

    tau = tau_v / tau_c

    return tau / K * 2

def get_rotation_axis(mesh, index=0):
    bounding_box = mesh.bounding_box.extents
    axis = np.argmin(bounding_box)
    if axis == 2:
        rot_ax = [1, 0, 0]
        # rot_angle = np.deg2rad(-90) if index == 0 else np.deg2rad(90)
        rot_angle = np.deg2rad(-30) if index == 0 else np.deg2rad(30)
    elif axis == 0:
        \
        rot_ax = [0, 0, 1]
        # rot_angle = np.deg2rad(90) if index == 0 else np.deg2rad(-90)
        rot_angle = np.deg2rad(30) if index == 0 else np.deg2rad(-30)
    else:
        rot_ax = [0, 1, 0]
        # rot_angle = 0 if index == 0 else np.deg2rad(180)
        rot_angle = np.deg2rad(60) if index == 0 else np.deg2rad(120)
    return rot_ax, rot_angle

def get_random_rotation(mesh):
    # bounding_box = mesh.bounding_box.extents
    # axis = np.argmin(bounding_box)
    axis = np.random.choice([0, 1, 2])
    if axis == 2:
        rot_ax = [1, 0, 0]
        rot_angle = np.random.uniform(-90, 90)
        rot_angle = np.deg2rad(rot_angle)
    elif axis == 0:
        rot_ax = [0, 0, 1]
        rot_angle = np.random.uniform(-90, 90)
        rot_angle = np.deg2rad(rot_angle)
    else:
        rot_ax = [0, 1, 0]
        rot_angle = np.random.uniform(-90, 90)
        rot_angle = np.deg2rad(rot_angle)
    return rot_ax, rot_angle


def filter_largest_components(mesh):
    """
    Keep the n largest connected components (by face count) in the mesh.
    Optionally, filter out components with fewer than min_faces.
    """
    min_faces = int(np.ceil(len(mesh.faces) * 0.3))
    # Get connected components as lists of face indices
    components = trimesh.graph.connected_components(mesh.face_adjacency, min_len=min_faces)
    # Sort by size (number of faces)
    components = sorted(components, key=len, reverse=True)
    # Keep only the n largest
    faces_to_keep = np.concatenate(components)
    # Create a new mesh with only those faces
    filtered = mesh.submesh([faces_to_keep], append=True)
    return filtered

def preprocess_mesh(path, out_path, index=0):
    """
    Rotate a mesh around a specific axis by a given angle (in degrees).
    axis: 'x', 'y', or 'z'
    """
    mesh = trimesh.load_mesh(path)
    # Keep only the largest connected component
    mesh = filter_largest_components(mesh)
    mesh.update_faces(mesh.unique_faces())
    mesh.remove_unreferenced_vertices()
    mesh.update_faces(mesh.nondegenerate_faces())
    mesh.remove_infinite_values()
    mesh.apply_translation(-mesh.centroid)
    # rot_ax, rot_angle = get_rotation_axis(mesh, index)
    rot_ax, rot_angle = get_random_rotation(mesh)
    rot = trimesh.transformations.rotation_matrix(rot_angle, rot_ax)
    mesh.apply_transform(rot)
    tau = calculate_sim_time(mesh)
    mesh.export(out_path)

    return tau, mesh.bounding_box.extents

def plot_data(data_dict, path):
    x = data_dict.get("Time")
    for key in data_dict.keys():
        if key != "Time":
            y = data_dict.get(key)
            plt.plot(x, y)
            plt.xlabel("Time")
            plt.ylabel(key)
            plt.title(f"Plot of {key} vs Time")
            plt.grid()
            name = f"{key}_vs_Time.png"
            plt.savefig(os.path.join(path, name))
            plt.close()

def simulation_setup(path, check_processed=True):
    ## ----- VARIABLES -----

    # Solver
    step_size = 4e-4
    max_iter_per_time_step = 50
    data_freq = 50

    base_dir, file_name = os.path.split(path)
    root, _ = os.path.splitext(file_name)
    result_dir = os.path.join(base_dir, "..", RESULTS_DIR)
    save_dir = os.path.join(result_dir, root)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(save_dir, exist_ok=True)

    with open(os.path.join(save_dir, "params.txt"), "w") as f:
        f.write(f"Contact angle (deg): {CONTACT_ANGLE}\n")
        f.write(f"Gravity vector (m/s^2): {GRAVITY_VEC.tolist()}\n")
        f.write(f"Consistency index: {CONSISTENCY_INDEX}\n")
        f.write(f"Power law index: {POWER_LAW_INDEX}\n")
        f.write(f"Velocity (m/s): {VELOCITY}\n")
        f.write(f"Step size: {step_size}\n")
        f.write(f"Max iterations per time step: {max_iter_per_time_step}\n")

    for i in range(3):
        ## ----- PREPROCESSING -----
        shape_name = root.lower().replace(" ", "_") + f"_{i}"
        cur_save_dir = os.path.join(save_dir, shape_name)
        if os.path.isfile(os.path.join(cur_save_dir, "data.csv")) and check_processed:
            print(f"Skipping {shape_name}: data.csv already exists.")
            continue
        os.makedirs(cur_save_dir, exist_ok=True)
        out_path = os.path.join(cur_save_dir, f"{shape_name}.stl")
        tau, bounding_box = preprocess_mesh(path, out_path, index=i)

        x_max, y_max, z_max = 0.9 * bounding_box / 1e3
        x_min, y_min, z_min = -x_max, -y_max, -z_max

        n_steps = int(np.ceil(tau / step_size))

        if i == 0:
            with open(os.path.join(save_dir, "simulation_info.txt"), "a") as f:
                f.write(f"Simulation Time: {tau:.6f} s\n")
                f.write(f"Number of time steps: {n_steps}\n")
                f.write(f"Time step size: {step_size} s\n")
                f.write(f"Characteristic Length: {max(bounding_box):.6f} mm\n")

        config_dict = pyfluent.logger.get_default_config()
        config_dict['handlers']['pyfluent_file']['filename'] = os.path.join(cur_save_dir, f'out.log')
        pyfluent.logger.enable(custom_config=config_dict)
        pyfluent.logger.set_global_level('INFO')

        try:
            ## ----- MESHING -----

            meshing = pyfluent.launch_fluent(mode=pyfluent.FluentMode.MESHING, processor_count=10, product_version="25.2.0", ui_mode=pyfluent.UIMode.NO_GUI)

            workflow = meshing.workflow

            # Initialize Workflow
            workflow.InitializeWorkflow(WorkflowType=r'Fault-tolerant Meshing')

            # Import CAD Model
            workflow.TaskObject['Import CAD and Part Management'].Arguments.set_state({r'Append': False, r'Context': 0,r'FMDFileName': out_path, r'ObjectSetting': r'DefaultObjectSetting',})
            workflow.TaskObject['Import CAD and Part Management'].Execute()

            # Describe Geometry and Flow
            workflow.TaskObject['Describe Geometry and Flow'].Arguments.set_state({r'AddEnclosure': r'Yes',r'CloseCaps': r'No',r'DescribeGeometryAndFlowOptions': {r'MovingObjects': r'No',},r'LocalRefinementRegions': r'No',})
            workflow.TaskObject['Describe Geometry and Flow'].UpdateChildTasks(SetupTypeChanged=False)
            workflow.TaskObject['Describe Geometry and Flow'].Execute()

            # Create External Flow Boundaries
            workflow.TaskObject['Create External Flow Boundaries'].Arguments.set_state({r'BoundingBoxObject': {r'YmaxRatio': 2,r'YminRatio': 10,},r'ObjectSelectionList': [shape_name],})
            workflow.TaskObject['Create External Flow Boundaries'].Execute()

            # Indentify Regions
            workflow.TaskObject['Identify Regions'].Arguments.set_state({r'MaterialPointsName': r'fluid-region',r'ObjectSelectionList': [r'tunnel'],})
            workflow.TaskObject['Identify Regions'].AddChildAndUpdate(DeferUpdate=False)

            # Define Leakage Threshold
            workflow.TaskObject['Define Leakage Threshold'].AddChildAndUpdate(DeferUpdate=False)

            # Update Region Settings
            workflow.TaskObject['Update Region Settings'].Arguments.set_state({r'AllRegionFilterCategories': [r'2', r'1'],r'AllRegionLeakageSizeList': [r'none', r'none'],r'AllRegionLinkedConstructionSurfaceList': [r'n/a', r'no'],r'AllRegionMeshMethodList': [r'none', r'wrap'],r'AllRegionNameList': [shape_name, r'fluid-region'],r'AllRegionOversetComponenList': [r'no', r'no'],r'AllRegionSourceList': [r'object', r'mpt'],r'AllRegionTypeList': [r'void', r'fluid'],r'AllRegionVolumeFillList': [r'none', r'hexcore'],})
            workflow.TaskObject['Update Region Settings'].Execute()

            # Set and Create Surface Mesh
            workflow.TaskObject['Choose Mesh Control Options'].Execute()
            workflow.TaskObject['Generate the Surface Mesh'].Arguments.set_state({r'AdvancedOptions': True,})
            workflow.TaskObject['Generate the Surface Mesh'].Execute()

            # Update Boundaries - Define Inlets and Outlets
            workflow.TaskObject['Update Boundaries'].Arguments.set_state({r'BoundaryZoneList': [r'tunnel-ymin', r'tunnel-ymax'],r'BoundaryZoneTypeList': [r'pressure-outlet', r'velocity-inlet'],r'OldBoundaryZoneList': [r'tunnel-ymin', r'tunnel-ymax'],r'OldBoundaryZoneTypeList': [r'wall', r'wall'],})
            workflow.TaskObject['Update Boundaries'].Execute()

            # Add Inflation
            workflow.TaskObject['Add Boundary Layers'].Arguments.set_state({r'LocalPrismPreferences': {r'Continuous': r'Continuous',},})
            workflow.TaskObject['Add Boundary Layers'].AddChildAndUpdate(DeferUpdate=False)

            # Generate Volume Mesh
            workflow.TaskObject['Generate the Volume Mesh'].Arguments.set_state({r'AdvancedOptions': True,r'AllRegionNameList': [shape_name, r'fluid-region'],r'AllRegionSizeList': [r'2.8247793', r'2.8247793'],r'AllRegionVolumeFillList': [r'none', r'hexcore'],r'QualityMethod': r'Enhanced Orthogonal',})
            workflow.TaskObject['Generate the Volume Mesh'].Arguments.set_state({r'AdvancedOptions': True,})
            workflow.TaskObject['Generate the Volume Mesh'].Execute()

            solver = meshing.switch_to_solver()

            ## ----- FLUENT SOLVER -----

            ## General Settings
            solver.setup.general.solver.time.set_state("unsteady-1st-order")
            solver.setup.general.operating_conditions.gravity.enable = True
            solver.setup.general.operating_conditions.gravity.components.set_state(GRAVITY_VEC)

            # Viscous Model
            solver.setup.models.viscous.model.set_state("laminar")

            # Add Sauce Material
            sauce = solver.setup.materials.fluid.create("sauce")
            sauce.viscosity.option.set_state("non-newtonian-power-law")
            sauce.viscosity.non_newtonian_power_law.consistency_index.set_state(CONSISTENCY_INDEX)
            sauce.viscosity.non_newtonian_power_law.power_law_index.set_state(POWER_LAW_INDEX)
            sauce.density.value.set_state(DENSITY)
            sauce.chemical_formula.set_state("")

            # Enable Multiphase Model
            solver.setup.models.multiphase.models = "vof"
            solver.tui.define.phases.set_domain_properties.change_phases_names('sauce', 'air')
            solver.tui.define.phases.set_domain_properties.phase_domains.sauce.material('yes')
            solver.tui.define.phases.set_domain_properties.phase_domains.air.material('yes')

            # Surface Tension
            solver.tui.define.phases.set_domain_properties.interaction_domain.forces.surface_tension.sfc_tension_coeff('yes', 'constant', str(SURFACE_TENSION))
            solver.tui.define.phases.set_domain_properties.interaction_domain.forces.surface_tension.sfc_modeling('yes')
            solver.tui.define.phases.set_domain_properties.interaction_domain.forces.surface_tension.sfc_model_type('yes')

            # Wall Adhesion
            solver.tui.define.phases.set_domain_properties.interaction_domain.forces.surface_tension.wall_adhesion('yes')

            # Setup Boundary Conditions
            solver.setup.boundary_conditions.velocity_inlet['tunnel-ymax'].phase['mixture'].momentum.velocity.set_state(VELOCITY)

            # Setup Contact Angle
            solver.setup.boundary_conditions.wall['solid-1'].phase['mixture'].multiphase.contact_angles['sauce-air'].set_state(np.deg2rad(CONTACT_ANGLE)) # 28.75 degrees

            # Solver Settings
            solver.settings.solution.methods.multiphase_numerics.solution_stabilization.execute_settings_optimization = True

            # Add Cell Register to Calculate Volume Integral
            solver.tui.solve.cell_registers.add('"enclosure"', 'type', 'hexahedron', 'inside?', 'yes', 'max-point', str(x_max), str(y_max), str(z_max), 'min-point', str(x_min), str(y_min), str(z_min), 'create-volume-surface', 'no')

            # Initialize Solution
            solver.settings.solution.initialization.initialization_type.set_state("hybrid")
            solver.settings.solution.initialization.hybrid_initialize()

            # Patch Initialization
            solver.settings.solution.initialization.patch.calculate_patch(domain="sauce", cell_zones=["fluid-region"], variable="mp", value=1)

            # Calculation Activities
            autosave = solver.file.auto_save

            autosave.case_frequency = "if-mesh-is-modified"
            autosave.data_frequency = data_freq
            autosave.save_data_file_every.frequency_type = "time-step"
            autosave.append_file_name_with.file_suffix_type = "time-step"

            # Solver Settings
            solver.settings.solution.run_calculation.transient_controls.time_step_size.set_state(step_size)
            solver.settings.solution.run_calculation.transient_controls.max_iter_per_time_step.set_state(max_iter_per_time_step)
            solver.settings.solution.run_calculation.transient_controls.time_step_count.set_state(1)

            solver.file.batch_options.confirm_overwrite = True
            solver.file.write(
                file_name=os.path.join(cur_save_dir, f"solver_case.cas.h5"),
                file_type="case",
            )

            solver.settings.solution.report_definitions.volume.create('adhered_mass')
            report = solver.settings.solution.report_definitions.volume['adhered_mass']
            report.report_type.set_state('volume-mass')
            report.phase.set_state('sauce')
            report.cell_zones.set_state(['enclosure'])

            mass_integrals = []

            for k in range(n_steps):
                # Run simulation for 1 timestep
                solver.settings.solution.run_calculation.calculate()
                if k % PLOT_FREQ == 0:
                    output = solver.settings.solution.report_definitions.compute(report_defs=['adhered_mass'])
                    mass_value = output[0]['adhered_mass'][0]
                    mass_integrals.append(mass_value)

            ## ----- CFD-POST PROCESSING -----

            graphics = solver.settings.results.graphics

            # Set View
            contour_view = graphics.views.display_states.create("contour_view")
            contour_view.front_faces_transparent = "disable"
            contour_view.view_name = "isometric"

            # Define the contour for molecular viscosity
            viscosity_contour = solver.settings.results.graphics.contour.create(
                name="viscosity-contour"
            )
            viscosity_contour.field = "sauce-viscosity-lam"
            viscosity_contour.surfaces_list = ["solid-1"]
            # viscosity_contour.range_option.option.set_state('auto-range-off')
            # viscosity_contour.range_option.auto_range_off.clip_to_range.set_state(True)
            # viscosity_contour.range_option.auto_range_off.maximum.set_state(25.0)
            viscosity_contour.display_state_name = contour_view.name()
            viscosity_contour.display()

            graphics.views.auto_scale()
            graphics.picture.save_picture(file_name=os.path.join(cur_save_dir, f"viscosity_contour.png"))

            # Set View
            contour_view = graphics.views.display_states.create("contour_view_top")
            contour_view.front_faces_transparent = "disable"
            contour_view.view_name = "top"

            # Define the contour for molecular viscosity
            viscosity_contour_top = solver.settings.results.graphics.contour.create(
                name="viscosity-contour-top"
            )

            viscosity_contour_top.field = "sauce-viscosity-lam"
            viscosity_contour_top.surfaces_list = ["solid-1"]
            # viscosity_contour_top.range_option.option.set_state('auto-range-off')
            # viscosity_contour_top.range_option.auto_range_off.clip_to_range.set_state(True)
            # viscosity_contour_top.range_option.auto_range_off.maximum.set_state(25.0)
            viscosity_contour_top.display_state_name = contour_view.name()
            viscosity_contour_top.display()

            graphics.views.auto_scale()
            graphics.picture.save_picture(file_name=os.path.join(cur_save_dir, f"viscosity_contour_top.png"))

            # Set View
            contour_view = graphics.views.display_states.create("contour_view_front")
            contour_view.front_faces_transparent = "disable"
            contour_view.view_name = "front"

            # Define the contour for molecular viscosity
            viscosity_contour_front = solver.settings.results.graphics.contour.create(
                name="viscosity-contour-front"
            )

            viscosity_contour_front.field = "sauce-viscosity-lam"
            viscosity_contour_front.surfaces_list = ["solid-1"]
            # viscosity_contour_front.range_option.option.set_state('auto-range-off')
            # viscosity_contour_front.range_option.auto_range_off.clip_to_range.set_state(True)
            # viscosity_contour_front.range_option.auto_range_off.maximum.set_state(25.0)
            viscosity_contour_front.display_state_name = contour_view.name()
            viscosity_contour_front.display()

            graphics.views.auto_scale()
            graphics.picture.save_picture(file_name=os.path.join(cur_save_dir, f"viscosity_contour_front.png"))

            solver.settings.file.write(
                file_type="case-data", file_name=os.path.join(cur_save_dir, f"data.cas.h5")
            )

            # Finish - exit solver
            solver.exit()

            data = {
                "Time": np.arange(0, n_steps, step=PLOT_FREQ) * step_size,
                "Adhesion": mass_integrals,
            }

            plot_data(data, cur_save_dir)

            df = pd.DataFrame(data)
            df.to_csv(os.path.join(cur_save_dir, f"data.csv"), index=False)
        except Exception as e:
            print(f"An error occurred while processing {shape_name}: {e}")
            continue

if __name__ == "__main__":
    if not os.getenv('FLUENT_PROD_DIR'):
        flglobals = pyfluent.setup_for_fluent(product_version="25.2.0", mode=pyfluent.FluentMode.MESHING, dimension=pyfluent.Dimension.THREE, precision="double", processor_count=10)
        pyfluent.LAUNCH_FLUENT_IP="localhost"
        globals().update(flglobals)

    cwd = os.getcwd()
    data_dir = os.path.join(cwd, DATA_DIR)
    
    for file in os.listdir(data_dir):
        file_path = os.path.join(data_dir, file)
        simulation_setup(file_path)

    # files = ["Cavatelli", "Cannolichi Rigati", "Rachette", "Orecchiette", "Farfalle", "Anellini", "Maccheroni", "Farfalline", "Ziti"]

    # for file in files:
    #     file_path = os.path.join(data_dir, f"{file}.stl")
    #     print(f"Processing {file_path}...")
    #     simulation_setup(file_path)