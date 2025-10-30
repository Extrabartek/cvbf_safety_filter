"""Script to save CBVF data for Simulink integration - Matplotlib version"""
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import os
import time

# Matplotlib imports
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from skimage.measure import marching_cubes

from scipy.io import savemat

# Set up JAX configuration
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '1.0'
jax.config.update("jax_enable_x64", True)

import cbvf_reachability as cbvf
from dyn_sys.DVNonlinearCar4wheels import DvNonlinearCar
from controllers.CBVFInterpolator3D import CBVFInterpolator


def find_steady_state_by_simulation(dyn, steering_angles, sim_time=1200.0, dt=0.005):
    """Find steady-state by running simulation until convergence with progress bar"""
    from tqdm import tqdm

    cpu = jax.devices('cpu')[0]
    gpu = jax.devices('gpu')[0]
    # Ensure steering angles are on CPU for simulation
    unique_deltas = jnp.unique(jax.device_put(steering_angles, cpu))
    print(f"Computing steady-state for {len(unique_deltas)} unique steering angles")

    with jax.default_device(cpu):
        def simulate_single_delta(delta):
            """Simulate steady-state for a single steering angle"""

            def cond_fun(carry):
                state, step = carry
                return step < int(sim_time / dt)

            def body_fun(carry):
                state, step = carry
                state_dot = dyn.open_loop_dynamics(state, step * dt)
                # state_dot = 0
                new_state = state + state_dot * dt
                new_state = new_state.at[2].set(state[2] + jnp.clip(delta - state[2], -0.0025, 0.0025))
                return new_state, step + 1

            initial_state = jnp.array([0.0, 0.0, 0.0])
            final_state, _ = jax.lax.while_loop(cond_fun, body_fun, (initial_state, 0))
            return final_state[:2]

        # Process in chunks for progress tracking
        chunk_size = max(1, len(unique_deltas) // 10)  # 20 progress updates
        chunks = [unique_deltas[i:i + chunk_size] for i in range(0, len(unique_deltas), chunk_size)]

        print("Running vectorized simulation...")
        time_start = time.time()

        all_results = []
        with tqdm(total=len(unique_deltas), desc="Computing steady states") as pbar:
            for chunk in chunks:
                vectorized_simulate = jax.vmap(simulate_single_delta)
                chunk_results = vectorized_simulate(chunk)
                all_results.append(chunk_results)
                pbar.update(len(chunk))

        # Concatenate all results
        steady_states = jnp.concatenate(all_results, axis=0)

        time_end = time.time()
        print(f"Simulation completed in {time_end - time_start:.2f} seconds.")
    return jax.device_put(steady_states[:, 0], gpu), jax.device_put(steady_states[:, 1], gpu)


# Create dynamics
car_params = {'m': 1708, 'Vx': 100.0 / 3.6, 'Lf': 1.536, 'Lr': 1.575, 'Iz': 2985.216, 'mu': 1.0, 'Mz': 5000,
              'Cf': 1.5745 * 10e5, 'Cr': 1.6426 * 10e5, 'Wf': 0.75, 'Wr': 0.75}  # bmw stats

dynamics = DvNonlinearCar(car_params=car_params)

plotting = False

# Grid parameters
x1_lim = 70
x2_lim = 45
x3_lim = 15

x1_lim = x1_lim * jnp.pi / 180
x2_lim = x2_lim * jnp.pi / 180
x3_lim = x3_lim * jnp.pi / 180

# These offsets define the safe region boundaries for each state dimension
# Modify these values to change the safe region size independently for each axis
x1_offset = 10 * jnp.pi / 180  # degrees converted to radians
x2_offset = 10 * jnp.pi / 180  # degrees converted to radians
x3_offset = 22.5 * jnp.pi / 180  # degrees converted to radians

# Create grid
grid = cbvf.Grid.from_lattice_parameters_and_boundary_conditions(
    cbvf.sets.Box(np.array([-x1_lim, -x2_lim, -x3_lim]), np.array([x1_lim, x2_lim, x3_lim])),
    (135, 135, 105)
)

# Ensure grid data is available for CPU computation
print("Finding steady-state values by simulation on CPU...")
steady_yaw_rate, steady_side_slip = find_steady_state_by_simulation(dynamics, grid.states[..., 2])

# Extract states - corrected order based on your DvNonlinearCar class
yaw_rate = grid.states[..., 0]  # ψ̇ (x1)
side_slip = grid.states[..., 1]  # β (x2)
steering_angle = grid.states[..., 2]  # δ (x3)

# Define safe region as deviations from steady-state values
x1_component = jnp.abs(yaw_rate - steady_yaw_rate) - x1_offset  # yaw rate deviation
x2_component = jnp.abs(side_slip - steady_side_slip) - x2_offset  # side slip deviation
x3_component = jnp.abs(steering_angle) - x3_offset  # steering angle from zero

# Initial values
values_vi = jnp.maximum(jnp.maximum(x1_component, x2_component), x3_component)
initial_values = jnp.maximum(jnp.maximum(x1_component, x2_component), x3_component)

# Time vector
times = np.linspace(0, -0.6, 100)

del steady_side_slip, steady_yaw_rate

# Plot steady-state results using Matplotlib
if plotting:
    print("Creating steady-state plots...")
    cpu = jax.devices('cpu')[0]

    with jax.default_device(cpu):
        # Move plotting data to CPU
        steering_plot = jax.device_put((grid.states[..., 2])[0, 0, :], cpu)
        yaw_rate_plot = jax.device_put(steady_yaw_rate, cpu)
        side_slip_plot = jax.device_put(steady_side_slip, cpu)

        # Extract data for plotting
        steering_flat = np.array(steering_plot.flatten())
        yaw_rate_ss_flat = np.array(yaw_rate_plot.flatten())
        side_slip_ss_flat = np.array(side_slip_plot.flatten())

        # 3D scatter plot of steady-state manifold
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Convert to degrees for plotting
        steering_sample = steering_flat * 180 / np.pi
        yaw_rate_sample = yaw_rate_ss_flat * 180 / np.pi
        side_slip_sample = side_slip_ss_flat * 180 / np.pi

        scatter = ax.scatter(steering_sample, yaw_rate_sample, side_slip_sample,
                             c=steering_sample, cmap='viridis', alpha=0.6, s=1)

        ax.set_xlabel('Steering Angle [deg]', fontsize=14)
        ax.set_ylabel('Yaw Rate [deg/s]', fontsize=14)
        ax.set_zlabel('Side Slip Angle [deg]', fontsize=14)
        ax.set_title(f'Steady-State Manifold (Vx = {dynamics.car_params["Vx"]:.1f} m/s)', fontsize=16)

        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax, shrink=0.8, aspect=30)
        cbar.set_label('Steering Angle [deg]', fontsize=12)

        ax.view_init(elev=20, azim=45)
        plt.tight_layout()
        plt.show()

        # Clean up CPU plotting data
        del steering_plot, yaw_rate_plot, side_slip_plot
        del steering_flat, yaw_rate_ss_flat, side_slip_ss_flat

# Plot safe region boundary using existing matplotlib code
if plotting:
    with jax.default_device(cpu):
        # Move data to CPU for plotting
        values_cpu = jax.device_put(values_vi, cpu)
        grid_states_cpu = jax.device_put(grid.states, cpu)

        # Convert to numpy for plotting
        values_np = np.array(values_cpu)
        states_np = np.array(grid_states_cpu)

        # Extract coordinates
        yaw_rates = states_np[..., 0] * 180 / np.pi  # Convert to deg/s
        side_slips = states_np[..., 1] * 180 / np.pi  # Convert to degrees
        steering_angles = states_np[..., 2] * 180 / np.pi  # Convert to degrees

        # Extract the isosurface using marching cubes
        verts, faces, normals, values = marching_cubes(values_np, level=0.001, step_size=4)

        # Convert voxel coordinates to actual axis units
        scale_x = (yaw_rates.max() - yaw_rates.min()) / (values_np.shape[0] - 1)
        scale_y = (side_slips.max() - side_slips.min()) / (values_np.shape[1] - 1)
        scale_z = (steering_angles.max() - steering_angles.min()) / (values_np.shape[2] - 1)

        offset_x = yaw_rates.min()
        offset_y = side_slips.min()
        offset_z = steering_angles.min()

        verts[:, 0] = verts[:, 0] * scale_x + offset_x
        verts[:, 1] = verts[:, 1] * scale_y + offset_y
        verts[:, 2] = verts[:, 2] * scale_z + offset_z

        # Plotting the safe region boundary
        fig = plt.figure(figsize=(13, 13))
        plt.rcParams['font.size'] = 30
        ax = fig.add_subplot(111, projection='3d', proj_type='persp')

        mesh = Poly3DCollection(verts[faces], alpha=0.3, facecolor='red', edgecolor='none')
        ax.add_collection3d(mesh)

        # Much simpler approach - create structured grid lines
        n_grid_lines = 6

        # 1. Horizontal lines (constant steering angles)
        steer_levels = np.linspace(steering_angles.min(), steering_angles.max(), n_grid_lines)
        for steer_val in steer_levels:
            # Find the 2D contour at this steering level in the original data
            steer_idx = int((steer_val - steering_angles.min()) / (steering_angles.max() - steering_angles.min()) * (
                        values_np.shape[2] - 1))
            steer_idx = np.clip(steer_idx, 0, values_np.shape[2] - 1)

            # Extract 2D slice and find contour
            from skimage.measure import find_contours

            try:
                contours = find_contours(values_np[:, :, steer_idx], 0.001)
                for contour in contours[:1]:  # Take only the main contour
                    # Convert back to real coordinates
                    yaw_line = contour[:, 0] * scale_x + offset_x
                    slip_line = contour[:, 1] * scale_y + offset_y
                    steer_line = np.full_like(yaw_line, steer_val)

                    ax.plot(yaw_line, slip_line, steer_line, 'k-', alpha=0.6, linewidth=1.0)
            except:
                pass

        # 2. Vertical lines (constant yaw rate OR constant side slip)
        yaw_levels = np.linspace(yaw_rates.min(), yaw_rates.max(), n_grid_lines)
        for yaw_val in yaw_levels:
            # Find contour in the yaw-steering plane
            yaw_idx = int((yaw_val - yaw_rates.min()) / (yaw_rates.max() - yaw_rates.min()) * (values_np.shape[0] - 1))
            yaw_idx = np.clip(yaw_idx, 0, values_np.shape[0] - 1)

            try:
                contours = find_contours(values_np[yaw_idx, :, :], 0.001)
                for contour in contours[:1]:  # Take only the main contour
                    slip_line = contour[:, 0] * scale_y + offset_y
                    steer_line = contour[:, 1] * scale_z + offset_z
                    yaw_line = np.full_like(slip_line, yaw_val)

                    ax.plot(yaw_line, slip_line, steer_line, 'b-', alpha=0.5, linewidth=0.8)
            except:
                pass

        from matplotlib.patches import Patch

        legend_elements = [Patch(facecolor='red', alpha=0.3, label='Target Set')]
        ax.legend(handles=legend_elements, loc='upper right')

        ax.set_xlabel('Yaw Rate [deg/s]', labelpad=30)
        ax.set_ylabel('Side Slip Angle [deg]', labelpad=30)
        ax.set_zlabel('Steering Angle [deg]', labelpad=16)

        # Set axis limits based on data
        ax.set_xlim([yaw_rates.min()/2, yaw_rates.max()/2])
        ax.set_ylim([side_slips.min()/2, side_slips.max()/2])
        ax.set_zlim([steering_angles.min(), steering_angles.max()])
        ax.view_init(elev=15, azim=45)
        plt.tight_layout()
        # plt.savefig('figs/target_set_steered.pdf')
        plt.show()
        exit(0)

# Solver settings
gamma = 2.5
solver_settings = cbvf.SolverSettings.with_accuracy(
    "cbvf",
    hamiltonian_postprocessor=cbvf.solver.identity,
    gamma=gamma
)

print("Computing CBVF values...")
cbvf_values = -cbvf.solve_cbvf(
    solver_settings=solver_settings,
    dynamics=dynamics,
    grid=grid,
    times=times,
    initial_values=values_vi,
    target_values=initial_values,
)

# Get device references
cpu = jax.devices('cpu')[0]

# Move data to CPU and switch computation
target_values_cpu = jax.device_put(cbvf_values, cpu)
grid_cpu = jax.device_put(grid, cpu)
times_cpu = jax.device_put(times, cpu)

# Initialize gradient arrays for 3D
gradient_x1 = np.zeros(grid_cpu.shape)
gradient_x2 = np.zeros(grid_cpu.shape)
gradient_x3 = np.zeros(grid_cpu.shape)
entry_times_grid = np.zeros(grid_cpu.shape)
entry_values_grid = np.zeros(grid_cpu.shape)

with jax.default_device(cpu):
    interpolator = CBVFInterpolator(grid=grid_cpu,
                                    cbvf_values=target_values_cpu,
                                    times=times_cpu, gamma=gamma)

    states_flat = grid_cpu.states.reshape(-1, 3)

    # Compute gradients at safe region entry points
    entry_times, entry_values, spatial_grads, time_grads = interpolator.compute_safe_entry_gradients_efficient(
        states_flat)

    # Reshape results back to grid shape
    gradient_x1 = spatial_grads[:, 0].reshape(grid.shape)
    gradient_x2 = spatial_grads[:, 1].reshape(grid.shape)
    gradient_x3 = spatial_grads[:, 2].reshape(grid.shape)
    time_grads = time_grads.reshape(grid.shape)
    entry_times_grid = entry_times.reshape(grid.shape)
    entry_values_grid = entry_values.reshape(grid.shape)

# Plot 2D slices of the 3D gradient data using matplotlib
if plotting:
    mid_idx = grid.shape[2] // 2  # Middle slice along x3 dimension

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))

    # Convert coordinates to degrees for display
    x1_coords = grid.coordinate_vectors[0] * 180 / np.pi
    x2_coords = grid.coordinate_vectors[1] * 180 / np.pi

    # ∂V/∂x1
    im1 = ax1.contourf(x1_coords, x2_coords, gradient_x1[:, :, mid_idx].T,
                       levels=20, cmap='RdBu_r')
    ax1.contour(x1_coords, x2_coords, gradient_x1[:, :, mid_idx].T,
                levels=[0], colors='black', linewidths=2)
    ax1.set_xlabel('Yaw Rate [deg/s]', fontsize=12)
    ax1.set_ylabel('Side Slip Angle [deg]', fontsize=12)
    ax1.set_title('∂V/∂x1', fontsize=14)
    plt.colorbar(im1, ax=ax1)

    # ∂V/∂x2
    im2 = ax2.contourf(x1_coords, x2_coords, gradient_x2[:, :, mid_idx].T,
                       levels=20, cmap='RdBu_r')
    ax2.contour(x1_coords, x2_coords, gradient_x2[:, :, mid_idx].T,
                levels=[0], colors='black', linewidths=2)
    ax2.set_xlabel('Yaw Rate [deg/s]', fontsize=12)
    ax2.set_ylabel('Side Slip Angle [deg]', fontsize=12)
    ax2.set_title('∂V/∂x2', fontsize=14)
    plt.colorbar(im2, ax=ax2)

    # ∂V/∂x3
    im3 = ax3.contourf(x1_coords, x2_coords, gradient_x3[:, :, mid_idx].T,
                       levels=20, cmap='RdBu_r')
    ax3.contour(x1_coords, x2_coords, gradient_x3[:, :, mid_idx].T,
                levels=[0], colors='black', linewidths=2)
    ax3.set_xlabel('Yaw Rate [deg/s]', fontsize=12)
    ax3.set_ylabel('Side Slip Angle [deg]', fontsize=12)
    ax3.set_title('∂V/∂x3', fontsize=14)
    plt.colorbar(im3, ax=ax3)

    # ∂V/∂t
    im4 = ax4.contourf(x1_coords, x2_coords, time_grads[:, :, mid_idx].T,
                       levels=20, cmap='RdBu_r')
    ax4.contour(x1_coords, x2_coords, time_grads[:, :, mid_idx].T,
                levels=[0], colors='black', linewidths=2)
    ax4.set_xlabel('Yaw Rate [deg/s]', fontsize=12)
    ax4.set_ylabel('Side Slip Angle [deg]', fontsize=12)
    ax4.set_title('∂V/∂t', fontsize=14)
    plt.colorbar(im4, ax=ax4)

    plt.suptitle(f'Gradients of the CBVF Value Function (gamma={gamma}) - Middle x3 slice',
                 fontsize=16)
    plt.tight_layout()
    plt.show()

savemat(f'data/cbvf_data_3d_{car_params['Mz']}_bmw_mu_{car_params['mu']}_tmax_{np.min(times)}.mat', {
    'grid_x1': grid_cpu.states[:, 0, 0, 0],
    'grid_x2': grid_cpu.states[0, :, 0, 1],
    'grid_x3': grid_cpu.states[0, 0, :, 2],
    'cbvf_values': target_values_cpu,
    'gradient_x1': gradient_x1,
    'gradient_x2': gradient_x2,
    'gradient_x3': gradient_x3,
    'time_grads': time_grads,
    'entry_times': entry_times_grid,
    'times': times,
    'dynamics': dynamics,
    'gamma': gamma,
    'initial_values': initial_values,
    'car_params': car_params,
    'x1_lim': x1_lim,
    'x2_lim': x2_lim,
    'x3_lim': x3_lim
})

print("Done.")