"""Script to save CBVF data for Simulink integration"""
import pickle
import numpy as np
import jax
import jax.numpy as jnp
import os

import plotly.graph_objects as go
from scipy.io import savemat


# Set up JAX configuration
os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '1.0'
jax.config.update("jax_enable_x64", True)

import cbvf_reachability as cbvf
from dyn_sys.DvNonlinearCar import DvNonlinearCar
from controllers.CBVFInterpolator3D import CBVFInterpolator

# Create dynamics
car_params = {'m': 1380, 'Vx': 100.0/3.6, 'Lf': 1.384, 'Lr': 2.79-1.384, 'Iz': 2634.5, 'mu': 1.0, 'Mz': 2000,
              'Cf': 120000, 'Cr': 190000} # camry stats

dynamics = DvNonlinearCar(car_params=car_params)

# Grid parameters
x1_lim = 200
x2_lim = 45
x3_lim = 25

x1_lim = x1_lim * jnp.pi / 180
x2_lim = x2_lim * jnp.pi / 180
x3_lim = x3_lim * jnp.pi / 180

# These offsets define the safe region boundaries for each state dimension
# Modify these values to change the safe region size independently for each axis
x1_offset = 10 * jnp.pi / 180  # degrees converted to radians
x2_offset = 5 * jnp.pi / 180  # degrees converted to radians
x3_offset = 22 * jnp.pi / 180  # degrees converted to radians

# Create grid
grid = cbvf.Grid.from_lattice_parameters_and_boundary_conditions(
    cbvf.sets.Box(np.array([-x1_lim, -x2_lim, -x3_lim]), np.array([x1_lim, x2_lim, x3_lim])),
    (135, 135, 105)
)

x1_component = jnp.abs(grid.states[..., 0]) - x1_offset
x2_component = jnp.abs(grid.states[..., 1]) - x2_offset
x3_component = jnp.abs(grid.states[..., 2]) - x3_offset

# Initial values - now using all 3 dimensions
values_vi = jnp.maximum(jnp.maximum(x1_component, x2_component), x3_component)
initial_values = jnp.maximum(jnp.maximum(x1_component, x2_component), x3_component)

# Time vector
times = np.linspace(0, -0.35, 100)

# Solver settings
gamma = 5.0
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
    entry_times, entry_values, spatial_grads, time_grads = interpolator.compute_safe_entry_gradients_efficient(states_flat)

    # Reshape results back to grid shape
    gradient_x1 = spatial_grads[:, 0].reshape(grid.shape)
    gradient_x2 = spatial_grads[:, 1].reshape(grid.shape)
    gradient_x3 = spatial_grads[:, 2].reshape(grid.shape)
    time_grads = time_grads.reshape(grid.shape)
    entry_times_grid = entry_times.reshape(grid.shape)
    entry_values_grid = entry_values.reshape(grid.shape)

# # Plot 2D slices of the 3D data (middle slice along x3 dimension)
# from plotly.subplots import make_subplots
#
# mid_idx = grid.shape[2] // 2  # Middle slice
#
# fig = make_subplots(
#     rows=2, cols=2,
#     subplot_titles=('∂V/∂x1', '∂V/∂x2', '∂V/∂x3', '∂V/∂t'),
#     specs=[[{'type': 'surface'}, {'type': 'surface'}],
#            [{'type': 'surface'}, {'type': 'surface'}]]
# )
#
# # Add ∂V/∂x1
# fig.add_trace(
#     go.Surface(
#         z=gradient_x1[:, :, mid_idx],
#         x=grid.coordinate_vectors[0],
#         y=grid.coordinate_vectors[1],
#         colorscale="RdBu",
#         name="∂V/∂x1",
#         contours={"z": {"show": True, "start": 0.0, "end": 0.0, "size": 1}}
#     ),
#     row=1, col=1
# )
#
# # Add ∂V/∂x2
# fig.add_trace(
#     go.Surface(
#         z=gradient_x2[:, :, mid_idx],
#         x=grid.coordinate_vectors[0],
#         y=grid.coordinate_vectors[1],
#         colorscale="RdBu",
#         name="∂V/∂x2",
#         contours={"z": {"show": True, "start": 0.0, "end": 0.0, "size": 1}}
#     ),
#     row=1, col=2
# )
#
# # Add ∂V/∂x3
# fig.add_trace(
#     go.Surface(
#         z=gradient_x3[:, :, mid_idx],
#         x=grid.coordinate_vectors[0],
#         y=grid.coordinate_vectors[1],
#         colorscale="RdBu",
#         name="∂V/∂x3",
#         contours={"z": {"show": True, "start": 0.0, "end": 0.0, "size": 1}}
#     ),
#     row=2, col=1
# )
#
# # Add ∂V/∂t
# fig.add_trace(
#     go.Surface(
#         z=time_grads[:, :, mid_idx],
#         x=grid.coordinate_vectors[0],
#         y=grid.coordinate_vectors[1],
#         colorscale="RdBu",
#         name="∂V/∂t",
#         contours={"z": {"show": True, "start": 0.0, "end": 0.0, "size": 1}}
#     ),
#     row=2, col=2
# )
#
# fig.update_layout(
#     title=f"Gradients of the CBVF Value Function (gamma={gamma}) - Middle x3 slice",
#     scene=dict(xaxis_title="x1", yaxis_title="x2", zaxis_title="∂V/∂x1"),
#     scene2=dict(xaxis_title="x1", yaxis_title="x2", zaxis_title="∂V/∂x2"),
#     scene3=dict(xaxis_title="x1", yaxis_title="x2", zaxis_title="∂V/∂x3"),
#     scene4=dict(xaxis_title="x1", yaxis_title="x2", zaxis_title="∂V/∂t"),
#     height=1000,
#     font=dict(size=18),
# )
# fig.show()


savemat('data/cbvf_data_3d.mat', {
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