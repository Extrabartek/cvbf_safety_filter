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
# from dyn_sys.MzNonlinearCar import MzNonlinearCar
from dyn_sys.MzNonlinearCar4wheels import MzNonlinearCar
from  controllers.CBVFInterpolator import CBVFInterpolator

# Create dynamics
car_params = {'m': 1900, 'Vx': 110.0/3.6, 'Lf': 1.41, 'Lr': 1.47, 'Iz': 3500, 'mu': 1.075, 'Mz': 0.1,
                               'Cf': 52000, 'Cr': 75000, 'Wf': 0.788, 'Wr': 0.782,}

dynamics = MzNonlinearCar(car_params=car_params)

# Grid parameters
x1_lim = 55
x2_lim = 25

x1_lim = x1_lim * jnp.pi / 180
x2_lim = x2_lim * jnp.pi / 180

# Create grid
grid = cbvf.Grid.from_lattice_parameters_and_boundary_conditions(
    cbvf.sets.Box(np.array([-x1_lim, -x2_lim]), np.array([x1_lim, x2_lim])),
    (401, 400)
)

# Initial values
values_vi = jnp.linalg.norm(grid.states[..., :2], axis=-1, ord=2) - 6 * jnp.pi / 180
initial_values = jnp.linalg.norm(grid.states[..., :2], axis=-1, ord=2) - 6 * (jnp.pi / 180)

# Time vector
times = np.linspace(0, -0.5, 100)

# Solver settings
gamma = 0.0
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

# Check both gradient components at safe region entry points
gradient_x1 = np.zeros(grid_cpu.shape)
gradient_x2 = np.zeros(grid_cpu.shape)
entry_times_grid = np.zeros(grid_cpu.shape)
entry_values_grid = np.zeros(grid_cpu.shape)

with jax.default_device(cpu):

    interpolator = CBVFInterpolator(grid=grid_cpu,
                                    cbvf_values=target_values_cpu,  # Use CPU version
                                    times=times_cpu, gamma=gamma)

    states_flat = grid_cpu.states.reshape(-1, 2)

    # Compute gradients at safe region entry points
    entry_times, entry_values, spatial_grads, time_grads = interpolator.compute_safe_entry_gradients_efficient(states_flat)
    # spatial_grads = interpolator.interpolate_spatial_gradient_vectorized(states_flat, times_cpu[-2])

    # Reshape results back to grid shape
    gradient_x1 = spatial_grads[:, 0].reshape(grid.shape)
    gradient_x2 = spatial_grads[:, 1].reshape(grid.shape)
    time_grads = time_grads.reshape(grid.shape)
    entry_times_grid = entry_times.reshape(grid.shape)
    entry_values_grid = entry_values.reshape(grid.shape)


# from plotly.subplots import make_subplots
#
# fig = make_subplots(
#     rows=2, cols=2,
#     subplot_titles=('∂V/∂x1', '∂V/∂x2', '∂V/∂t'),
#     specs=[[{'type': 'surface'}, {'type': 'surface'}], [{'type': 'surface'}, {'type': 'surface'}]]
# )
#
# # Add ∂V/∂x1
# fig.add_trace(
#     go.Surface(
#         z=gradient_x1,
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
#         z=gradient_x2,
#         x=grid.coordinate_vectors[0],
#         y=grid.coordinate_vectors[1],
#         colorscale="RdBu",
#         name="∂V/∂x2",
#         contours={"z": {"show": True, "start": 0.0, "end": 0.0, "size": 1}}
#     ),
#     row=1, col=2
# )
#
# # Add ∂V/∂t
# fig.add_trace(
#     go.Surface(
#         z=time_grads,
#         x=grid.coordinate_vectors[0],
#         y=grid.coordinate_vectors[1],
#         colorscale="RdBu",
#         name="∂V/∂t",
#         contours={"z": {"show": True, "start": 0.0, "end": 0.0, "size": 1}}
#     ),
#     row=2, col=1
# )
#
# fig.update_layout(
#     title=f"Gradients of the CBVF Value Function (gamma={gamma})",
#     scene=dict(xaxis_title="x1", yaxis_title="x2", zaxis_title="∂V/∂x1"),
#     scene2=dict(xaxis_title="x1", yaxis_title="x2", zaxis_title="∂V/∂x2"),
#     height=1000,
#     font=dict(size=18),
# )
# fig.show()

# Save data
savemat('data/cbvf_data_port_comp_4w_cooked.mat', {
    'grid_x1': grid_cpu.states[:, 0, 0],
    'grid_x2': grid_cpu.states[0, :, 1],
    'cbvf_values': target_values_cpu,
    'gradient_x1': gradient_x1,
    'gradient_x2': gradient_x2,
    'time_grads': time_grads,
    'entry_times': entry_times_grid,
    'times': times,
    'gamma': gamma,
    'initial_values': initial_values,
    'car_params': car_params,
    'x1_lim': x1_lim,
    'x2_lim': x2_lim
})


print("Done.")
