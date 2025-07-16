import os
import jax

os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '1.0'
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import plotly.io as pio
pio.renderers.default = "browser"

import matplotlib.pyplot as plt
import plotly.graph_objects as go
# import plotly.subplots as sp

import cbvf_reachability as cbvf

from tqdm import tqdm
from dyn_sys.MzNonlinearCar import MzNonlinearCar
from controllers.CBVF_QP import CBVFQPController
from controllers.CBVFInterpolator import CBVFInterpolator


car_params = {'m': 1430, 'Vx': 30.0, 'Lf': 1.05, 'Lr': 1.61, 'Iz': 2059.2, 'mu': 1.0, 'Mz': 1e4 * 1.5,
                               'Cf': 9 * 10e3, 'Cr': 10 * 10e3}
dynamics = MzNonlinearCar(car_params=car_params)

# limits of the grid in degrees
x1_lim = 200
x2_lim = 45

x1_lim = x1_lim * jnp.pi / 180
x2_lim = x2_lim * jnp.pi / 180

grid = cbvf.Grid.from_lattice_parameters_and_boundary_conditions(cbvf.sets.Box(np.array([-x1_lim, -x2_lim]),
                                                                           np.array([x1_lim, x2_lim])),
                                                                           (600, 600))
values_vi = jnp.linalg.norm(grid.states[..., :2], axis=-1) - 6 * jnp.pi / 180
initial_values = jnp.linalg.norm(grid.states[..., :2], axis=-1) - 6 * (jnp.pi / 180)

times = np.linspace(0, -0.25, 360)

gamma = 0.0

jax.clear_caches()
solver_settings = cbvf.SolverSettings.with_accuracy("cbvf",
                                                  hamiltonian_postprocessor=cbvf.solver.identity,
                                                  gamma=gamma)
cbvf_values = -cbvf.solve_cbvf(solver_settings=solver_settings,
                              dynamics=dynamics,
                              grid=grid,
                              times=times,
                              initial_values=values_vi,
                              target_values=initial_values,)


x_init = jnp.array([-0 * (jnp.pi/180), 15 * (jnp.pi/180)])
target = np.array([-70 * (jnp.pi/180), -30 * (jnp.pi/180)])

# Get device references
cpu = jax.devices('cpu')[0]

# Move data to CPU and switch computation
cbvf_values_cpu = jax.device_put(cbvf_values, cpu)
grid_cpu = jax.device_put(grid, cpu)
times_cpu = jax.device_put(times, cpu)


with jax.default_device(cpu):

    # coefficients of a PD controller
    k_p = 200000
    k_d = 500000

    interpolator = CBVFInterpolator(grid=grid_cpu,
                                    cbvf_values=cbvf_values_cpu,
                                    times=times_cpu, gamma=1000)

    safe_controller = CBVFQPController(value_fn=interpolator,
                                       gamma=1000,
                                       verbose=False,)

    del cbvf_values_cpu, grid_cpu

    sim_time = np.linspace(0, 0.7, 350)
    dt = sim_time[1] - sim_time[0]

    trajectory = np.zeros((len(sim_time), 2))
    trajectory_pd = np.zeros((len(sim_time), 2))
    control_his = np.zeros(len(sim_time))
    control_ref = np.zeros(len(sim_time))
    constrain_his = np.zeros(len(sim_time))
    cbvf_vals = np.zeros(len(sim_time))
    cbvf_grad_x = np.zeros((len(sim_time), 2))
    cbvf_grad_t = np.zeros(len(sim_time))
    a_term = np.zeros(len(sim_time))
    constrain_rest_hist = np.zeros(len(sim_time))

    state = x_init.copy()
    state_pd = x_init.copy()

    for i, time in enumerate(tqdm(sim_time, desc="Simulation", unit="sim_s", unit_scale=dt)):
        grad_time = interpolator.compute_safe_entry_gradients_efficient(jnp.array(state).reshape(1, -1))[0]
        _, cbvf_grad_x[i], cbvf_grad_t[i] = safe_controller.cbvf.get_value_and_gradient(state, grad_time[0])
        cbvf_vals[i], _, _ = safe_controller.cbvf.get_value_and_gradient(state, times_cpu[-1])
        # u = k_p * (target[0] - state[0]) + k_d * (target[1] - state[1])
        u = -15000
        control_ref[i] = u
        u_prev = 0.0 if i == 0 else control_his[i - 1]
        u_safe, constrain_his[i], a_term[i] = safe_controller.compute_safe_control(state=state, time=time,
                                                      u_ref=u, dynamics=dynamics,
                                                      u_prev=u_prev,
                                                      u_max_mag = 15000,
                                                      gradient_time=grad_time,)
        state_dot = dynamics.open_loop_dynamics(state=state, time=time) + dynamics.control_jacobian(state, time) @ np.array([u_safe])
        state_dot_pd = dynamics.open_loop_dynamics(state=state_pd, time=time) + dynamics.control_jacobian(state_pd, time) @ np.array([u])
        state = state + state_dot * dt
        state_pd = state_pd + state_dot_pd * dt

        trajectory[i, :] = state
        trajectory_pd[i, :] = state_pd
        control_his[i] = u_safe

    constrain_rest_hist = constrain_his - a_term

# Plot results
fig, axes = plt.subplots(8, 1, figsize=(12, 12))

# State trajectory
axes[0].plot(sim_time, trajectory[:, 0], label='Yaw Rate (x1)')
axes[0].plot(sim_time, trajectory[:, 1], label='Side Slip (x2)')
axes[0].set_ylabel('States')
axes[0].legend()
axes[0].grid(True)

# Control input
axes[1].plot(sim_time, control_his, 'r-', label='Control Input')
axes[1].plot(sim_time, control_ref, 'g--', label='Reference Control')
axes[1].axhline(dynamics.control_space.lo, color='black', linestyle='--', label='Control Limit')
axes[1].axhline(dynamics.control_space.hi, color='black', linestyle='--')
axes[1].set_ylabel('Control')
# axes[1].legend()
axes[1].grid(True)

# QP Constraints
axes[2].plot(sim_time, constrain_his, 'r-', label='QP Constraint')
axes[2].axhline(0, color='black', linestyle='--', label='Zero Constraint')
axes[2].set_ylabel('QP Constraint')
axes[2].legend()
axes[2].grid(True)

# CBVF Values
axes[3].plot(sim_time, cbvf_vals, 'r-', label='CBVF Value')
axes[3].axhline(0, color='black', linestyle='--', label='Zero Constraint')
axes[3].set_ylabel('CBVF Value')
axes[3].legend()
axes[3].grid(True)

# CBVF Gradients X
axes[4].plot(sim_time, cbvf_grad_x[:, 0], 'r-', label='CBVF Grad X1')
axes[4].plot(sim_time, cbvf_grad_x[:, 1], 'b-', label='CBVF Grad X2')
axes[4].set_ylabel('CBVF Gradients')
axes[4].legend()
axes[4].grid(True)

# CBVF Gradients Tgamma
axes[5].plot(sim_time, cbvf_grad_t, 'r-', label='CBVF Grad T')
axes[5].set_ylabel('CBVF Gradients')
axes[5].legend()
axes[5].grid(True)

# A Term in Constraint
axes[6].plot(sim_time, a_term, 'r-', label='a_term')
axes[6].set_ylabel('A Term')
axes[6].legend()
axes[6].grid(True)

# Controlled Constraint
axes[7].plot(sim_time, constrain_rest_hist, 'r-', label='Controlled Constraint')
axes[7].set_ylabel('Rest')
axes[7].legend()
axes[7].grid(True)

plt.tight_layout()
plt.show()

# Phase portrait
plt.figure(figsize=(8, 8))
plt.plot(trajectory[:, 0], trajectory[:, 1], 'b-', linewidth=2)
plt.plot(trajectory_pd[:, 0], trajectory_pd[:, 1], 'k--', linewidth=2, label='No Safety Controller')
plt.scatter(x_init[0], x_init[1], color='g', s=100, marker='o', label='Start')
plt.scatter(trajectory[-1, 0], trajectory[-1, 1], color='r', s=100, marker='s', label='End')
# plt.scatter(target[0], target[1], color='y', s=100, marker='o', label='Target')
# plt.plot(trajectory_pd[:, 0], trajectory_pd[:, 1], 'k--', linewidth=2, label='PD Trajectory')


plt.xlabel('Position')
plt.ylabel('Velocity')
plt.title('Phase Portrait')
plt.legend()
plt.grid(True)
# plt.axis('equal')
plt.xlim(-x1_lim, x1_lim)
plt.ylim(-x2_lim, x2_lim)
plt.contour(grid.coordinate_vectors[0],
            grid.coordinate_vectors[1],
            cbvf_values[-1, :, :].T,
            levels=0,
            colors="black",
            linewidths=3)
plt.contour(grid.coordinate_vectors[0],
                grid.coordinate_vectors[1],
                values_vi.T,
                levels=0,
                colors="green",
                linewidths=3,
                linestyles='--')
plt.show()

