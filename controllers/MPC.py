import do_mpc
import numpy as np

from casadi import sin, cos, exp, pi, fabs

car = {'m': 1430, 'Vx': 30.0, 'Lf': 1.05, 'Lr': 1.61, 'Iz': 2059.2, 'mu': 1.0, 'Mz': 1e4 * 1.5,
                               'Cf': 9 * 10e3, 'Cr': 10 * 10e3}

model = do_mpc.model.Model('continuous')

vx = model.set_variable(var_type='_x', var_name='vx', shape=(1, 1))
vy = model.set_variable(var_type='_x', var_name='vy', shape=(1, 1))
r = model.set_variable(var_type='_x', var_name='r', shape=(1, 1))
phi = model.set_variable(var_type='_x', var_name='phi', shape=(1, 1))
xp = model.set_variable(var_type='_x', var_name='xp', shape=(1, 1))
yp = model.set_variable(var_type='_x', var_name='yp', shape=(1, 1))
delta = model.set_variable(var_type='_x', var_name='delta', shape=(1, 1))
d_delta = model.set_variable(var_type='_u', var_name='d_delta', shape=(1, 1))
vy_dot = model.set_variable(var_type='_x', var_name='vy_dot', shape=(1, 1))

r_dot = model.set_variable(var_type='_x', var_name='r_dot', shape=(1, 1))


model.set_rhs('vx', vy * r)
model.set_rhs('vy', vy_dot)
model.set_rhs('vy_dot', -(car['Cf']+car['Cr'])/(car['m']*vx) * vy + \
                                    ((car['Lr'] * car['Cr'] - car['Lf'] * \
                                      car['Cf'])/(car['m']*vx) - vx)* r + car['Cf']/car['m'] * delta)
model.set_rhs('r_dot', (car['Lr'] * car['Cr'] - car['Lf'] * car['Cf']) / (car['Iz'] * vx) * vy + \
                (car['Lf']**2 * car['Cf'] - car['Lr']**2 * car['Cr']) / (car['Iz'] * vx) * r + \
                (car['Lf'] * car['Cf'] / car['Iz']) * delta)
model.set_rhs('r', r_dot)

model.set_rhs('phi', r)
model.set_rhs('xp', vx * cos(phi) - vy * sin(phi))
model.set_rhs('yp', vx * sin(phi) + vy * cos(phi))
model.set_rhs('delta', d_delta)

model.setup()

mpc = do_mpc.controller.MPC(model)

setup_mpc = {
    'n_horizon': 10,
    't_step': 0.1,
    'n_robust': 0,
    'store_full_solution': True,
}

mpc.set_param(**setup_mpc)

# set up the reference trajectory (sigmoid)
sigmoid = 1 / (1 + exp(-0.1 * (xp - 5)))

lterm = (yp - sigmoid) ** 2
mterm = (yp - 1.0) ** 2

mpc.set_objective(lterm=lterm, mterm=mterm)

mpc.set_rterm(
    d_delta=0.1,  # steering rate
)


# longitudinal velocity limits
mpc.bounds['lower', '_x', 'vx'] = 0.0
mpc.bounds['upper', '_x', 'vx'] = 170 / 3.6

# side slip angle limits
# mpc.bounds['lower', '_x', 'vy'] = -5 * pi / 180 * vx
# mpc.bounds['upper', '_x', 'vy'] = 5 * pi / 180 * vx

mpc.set_nl_cons('sideslipe', fabs(vy/vx), 5 * pi / 180)

# side slip rate limits
# mpc.bounds['lower', '_x', 'vy_dot'] = -25 * pi / 180 * vx
# mpc.bounds['upper', '_x', 'vy_dot'] = 25 * pi / 180 * vx

mpc.set_nl_cons('sideslipr', fabs(vy_dot/vx), 25 * pi / 180)

# bound on the lateral acceleration
# mpc.bounds['lower', '_x', 'vy_dot'] = -0.85 * car['mu'] * 9.81 - vx * r
# mpc.bounds['upper', '_x', 'vy_dot'] = 0.85 * car['mu'] * 9.81 - vx * r

mpc.set_nl_cons('lat_acc', fabs(vy_dot + vx * r), 0.85 * car['mu'] * 9.81)

# steering angle limits
mpc.bounds['lower', '_x', 'delta'] = -2.76 * 360 * pi / 180 / 15.3
mpc.bounds['upper', '_x', 'delta'] = 2.76 * 360 * pi / 180 / 15.3

# steering rate limits
mpc.bounds['lower', '_u', 'd_delta'] = -800 * pi / 180 / 15.3
mpc.bounds['upper', '_u', 'd_delta'] = 800 * pi / 180 / 15.3

mpc.setup()

simulator = do_mpc.simulator.Simulator(model)
simulator.set_param(t_step=0.002)
simulator.setup()

x0 = np.array([car['Vx'], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]).reshape(-1, 1)

simulator.x0 = x0
mpc.x0 = x0

mpc.set_initial_guess()

import matplotlib.pyplot as plt
import matplotlib as mpl
# Customizing Matplotlib:
mpl.rcParams['font.size'] = 18
mpl.rcParams['lines.linewidth'] = 3
mpl.rcParams['axes.grid'] = True

mpc_graphics = do_mpc.graphics.Graphics(mpc.data)
sim_graphics = do_mpc.graphics.Graphics(simulator.data)


# # We just want to create the plot and not show it right now. This "inline magic" supresses the output.
fig, ax = plt.subplots(2, sharex=True, figsize=(16,9))
fig.align_ylabels()

for g in [sim_graphics, mpc_graphics]:
    # Plot the angle positions (phi_1, phi_2, phi_2) on the first axis:
    # g.add_line(var_type='_x', var_name='xp', axis=ax[0])
    g.add_line(var_type='_x', var_name='yp', axis=ax[0])

    # Plot the set motor positions (phi_m_1_set, phi_m_2_set) on the second axis:
    g.add_line(var_type='_u', var_name='d_delta', axis=ax[1])

ax[0].set_ylabel('x/y position [m]')
ax[1].set_ylabel('control input [rad/s]')
ax[1].set_xlabel('time [s]')
#
# u0 = np.zeros((1,1))
# for i in range(200):
#     simulator.make_step(u0)
# #
# sim_graphics.plot_results()
# # # Reset the limits on all axes in graphic to show the data.
# sim_graphics.reset_axes()
# # # Show the figure:
# fig.show()
#
# print(mpc.bounds['lower', '_x', 'vy'])
#
# u0 = mpc.make_step(x0)
#
sim_graphics.clear()
#
# mpc_graphics.plot_predictions()
# mpc_graphics.reset_axes()
# # Show the figure:
# fig.show()

simulator.reset_history()
simulator.x0 = x0
mpc.reset_history()

for i in range(2000):
    if i % 10 == 0:
        u0 = mpc.make_step(x0)
    x0 = simulator.make_step(u0)

# Plot predictions from t=0
mpc_graphics.plot_predictions(t_ind=0)
# Plot results until current time
sim_graphics.plot_results()
sim_graphics.reset_axes()
fig.show()

