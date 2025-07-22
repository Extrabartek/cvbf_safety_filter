import do_mpc

from casadi import sin, cos, exp, pi

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

vy_dot = -(car['Cf']+car['Cr'])/(car['m']*vx) * vy + \
              ((car['Lr'] * car['Cr'] - car['Lf'] * car['Cf'])/(car['m']*vx) - vx)* r + car['Cf']/car['m'] * delta
r_dot = (car['Lr'] * car['Cr'] - car['Lf'] * car['Cf']) / (car['Iz'] * vx) * vy + \
                (car['Lf']**2 * car['Cf'] - car['Lr']**2 * car['Cr']) / (car['Iz'] * vx) * r + \
                (car['Lf'] * car['Cf'] / car['Iz']) * delta

model.set_rhs('vx', vy * r)
model.set_rhs('vy', vy_dot)
model.set_rhs('r', r_dot)

model.set_rhs('phi', r)
model.set_rhs('xp', vx * cos(phi) - vy * sin(phi))
model.set_rhs('yp', vx * sin(phi) + vy * cos(phi))
model.set_rhs('delta', d_delta)

model.setup()

mpc = do_mpc.controller.MPC(model)

setup_mpc = {
    'n_horizon': 30,
    't_step': 0.01,
    'n_robust': 0,
    'store_full_solution': True,
}

mpc.set_param(**setup_mpc)

# set up the reference trajectory (sigmoid)
sigmoid = 1 / (1 + exp(-0.1 * (xp - 5)))

lterm = (yp - sigmoid) ** 2
mterm = (yp - 1.0) ** 2

mpc.set_objective(lterm=lterm, mterm=mterm)

# longitudinal velocity limits
mpc.bounds['lower', '_x', 'vx'] = 0.0
mpc.bounds['upper', '_x', 'vx'] = 170 / 3.6

# side slip angle limits
mpc.bounds['lower', '_x', 'vy'] = -5 * pi / 180 * vx
mpc.bounds['upper', '_x', 'vy'] = 5 * pi / 180 * vx





