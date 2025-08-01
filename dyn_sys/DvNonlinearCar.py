import cbvf_reachability as hj
import jax.numpy as jnp
from tire_models.pacejka_magic_formula import pacejka_lateral_force

class DvNonlinearCar(hj.ControlAndDisturbanceAffineDynamics):

    """

    The car parameters are as follows:
        m           % Vehicle mass
        Vx          % Longitudinal velocity
        Lf          % Distance from CG to front axle
        Lr          % Distance from CG to rear axle
        Iz          % Yaw moment of inertia
        mu          % Friction coefficient
        dv          % Max steering rate
        Cf          % Front cornering stiffness
        Cr          % Rear cornering stiffness

        These are computed:
        Fzf         % Front axle vertical load
        Fzr         % Rear axle vertical load
    """

    def __init__(self,
                 car_params=None,
                 control_mode="min",
                 disturbance_mode="max",
                 control_space=None,
                 disturbance_space=None):

        if car_params is None:
            self.car_params = {'m': 1430, 'Vx': 30, 'Lf': 1.05, 'Lr': 1.61, 'Iz': 2059.2, 'mu': 1.0, 'dv': 0.7,
                               'Cf': 9 * 10e3, 'Cr': 10 * 10e3}

        else:
            self.car_params = car_params

        self.car_params['Fzf'] = (self.car_params['m'] * 9.81 * self.car_params['Lr']
                               / (self.car_params['Lf'] + self.car_params['Lr']))
        self.car_params['Fzr'] = (self.car_params['m'] * 9.81 * self.car_params['Lf']
                               / (self.car_params['Lf'] + self.car_params['Lr']))

        if control_space is None:
            control_space = hj.sets.Box(jnp.array([-self.car_params['dv']]),
                                        jnp.array([self.car_params['dv']]))
        if disturbance_space is None:
            disturbance_space = hj.sets.Ball(jnp.zeros(1), 1)
        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def open_loop_dynamics(self, state, time):
        """
        Dynamics:

        x1 - yaw rate, x2 - sideslip angle, x3 - steering angle

        x1_dot = 1/Iz * (Lf*Ff - LR*Fr)
        x2_dot = arctan((Ff+Fr)/(m*Vx) - x1)
        x3_dot = u
        """

        x1, x2, x3 = state
        alpha_f = jnp.atan(x2 + self.car_params['Lf'] * x1 / self.car_params['Vx']) + x3
        alpha_r = jnp.atan(x2 - self.car_params['Lf'] * x1 / self.car_params['Vx'])
        f_f = pacejka_lateral_force(alpha_f, self.car_params['Fzf'], self.car_params['mu'], self.car_params['Cf'])
        f_r = pacejka_lateral_force(alpha_r, self.car_params['Fzr'], self.car_params['mu'], self.car_params['Cr'])

        x1_dot = 1/self.car_params['Iz'] * (self.car_params['Lf'] * f_f - self.car_params['Lr'] * f_r)
        x2_dot = jnp.atan((f_f+f_r) / (self.car_params['m'] * self.car_params['Vx']) - x1)

        return jnp.array([x1_dot, x2_dot, 0])

    def control_jacobian(self, state, time):
        return jnp.array([[0],
                          [0],
                          [self.car_params['dv']]])

    def disturbance_jacobian(self, state, time):
        return jnp.array([[0],
                          [0],
                          [0]])