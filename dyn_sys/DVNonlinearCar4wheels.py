import cbvf_reachability as hj
import jax.numpy as jnp

from utils.tpf_handler import read_tpf
from tire_models.pacejka_62 import pacejka_62_scalar, preprocess_tpf_for_jax


class DvNonlinearCar(hj.ControlAndDisturbanceAffineDynamics):

    """

    The car parameters are as follows:
        m           % Vehicle mass
        Vx          % Longitudinal velocity
        Lf          % Distance from CG to front axle
        Lr          % Distance from CG to rear axle
        Iz          % Yaw moment of inertia
        mu          % Friction coefficient
        Mz          % Max yaw moment
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
            self.car_params = {'m': 1430, 'Vx': 30, 'Lf': 1.05, 'Lr': 1.61, 'Iz': 2059.2, 'mu': 1.0, 'Mz': 10e3 * 0.5,
                               'Cf': 9 * 10e3, 'Cr': 10 * 10e3, 'Wf': 0.755, 'Wr': 0.755,}

        else:
            self.car_params = car_params

        self.car_params['Fz_Fl'] = (0.5 * self.car_params['m'] * 9.81 * self.car_params['Lr']
                               / (self.car_params['Lf'] + self.car_params['Lr']))
        self.car_params['Fz_Fr'] = (0.5 * self.car_params['m'] * 9.81 * self.car_params['Lr']
                                    / (self.car_params['Lf'] + self.car_params['Lr']))
        self.car_params['Fz_Rl'] = (0.5 * self.car_params['m'] * 9.81 * self.car_params['Lf']
                               / (self.car_params['Lf'] + self.car_params['Lr']))
        self.car_params['Fz_Rr'] = (0.5 * self.car_params['m'] * 9.81 * self.car_params['Lf']
                                    / (self.car_params['Lf'] + self.car_params['Lr']))

        self.tire_data = preprocess_tpf_for_jax(read_tpf('tire_models/tire_data/TNO_car205_60R15.tir'))

        if control_space is None:
            control_space = hj.sets.Box(jnp.array([-self.car_params['Mz']]),
                                        jnp.array([self.car_params['Mz']]))
        if disturbance_space is None:
            disturbance_space = hj.sets.Ball(jnp.zeros(1), 1)
        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def open_loop_dynamics(self, state, time):
        """
        Dynamics:

        x1 - yaw rate, x2 - sideslip angle
        """

        x1, x2, x3 = state
        alpha_fl = -x3 + jnp.atan(x2 + self.car_params['Lf'] * x1 / (self.car_params['Vx'] - self.car_params['Wf'] * x1))
        alpha_fr = -x3 + jnp.atan(x2 + self.car_params['Lf'] * x1 / (self.car_params['Vx'] + self.car_params['Wf'] * x1))
        alpha_rl = jnp.atan(x2 - self.car_params['Lr'] * x1 / (self.car_params['Vx'] - self.car_params['Wr'] * x1))
        alpha_rr = jnp.atan(x2 - self.car_params['Lr'] * x1 / (self.car_params['Vx'] + self.car_params['Wr'] * x1))

        f_fl = self.car_params['mu'] * pacejka_62_scalar(self.tire_data, self.car_params['Fz_Fl'], 0.0, alpha_fl, 0.0, self.car_params['Vx'], 4)[1]
        f_fr = self.car_params['mu'] * pacejka_62_scalar(self.tire_data, self.car_params['Fz_Fl'], 0.0, alpha_fr, 0.0, self.car_params['Vx'], 4)[1]
        f_rl = self.car_params['mu'] * pacejka_62_scalar(self.tire_data, self.car_params['Fz_Rl'], 0.0, alpha_rl, 0.0, self.car_params['Vx'], 4)[1]
        f_rr = self.car_params['mu'] * pacejka_62_scalar(self.tire_data, self.car_params['Fz_Rr'], 0.0, alpha_rr, 0.0, self.car_params['Vx'], 4)[1]

        x1_dot = (((self.car_params['Wf'] * f_fl * jnp.sin(x3) - f_fr * jnp.sin(x3)) + self.car_params['Lf']
                  * (f_fl * jnp.cos(x3) + f_fr * jnp.cos(x3)) - self.car_params['Lr'] * (f_rl + f_rr))/self.car_params['Iz'])
        x2_dot = ((f_fl + f_fr) * jnp.cos(x3) + f_rl + f_rr) / self.car_params['m'] / self.car_params['Vx'] - x1

        return jnp.array([x1_dot, x2_dot, 0.0])

    def control_jacobian(self, state, time):
        return jnp.array([[1/self.car_params['Iz']],
                          [0],
                          [0]])

    def disturbance_jacobian(self, state, time):
        return jnp.array([[0],
                          [0],
                          [0]])
