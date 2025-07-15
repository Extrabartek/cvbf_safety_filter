import cbvf_reachability as hj
import jax.numpy as jnp

class DoubleIntegrator(hj.ControlAndDisturbanceAffineDynamics):

    def __init__(self,
                 control_mode="min",
                 disturbance_mode="max",
                 control_space=None,
                 disturbance_space=None):

        if control_space is None:
            control_space = hj.sets.Box(jnp.array([-0.5]),
                                        jnp.array([0.5]))
        if disturbance_space is None:
            disturbance_space = hj.sets.Box(jnp.array([-0.0]),
                                        jnp.array([0.0]))
        super().__init__(control_mode, disturbance_mode, control_space, disturbance_space)

    def open_loop_dynamics(self, state, time):
        x1, x2 = state

        return jnp.array([x2, 0])

    def control_jacobian(self, state, time):
        return jnp.array([[0],
                          [1]])

    def disturbance_jacobian(self, state, time):
        return jnp.array([[0],
                          [0]])