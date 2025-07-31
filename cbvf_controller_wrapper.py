import numpy as np
import jax
import jax.numpy as jnp
from pathlib import Path
import pickle


class CBVFControllerWrapper:
    """Wrapper for CBVF controller to be used in Simulink"""

    def __init__(self, data_path=None):
        """Initialize controller with precomputed CBVF data

        Args:
            data_path: Path to saved CBVF data (pickle file)
        """
        self.initialized = False
        self.controller = None
        self.interpolator = None
        self.dynamics = None

        if data_path:
            self.load_data(data_path)

    def load_data(self, data_path):
        """Load precomputed CBVF data and initialize controller"""
        with open(data_path, 'rb') as f:
            data = pickle.load(f)

        # Extract components
        self.grid = data['grid']
        self.cbvf_values = data['cbvf_values']
        self.times = data['times']
        self.dynamics = data['dynamics']
        self.gamma = data.get('gamma', 1000)

        # Initialize interpolator
        from CBVFInterpolator import CBVFInterpolator
        self.interpolator = CBVFInterpolator(
            grid=self.grid,
            cbvf_values=self.cbvf_values,
            times=self.times,
            gamma=self.gamma
        )

        # Initialize controller
        from CBVF_QP import CBVFQPController
        self.controller = CBVFQPController(
            value_fn=self.interpolator,
            gamma=self.gamma,
            verbose=False
        )

        self.initialized = True

    def compute_control(self, state, time, u_ref, u_prev=0.0, u_max_mag=15000):
        """Compute safe control for given state

        Args:
            state: Current state [x1, x2] as numpy array
            time: Current simulation time
            u_ref: Reference control input
            u_prev: Previous control input (default: 0)
            u_max_mag: Maximum control magnitude (default: 15000)

        Returns:
            u_safe: Safe control input
            cbvf_val: Current CBVF value
            constraint_val: Constraint value (should be >= 0)
        """
        if not self.initialized:
            raise RuntimeError("Controller not initialized. Call load_data() first.")

        # Convert inputs to proper format
        state = np.array(state).flatten()

        # Compute gradient time
        gradient_time = self.interpolator.compute_safe_entry_gradients_efficient(
            jnp.array(state.reshape(1, -1))
        )[0]

        # Get CBVF value at current state
        cbvf_val, _, _ = self.controller.cbvf.get_value_and_gradient(
            state, self.times[-1]
        )

        # Compute safe control
        u_safe, constraint_val, _ = self.controller.compute_safe_control(
            state=state,
            time=time,
            u_ref=u_ref,
            dynamics=self.dynamics,
            u_prev=u_prev,
            u_max_mag=u_max_mag,
            gradient_time=gradient_time
        )

        return float(u_safe), float(cbvf_val), float(constraint_val)

    def get_cbvf_value(self, state):
        """Get CBVF value at given state (at final time)"""
        if not self.initialized:
            raise RuntimeError("Controller not initialized. Call load_data() first.")

        state = np.array(state).flatten()
        return float(self.interpolator.interpolate_value(state, self.times[-1]))


# Singleton instance for MATLAB
_controller_instance = None


def initialize_controller(data_path):
    """Initialize the global controller instance"""
    global _controller_instance
    _controller_instance = CBVFControllerWrapper(data_path)
    return True


def compute_safe_control(state, time, u_ref, u_prev=0.0, u_max_mag=15000):
    """Compute safe control using global controller instance"""
    if _controller_instance is None:
        raise RuntimeError("Controller not initialized. Call initialize_controller() first.")

    return _controller_instance.compute_control(state, time, u_ref, u_prev, u_max_mag)


def get_cbvf_value(state):
    """Get CBVF value using global controller instance"""
    if _controller_instance is None:
        raise RuntimeError("Controller not initialized. Call initialize_controller() first.")

    return _controller_instance.get_cbvf_value(state)