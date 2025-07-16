import jax
import jax.numpy as jnp
from typing import Tuple, Optional, Callable
from jax.scipy.ndimage import map_coordinates
from cbvf_reachability.grid import Grid
from cbvf_reachability.finite_differences import upwind_first


class CBVFInterpolator:
    """
    A class for interpolating CBVF values and computing gradients at arbitrary points in space and time.
    """

    def __init__(self,
                 grid: Grid,
                 cbvf_values: jnp.ndarray,
                 times: jnp.ndarray,
                 gamma: float = 0.0,
                 upwind_scheme: Optional[Callable] = None):
        """
        Initialize CBVF interpolator with full space-time data.

        Args:
            grid: Grid object defining the spatial discretization
            cbvf_values: CBVF values at all grid points and times
                         Shape: (n_times, *grid.shape)
            times: Array of time values corresponding to cbvf_values
            gamma: CBVF relaxation parameter
            upwind_scheme: Upwind scheme for gradient computation (default: WENO5)
        """
        self.grid = grid
        self.cbvf_values = cbvf_values
        self.times = times
        self.gamma = gamma
        self.upwind_scheme = upwind_scheme or upwind_first.WENO5

        # Validate inputs
        if cbvf_values.shape[0] != len(times):
            raise ValueError(f"Time dimension mismatch: cbvf_values has {cbvf_values.shape[0]} time steps, "
                             f"but times array has length {len(times)}")

        self.interpolator = self._create_interpolator()
        # Create gradient functions for spatial and temporal derivatives
        self.spatial_grad_interpolator = jax.grad(self.interpolator, argnums=0)
        self.time_grad_interpolator = jax.grad(self.interpolator, argnums=1)

    def _create_interpolator(self) -> Callable:
        t_range = (self.times[0], self.times[-1])
        x1_range = (self.grid.domain.lo[0], self.grid.domain.hi[0])
        x2_range = (self.grid.domain.lo[1], self.grid.domain.hi[1])
        data = self.cbvf_values

        def _interpolator(state, time):
            t_idx = jnp.abs((time - t_range[0]) / (t_range[1] - t_range[0]) * (data.shape[0] - 1))
            x1_idx = (state[0] - x1_range[0]) / (x1_range[1] - x1_range[0]) * (data.shape[1] - 1)
            x2_idx = (state[1] - x2_range[0]) / (x2_range[1] - x2_range[0]) * (data.shape[2] - 1)

            coordinates = jnp.array([[t_idx], [x1_idx], [x2_idx]])
            return map_coordinates(data, coordinates, order=1, mode='nearest', cval=0.0)[0]

        return _interpolator

    def interpolate_value(self, state: jnp.ndarray, time: float) -> float:
        """Interpolate CBVF value at given state and time."""
        return self.interpolator(state, time)

    def interpolate_spatial_gradient(self, state: jnp.ndarray, time: float) -> jnp.ndarray:
        """Compute spatial gradient [∂V/∂x1, ∂V/∂x2] at given state and time."""
        return self.spatial_grad_interpolator(state, time)

    def compute_time_gradient(self, state: jnp.ndarray, time: float) -> float:
        """Compute time gradient ∂V/∂t at given state and time."""
        return self.time_grad_interpolator(state, time)

    def get_value_and_gradient(self, state: jnp.ndarray, time: float) -> Tuple[float, jnp.ndarray, float]:
        """
        Get value and all gradients at once (more efficient).

        Returns:
            value: Interpolated CBVF value
            spatial_grad: Spatial gradient [∂V/∂x1, ∂V/∂x2]
            time_grad: Time gradient ∂V/∂t
        """
        value = self.interpolator(state, time)
        spatial_grad = self.spatial_grad_interpolator(state, time)
        time_grad = self.time_grad_interpolator(state, time)

        return value, spatial_grad, time_grad

    def interpolate_spatial_gradient_vectorized(self, states: jnp.ndarray, time: float) -> jnp.ndarray:
        # Vectorize over the first axis (states)
        vectorized_grad = jax.vmap(self.spatial_grad_interpolator, in_axes=(0, None))
        return vectorized_grad(states, time)

    def compute_safe_entry_gradients_efficient(self, states: jnp.ndarray) -> Tuple[
        jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        More efficient version using direct grid interpolation instead of repeated calls.

        Args:
            states: Array of states to analyze, shape (n_states, 2)

        Returns:
            entry_times: Time when each state first enters safe region, shape (n_states,)
            entry_values: CBVF values at entry times, shape (n_states,)
            spatial_grads: Spatial gradients at entry, shape (n_states, 2)
            time_grads: Time gradients at entry, shape (n_states,)
        """
        # Ensure times is a JAX array for proper indexing
        times_jax = jnp.array(self.times)

        # Convert states to grid coordinates for efficient lookup
        x1_range = (self.grid.domain.lo[0], self.grid.domain.hi[0])
        x2_range = (self.grid.domain.lo[1], self.grid.domain.hi[1])

        def state_to_indices(state):
            x1_idx = (state[0] - x1_range[0]) / (x1_range[1] - x1_range[0]) * (self.cbvf_values.shape[1] - 1)
            x2_idx = (state[1] - x2_range[0]) / (x2_range[1] - x2_range[0]) * (self.cbvf_values.shape[2] - 1)
            return jnp.array([x1_idx, x2_idx])

        def find_safe_entry_efficient(state):
            # Get spatial indices for this state
            spatial_coords = state_to_indices(state)

            # Extract time series for this spatial location via interpolation
            def get_value_at_time_idx(t_idx):
                coords = jnp.array([[t_idx], [spatial_coords[0]], [spatial_coords[1]]])
                return map_coordinates(self.cbvf_values, coords, order=1, mode='nearest')[0]

            # Sample at all time indices
            time_indices = jnp.arange(len(times_jax))
            values_over_time = jax.vmap(get_value_at_time_idx)(time_indices.astype(float))

            # Find safe entry - we want the LAST (most negative time) occurrence
            safe_mask = values_over_time <= 0.0
            has_safe_entry = jnp.any(safe_mask)

            # Find the maximum index where safe_mask is True (earliest/most negative time)
            safe_indices = jnp.where(safe_mask, jnp.arange(len(safe_mask)), -1)
            entry_idx = jnp.where(has_safe_entry,
                                  jnp.max(safe_indices),  # Last True index (most negative time)
                                  0)  # Default to most negative time if never safe
            entry_time = times_jax[entry_idx]

            # Get gradients at entry time using existing methods
            value, spatial_grad, time_grad = self.get_value_and_gradient(state, entry_time)

            return entry_time, value, spatial_grad, time_grad

        # Vectorize and compute
        vectorized_compute = jax.vmap(find_safe_entry_efficient)
        return vectorized_compute(states)