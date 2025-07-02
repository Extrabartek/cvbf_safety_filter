import jax
import jax.numpy as jnp
from typing import Tuple, Optional, Callable
from cbvf_reachability.grid import Grid
from cbvf_reachability.finite_differences import upwind_first


class CBVFInterpolator:
    """
    A class for interpolating CBVF values and computing gradients at arbitrary points in space and time.

    This class wraps the full space-time CBVF computation results and provides:
    - Interpolation of value function at any point in space-time
    - Computation of spatial gradients using high-order upwind schemes
    - Computation of time gradients
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

        # Use WENO5 by default for high-order accuracy
        self.upwind_scheme = upwind_scheme or upwind_first.WENO5

        # Validate inputs
        if cbvf_values.shape[0] != len(times):
            raise ValueError(f"Time dimension mismatch: cbvf_values has {cbvf_values.shape[0]} time steps, "
                             f"but times array has length {len(times)}")

        # Precompute spatial gradients at all time steps for efficiency
        # self._precompute_all_gradients()

    def _precompute_all_gradients(self):
        """Precompute spatial gradients at all grid points and times."""
        n_times = len(self.times)

        # Initialize storage for gradients
        grad_shape = (n_times,) + self.grid.shape + (self.grid.ndim,)
        self.grad_values_all_times = jnp.zeros(grad_shape)
        self.left_grad_values_all_times = jnp.zeros(grad_shape)
        self.right_grad_values_all_times = jnp.zeros(grad_shape)

        # Compute gradients for each time step
        for t_idx in range(n_times):
            values_t = self.cbvf_values[t_idx]

            # Compute left and right derivatives using upwind scheme
            left_grads, right_grads = self.grid.upwind_grad_values(
                self.upwind_scheme, values_t
            )

            # Store gradients
            self.left_grad_values_all_times = self.left_grad_values_all_times.at[t_idx].set(left_grads)
            self.right_grad_values_all_times = self.right_grad_values_all_times.at[t_idx].set(right_grads)
            self.grad_values_all_times = self.grad_values_all_times.at[t_idx].set(
                (left_grads + right_grads) / 2
            )

    def _interpolate_time_index(self, time: float) -> Tuple[int, float]:
        """
        Find time indices and interpolation weight for given time.

        Returns:
            (lower_index, weight) where weight is for linear interpolation
        """
        # Handle boundary cases
        if time <= self.times[0]:
            return 0, 0.0
        if time >= self.times[-1]:
            return len(self.times) - 2, 1.0

        # Binary search for time interval
        idx = jnp.searchsorted(self.times, time) - 1

        # Compute interpolation weight
        dt = self.times[idx + 1] - self.times[idx]
        weight = (time - self.times[idx]) / dt

        return idx, weight

    @jax.jit
    def interpolate_value(self, state: jnp.ndarray, time: float) -> float:
        """
        Interpolate CBVF value at arbitrary state and time.

        Args:
            state: State vector (JAX array)
            time: Time value

        Returns:
            Interpolated CBVF value B_γ(x,t)
        """
        # Get time interpolation indices
        t_idx, t_weight = self._interpolate_time_index(time)

        # Interpolate in space at both time points
        value_t0 = self.grid.interpolate(self.cbvf_values[t_idx], state)
        value_t1 = self.grid.interpolate(self.cbvf_values[t_idx + 1], state)

        # Linear interpolation in time
        return (1 - t_weight) * value_t0 + t_weight * value_t1

    @jax.jit
    def interpolate_spatial_gradient(self, state: jnp.ndarray, time: float,
                                     use_upwind: bool = False,
                                     velocity: Optional[jnp.ndarray] = None) -> jnp.ndarray:
        """
        Compute spatial gradient of CBVF at arbitrary state and time.

        Args:
            state: State vector (JAX array)
            time: Time value
            use_upwind: If True, use upwind-biased gradients based on velocity
            velocity: Velocity vector for upwind direction (required if use_upwind=True)

        Returns:
            Spatial gradient vector ∇_x B_γ(x,t)
        """
        # Get time interpolation indices
        t_idx, t_weight = self._interpolate_time_index(time)

        if use_upwind and velocity is not None:
            # Interpolate upwind gradients at both time points
            grad_left_t0 = self.grid.interpolate(self.left_grad_values_all_times[t_idx], state)
            grad_left_t1 = self.grid.interpolate(self.left_grad_values_all_times[t_idx + 1], state)
            grad_right_t0 = self.grid.interpolate(self.right_grad_values_all_times[t_idx], state)
            grad_right_t1 = self.grid.interpolate(self.right_grad_values_all_times[t_idx + 1], state)

            # Linear interpolation in time
            grad_left = (1 - t_weight) * grad_left_t0 + t_weight * grad_left_t1
            grad_right = (1 - t_weight) * grad_right_t0 + t_weight * grad_right_t1

            # Select upwind gradient component-wise based on velocity sign
            grad = jnp.where(velocity > 0, grad_left, grad_right)
        else:
            # Interpolate central gradients at both time points
            grad_t0 = self.grid.interpolate(self.grad_values_all_times[t_idx], state)
            grad_t1 = self.grid.interpolate(self.grad_values_all_times[t_idx + 1], state)

            # Linear interpolation in time
            grad = (1 - t_weight) * grad_t0 + t_weight * grad_t1

        return grad

    @jax.jit
    def compute_time_gradient(self, state: jnp.ndarray, time: float) -> float:
        """
        Compute time gradient of CBVF at given state and time using finite differences.

        Args:
            state: State vector (JAX array)
            time: Time value

        Returns:
            Time gradient ∂_t B_γ(x,t)
        """
        # Use central differences when possible, forward/backward at boundaries
        dt = 1e-4  # Small time step for finite difference

        if time - dt >= self.times[0] and time + dt <= self.times[-1]:
            # Central difference
            value_plus = self.interpolate_value(state, time + dt)
            value_minus = self.interpolate_value(state, time - dt)
            return (value_plus - value_minus) / (2 * dt)
        elif time - dt < self.times[0]:
            # Forward difference at start
            value_plus = self.interpolate_value(state, time + dt)
            value_current = self.interpolate_value(state, time)
            return (value_plus - value_current) / dt
        else:
            # Backward difference at end
            value_current = self.interpolate_value(state, time)
            value_minus = self.interpolate_value(state, time - dt)
            return (value_current - value_minus) / dt

    def get_value_and_gradients(self, state: jnp.ndarray, time: float,
                                use_upwind: bool = False,
                                velocity: Optional[jnp.ndarray] = None) -> Tuple[float, jnp.ndarray, float]:
        """
        Get CBVF value and all gradients at once (efficient for CBVF-QP).

        Args:
            state: State vector (JAX array)
            time: Time value
            use_upwind: If True, use upwind-biased spatial gradients
            velocity: Velocity vector for upwind direction

        Returns:
            Tuple of (cbvf_value, spatial_gradient, time_gradient)
        """
        cbvf_value = self.interpolate_value(state, time)
        spatial_grad = self.interpolate_spatial_gradient(state, time, use_upwind, velocity)
        time_grad = self.compute_time_gradient(state, time)

        return cbvf_value, spatial_grad, time_grad

    @jax.jit
    def compute_hessian(self, state: jnp.ndarray, time: float) -> jnp.ndarray:
        """
        Compute Hessian (second-order spatial derivatives) of CBVF.

        This uses finite differences on the gradient field for higher accuracy.

        Args:
            state: State vector (JAX array)
            time: Time value

        Returns:
            Hessian matrix ∇²_x B_γ(x,t)
        """
        # Create perturbation for finite differences
        eps = 1e-4  # Small perturbation
        n_dims = len(state)
        hessian = jnp.zeros((n_dims, n_dims))

        # Get gradient at current state
        grad_center = self.interpolate_spatial_gradient(state, time)

        # Compute mixed partial derivatives
        for i in range(n_dims):
            # Perturb in i-th direction
            state_plus = state.at[i].add(eps)
            state_minus = state.at[i].add(-eps)

            # Get gradients at perturbed states
            grad_plus = self.interpolate_spatial_gradient(state_plus, time)
            grad_minus = self.interpolate_spatial_gradient(state_minus, time)

            # Central difference for i-th row of Hessian
            hessian = hessian.at[i, :].set((grad_plus - grad_minus) / (2 * eps))

        # Symmetrize (since Hessian should be symmetric for smooth functions)
        hessian = (hessian + hessian.T) / 2

        return hessian

    def validate_gradients(self, state: jnp.ndarray, time: float, eps: float = 1e-5) -> dict:
        """
        Validate gradient computation using finite differences.

        Useful for debugging and verifying gradient accuracy.

        Args:
            state: State to check gradients at
            time: Time to check gradients at
            eps: Finite difference step size

        Returns:
            Dictionary with gradient comparisons
        """
        # Compute gradients using interpolation
        spatial_grad_interp = self.interpolate_spatial_gradient(state, time)
        time_grad_interp = self.compute_time_gradient(state, time)

        # Compute spatial gradient using finite differences
        value_center = self.interpolate_value(state, time)
        spatial_grad_fd = jnp.zeros_like(state)

        for i in range(len(state)):
            state_plus = state.at[i].add(eps)
            state_minus = state.at[i].add(-eps)

            value_plus = self.interpolate_value(state_plus, time)
            value_minus = self.interpolate_value(state_minus, time)

            spatial_grad_fd = spatial_grad_fd.at[i].set((value_plus - value_minus) / (2 * eps))

        # Compute time gradient using finite differences
        value_plus_t = self.interpolate_value(state, time + eps)
        value_minus_t = self.interpolate_value(state, time - eps)
        time_grad_fd = (value_plus_t - value_minus_t) / (2 * eps)

        # Compare
        spatial_error = jnp.linalg.norm(spatial_grad_interp - spatial_grad_fd)
        spatial_relative_error = spatial_error / (jnp.linalg.norm(spatial_grad_fd) + 1e-10)

        time_error = jnp.abs(time_grad_interp - time_grad_fd)
        time_relative_error = time_error / (jnp.abs(time_grad_fd) + 1e-10)

        return {
            'spatial_gradient_interpolated': spatial_grad_interp,
            'spatial_gradient_finite_diff': spatial_grad_fd,
            'spatial_absolute_error': spatial_error,
            'spatial_relative_error': spatial_relative_error,
            'time_gradient_interpolated': time_grad_interp,
            'time_gradient_finite_diff': time_grad_fd,
            'time_absolute_error': time_error,
            'time_relative_error': time_relative_error
        }

    def get_time_bounds(self) -> Tuple[float, float]:
        """Get the time range for which CBVF data is available."""
        return float(self.times[0]), float(self.times[-1])

    def get_spatial_bounds(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get the spatial domain bounds."""
        return self.grid.domain.lo, self.grid.domain.hi
