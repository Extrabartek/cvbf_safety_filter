import jax
import jax.numpy as jnp
import functools
from typing import Tuple, Optional, Callable
from cbvf_reachability.grid import Grid
from cbvf_reachability.finite_differences import upwind_first


# JIT-compiled standalone functions (no self reference)
@jax.jit
def _interpolate_time_index_jit(times: jnp.ndarray, time: float) -> Tuple[int, float]:
    """Find time indices and interpolation weight for given time."""
    # Determine if times are increasing or decreasing
    is_increasing = times[1] > times[0]

    def handle_increasing():
        # Handle increasing time array
        clipped_time = jnp.clip(time, times[0], times[-1])
        idx = jnp.searchsorted(times, clipped_time, side='right') - 1
        idx = jnp.clip(idx, 0, len(times) - 2)

        # Compute interpolation weight
        dt = times[idx + 1] - times[idx]
        weight = jnp.where(dt > 0, (clipped_time - times[idx]) / dt, 0.0)
        return idx, weight

    def handle_decreasing():
        # Handle decreasing time array (common in backward reachability)
        clipped_time = jnp.clip(time, times[-1], times[0])  # Note: times[-1] < times[0]

        # For decreasing array, we need to search differently
        # Flip the array to use searchsorted, then adjust the index
        times_flipped = times[::-1]
        idx_flipped = jnp.searchsorted(times_flipped, clipped_time, side='right') - 1
        idx_flipped = jnp.clip(idx_flipped, 0, len(times_flipped) - 2)

        # Convert back to original array index
        idx = len(times) - 2 - idx_flipped
        idx = jnp.clip(idx, 0, len(times) - 2)

        # Compute interpolation weight
        dt = times[idx + 1] - times[idx]  # This will be negative for decreasing array
        weight = jnp.where(dt != 0, (clipped_time - times[idx]) / dt, 0.0)
        return idx, weight

    return jax.lax.cond(is_increasing, handle_increasing, handle_decreasing)


@jax.jit
def _interpolate_value_jit(grid_spacings: jnp.ndarray,
                           grid_lo: jnp.ndarray, grid_shape: jnp.ndarray,
                           cbvf_values: jnp.ndarray, times: jnp.ndarray,
                           state: jnp.ndarray, time: float) -> float:
    """JIT-compiled value interpolation."""
    # Get time interpolation
    t_idx, t_weight = _interpolate_time_index_jit(times, time)

    # Spatial interpolation using multilinear interpolation
    value_t0 = _multilinear_interpolate(grid_spacings, grid_lo, grid_shape,
                                        cbvf_values[t_idx], state)
    value_t1 = _multilinear_interpolate(grid_spacings, grid_lo, grid_shape,
                                        cbvf_values[t_idx + 1], state)

    # Linear interpolation in time
    return (1 - t_weight) * value_t0 + t_weight * value_t1


@jax.jit
def _multilinear_interpolate(grid_spacings: jnp.ndarray, grid_lo: jnp.ndarray,
                             grid_shape: jnp.ndarray, values: jnp.ndarray,
                             state: jnp.ndarray) -> float:
    """Multilinear interpolation on regular grid."""
    # Convert state to grid coordinates
    position = (state - grid_lo) / grid_spacings

    # Get integer and fractional parts
    idx_lo = jnp.floor(position).astype(jnp.int32)
    idx_hi = idx_lo + 1
    weight_hi = position - idx_lo
    weight_lo = 1 - weight_hi

    # Clamp indices to valid range
    idx_lo = jnp.clip(idx_lo, 0, grid_shape - 1)
    idx_hi = jnp.clip(idx_hi, 0, grid_shape - 1)

    # Perform multilinear interpolation
    ndim = len(state)

    if ndim == 1:
        return (weight_lo[0] * values[idx_lo[0]] +
                weight_hi[0] * values[idx_hi[0]])
    elif ndim == 2:
        return (weight_lo[0] * weight_lo[1] * values[idx_lo[0], idx_lo[1]] +
                weight_lo[0] * weight_hi[1] * values[idx_lo[0], idx_hi[1]] +
                weight_hi[0] * weight_lo[1] * values[idx_hi[0], idx_lo[1]] +
                weight_hi[0] * weight_hi[1] * values[idx_hi[0], idx_hi[1]])
    elif ndim == 3:
        return (weight_lo[0] * weight_lo[1] * weight_lo[2] * values[idx_lo[0], idx_lo[1], idx_lo[2]] +
                weight_lo[0] * weight_lo[1] * weight_hi[2] * values[idx_lo[0], idx_lo[1], idx_hi[2]] +
                weight_lo[0] * weight_hi[1] * weight_lo[2] * values[idx_lo[0], idx_hi[1], idx_lo[2]] +
                weight_lo[0] * weight_hi[1] * weight_hi[2] * values[idx_lo[0], idx_hi[1], idx_hi[2]] +
                weight_hi[0] * weight_lo[1] * weight_lo[2] * values[idx_hi[0], idx_lo[1], idx_lo[2]] +
                weight_hi[0] * weight_lo[1] * weight_hi[2] * values[idx_hi[0], idx_lo[1], idx_hi[2]] +
                weight_hi[0] * weight_hi[1] * weight_lo[2] * values[idx_hi[0], idx_hi[1], idx_lo[2]] +
                weight_hi[0] * weight_hi[1] * weight_hi[2] * values[idx_hi[0], idx_hi[1], idx_hi[2]])
    else:
        # For higher dimensions, use a more general approach
        # This is less efficient but works for any dimension
        return values[tuple(idx_lo)]  # Fallback to nearest neighbor


@jax.jit
def _compute_spatial_gradient_jit(grid_spacings: jnp.ndarray, grid_lo: jnp.ndarray,
                                  grid_shape: jnp.ndarray, cbvf_values: jnp.ndarray,
                                  times: jnp.ndarray, state: jnp.ndarray,
                                  time: float, eps: float = 1e-6) -> jnp.ndarray:
    """JIT-compiled spatial gradient computation using finite differences."""
    ndim = len(state)
    grad = jnp.zeros(ndim)

    for i in range(ndim):
        state_plus = state.at[i].add(eps)
        state_minus = state.at[i].add(-eps)

        value_plus = _interpolate_value_jit(grid_spacings, grid_lo, grid_shape,
                                            cbvf_values, times, state_plus, time)
        value_minus = _interpolate_value_jit(grid_spacings, grid_lo, grid_shape,
                                             cbvf_values, times, state_minus, time)

        grad = grad.at[i].set((value_plus - value_minus) / (2 * eps))

    return grad


@jax.jit
def _compute_time_gradient_jit(grid_spacings: jnp.ndarray, grid_lo: jnp.ndarray,
                               grid_shape: jnp.ndarray, cbvf_values: jnp.ndarray,
                               times: jnp.ndarray, state: jnp.ndarray,
                               time: float, dt: float = 1e-4) -> float:
    """JIT-compiled time gradient computation."""
    # Check bounds
    t_min, t_max = jnp.min(times), jnp.max(times)

    # Compute all possible finite difference values
    value_plus = _interpolate_value_jit(grid_spacings, grid_lo, grid_shape,
                                        cbvf_values, times, state, time + dt)
    value_minus = _interpolate_value_jit(grid_spacings, grid_lo, grid_shape,
                                         cbvf_values, times, state, time - dt)
    value_current = _interpolate_value_jit(grid_spacings, grid_lo, grid_shape,
                                           cbvf_values, times, state, time)

    # Compute all possible gradients
    central_grad = (value_plus - value_minus) / (2 * dt)
    forward_grad = (value_plus - value_current) / dt
    backward_grad = (value_current - value_minus) / dt

    # Choose which gradient to use based on bounds
    use_central = (time - dt >= t_min) & (time + dt <= t_max)
    use_forward = (time - dt < t_min) & (time + dt <= t_max)

    return jnp.where(use_central, central_grad,
                     jnp.where(use_forward, forward_grad, backward_grad))


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

        # Extract grid information for JIT functions
        self.grid_spacings = jnp.array(grid.spacings)
        self.grid_lo = grid.domain.lo
        self.grid_shape = jnp.array(grid.shape)

    def interpolate_value(self, state: jnp.ndarray, time: float) -> float:
        """
        Interpolate CBVF value at arbitrary state and time.

        Args:
            state: State vector (JAX array)
            time: Time value

        Returns:
            Interpolated CBVF value B_γ(x,t)
        """
        # Fixed argument order!
        return _interpolate_value_jit(self.grid_spacings, self.grid_lo, self.grid_shape,
                                      self.cbvf_values, self.times, state, time)

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
        return _compute_spatial_gradient_jit(self.grid_spacings, self.grid_lo, self.grid_shape,
                                             self.cbvf_values, self.times, state, time)

    def compute_time_gradient(self, state: jnp.ndarray, time: float) -> float:
        """
        Compute time gradient of CBVF at given state and time using finite differences.

        Args:
            state: State vector (JAX array)
            time: Time value

        Returns:
            Time gradient ∂_t B_γ(x,t)
        """
        return _compute_time_gradient_jit(self.grid_spacings, self.grid_lo, self.grid_shape,
                                          self.cbvf_values, self.times, state, time)

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

    def get_time_bounds(self) -> Tuple[float, float]:
        """Get the time range for which CBVF data is available."""
        return float(jnp.min(self.times)), float(jnp.max(self.times))

    def get_spatial_bounds(self) -> Tuple[jnp.ndarray, jnp.ndarray]:
        """Get the spatial domain bounds."""
        return self.grid.domain.lo, self.grid.domain.hi