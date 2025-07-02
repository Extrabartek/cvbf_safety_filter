import jax.numpy as jnp
import numpy as np
import qpsolvers



class CBVFQPController:
    """
    CBVF-QP Safety Controller that leverages existing HJ reachability infrastructure.

    Implements the Robust CBVF-QP formulation:
    u* = argmin ||u - u_ref||^2
    s.t. a(x,t) + ∇_x B_γ(x,t) · q(x) u + γ B_γ(x,t) ≥ 0

    Where:
    - a(x,t) = ∂_t B_γ + ∇_x B_γ · p(x) + min_d ∇_x B_γ · r(x) d
    - Uses existing dynamics.control_jacobian() and dynamics.disturbance_jacobian()
    """

    def __init__(self,
                 gamma: float,
                 solver: str = 'osqp',
                 verbose: bool = False):
        """
        Initialize CBVF-QP Controller

        Args:
            gamma: CBVF relaxation parameter (γ ≥ 0)
            solver: QP solver to use ('osqp', 'quadprog', 'cvxopt', etc.)
            verbose: Whether to print solver information
        """
        self.gamma = gamma
        self.solver = solver
        self.verbose = verbose

    def compute_safe_control(self,
                             state: jnp.ndarray,
                             time: float,
                             u_ref: jnp.ndarray,
                             cbvf_value: float,
                             cbvf_grad_x: jnp.ndarray,
                             cbvf_grad_t: float,
                             dynamics) -> jnp.ndarray:
        """
        Compute safe control using CBVF-QP formulation

        Args:
            state: Current state (jnp.ndarray)
            time: Current time
            u_ref: Reference control input (jnp.ndarray)
            cbvf_value: CBVF value B_γ(x,t) at current state/time
            cbvf_grad_x: Spatial gradient ∇_x B_γ(x,t) (jnp.ndarray)
            cbvf_grad_t: Time gradient ∂_t B_γ(x,t) (scalar)
            dynamics: Dynamics object with control_jacobian, etc.

        Returns:
            Safe control input u_safe (jnp.ndarray)
        """

        # Get system matrices using existing dynamics interface
        p_x = dynamics.open_loop_dynamics(state, time)
        q_x = dynamics.control_jacobian(state, time)
        r_x = dynamics.disturbance_jacobian(state, time)

        # Convert JAX arrays to numpy for qpsolvers
        state_np = np.array(state)
        u_ref_np = np.array(u_ref)
        cbvf_grad_x_np = np.array(cbvf_grad_x)
        p_x_np = np.array(p_x)
        q_x_np = np.array(q_x)
        r_x_np = np.array(r_x)

        # Compute the 'a' term: a(x,t) = ∂_t B_γ + ∇_x B_γ · p(x) + min_d ∇_x B_γ · r(x) d
        a_term = float(cbvf_grad_t) + np.dot(cbvf_grad_x_np, p_x_np)

        # Add worst-case disturbance term using dynamics disturbance bounds
        if hasattr(dynamics, 'disturbance_space') and r_x_np.size > 0:
            # Get disturbance bounds from dynamics object
            d_bounds = dynamics.disturbance_space

            # Compute disturbance effect on constraint
            disturbance_term = np.dot(cbvf_grad_x_np, r_x_np)

            # For each component, choose worst-case disturbance
            if hasattr(d_bounds, 'lower') and hasattr(d_bounds, 'upper'):
                d_min = np.array(d_bounds.lower)
                d_max = np.array(d_bounds.upper)
            elif hasattr(d_bounds, 'radius'):
                # For Ball constraints
                d_max = d_bounds.radius * np.ones(r_x_np.shape[1])
                d_min = -d_max
            else:
                # Default small disturbance if bounds not available
                d_max = 0.1 * np.ones(r_x_np.shape[1])
                d_min = -d_max

            # Choose disturbance that minimizes the constraint
            worst_case_d = np.where(disturbance_term >= 0, d_min, d_max)
            a_term += np.dot(disturbance_term.flatten(), worst_case_d)

        # Set up QP problem: min 0.5 * u^T P u + q^T u
        n_controls = len(u_ref_np)

        # Objective: minimize ||u - u_ref||^2
        P = 2.0 * np.eye(n_controls)  # Quadratic term
        q = -2.0 * u_ref_np  # Linear term

        # Safety constraint: a(x,t) + ∇_x B_γ · q(x) u + γ B_γ(x,t) ≥ 0
        # Reformulated as: -∇_x B_γ · q(x) u ≤ a(x,t) + γ B_γ(x,t)
        constraint_coeff = -np.dot(cbvf_grad_x_np, q_x_np).reshape(1, -1)
        constraint_bound = np.array([a_term + self.gamma * float(cbvf_value)])

        # Add control bounds from dynamics object if available
        if hasattr(dynamics, 'control_space'):
            control_bounds = dynamics.control_space

            if hasattr(control_bounds, 'lower') and hasattr(control_bounds, 'upper'):
                u_min = np.array(control_bounds.lower)
                u_max = np.array(control_bounds.upper)

                # Add inequality constraints: u_min <= u <= u_max
                G_bounds = np.vstack([-np.eye(n_controls), np.eye(n_controls)])
                h_bounds = np.hstack([-u_min, u_max])

                # Combine with safety constraint
                G = np.vstack([constraint_coeff, G_bounds])
                h = np.hstack([constraint_bound, h_bounds])
            else:
                G = constraint_coeff
                h = constraint_bound
        else:
            G = constraint_coeff
            h = constraint_bound

        # Solve QP
        try:
            u_safe = qpsolvers.solve_qp(P, q, G, h, solver=self.solver, verbose=self.verbose)

            if u_safe is None:
                if self.verbose:
                    print("QP solver failed, using reference control")
                return u_ref

            return jnp.array(u_safe)

        except Exception as e:
            if self.verbose:
                print(f"QP solver error: {e}, using reference control")
            return u_ref