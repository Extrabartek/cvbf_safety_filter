import jax.numpy as jnp
import numpy as np
import qpsolvers


def debug_quadprog_call(P, q, G, h, verbose=True):
    """
    Call quadprog directly to get detailed status information.
    """
    try:
        import quadprog

        # Convert to quadprog format:
        # quadprog minimizes (1/2)x^T G x - a^T x subject to C^T x >= b
        # Our problem: minimize (1/2)u^T P u + q^T u subject to G u <= h
        # Conversion: G_qp = P, a_qp = -q, C_qp^T = -G, b_qp = -h

        G_qp = P
        a_qp = -q

        if G.shape[0] > 0:
            C_qp = -G.T
            b_qp = -h
            meq = 0  # Number of equality constraints (we only have inequalities)
        else:
            C_qp = None
            b_qp = None
            meq = 0

        if False:
            print(f"=== Direct Quadprog Call ===")
            print(f"G_qp (P) shape: {G_qp.shape}, condition: {np.linalg.cond(G_qp):.2e}")
            print(f"a_qp (-q): {a_qp}")
            if C_qp is not None:
                print(f"C_qp (-G^T) shape: {C_qp.shape}")
                print(f"b_qp (-h): {b_qp}")
            print(f"meq (equality constraints): {meq}")

        # Call quadprog directly
        if C_qp is not None:
            result = quadprog.solve_qp(G_qp, a_qp, C_qp, b_qp, meq)
        else:
            # Unconstrained case
            result = quadprog.solve_qp(G_qp, a_qp)

        if False:
            print(f"Quadprog raw result: {result}")
            print(f"Result type: {type(result)}")
            if len(result) >= 2:
                solution, obj_value = result[0], result[1]
                print(f"Solution: {solution}")
                print(f"Objective value: {obj_value}")
                if len(result) > 2:
                    print(f"Additional info: {result[2:]}")

        return result[0] if result[0] is not None else None

    except ImportError:
        if verbose:
            print("quadprog not available, using qpsolvers interface")
        return qpsolvers.solve_qp(P, q, G, h, solver='quadprog', verbose=verbose)

    except Exception as e:
        if verbose:
            print(f"Quadprog error: {e}")
            print(f"Error type: {type(e)}")

            # Check for specific quadprog error conditions
            error_str = str(e).lower()
            if 'matrix not positive definite' in error_str:
                print("DIAGNOSIS: P matrix is not positive definite")
            elif 'constraints inconsistent' in error_str or 'infeasible' in error_str:
                print("DIAGNOSIS: Constraints are inconsistent/infeasible")
            elif 'unbounded' in error_str:
                print("DIAGNOSIS: Problem is unbounded")
            else:
                print(f"DIAGNOSIS: Unknown quadprog error - {e}")
        return None


def find_safe_entry_time_efficient(cbvf_interpolator, state, safe_threshold=0.0, debug=False):
    """
    Find the optimal time for gradient computation in CBVF-QP controller using binary search.

    For backward reachability with negative times:
    - If state is safe at some point, returns the time of entry into safe set
    - If state is never safe, returns the time with maximum CBVF value (least unsafe)

    This ensures we get meaningful gradients even for states outside the safe set.
    """
    times = cbvf_interpolator.times

    if len(times) < 2:
        return times[0]

    # Determine time ordering
    ascending = abs(times[1]) > abs(times[0])

    def is_safe(time_idx):
        """Check if state is safe at given time index."""
        t = times[time_idx]
        v = cbvf_interpolator.interpolate_value(state, t)
        return v >= safe_threshold

    # Binary search for safe entry time
    def binary_search_safe_entry():
        """Binary search for the earliest safe time."""
        if ascending:
            # For ascending times: search from earliest (most negative) to latest
            left, right = 0, len(times) - 1

            # Check if any point is safe
            if not is_safe(right):  # Latest time not safe
                return None

            # Binary search for first safe time
            while left < right:
                mid = (left + right) // 2
                if is_safe(mid):
                    right = mid
                else:
                    left = mid + 1
            return left
        else:
            # For descending times: search from latest (most negative) to earliest
            left, right = 0, len(times) - 1

            # Check if any point is safe
            if not is_safe(left):  # Latest time not safe
                return None

            # Binary search for first safe time in descending order
            while left < right:
                mid = (left + right + 1) // 2
                if is_safe(mid):
                    left = mid
                else:
                    right = mid - 1
            return left

    # Try to find safe entry time with binary search
    safe_idx = binary_search_safe_entry()

    if safe_idx is not None:
        entry_time = times[safe_idx]
        if debug:
            entry_value = cbvf_interpolator.interpolate_value(state, entry_time)
            print(f"CBVF profile for state {state}:")
            print(f"  Safe threshold: {safe_threshold}")
            print(f"  Entry time: {entry_time} (value: {entry_value:.5f})")
        return entry_time

    # If never safe, find time with maximum value using sampling
    # Sample fewer points for efficiency while maintaining accuracy
    sample_size = min(20, len(times))  # Sample at most 20 points
    sample_indices = [i * (len(times) - 1) // (sample_size - 1) for i in range(sample_size)]

    max_value = float('-inf')
    max_value_time = times[0]

    for idx in sample_indices:
        t = times[idx]
        v = cbvf_interpolator.interpolate_value(state, t)
        if v > max_value:
            max_value = v
            max_value_time = t

    if debug:
        print(f"CBVF profile for state {state}:")
        print(f"  Safe threshold: {safe_threshold}")
        print(f"  Never safe, using max value time: {max_value_time} (value: {max_value:.5f})")

    return max_value_time

# Simple enhanced version of your existing controller
class CBVFQPController:
    def __init__(self, value_fn, gamma, solver='quadprog', verbose=False):
        self.cbvf = value_fn
        self.gamma = gamma
        self.solver = solver
        self.verbose = verbose
        self.safe_threshold = 0.0  # Safe set threshold

    def compute_safe_control(self, state, time, u_ref, dynamics, u_prev, u_max_mag=None):
        gradient_time = find_safe_entry_time_efficient(self.cbvf, state, self.safe_threshold, debug=False)
        if self.verbose:
            print(f"Current state: {state}, time: {time}, reference control: {u_ref}")
            print(f"Gradient time for safe entry: {gradient_time}")
        cbvf_value, cbvf_grad_x, cbvf_grad_t = self.cbvf.get_value_and_gradients(state, gradient_time)

        # Get system matrices
        p_x = dynamics.open_loop_dynamics(state, time)
        q_x = dynamics.control_jacobian(state, time)
        r_x = dynamics.disturbance_jacobian(state, time)

        # Convert to numpy
        u_ref_np = np.array([u_ref])
        cbvf_grad_x_np = np.array(cbvf_grad_x)
        p_x_np = np.array(p_x)
        q_x_np = np.array(q_x)
        r_x_np = np.array(r_x)

        # Compute a term
        a_term = float(cbvf_grad_t) + np.dot(cbvf_grad_x_np, p_x_np)

        # Add disturbance term
        if hasattr(dynamics, 'disturbance_space') and r_x_np.size > 0:
            d_bounds = dynamics.disturbance_space
            disturbance_term = np.dot(cbvf_grad_x_np, r_x_np)

            if hasattr(d_bounds, 'lo') and hasattr(d_bounds, 'hi'):
                d_min = np.array(d_bounds.lo)
                d_max = np.array(d_bounds.hi)
            elif hasattr(d_bounds, 'radius'):
                d_max = d_bounds.radius * np.ones(r_x_np.shape[1])
                d_min = -d_max
            else:
                d_max = 0.1 * np.ones(r_x_np.shape[1])
                d_min = -d_max

            worst_case_d = np.where(disturbance_term >= 0, d_min, d_max)
            a_term += np.dot(disturbance_term.flatten(), worst_case_d)

        # Set up QP
        n_controls = len(u_ref_np)
        P = 2.0 * np.eye(n_controls)
        q = np.array(-2.0 * u_ref_np)

        # Safety constraint
        constraint_coeff = -np.dot(cbvf_grad_x_np, q_x_np)
        constraint_bound = np.array([a_term + self.gamma * float(cbvf_value)])

        # Add control bounds
        if u_max_mag is not None:
            G_bounds = np.vstack([-np.eye(n_controls), np.eye(n_controls)])
            G = np.vstack([constraint_coeff, G_bounds])
            h = np.concatenate([constraint_bound.flatten(), [u_max_mag, u_max_mag]])
        elif hasattr(dynamics, 'control_space'):
            control_bounds = dynamics.control_space
            if hasattr(control_bounds, 'lo') and hasattr(control_bounds, 'hi'):
                G_bounds = np.vstack([-np.eye(n_controls), np.eye(n_controls)])
                h = np.hstack([constraint_bound.flatten(), h_bounds.flatten()])

                G = np.vstack([constraint_coeff, G_bounds])
                h = np.hstack([constraint_bound, h_bounds])
            else:
                G = constraint_coeff
                h = constraint_bound
        else:
            G = constraint_coeff
            h = constraint_bound

        if self.verbose:
            print(f"QP Setup - P: {P.shape}, q: {q.shape}, G: {G.shape}, h: {h.shape}")
            print(f"P condition number: {np.linalg.cond(P):.2e}")
            print(f"Constraint matrix G:\n{G}")
            print(f"Constraint bounds h: {h}")
            print(f"Reference control: {u_ref_np}")

            # Check constraint violations at reference
            if G.shape[0] > 0:
                violations = G @ u_ref_np - h
                print(f"Constraint violations at u_ref: {violations}")
                print(f"Max violation: {np.max(violations):.6f}")

        # if self.verbose:
        #     # Debug: Analyze constraint feasibility
        #     print("\n=== CONSTRAINT FEASIBILITY ANALYSIS ===")
        #
        #     # Safety constraint: constraint_coeff * u <= constraint_bound
        #     # Rearranging: u <= constraint_bound / constraint_coeff (if coeff < 0)
        #     #              u >= constraint_bound / constraint_coeff (if coeff > 0)
        #     safety_coeff = constraint_coeff[0, 0]
        #     safety_bound = constraint_bound[0]
        #
        #     print(f"\n1. Safety constraint: {safety_coeff:.6e} * u <= {safety_bound:.6f}")
        #     if abs(safety_coeff) > 1e-10:  # Avoid division by very small numbers
        #         if safety_coeff > 0:
        #             u_max_from_safety = safety_bound / safety_coeff
        #             print(f"   → u <= {u_max_from_safety:.2f}")
        #             safety_feasible_range = (-np.inf, u_max_from_safety)
        #         else:
        #             u_min_from_safety = safety_bound / safety_coeff
        #             print(f"   → u >= {u_min_from_safety:.2f}")
        #             safety_feasible_range = (u_min_from_safety, np.inf)
        #     else:
        #         print(f"   → Coefficient too small, constraint is: 0 <= {safety_bound:.6f}")
        #         if safety_bound >= 0:
        #             print("   → Always satisfied")
        #             safety_feasible_range = (-np.inf, np.inf)
        #         else:
        #             print("   → IMPOSSIBLE - constraint cannot be satisfied!")
        #             safety_feasible_range = (np.inf, -np.inf)
        #
        #     # Control bounds
        #     print(f"\n2. Control bounds:")
        #     print(f"   Lower bound: u >= {-u_max_mag}")
        #     print(f"   Upper bound: u <= {u_max_mag}")
        #     control_feasible_range = (-u_max_mag, u_max_mag)
        #
        #     # Check intersection of feasible ranges
        #     print(f"\n=== FEASIBLE RANGES ===")
        #     print(f"Safety constraint allows: u ∈ [{safety_feasible_range[0]:.2f}, {safety_feasible_range[1]:.2f}]")
        #     print(f"Control bounds allow:     u ∈ [{control_feasible_range[0]:.2f}, {control_feasible_range[1]:.2f}]")
        #
        #     # Find intersection
        #     feasible_min = max(safety_feasible_range[0], control_feasible_range[0])
        #     feasible_max = min(safety_feasible_range[1], control_feasible_range[1])
        #
        #     if feasible_min <= feasible_max:
        #         print(f"\nFEASIBLE REGION: u ∈ [{feasible_min:.2f}, {feasible_max:.2f}]")
        #         print(f"Optimal feasible control closest to u_ref={u_ref_np[0]}: ", end="")
        #         u_optimal = np.clip(u_ref_np[0], feasible_min, feasible_max)
        #         print(f"{u_optimal:.2f}")
        #     else:
        #         print(f"\nINFEASIBLE! No value of u can satisfy all constraints.")
        #         print(f"Safety requires u ∈ [{safety_feasible_range[0]:.2f}, {safety_feasible_range[1]:.2f}]")
        #         print(
        #             f"But control bounds restrict u ∈ [{control_feasible_range[0]:.2f}, {control_feasible_range[1]:.2f}]")
        #         print(f"These ranges don't overlap!")
        #
        #         # Suggest which constraint to relax
        #         if safety_coeff < 0 and u_min_from_safety > u_max_mag:
        #             violation_amount = u_min_from_safety - u_max_mag
        #             print(f"\nSuggestion: Either:")
        #             print(f"  1. Increase control bound to at least {u_min_from_safety:.2f}")
        #             print(f"  2. Relax safety constraint (reduce gamma or modify CBVF)")
        #             print(f"  3. Use u = {u_max_mag} (violates safety by {violation_amount:.2f})")
        #
        #     # Also check the actual safety value at different controls
        #     print(f"\n=== SAFETY VALUES AT KEY CONTROLS ===")
        #     for u_test, label in [(u_ref_np[0], "u_ref"),
        #                           (-u_max_mag, "u_min"),
        #                           (u_max_mag, "u_max"),
        #                           (0, "u=0")]:
        #         safety_val = a_term + safety_coeff * u_test + self.gamma * float(cbvf_value)
        #         print(
        #             f"At {label}={u_test:8.2f}: safety = {safety_val:10.6f} {'(SAFE)' if safety_val >= 0 else '(UNSAFE)'}")
        #
        #     print("=" * 50 + "\n")

        # Use direct quadprog call for debugging
        if self.solver == 'quadprog':
            u_safe = debug_quadprog_call(P, q, G, h, self.verbose)
        else:
            u_safe = qpsolvers.solve_qp(P, q, G, h, solver=self.solver, verbose=self.verbose)

        if u_safe is None:
            print("QP solver failed, using previous control")
            return u_prev, (a_term + np.dot(cbvf_grad_x_np @ q_x_np, u_prev) + self.gamma * float(cbvf_value))[0]

        return u_safe[0], a_term + np.dot(cbvf_grad_x_np @ q_x_np, u_safe) + self.gamma * float(cbvf_value)