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

def find_safe_entry_time_efficient(cbvf_interpolator, state, safe_threshold=0.0):
    """
    More efficient version using binary search if times are sorted.
    """
    times = -cbvf_interpolator.times

    # Binary search for the first safe time
    left, right = 0, times.shape[0] - 1
    safe_entry_time = None

    while left <= right:
        mid = (left + right) // 2
        cbvf_value = cbvf_interpolator.interpolate_value(state, times[mid])

        if cbvf_value >= safe_threshold:
            safe_entry_time = float(mid)
            right = mid - 1  # Look for earlier safe time
        else:
            left = mid + 1  # Need later time to be safe

    # Return the safe entry time or latest time if never safe
    return safe_entry_time if safe_entry_time is not None else float(times[-1])


# Simple enhanced version of your existing controller
class CBVFQPController:
    def __init__(self, value_fn, gamma, solver='quadprog', verbose=False):
        self.cbvf = value_fn
        self.gamma = gamma
        self.solver = solver
        self.verbose = verbose
        self.safe_threshold = 0.0  # Safe set threshold

    def compute_safe_control(self, state, time, u_ref, dynamics):
        try:
            gradient_time = find_safe_entry_time_efficient(self.cbvf, state, self.safe_threshold)
            if self.verbose:
                print(f"Current state: {state}, time: {time}, reference control: {u_ref}")
                print(f"Gradient time for safe entry: {gradient_time}")
            cbvf_value, cbvf_grad_x, cbvf_grad_t = self.cbvf.get_value_and_gradients(state, gradient_time)

            # Get system matrices
            p_x = dynamics.open_loop_dynamics(state, time)
            q_x = dynamics.control_jacobian(state, time)
            r_x = dynamics.disturbance_jacobian(state, time)

            # Convert to numpy
            u_ref_np = np.array(u_ref)
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
            q = -2.0 * u_ref_np

            # Safety constraint
            constraint_coeff = -np.dot(cbvf_grad_x_np, q_x_np).reshape(1, -1)
            constraint_bound = np.array([a_term + self.gamma * float(cbvf_value)])

            # Add control bounds
            if hasattr(dynamics, 'control_space'):
                control_bounds = dynamics.control_space
                if hasattr(control_bounds, 'lo') and hasattr(control_bounds, 'hi'):
                    u_min = np.array(control_bounds.lo) * 3
                    u_max = np.array(control_bounds.hi) * 3

                    G_bounds = np.vstack([-np.eye(n_controls), np.eye(n_controls)])
                    h_bounds = np.hstack([-u_min, u_max])

                    G = np.vstack([constraint_coeff, G_bounds])
                    h = np.hstack([constraint_bound, h_bounds])
                else:
                    G = constraint_coeff
                    h = constraint_bound
            else:
                G = constraint_coeff
                h = constraint_bound

            if False:
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

            # Use direct quadprog call for debugging
            if self.solver == 'quadprog':
                u_safe = debug_quadprog_call(P, q, G, h, self.verbose)
            else:
                u_safe = qpsolvers.solve_qp(P, q, G, h, solver=self.solver, verbose=self.verbose)

            if u_safe is None:
                if self.verbose:
                    print("QP solver failed, using reference control")
                return u_ref

            return jnp.array(u_safe)

        except Exception as e:
            if self.verbose:
                print(f"CBVF-QP controller error: {e}")
                import traceback
                traceback.print_exc()
            return u_ref