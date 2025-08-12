import jax
import jax.numpy as jnp
from jax import jit, vmap, grad
from functools import partial
from typing import Dict


# Enable 64-bit precision if needed (for better numerical accuracy)
jax.config.update("jax_enable_x64", True)

@partial(jit, static_argnums=(6,))
def pacejka_62_scalar(
        tpf_params: Dict[str, float],
        F_z: float,
        kappa: float,
        alpha_F: float,
        gamma: float,
        Vwx: float,
        tyre_mode: int
) -> jnp.ndarray:
    """
    JAX-optimized Pacejka 6.2 tire model for scalar inputs

    Parameters:
    -----------
    tpf_params : dict
        Tire parameters (should be converted to JAX arrays)
    F_z : float
        Vertical force [N]
    kappa : float
        Longitudinal slip [-]
    alpha_F : float
        Slip angle [rad]
    gamma : float
        Camber angle [rad]
    Vwx : float
        Wheel velocity [m/s]
    tyre_mode : int
        Mode of slip forces (static argument for JIT)

    Returns:
    --------
    jnp.ndarray : [F_x, F_y, F_z, M_x, M_y, M_z, trail, K_xkappa, K_yalpha, mu_x, mu_y]
    """

    # Extract frequently used parameters (for cleaner code)
    tp = tpf_params

    # Limitations using JAX operations
    kappa = jnp.clip(kappa, tp['kappa_min'], tp['kappa_max'])
    alpha_F = jnp.clip(alpha_F, tp['alpha_min'], tp['alpha_max'])
    gamma = jnp.clip(gamma, tp['gamma_min'], tp['gamma_max'])
    F_z = jnp.clip(F_z, tp['F_z_min'], tp['F_z_max'])
    pres_i = jnp.clip(tp['p_i0'], tp['p_lim_min'], tp['p_lim_max'])

    # Dimensionless factors
    dp_i = (pres_i - tp['p_nom']) / tp['p_nom']
    F_z0 = tp['F_z0'] * tp['LFZ0']
    df_z = (F_z - F_z0) / F_z0

    # Initialize variables
    alpha_M = alpha_F

    # Low speed correction (disabled)
    corrVxlow_cos = 0.0
    corrVxlow_sin = 1.0

    # Turnslip factors (all = 1 for simplified version)
    zeta = jnp.ones(9)  # zeta0 through zeta8

    # === PURE SLIP MODE ===

    # Longitudinal force components
    C_x = tp['PCX1'] * tp['LCX']

    mu_x = ((tp['PDX1'] + tp['PDX2'] * df_z) *
            (1 + tp['PPX3'] * dp_i + tp['PPX4'] * dp_i ** 2) *
            jnp.maximum(1 - tp['PDX3'] * gamma ** 2, 1e-5) * tp['LMUX'])

    D_x = mu_x * F_z * zeta[1]

    K_xkappa = (F_z * (tp['PKX1'] + tp['PKX2'] * df_z) *
                (1 + tp['PPX1'] * dp_i + tp['PPX2'] * dp_i ** 2) *
                jnp.exp(jnp.minimum(tp['PKX3'] * df_z, 50.0)) * tp['LKX'])

    # Add small epsilon to avoid division by zero
    eps = 1e-10
    K_xkappa = K_xkappa + eps * jnp.sign(K_xkappa + eps)

    B_x = K_xkappa / (C_x * D_x + eps)

    S_Vx = F_z * (tp['PVX1'] + tp['PVX2'] * df_z) * tp['LVX'] * tp['LMUX'] * zeta[1] * corrVxlow_sin
    S_Hx = (tp['PHX1'] + tp['PHX2'] * df_z) * tp['LHX'] * corrVxlow_sin

    # Lateral force components
    C_y = tp['PCY1'] * tp['LCY']

    mu_y_nogamma = ((tp['PDY1'] + tp['PDY2'] * df_z) *
                    (1 + tp['PPY3'] * dp_i + tp['PPY4'] * dp_i ** 2) * tp['LMUY'])

    mu_y = mu_y_nogamma * jnp.maximum(1 - tp['PDY3'] * gamma ** 2, 1e-5)

    D_y = mu_y * F_z * zeta[2]

    K_yalpha0 = (tp['PKY1'] * F_z0 * (1 + tp['PPY1'] * dp_i) *
                 jnp.sin(tp['PKY4'] * jnp.arctan(F_z /
                                                 jnp.maximum(tp['PKY2'] * F_z0 * (1 + tp['PPY2'] * dp_i), eps))) * tp[
                     'LKY'])

    K_yalpha = (tp['PKY1'] * F_z0 * (1 + tp['PPY1'] * dp_i) *
                jnp.sin(tp['PKY4'] * jnp.arctan(F_z /
                                                jnp.maximum((tp['PKY2'] + tp['PKY5'] * gamma ** 2) * F_z0 *
                                                            (1 + tp['PPY2'] * dp_i), eps))) *
                (1 - tp['PKY3'] * jnp.abs(gamma)) * tp['LKY'] * zeta[3])

    B_y = K_yalpha / (C_y * D_y + eps)

    K_ygamma0 = (tp['PKY6'] + tp['PKY7'] * df_z) * F_z * tp['LKYC'] * (1 + tp['PPY5'] * dp_i)

    S_Vy0 = F_z * (tp['PVY1'] + tp['PVY2'] * df_z) * tp['LVY'] * tp['LMUY'] * corrVxlow_sin
    S_Vygamma = F_z * (tp['PVY3'] + tp['PVY4'] * df_z) * gamma * tp['LKYC'] * tp['LMUY'] * zeta[2] * corrVxlow_sin
    S_Vy = S_Vy0 * zeta[2] + S_Vygamma

    S_Hy0 = (tp['PHY1'] + tp['PHY2'] * df_z) * tp['LHY'] * corrVxlow_sin
    S_Hygamma = (K_ygamma0 * gamma - S_Vygamma) / (K_yalpha + eps) * (zeta[0] + zeta[4] - 1)
    S_Hy = S_Hy0 + S_Hygamma

    # === COMBINED SLIP ===

    # Use JAX conditional for combined slip calculations
    use_combined = jnp.logical_or(tp['RCX1'] > 0, tp['RCY1'] > 0)

    def combined_slip_calc():
        S_Hxalpha = tp['RHX1']
        C_xalpha = tp['RCX1']
        B_xalpha = jnp.maximum((tp['RBX1'] + tp['RBX3'] * gamma ** 2) *
                               jnp.cos(jnp.arctan(tp['RBX2'] * kappa)) * tp['LXAL'], 0)
        E_xalpha = jnp.minimum(tp['REX1'] + tp['REX2'] * df_z, 1)

        alpha_s = alpha_F + S_Hxalpha

        G_xalpha_num = jnp.cos(C_xalpha * jnp.arctan(B_xalpha * alpha_s -
                                                     E_xalpha * (B_xalpha * alpha_s - jnp.arctan(B_xalpha * alpha_s))))
        G_xalpha_den = jnp.cos(C_xalpha * jnp.arctan(B_xalpha * S_Hxalpha -
                                                     E_xalpha * (B_xalpha * S_Hxalpha - jnp.arctan(
            B_xalpha * S_Hxalpha))))
        G_xalpha = G_xalpha_num / (G_xalpha_den + eps)

        S_Hykappa = tp['RHY1'] + tp['RHY2'] * df_z
        kappa_s = kappa + S_Hykappa
        C_ykappa = tp['RCY1']
        E_ykappa = jnp.minimum(tp['REY1'] + tp['REY2'] * df_z, 1)
        B_ykappa = jnp.maximum((tp['RBY1'] + tp['RBY4'] * gamma ** 2) *
                               jnp.cos(jnp.arctan(tp['RBY2'] * (alpha_F - tp['RBY3']))) * tp['LYKA'], 0)

        G_ykappa_num = jnp.cos(C_ykappa * jnp.arctan(B_ykappa * kappa_s -
                                                     E_ykappa * (B_ykappa * kappa_s - jnp.arctan(B_ykappa * kappa_s))))
        G_ykappa_den = jnp.cos(C_ykappa * jnp.arctan(B_ykappa * S_Hykappa -
                                                     E_ykappa * (B_ykappa * S_Hykappa - jnp.arctan(
            B_ykappa * S_Hykappa))))
        G_ykappa = G_ykappa_num / (G_ykappa_den + eps)

        kappa_x = kappa + S_Hx
        alpha_y = alpha_F + S_Hy

        return G_xalpha, G_ykappa, kappa_x, alpha_y, kappa_s, C_ykappa, E_ykappa, S_Hykappa

    def simple_combined_calc():
        kappa_s = kappa + S_Hx + S_Vx / (K_xkappa + eps)
        alpha_s = jnp.tan(alpha_F) + S_Vy / (K_yalpha + eps) + S_Hy

        combined_mag = jnp.sqrt(kappa_s ** 2 + alpha_s ** 2) + eps
        G_xalpha = jnp.abs(kappa_s) / combined_mag
        G_ykappa = jnp.abs(alpha_s) / combined_mag

        kappa_x = combined_mag * jnp.sign(kappa_s) - S_Vx / (K_xkappa + eps)
        alpha_y = jnp.arctan(combined_mag) * jnp.sign(alpha_s) - S_Vy / (K_yalpha + eps)

        C_ykappa = 0.0
        E_ykappa = 0.0
        S_Hykappa = 0.0

        return G_xalpha, G_ykappa, kappa_x, alpha_y, kappa_s, C_ykappa, E_ykappa, S_Hykappa

    # Use lax.cond for conditional execution
    G_xalpha, G_ykappa, kappa_x, alpha_y, kappa_s, C_ykappa, E_ykappa, S_Hykappa = jax.lax.cond(
        use_combined,
        combined_slip_calc,
        simple_combined_calc
    )

    # Curvature factors
    E_x = jnp.minimum((tp['PEX1'] + tp['PEX2'] * df_z + tp['PEX3'] * df_z ** 2) *
                      (1 - tp['PEX4'] * jnp.sign(kappa_x)) * tp['LEX'], 1)

    E_y = jnp.minimum((tp['PEY1'] + tp['PEY2'] * df_z) *
                      (1 + tp['PEY5'] * gamma ** 2 - (tp['PEY3'] + tp['PEY4'] * gamma) *
                       jnp.sign(alpha_y)) * tp['LEY'], 1)

    # Pure forces
    F_xp = (D_x * jnp.minimum(jnp.abs(Vwx) * C_x * jnp.abs(B_x * kappa_x), 1) *
            jnp.sign(B_x * kappa_x) * corrVxlow_cos +
            (D_x * jnp.sin(C_x * jnp.arctan(B_x * kappa_x - E_x *
                                            (B_x * kappa_x - jnp.arctan(B_x * kappa_x)))) + S_Vx) * (1 - corrVxlow_cos))

    # Apply tire mode using JAX operations
    mode_1_3 = jnp.logical_or(tyre_mode == 1, tyre_mode == 3)
    mode_4_5 = jnp.logical_or(tyre_mode == 4, tyre_mode == 5)
    F_x = jnp.where(mode_1_3, F_xp, jnp.where(mode_4_5, G_xalpha * F_xp, 0.0))

    # Plysteer components
    D_Vykappa = (mu_y * F_z * (tp['RVY1'] + tp['RVY2'] * df_z + tp['RVY3'] * gamma) *
                 jnp.cos(jnp.arctan(tp['RVY4'] * alpha_F)) * zeta[2])
    S_Vykappa = D_Vykappa * jnp.sin(tp['RVY5'] * jnp.arctan(tp['RVY6'] * kappa)) * tp['LVYKA'] * corrVxlow_sin

    # Pure lateral force
    F_yp = (D_y * jnp.minimum(jnp.abs(Vwx) * C_y * jnp.abs(B_y * alpha_y), 1) *
            jnp.sign(B_y * alpha_y) * corrVxlow_cos +
            (D_y * jnp.sin(C_y * jnp.arctan(B_y * alpha_y - E_y *
                                            (B_y * alpha_y - jnp.arctan(B_y * alpha_y)))) + S_Vy) * (1 - corrVxlow_cos))

    mode_2_3 = jnp.logical_or(tyre_mode == 2, tyre_mode == 3)
    F_y = jnp.where(mode_2_3, F_yp, jnp.where(mode_4_5, G_ykappa * F_yp + S_Vykappa, 0.0))

    # Moments calculation
    M_x = calculate_overturning_moment(tp, F_z, F_y, gamma, dp_i, F_z0)
    M_y = calculate_rolling_moment(tp, F_z0, F_x, Vwx, gamma, F_z, corrVxlow_cos)

    # Self-aligning moment (simplified for brevity)
    trail, M_z = calculate_aligning_moment(
        tp, df_z, dp_i, gamma, F_z, F_z0, zeta, corrVxlow_sin,
        alpha_M, K_yalpha, K_xkappa, kappa, alpha_F, S_Hy, S_Vy,
        S_Hy0, S_Vy0, C_y, K_yalpha0, mu_y_nogamma, B_y, Vwx,
        tyre_mode, F_yp, F_x, use_combined, kappa_s, C_ykappa,
        E_ykappa, S_Hykappa,)

    return jnp.array([F_x, F_y, F_z, M_x, M_y, M_z, trail, K_xkappa, K_yalpha, mu_x, mu_y])


@jit
def calculate_overturning_moment(tp, F_z, F_y, gamma, dp_i, F_z0):
    """Calculate overturning moment Mx"""
    M_x = (tp['R_0'] * F_z * tp['LMX'] *
           (tp['QSX1'] * tp['LVMX'] -
            tp['QSX2'] * gamma * (1 + tp['PPMX1'] * dp_i) +
            tp['QSX3'] * F_y / F_z0 +
            tp['QSX4'] * jnp.cos(tp['QSX5'] * jnp.arctan((tp['QSX6'] * F_z / F_z0) ** 2)) *
            jnp.sin(tp['QSX7'] * gamma + tp['QSX8'] * jnp.arctan(tp['QSX9'] * F_y / F_z0)) +
            tp['QSX10'] * jnp.arctan(tp['QSX11'] * F_z / F_z0) * gamma) +
           tp['R_0'] * tp['LMY'] * (F_y * (tp['QSX13'] + tp['QSX14'] * jnp.abs(gamma)) -
                                    F_z * tp['QSX12'] * gamma * jnp.abs(gamma)))
    return M_x


@jit
def calculate_rolling_moment(tp, F_z0, F_x, Vwx, gamma, F_z, corrVxlow_cos):
    """Calculate rolling resistance moment My"""
    M_y = (-tp['R_0'] * F_z0 * tp['LMY'] *
           (tp['QSY1'] +
            tp['QSY2'] * F_x / F_z0 +
            tp['QSY3'] * jnp.abs(Vwx / tp['LONGVL']) +
            tp['QSY4'] * (Vwx / tp['LONGVL']) ** 4 +
            tp['QSY5'] * gamma ** 2 +
            tp['QSY6'] * F_z / F_z0 * gamma ** 2) *
           (F_z / F_z0) ** tp['QSY7'] * (tp['p_i0'] / tp['p_nom']) ** tp['QSY8'])
    M_y = M_y * (1 - corrVxlow_cos) * jnp.sign(Vwx)
    return M_y


@jit
def calculate_aligning_moment(tp, df_z, dp_i, gamma, F_z, F_z0, zeta, corrVxlow_sin,
                              alpha_M, K_yalpha, K_xkappa, kappa, alpha_F, S_Hy, S_Vy,
                              S_Hy0, S_Vy0, C_y, K_yalpha0, mu_y_nogamma, B_y, Vwx,
                              tyre_mode, F_yp, F_x, use_combined, kappa_s, C_ykappa,
                              E_ykappa, S_Hykappa):
    """Calculate self-aligning moment Mz and trail (simplified)"""
    eps = 1e-10

    # Pneumatic trail factors
    B_t = ((tp['QBZ1'] + tp['QBZ2'] * df_z + tp['QBZ3'] * df_z ** 2) *
           (1 + tp['QBZ4'] * gamma + tp['QBZ5'] * jnp.abs(gamma)) * tp['LKY'] / tp['LMUY'])
    C_t = tp['QCZ1']
    D_t = (F_z * (tp['QDZ1'] + tp['QDZ2'] * df_z) * (1 - tp['PPZ1'] * dp_i) *
           (1 + tp['QDZ3'] * gamma + tp['QDZ4'] * gamma ** 2) *
           tp['R_0'] / F_z0 * tp['LTR'] * zeta[5])

    S_Ht = tp['QHZ1'] + tp['QHZ2'] * df_z + (tp['QHZ3'] + tp['QHZ4'] * df_z) * gamma * corrVxlow_sin

    # Residual moment factors
    D_y0 = mu_y_nogamma * F_z * zeta[2]
    B_y0 = K_yalpha0 / (C_y * D_y0 + eps)

    D_r = (F_z * tp['R_0'] * tp['LMUY'] * jnp.cos(alpha_M) *
           ((tp['QDZ6'] + tp['QDZ7'] * df_z) * tp['LRES'] * zeta[2] +
            (tp['QDZ8'] + tp['QDZ9'] * df_z) * gamma * tp['LKZC'] * (1 + tp['PPZ2'] * dp_i) * zeta[0] +
            (tp['QDZ10'] + tp['QDZ11'] * df_z) * gamma * jnp.abs(gamma) * tp['LKZC'] * zeta[0]) *
           corrVxlow_sin - zeta[8] + 1)

    B_r = (tp['QBZ9'] * tp['LKY'] / tp['LMUY'] + tp['QBZ10'] * B_y * C_y) * zeta[6]

    # Slip angles for trail and residual moment
    alpha_t = alpha_M + S_Ht
    alpha_r = alpha_M + S_Hy + S_Vy / (K_yalpha + eps)

    # Equivalent slip angles based on tire mode
    mode_2_3 = jnp.logical_or(tyre_mode == 2, tyre_mode == 3)
    mode_4_5 = jnp.logical_or(tyre_mode == 4, tyre_mode == 5)

    alpha_teq = jnp.where(
        mode_2_3, alpha_t,
        jnp.where(mode_4_5,
                  jnp.arctan(jnp.sqrt(jnp.tan(alpha_t) ** 2 + (K_xkappa * kappa / (K_yalpha + eps)) ** 2)) * jnp.sign(
                      alpha_t),
                  0.0)
    )

    alpha_req = jnp.where(
        mode_2_3, alpha_r,
        jnp.where(mode_4_5,
                  jnp.arctan(jnp.sqrt(jnp.tan(alpha_r) ** 2 + (K_xkappa * kappa / (K_yalpha + eps)) ** 2)) * jnp.sign(
                      alpha_r),
                  0.0)
    )

    # Trail and residual moment
    E_t = jnp.minimum((tp['QEZ1'] + tp['QEZ2'] * df_z + tp['QEZ3'] * df_z ** 2) *
                      (1 + (tp['QEZ4'] + tp['QEZ5'] * gamma) * (2 / jnp.pi) *
                       jnp.arctan(B_t * C_t * alpha_t)), 1)

    trail = (D_t * jnp.cos(C_t * jnp.arctan(B_t * alpha_teq - E_t *
                                            (B_t * alpha_teq - jnp.arctan(B_t * alpha_teq)))) *
             jnp.cos(alpha_M) * corrVxlow_sin)

    M_zr = D_r * jnp.cos(zeta[7] * jnp.arctan(B_r * alpha_req))

    # Lateral offset
    s = ((tp['SSZ1'] + tp['SSZ2'] * F_yp / F_z0 +
          (tp['SSZ3'] + tp['SSZ4'] * df_z) * gamma) * tp['R_0'] * tp['LS'])

    # Calculate F_yp0 for self-aligning moment
    alpha_y0 = jnp.where(use_combined,
                         alpha_F + S_Hy0,
                         jnp.arctan(
                             jnp.sqrt(kappa_s ** 2 + (jnp.tan(alpha_F) + S_Vy0 / (K_yalpha0 + eps) + S_Hy0) ** 2)) *
                         jnp.sign(jnp.tan(alpha_F) + S_Vy0 / (K_yalpha0 + eps) + S_Hy0) - S_Vy0 / (K_yalpha0 + eps))

    E_y0 = jnp.minimum((tp['PEY1'] + tp['PEY2'] * df_z) * (1 - tp['PEY3'] * jnp.sign(alpha_y0)) * tp['LEY'], 1)

    F_yp0 = (D_y0 * jnp.minimum(jnp.abs(Vwx) * C_y * jnp.abs(B_y0 * alpha_y0), 1) *
             jnp.sign(B_y0 * alpha_y0) * corrVxlow_sin +
             (D_y0 * jnp.sin(C_y * jnp.arctan(B_y0 * alpha_y0 - E_y0 *
                                              (B_y0 * alpha_y0 - jnp.arctan(B_y0 * alpha_y0)))) + S_Vy0) * (
                         1 - corrVxlow_sin))

    # Calculate G_ykappa0 for combined modes
    def calc_G_ykappa0_combined():
        B_ykappa0 = jnp.maximum(tp['RBY1'] * jnp.cos(jnp.arctan(tp['RBY2'] *
                                                                (alpha_F - tp['RBY3']))) * tp['LYKA'], 0.01)
        return (jnp.cos(C_ykappa * jnp.arctan(B_ykappa0 * kappa_s - E_ykappa *
                                              (B_ykappa0 * kappa_s - jnp.arctan(B_ykappa0 * kappa_s)))) /
                (jnp.cos(C_ykappa * jnp.arctan(B_ykappa0 * S_Hykappa - E_ykappa *
                                               (B_ykappa0 * S_Hykappa - jnp.arctan(B_ykappa0 * S_Hykappa)))) + eps))

    def calc_G_ykappa0_simple():
        alpha_s0 = jnp.tan(alpha_F) + S_Vy0 / (K_yalpha0 + eps) + S_Hy0
        return jnp.abs(alpha_s0) / (jnp.sqrt(kappa_s ** 2 + alpha_s0 ** 2) + eps)

    G_ykappa0 = jax.lax.cond(use_combined, calc_G_ykappa0_combined, calc_G_ykappa0_simple)

    # Final self-aligning moment
    M_z = jnp.where(
        mode_2_3, -trail * F_yp0 + M_zr,
        jnp.where(mode_4_5, -trail * F_yp0 * G_ykappa0 + M_zr + s * F_x, 0.0)
    )

    return trail, M_z


# Vectorized version for batch processing
pacejka_62_batch = vmap(pacejka_62_scalar, in_axes=(None, 0, 0, 0, 0, 0, None))


def preprocess_tpf_for_jax(tpf_dict: Dict) -> Dict:
    """
    Convert TPF dictionary to JAX-compatible format

    Parameters:
    -----------
    tpf_dict : dict
        Original TPF dictionary from read_tpf

    Returns:
    --------
    dict : JAX-compatible parameter dictionary
    """
    # Convert all values to JAX arrays and flatten nested structures
    jax_params = {}

    for key, value in tpf_dict.items():
        if isinstance(value, list):
            # Handle range parameters
            if key == 'p_lim':
                jax_params['p_lim_min'] = jnp.array(value[0], dtype=jnp.float64)
                jax_params['p_lim_max'] = jnp.array(value[1], dtype=jnp.float64)
            elif key == 'F_z':
                jax_params['F_z_min'] = jnp.array(value[0], dtype=jnp.float64)
                jax_params['F_z_max'] = jnp.array(value[1], dtype=jnp.float64)
            elif key == 'kappa':
                jax_params['kappa_min'] = jnp.array(value[0], dtype=jnp.float64)
                jax_params['kappa_max'] = jnp.array(value[1], dtype=jnp.float64)
            elif key == 'alpha':
                jax_params['alpha_min'] = jnp.array(value[0], dtype=jnp.float64)
                jax_params['alpha_max'] = jnp.array(value[1], dtype=jnp.float64)
            elif key == 'gamma':
                jax_params['gamma_min'] = jnp.array(value[0], dtype=jnp.float64)
                jax_params['gamma_max'] = jnp.array(value[1], dtype=jnp.float64)
        else:
            # Convert scalar values
            jax_params[key] = jnp.array(value, dtype=jnp.float64)

    return jax_params


# Optional: Create differentiable version for optimization/learning
pacejka_62_grad = grad(lambda *args: pacejka_62_scalar(*args)[0])  # Gradient w.r.t F_x


def create_tire_model(tpf_file: str):
    """
    Factory function to create a JIT-compiled tire model

    Parameters:
    -----------
    tpf_file : str
        Path to TPF file

    Returns:
    --------
    callable : JIT-compiled Pacejka model function
    """
    from utils.tpf_handler import read_tpf  # Import the TPF reader

    # Load and preprocess parameters
    tpf_dict = read_tpf(tpf_file)
    jax_params = preprocess_tpf_for_jax(tpf_dict)

    # Create partially applied function with fixed parameters
    @partial(jit, static_argnums=(5,))
    def tire_model(F_z, kappa, alpha_F, gamma, Vwx, tyre_mode=4):
        return pacejka_62_scalar(jax_params, F_z, kappa, alpha_F, gamma, Vwx, tyre_mode)

    # Also create batch version
    tire_model_batch = vmap(tire_model, in_axes=(0, 0, 0, 0, 0, None))

    return tire_model, tire_model_batch, jax_params
