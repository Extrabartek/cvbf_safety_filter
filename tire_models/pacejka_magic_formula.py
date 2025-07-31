import jax.numpy as jnp

def pacejka_lateral_force(alpha, fz, mu, cor):
    """
    Calculate the lateral force of the tire, using the pacejka magic formula

    :param alpha: angle of slip
    :param fz: vertical force
    :param mu: friction coefficient
    :param c: refrerence cornering stiffness
    :return: lateral force
    """

    c_alpha_ref = 1.0

    d = mu * fz
    b = cor / (c_alpha_ref * d)
    e = -0.1 * jnp.ones(jnp.shape(alpha))
    c = 1.3 * jnp.ones(jnp.shape(alpha))

    return d * jnp.sin(c * jnp.atan(b * alpha - e * (b * alpha - jnp.atan(b * alpha))))
