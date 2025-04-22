import jax
import jax.numpy as jnp
import jax.random as random
from jax.nn import sigmoid
import jax.tree_util as jtu
from jax import grad, jit, vmap, value_and_grad
import optax

@jit
def loss(model, images):
    """
    Loss per image, summed over all images
    """
    recon, lengths, curvatures = model()
    assert recon.ndim == 3
    assert recon.shape == images.shape
    recon_loss = jnp.mean((images - recon) ** 2, axis=(-1, -2))
    min_scale_knots = jax.nn.sigmoid((model.loc_params + model.knot_params)[..., 2]).min(axis = -1)
    # jax.debug.print("min_scale_knots: {min_scale_knots}", min_scale_knots=min_scale_knots)
    scale_multiplier_reg = 1e-3 * ((min_scale_knots - 1.0)**2).mean(axis=-1)
    # curvature_reg = 1e-2 * curvatures.mean(axis=-1)
    curvature_reg = 1e-3 * curvatures.mean(axis=-1)
    length_reg = 1e-3 * lengths.mean(axis=-1)
    return (recon_loss + scale_multiplier_reg + curvature_reg + length_reg).sum()


def fit(model, images, n_iter=1000):
    optim = optax.adam(1e-2)
    def make_step(model, images, opt_state):
        loss_value, grads = value_and_grad(loss)(model, images)
        updates, opt_state = optim.update(grads, opt_state, model)
        model = optax.apply_updates(model, updates)
        return model, opt_state, loss_value
    losses = []
    opt_state = optim.init(model)
    for step in range(n_iter):
        model, opt_state, loss_value = make_step(model, images, opt_state)
        losses.append(loss_value)
    return model, losses