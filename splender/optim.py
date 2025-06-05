import jax
import jax.numpy as jnp
import jax.random as random
from jax.nn import sigmoid
import jax.tree_util as jtu
from jax import grad, jit, vmap, value_and_grad
import optax

def loss(model, batch):
    """
    Loss per image, summed over all batch
    """
    # jax.debug.print("model: {model}", model=model.init_knots)
    # return model.init_knots.sum()
    recon, lengths, curvatures = model()
    assert recon.shape == batch.shape
    recon_loss = jnp.mean((batch - recon) ** 2, axis=(-1, -2, -3)[:batch.ndim - 1]) # don't mean over batch yet
    min_scale_knots = 1e-5 * jax.nn.sigmoid((model.loc_params + model.knot_params)[..., 2]).min(axis = -1)
    # jax.debug.print("min_scale_knots: {min_scale_knots}", min_scale_knots=min_scale_knots)
    scale_multiplier_reg = 1e-3 * ((min_scale_knots - 1.0)**2).mean(axis=(-1, -2)[:lengths.ndim - 1])
    # curvature_reg = 1e-2 * curvatures.mean(axis=(-1, -2)[:lengths.ndim - 1])
    # curvature_reg = 1e-3 * curvatures.mean(axis=(-1, -2)[:lengths.ndim - 1])
    curvature_reg = 5e-4 * curvatures.mean(axis=(-1, -2)[:lengths.ndim - 1])
    # curvature_reg = 1e-4 * curvatures.mean(axis=(-1, -2)[:lengths.ndim - 1])
    length_reg = 1e-5 * lengths.mean(axis=(-1, -2)[:lengths.ndim - 1])
    # jax.debug.print("curvature_reg: {curvature_reg}", curvature_reg=curvature_reg)
    # jax.debug.print("recon_loss: {recon_loss}", recon_loss=recon_loss)
    return recon_loss + scale_multiplier_reg + curvature_reg + length_reg + min_scale_knots, lengths

# @jit
def image_loss(model, batch):
    loss_per_image, _ = loss(model, batch)
    return loss_per_image.sum()

# @jit
def video_loss(model, batch):
    loss_per_video, lengths = loss(model, batch)
    knot_params = (model.loc_params + model.knot_params)[..., 2]
    anchor_knot_reg = 5e-1 * ((knot_params[:, :1, :1, :] - knot_params[:, :1, :, :])**2).mean(axis=(-1, -2)[:lengths.ndim - 1])
    equal_length_reg = 1e-2 * ((lengths[:, :, :-1] - lengths[:, :, 1:])**2).mean(axis=(-1, -2)[:lengths.ndim - 1])
    temporal_reg = 1e-1 * ((knot_params[:, :, :-1, :] - knot_params[:, :, 1:, :])**2).mean(axis=(-1, -2, -3)[:lengths.ndim])
    return (loss_per_video + anchor_knot_reg + equal_length_reg + temporal_reg).sum()


def fit(model, batch, n_iter=1000, lr=1e-3, video = True):
    model.__manual_post_init__()
    # assert batch has values in [0, 1]
    assert jnp.all(batch >= 0) and jnp.all(batch <= 1)

    loss = video_loss if video else image_loss

    optim = optax.adam(lr)
    @jit
    def make_step(model, batch, opt_state):
        loss_value, grads = value_and_grad(loss)(model, batch)
        # jax.debug.breakpoint()
        updates, opt_state = optim.update(grads, opt_state, model)
        model = optax.apply_updates(model, updates)
        return model, opt_state, loss_value
    losses = []
    opt_state = optim.init(model)
    for step in range(n_iter):
        model, opt_state, loss_value = make_step(model, batch, opt_state)
        losses.append(loss_value)
    return model, losses