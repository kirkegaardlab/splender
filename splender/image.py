from .utils import circle_image, knots2params, params2knots
import jax.numpy as jnp
from jax import grad, jit, vmap, value_and_grad
import jax
import jax.random as random
from functools import partial
from jax.nn import sigmoid, softplus
from jax.image import scale_and_translate
from interpax import Interpolator1D, Interpolator2D
from collections import namedtuple
from jax.scipy.special import logit
import optax
from tqdm import tqdm
import matplotlib.pyplot as plt

param_tuple = namedtuple("params", ["locations",
                                     "knots",
                                     "scale",
                                     "background_contrast",
                                     "background_brightness",
                                     "background_slope",
                                     "background_angle",
                                     "background_translation",
                                     "spline_contrast",
                                     "spline_brightness",
                                     "kernel",
                                     "blob",
                                     "opacity",
                                     ])


def blob(params):
    x = jnp.linspace(-1, 1, params.shape[0])
    spline = Interpolator1D(x, params, method='cubic')
    X, Y = jnp.meshgrid(x, x)
    D = jnp.sqrt(X**2 + Y**2)
    im = vmap(spline)(jnp.exp(-D))
    return im


def config_closure(config):
    res = config.res
    n_points_per_spline = config.n_points_per_spline
    eps = 1e-6

    def fit_spline(params):
        knots = params2knots(params)

        x, y, scale_knots = knots[..., 0], knots[..., 1], knots[..., 2]
        
        s = jnp.linspace(0, 1, len(x))

        x_spline = Interpolator1D(s, x, method="cubic2")
        y_spline = Interpolator1D(s, y, method="cubic2")
        scale_spline = Interpolator1D(s, scale_knots, method="cubic")
        return x_spline, y_spline, scale_spline
    
    def spline_curvature(x_spline, y_spline):
        d2x_ds2 = vmap(partial(x_spline, dx=2))
        d2y_ds2 = vmap(partial(y_spline, dx=2))
        s_fine = jnp.linspace(0, 1, n_points_per_spline)
        delta_s = s_fine[1] - s_fine[0]
        curvature = jnp.sqrt(d2x_ds2(s_fine)**2 + d2y_ds2(s_fine)**2 + eps) * delta_s
        # return curvature.var(0) + curvature.mean(0)
        return curvature.mean(0)
    
    def get_uniform_points(x_spline, y_spline):
        # Compute arc length
        dx_ds = vmap(partial(x_spline, dx=1))
        dy_ds = vmap(partial(y_spline, dx=1))
        s_fine = jnp.linspace(0, 1, n_points_per_spline)
        ds_vals = jnp.sqrt(dx_ds(s_fine)**2 + dy_ds(s_fine)**2 + eps)
        delta_s = s_fine[1] - s_fine[0]
        cumulative_length = jnp.concatenate([
            jnp.array([0.0]),
            jnp.cumsum(0.5 * (ds_vals[1:] + ds_vals[:-1]) * delta_s)
        ])
        s_uniform = jnp.interp(jnp.linspace(0, cumulative_length[-1], n_points_per_spline), cumulative_length, s_fine)
        return s_uniform, cumulative_length[-1]

    def fit_and_draw_spline(params):
        x_spline, y_spline, scale_spline = fit_spline(params)
        curvature = spline_curvature(x_spline, y_spline)
        s_uniform, length = get_uniform_points(x_spline, y_spline)
        x_uniform = x_spline(s_uniform)
        y_uniform = y_spline(s_uniform)
        scales_uniform = scale_spline(s_uniform)
        return x_uniform, y_uniform, scales_uniform, length, curvature
    
    def logistic_background(slope, angle, translation):
        x = jnp.linspace(-1, 1, res)
        y = jnp.linspace(-1, 1, res)
        xx, yy = jnp.meshgrid(x, y)
        angle = sigmoid(angle) * 2 * jnp.pi
        slope = sigmoid(slope) * 10
        translation = sigmoid(translation) * 2 - 1
        xx = jnp.cos(angle) * (xx - translation) - jnp.sin(angle) * (yy - translation)
        background = sigmoid(slope * xx)
        return background

    def draw_point(point, scale, blob_params, contrast_param, brightness_param, debug_mode=False): 
        im = blob(blob_params) * contrast_param + brightness_param
        to = res * point - im.shape[0] / 2 * scale + 0.5

        im = jnp.where(debug_mode, (0 * im).at[6, 6].set(1.0), im)

        return scale_and_translate(
            im, (res, res), (0, 1),
            scale * jnp.ones(2), to, method='cubic')
    
    return fit_and_draw_spline, fit_spline, get_uniform_points, draw_point, logistic_background, spline_curvature, blob

def model(params, config, debug_mode=False):

    fit_and_draw_spline,_,_,draw_point, logistic_background, _,_ = config_closure(config)
    
    knot_params = params.locations + params.knots
    
    def make_single_spline_image(knot_params, global_scale):
        def true_fun():
            return jnp.zeros((config.res, config.res)), 0., 0.
        
        def false_fun():
            x_uniform, y_uniform, scales_uniform, lengths, curvatures = fit_and_draw_spline(knot_params)
            scales_uniform = global_scale * scales_uniform
            image = jnp.sum(
                            vmap(draw_point, in_axes=(0, 0, None, None, None, None))
                                    (
                                        jnp.stack([y_uniform, x_uniform], axis=-1), 
                                        scales_uniform, 
                                        params.blob, 
                                        params.spline_contrast, 
                                        params.spline_brightness, 
                                        debug_mode
                                    ), 
                                axis=0)
            return image, lengths, curvatures
        
        return jax.lax.cond(jnp.all(knot_params[:2] == 0), true_fun, false_fun)

    spline_images, lengths, curvatures = vmap(make_single_spline_image, in_axes = (0, None))(knot_params, params.scale)
    image = jnp.max(spline_images, axis=0) * params.opacity + jnp.sum(spline_images, axis=0) * (1 - params.opacity)

    image = image + logistic_background(params.background_slope, params.background_angle, params.background_translation)

    image = jax.scipy.signal.convolve2d(image, params.kernel, mode='same', boundary='fill')

    image = sigmoid(params.background_contrast * image + params.background_brightness)
    return image, lengths, curvatures

def batch_model(params, config, debug_mode=False):
    return vmap(model, in_axes=(0, None, None))(params, config, debug_mode)

def loss(params, images, length_guess, regularization, config):
    recon, lengths, curvatures = batch_model(params, config)
    diff = recon - images
    recon_loss = jnp.mean(diff**2, axis=(1, 2))
    # scale_reg = 1e-5 * ((params[2] - 1.0)**2).mean()
    # scale = params[2]
    scale_reg = regularization[0] * ((params.scale - config.expected_width)**2).mean()
    # length_reg = 1e-4 * (lengths**2).mean()
    # curvature_reg = 2e-4 * curvatures.mean()
    curvature_reg = regularization[1] * curvatures.mean()
    # min_scale_knots = sigmoid((params[0] + params[1])[..., 2]).min(axis = -1)
    min_scale_knots = sigmoid((params.locations + params.knots)[..., 2]).min(axis = -1)
    scale_multiplier_reg = regularization[2] * ((min_scale_knots - config.expected_min_width)**2).mean()
    length_prior_reg = regularization[3] * ((lengths - length_guess)**2).mean()
    loss_terms = (recon_loss, params.scale, lengths, curvatures, min_scale_knots)
    return recon_loss.mean() + scale_reg + curvature_reg + scale_multiplier_reg + length_prior_reg, loss_terms

# @partial(jit, static_argnums=(4, 5, 6))
@partial(jit, static_argnums=(5, 6))
def update(params, opt_state, video, length_guess, regularization, opt, config):
    value, grads = value_and_grad(loss, has_aux=True)(params, video, length_guess, regularization, config)
    value, loss_terms = value
    updates, opt_state = opt.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, value, loss_terms

def masked_optimize_scan(params, opt_state, video, length_guess, regularization, opt, config):
    """Scanned step for optimization"""
    def step(carry, _):
        params, opt_state, prev_loss = carry
        params, opt_state, loss_value, loss_terms = update(params, opt_state, video, length_guess, regularization, opt, config)
        
        # return (params, opt_state, loss_value), loss_terms
        return (params, opt_state, loss_value), (loss_terms, params)

    # Run the scan
    # (final_params, final_opt_state, final_loss), loss_terms = jax.lax.scan(
    (final_params, final_opt_state, final_loss), (loss_terms, params) = jax.lax.scan(
        step, (params, opt_state, jnp.inf), xs=jnp.arange(config.max_iter // 2)  # Iterations
    )

    # return final_params, loss_terms
    return final_params, (loss_terms, params)

def masked_optimize_parallel(video, params, length_guess, config, mask, losses, lr, regularization, seeds):
    """Runs multiple optimizations in parallel with different random seeds."""
    
    # def single_optimization(seed):
    def single_optimization(seed, params, regularization):
        # Create optimizer
        opt = optax.adam(learning_rate=lr)
        opt = optax.chain(
            opt,
            optax.add_noise(eta=config.simulated_annealing_base_noise_variance, gamma=config.simulated_annealing_decay, seed=seed),  # Different seed per run
            optax.masked(optax.set_to_zero(), mask)
        )

        opt_state = opt.init(params)

        # Run optimization
        # final_params, loss_terms = masked_optimize_scan(params, opt_state, video, length_guess, regularization, opt, config)
        final_params, (loss_terms, params) = masked_optimize_scan(params, opt_state, video, length_guess, regularization, opt, config)
        # return final_params, loss_terms
        return final_params, (loss_terms, params)

    # Vectorize over different seeds
    # final_params, loss_terms = jax.vmap(single_optimization, in_axes = (0, None, None))(seeds, params, regularization)
    final_params, (loss_terms, params_history) = jax.vmap(single_optimization, in_axes = (0, None, None))(seeds, params, regularization)

    # from IPython import embed
    # embed()
    # print("final_params.shape", type(params))
    # print("final_params[0].shape", params.scale.shape)

    # Select best result
    # print all final losses
    recon_loss = loss_terms[0].mean(axis = -1)
    if config.verbose:
        print("Final losses:", recon_loss[:, -1][0])

    # plot all losses
    # plt.figure()
    # for loss in parallel_optimizations[1]:
    #     plt.plot(loss)
    # plt.show()

    best_idx = jnp.argmin(recon_loss[:, -1])  # Select lowest loss
    if config.verbose:
        print("Best loss:", recon_loss[best_idx, -1])
        print("Final loss mean and variance:", recon_loss[:, -1].mean(), recon_loss[:, -1].var())
    best_params = jax.tree_map(lambda x: x[best_idx], final_params)

    return best_params, loss_terms, params_history

def masked_optimize(video, params, length_guess, config, mask, losses, lr, regularization):
    opt = optax.adam(learning_rate=lr)
    opt = optax.chain(
        opt,
        optax.masked(optax.set_to_zero(), mask)
    )
    opt_state = opt.init(params)
    prev_loss = jnp.inf
    for _ in (bar:= tqdm(range(config.max_iter), ncols=100)):
        params, opt_state, loss_value, loss_terms = update(params, opt_state, video, length_guess, regularization, opt, config)
        worst_image_recon_loss = loss_terms[0].max()
        bar.set_description(f"Loss: {loss_value:.2g}")
        if abs(prev_loss - worst_image_recon_loss) < (config.atol + config.rtol * abs(worst_image_recon_loss)):
            break
        prev_loss = worst_image_recon_loss
        losses.append(loss_value)
    return params, losses

def optimize(video, params, length_guess, config):
    losses = []
    
    # first optimize everything other than knots and scale
    # mask = (True, True, True) + (False, ) * (len(params) - 4) + (True,)
    
    mask = (True, True, False) + (False, ) * (len(params) - 4) + (True,)
    # mask = (False, ) * (len(params) - 2) + (True, True,)
    lr = optax.cosine_decay_schedule(init_value=config.learning_rate, decay_steps=config.max_iter // 3)
    
    # synthetic images
    # regularization = jnp.array([
    #                             1e-5, 
    #                             2e-4, 
    #                             1e-7, 
    #                             1e-5
    #                             ])

    # real images
    # regularization = jnp.array([
    #                             1e-3, # base scale
    #                             2e-3, # curvature
    #                             1e-3, # scale multiplier
    #                             1e-4, # length prior
    #                             ])

    regularization = jnp.array([
                                1e-4, # base scale
                                2e-3, # curvature
                                1e-2, # scale multiplier
                                # 1e-3, # scale multiplier
                                # 2e-3, # scale multiplier
                                2e-4, # length prior
                                ])

    # sim2real
    # regularization = jnp.array([
    #                             1e-5, # base scale
    #                             1e-2, # curvature
    #                             4e-4, # scale multiplier
    #                             # 2e-3, # scale multiplier
    #                             # 2e-5, # length prior
    #                             0.0
    #                             ])

    # set all knots to zero, so that the model doesn't plot any splines
    params_sans_knots = (jnp.zeros_like(params[0]), jnp.zeros_like(params[1]),) + params[2:]
    params_sans_knots = param_tuple(*params_sans_knots)
    mask = param_tuple(*mask)
    # mask = param_tuple(*((jnp.ones_like(param) * mask).astype(jnp.bool) for (param, mask) in zip(params, mask)))
    # print(mask)

    bg_params, losses = masked_optimize(video, params_sans_knots, length_guess, config, mask, losses, lr, regularization)

    params = (params[0], params[1],) + bg_params[2:]
    params = param_tuple(*params)
    
    if config.plot:
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(video[0], cmap='gray', origin='lower', vmin=0, vmax=1)
        plt.subplot(1, 2, 2)
        plt.imshow(batch_model(params, config)[0][0], cmap='gray', origin='lower', vmin=0, vmax=1)
        plt.show()

    # next optimize everything other than blob
    mask = (False, ) * (len(params) - 5) + (True, True, False, True, True,)
    # mask = (False, False, False, True, True, True, True, True, False, False, False, True, True,)
    mask = param_tuple(*mask)
    lr = optax.cosine_decay_schedule(init_value= config.learning_rate, decay_steps=config.max_iter, exponent=0.6)

    # regularization = jnp.array([
    #                   1e-5, # base scale
    #                   7e-2, # curvature
    #                   1e-7, # scale multiplier
    #                   1e-4, # length prior
    #                   ])

    
    num_parallel_runs = 1 if config.simulated_annealing_base_noise_variance == 0 else 64
    rng = jax.random.PRNGKey(0)
    seeds = jax.random.randint(rng, (num_parallel_runs,), 0, 10000)  # Generate random seeds

    # video_thresholded = jnp.array([image > threshold_minimum(image) for image in video])
    # params, loss_terms, params_history = masked_optimize_parallel(video_thresholded, params, length_guess, config, mask, losses, lr, regularization, seeds)
    params, loss_terms, params_history = masked_optimize_parallel(video, params, length_guess, config, mask, losses, lr, regularization, seeds)

    if config.plot:
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(video[0], cmap='gray', origin='lower', vmin=0, vmax=1)
        plt.subplot(1, 2, 2)
        plt.imshow(batch_model(params, config)[0][0], cmap='gray', origin='lower', vmin=0, vmax=1)
        plt.show()

    # lastly optimize everything
    lr = optax.cosine_decay_schedule(init_value=config.learning_rate / 100, decay_steps=config.max_iter)
    # lr = optax.cosine_decay_schedule(init_value=1e-4, decay_steps=config.max_iter)
    mask = (False, ) * (len(params) - 2) + (False, False,)
    mask = param_tuple(*mask)

    config = config._replace(atol=0, rtol=0)
    params, losses = masked_optimize(video, params, length_guess, config, mask, losses, lr, regularization)

    if config.plot:
        plt.figure()
        plt.subplot(1, 2, 1)
        plt.imshow(video[0], cmap='gray', origin='lower', vmin=0, vmax=1)
        plt.subplot(1, 2, 2)
        plt.imshow(batch_model(params, config)[0][0], cmap='gray', origin='lower', vmin=0, vmax=1)
        plt.show()

    return params, losses, loss_terms, params_history


def get_init_knots(init_scale, config, prng_key = None):
    if prng_key is None:
        prng_key = random.PRNGKey(0)
    init_knots = random.uniform(prng_key, (config.n_splines, config.s_knots, 2)) * config.res * 0.8
    # sort the knots by the first coordinate to simplify the shape
    init_knots = jnp.sort(init_knots, axis=-2)
    init_knots = jnp.concatenate([init_knots, init_scale * jnp.ones((config.n_splines, config.s_knots, 1))], axis=-1)
    return init_knots

def get_splines(params, config):
    _, fit_spline, get_uniform_spatial_grid, _, _, _, blob  = config_closure(config)
    
    final_knot_params = params[0] + params[1]
    splines = []
    for i in range(config.n_splines):
        
        xspline, yspline, _ = fit_spline(final_knot_params[i])
        s_uniform,_ = get_uniform_spatial_grid(xspline, yspline)
        x = xspline(s_uniform)
        y = yspline(s_uniform)
        splines.append((x, y))
    splines = jnp.array(splines)
    splines = splines.transpose(0, 2, 1) * config.res
    return splines, blob(params[11])

def plot_splines(image, recon, splines, config):
    plt.figure()
    plt.imshow(image, cmap='gray', origin='lower', vmin=0, vmax=1, alpha=0.5)
    plt.imshow(recon, cmap='Reds', origin='lower', vmin=0, vmax=1, alpha=0.5)
    for spline in splines:
        plt.plot(spline[:, 0], spline[:, 1], '.-', color='orange')
    plt.show()

def get_init_params(image, 
                    init_knots, 
                    config,
                    s = 7, 
                    init_scale = None,
                    init_spline_contrast = 1.0,
                    init_spline_brightness = 0.0,
                    init_background_brightness = None,
                    init_background_contrast = None,
                    init_background_slope = -5.,
                    init_background_angle = 0.,
                    init_background_translation = 0.,
                    init_kernel = None,
                    init_blob = None,
                    prng_key = None,
                    ):

    
    assert image.ndim == 2 and image.shape[0] == image.shape[1], "image must have shape (res, res), but got shape {}".format(image.shape)


    n, s = init_knots.shape[:2] if init_knots is not None else (config.n_splines, s)

    # get average distance between knots
    # mean_distance = jnp.sqrt(jnp.sum((init_knots[:, 1:] - init_knots[:, :-1])**2, axis=-1)).mean()
    init_scale = image.shape[-1] / 100 if init_scale is None else init_scale


    if init_knots is None:
        init_knots = get_init_knots(init_scale, config, prng_key=prng_key)
    else: 
        assert init_knots.ndim == 3, "init_knots must have shape (n_splines, s_knots, 2), but got shape {}".format(init_knots.shape)

    if init_background_brightness is None:
        init_background_brightness = logit(max(image.min(), 1e-3))

    if init_background_contrast is None:
        init_background_contrast = logit(0.9)

    if init_kernel is None:
        init_kernel = jnp.zeros((3, 3))
        init_kernel = init_kernel.at[1, 1].set(1.0)

    if init_blob is None:
        init_blob = jnp.linspace(-1, 1, 13)**2
    
    # if there are more than n_points_per_spline knots, we need to subsample
    if s > config.n_points_per_spline:
        init_knots = init_knots[:, jnp.linspace(0, s-1, config.n_points_per_spline // 2).astype(int), :]

    # check that all knots are between 0 and 1
    if not (jnp.all(init_knots >= 0) and jnp.all(init_knots <= 1)):
        init_knots = init_knots / config.res

    # set all knots below 0 to 0 and all knots above 1 to 1
    eps = 1e-3
    init_knots = jnp.minimum(jnp.maximum(init_knots, eps), 1 - eps)


    def knots2params_or_zeros(knots):
        return jax.lax.cond(
            jnp.all(knots == 0),  # This is now a boolean scalar, not a function
            lambda: knots,        # If all zeros, return knots as is
            lambda: knots2params(knots)  # Otherwise, transform with knots2params
        )

    init_params = vmap(knots2params_or_zeros)(init_knots)[..., :2]

    # subtract mean
    init_param_mean = init_params.mean(1, keepdims=True)
    init_params = init_params - init_param_mean
    init_param_mean = jnp.concatenate([init_param_mean, 5 * jnp.ones((n, 1, 1))], axis=2)
    init_params = jnp.concatenate([init_params, jnp.zeros((n, s, 1))], axis=2)
    # init_opacity = jnp.ones(1)
    init_opacity = jnp.ones(1) * 0.0
    params = (init_param_mean, 
              init_params, 
              init_scale, 
              init_background_contrast, 
              init_background_brightness, 
              init_background_slope, 
              init_background_angle, 
              init_background_translation, 
              init_spline_contrast,
              init_spline_brightness,
              init_kernel,
              init_blob,
              init_opacity,
              )
    return params


def fit(image, 
        init_knots = None, 
        s = 7, 
        n_splines = None,
        init_scale = None,
        init_spline_contrast = 1.0,
        init_spline_brightness = 0.0,
        init_background_brightness = None,
        init_background_contrast = None,
        init_background_slope = -5.,
        init_background_angle = 0.,
        init_background_translation = 0.,
        init_kernel = None,
        n_points_per_spline = 100,
        length_guess = None,
        plot = False,
        return_uniform_spatial_grid = False,
        verbose = False,
        ):

    config = {
        "res": image.shape[-1], # the resolution of the image
        "n_splines": init_knots.shape[1] if init_knots is not None else n_splines if n_splines is not None else 1, # the number of splines
        "s_knots": s, # the number of spatial knots in each frame
        "n_points_per_spline": n_points_per_spline,
        # "max_iter": 1500,
        "max_iter": 2500,
        "rtol": 1e-15,
        "atol": 1e-15,
        "learning_rate": 1e-2,
        "expected_width": init_scale if init_scale is not None else 1.0,
        # "expected_min_width": 0.7,
        "expected_min_width": 1.0,
        # "expected_min_width": 0.5,
        # "learning_rate": 3e-4,
        # "learning_rate": 3e-5,
        # for synthetic images
        # "simulated_annealing_base_noise_variance": 1e-4,
        
        # for real images, turn off simulated annealing
        "simulated_annealing_base_noise_variance": 0.0 if init_knots is not None else 3e-5,
        "simulated_annealing_decay": 0.7,
        "plot": plot,
        "return_uniform_spatial_grid": return_uniform_spatial_grid,
        "verbose": verbose,
    }
    config = namedtuple("Config", config.keys())(*config.values())
    # check range of the image is between 0 and 1
    # if not (jnp.all(image >= 0) and jnp.all(image <= 1)):
    image_max = image.max()
    image_min = image.min()
    image = (image - image_min) / (image_max - image_min)

    if image.ndim == 2:
        image = image[None, ...]

    if init_knots is None:
        init_knots = [None] * image.shape[0]

    prng_keys = random.split(random.PRNGKey(0), image.shape[0])

    params = [partial(get_init_params, 
                            config = config,
                            s = s, 
                            init_scale = init_scale,
                            init_spline_contrast = init_spline_contrast,
                            init_spline_brightness = init_spline_brightness,
                            init_background_brightness = init_background_brightness,
                            init_background_contrast = init_background_contrast,
                            init_background_slope = init_background_slope,
                            init_background_angle = init_background_angle,
                            init_background_translation = init_background_translation,
                            init_kernel = init_kernel,
                            prng_key = prng_key,
                            )(image, init_knots) for (image, init_knots, prng_key) in zip(image, init_knots, prng_keys)]

    
    # stack params for batch processing

    params = param_tuple(*[jnp.stack([param[i] for param in params], axis=0) for i in range(len(params[0]))])


    # get length of init knots

    def length(init):
        init = init / image.shape[-1]
        init = init - init.mean(0)
        return jnp.sum(jnp.linalg.norm(init[1:] - init[:-1], axis=-1))
    
    if init_knots[0] is not None:
        length_guess = vmap(vmap(length))(init_knots) if length_guess is None else length_guess
        length_guess = length_guess / image.shape[-1]
    else:
        length_guess = 0.5 # 5 % of the image size
    
    if verbose:
        print("length_guess", length_guess)


    if config.plot:
        pass
        # # plot reconstruction
        # recon, lengths, curvatures = batch_model(params, config)
        # print("lengths", lengths)
        # splines = get_splines(params, config)
        # plot_splines(image, recon, splines, config)
    # print(params[0])
    # print(params[1])
    recon, lengths, curvatures = batch_model(params, config)
    for i in range(image.shape[0]):
        plt.imshow(recon[i], cmap='gray', vmin=0, vmax=1)
        plt.show()

    params, losses, loss_terms, params_history = optimize(image, params, length_guess, config)

    if verbose:
        print("Final knots and params:")
        print(params2knots(params[0] + params[1]), params[2], params[3], params[4])

    if config.plot:
        plt.figure()
        plt.plot(losses)
        plt.show()

    # if config.plot:
    #     pass
    #     # plot reconstruction
    #     # recon, lengths, curvatures  = model(params, config)
    #     # print("curvature", curvatures)
    #     # splines = get_splines(params, config)
    #     # if config.plot:
    #     #     plot_splines(image, recon, splines, config)
    #     # return splines, recon

    # else:
    recon, lengths, curvatures = batch_model(params, config, debug_mode=False)
    # splines = [get_splines([param[i] for param in params], config) for i in range(image.shape[0])]
    splines, blobs = zip(*[get_splines([param[i] for param in params], config) for i in range(image.shape[0])])
    return splines, recon, blobs, loss_terms, params_history, config