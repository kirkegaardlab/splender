from utils import circle_image, knots2params, params2knots
import jax
from jax import grad, jit, vmap
import jax.numpy as jnp
from jax.nn import sigmoid, softplus
import jax.random as random
from functools import partial
from interpax import Interpolator2D
import optax
from jax.image import scale_and_translate
from collections import namedtuple
from tqdm import tqdm
import optax
import matplotlib.pyplot as plt


def make_process_spline(config):
    res = config.res
    n_points_to_plot = config.n_points_to_plot
    n_frames = config.n_frames
    eps = 1e-6

    def fit_spline(params):
        knots = params2knots(params)

        s_sparse = jnp.linspace(0, 1, knots.shape[0])
        t_sparse = jnp.linspace(0, 1, knots.shape[1])
        
        x_spline = Interpolator2D(s_sparse, t_sparse, knots[..., 0], method="cubic")
        y_spline = Interpolator2D(s_sparse, t_sparse, knots[..., 1], method="cubic")
        return x_spline, y_spline
    
    def spline_curvature_t(x_spline, y_spline, t):
        d2x_ds2 = vmap(partial(x_spline, yq = t, dx=2))
        d2y_ds2 = vmap(partial(y_spline, yq = t, dx=2))
        s_fine = jnp.linspace(0, 1, n_points_to_plot)
        delta_s = s_fine[1] - s_fine[0]
        curvature = jnp.sqrt(d2x_ds2(s_fine)**2 + d2y_ds2(s_fine)**2 + eps) * delta_s
        return curvature.mean(0)

    def get_uniform_points(x_spline, y_spline):
        dx_ds = vmap(grad(x_spline))
        dy_ds = vmap(grad(y_spline))
        s_fine = jnp.linspace(0, 1, n_points_to_plot)
        ds_vals = jnp.sqrt(dx_ds(s_fine)**2 + dy_ds(s_fine)**2)

        delta_s = s_fine[1] - s_fine[0]

        cumulative_length = jnp.concatenate([
            jnp.array([0.0]),
            jnp.cumsum(0.5 * (ds_vals[1:] + ds_vals[:-1]) * delta_s)
        ])

        s_uniform = jnp.linspace(0, cumulative_length[-1], n_points_to_plot)
        s_uniform = jnp.interp(s_uniform, cumulative_length, s_fine)

        return s_uniform

    def get_uniform_spatial_grid(x_spline, y_spline):
        # Create a dense grid for evaluation.
        s_fine = jnp.linspace(0, 1, n_points_to_plot)
        t_fine = jnp.linspace(0, 1, n_frames)
        _, T = jnp.meshgrid(s_fine, t_fine, indexing='ij')

        # Make spatial coordinates uniform
        get_uniform_points_t = lambda t: get_uniform_points(partial(x_spline, yq = t), partial(y_spline, yq = t))
        S = vmap(get_uniform_points_t, out_axes = 1)(t_fine)

        return S, T

    def draw_spline(x_spline, y_spline, S, T):
        x_dense = vmap(x_spline)(S, T)
        y_dense = vmap(y_spline)(S, T)

        points = jnp.stack([y_dense.ravel(), x_dense.ravel()], axis=-1)
        return points

    def draw_point(point, scale): 
        im = circle_image(3)
        to = res * point
        to = to.at[0].add(-im.shape[0] / 2 * scale)
        to = to.at[1].add(-im.shape[1] / 2 * scale)
        return scale_and_translate(
            im, (res, res), (0, 1),
            scale * jnp.ones(2), to, method='cubic')

    def process_spline(knot_params, scale):
        x_spline, y_spline = fit_spline(knot_params)
        t_fine = jnp.linspace(0, 1, n_frames)
        get_curvature_t = lambda t: spline_curvature_t(x_spline, y_spline, t)
        curvature = vmap(get_curvature_t)(t_fine)
        S, T = get_uniform_spatial_grid(x_spline, y_spline)
        points = draw_spline(x_spline, y_spline, S, T)
        drawn = vmap(lambda pt: draw_point(pt, scale))(points)
        return drawn, curvature

    return process_spline, fit_spline, get_uniform_spatial_grid


@partial(jit, static_argnums=(1,))
def model(params, config, median_frame):
    # settings
    n_frames = config.n_frames
    n_points_to_plot = config.n_points_to_plot
    
    scale = params[2]
    knot_params = params[0] + params[1]

    process_spline,_,_ = make_process_spline(config)

    all_drawn_points, curvatures = vmap(process_spline, in_axes = (0, None))(knot_params, scale)
    
    drawn = jnp.max(all_drawn_points, axis=0)

    frame_idx = jnp.tile(jnp.arange(n_frames), (n_points_to_plot, 1))

    frames = frame_idx.ravel()
    
    video = jax.ops.segment_sum(drawn, frames, num_segments=n_frames)    
    
    
    video = sigmoid(params[3] * video + params[4])


    frame_conv = lambda frame: jax.scipy.signal.convolve2d(frame, params[6], mode='same', boundary='fill')
    video = vmap(frame_conv)(video)

    bg = params[8] * median_frame + params[9]

    video = params[7] * video + (1 - params[7]) * bg.squeeze(-1)

    return video, curvatures

@partial(jit, static_argnums=(3,))
def loss(params, video, median_frame, config):
    recon, curvatures = model(params, config, median_frame)
    diff = recon - video
    recon_loss = jnp.mean(diff**2)

    scale_reg = 1e-4 * (params[2] - 2.0)**2
    curvature_reg = 1e-5 * jnp.mean(curvatures)
    knot_params = params[0] + params[1]

    # regularize the first knot in each spline to be the same for all time points
    knot_reg = 1e-3 * jnp.mean((knot_params[:, :1, :1, :] - knot_params[:, :1, :, :])**2)
    return recon_loss + scale_reg + curvature_reg + knot_reg

@partial(jit, static_argnums=(4, 5))
def update(params, opt_state, video, median_frame, opt, config):
    # value, grads = jax.value_and_grad(loss, has_aux=True)(params, video, config)
    value, grads = jax.value_and_grad(loss)(params, video, median_frame, config)

    updates, opt_state = opt.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)

    return new_params, opt_state, value

def masked_optimize(video, params, config, mask, losses, max_iter):
    lr = optax.cosine_decay_schedule(init_value=config.learning_rate, decay_steps=config.max_iter)
    opt = optax.adam(learning_rate=lr)
    opt = optax.chain(
        opt,
        optax.masked(optax.set_to_zero(), mask)
    )
    opt_state = opt.init(params)
    median_frame = jnp.median(video, axis=0).reshape(1, config.res, config.res, 1)
    prev_loss = jnp.inf
    for _ in (bar:= tqdm(range(max_iter), ncols=100)):
        params, opt_state, loss_value = update(params, opt_state, video, median_frame, opt, config)
        bar.set_description(f"Loss: {loss_value:.2g}")
        losses.append(loss_value)
    return params, losses


def optimize(video, params, config):
    mask = (True, True, False, False, False, True, True, False, False, False)
    params, losses = masked_optimize(video, params, config, mask, losses = [], max_iter = 1000)
    
    # mask = (True, True, False, False, False, True, False, False, False, False)
    # params, losses = masked_optimize(video, params, config, mask, losses = losses, max_iter = config.max_iter)
    # # params, losses = masked_optimize(video, params, config, mask, losses = [], max_iter = config.max_iter)

    mask = (False, False, False, False, False, False, False, False, False, False)
    params, losses = masked_optimize(video, params, config, mask, losses = losses, max_iter = config.max_iter)
    return params, losses

def get_init_knots(config, prng_key = None):
    if prng_key is None:
        prng_key = random.PRNGKey(0)
    return random.uniform(prng_key, (config.n_splines, config.s_knots, config.t_knots, 2))

def get_init(video, 
        init_knots = None, 
        n = 1,
        s = 7, 
        t = 3,
        init_scale = None,
        init_background_brightness = None,
        init_background_contrast = None,
        plot = False,
        return_uniform_spatial_grid = False,
        verbose = False,
        ):

    
    n, s, t = init_knots.shape[:3] if init_knots is not None else (n, s, t)
    
    config = {
        "n_frames": video.shape[0], # the number of frames that we interpolate the t_knots over
        "res": video.shape[1], # the resolution of the video
        "n_splines": n, # the number of splines
        "s_knots": s, # the number of spatial knots in each frame
        "t_knots": t, # the number of knots over time
        "n_points_to_plot": 100,
        "max_iter": 2500,
        "rtol": 1e-10,
        "atol": 1e-10,
        "learning_rate": 1e-2,
        "filter_size": 11,
        "plot": plot,
        "return_uniform_spatial_grid": return_uniform_spatial_grid,
        "verbose": verbose,
    }
    config = namedtuple("Config", config.keys())(*config.values())

    init_scale = config.res / 100 if init_scale is None else init_scale
    
    if init_knots is None:
        init_knots = get_init_knots(config)
    elif init_knots.ndim == 3:
        init_knots = jnp.stack([init_knots,] * t, axis=2)
    else: 
        assert init_knots.ndim == 4, "init_knots must have shape (n_splines, s_knots, t_knots, 2)"

    if init_background_brightness is None:
        init_background_brightness = jax.scipy.special.logit(max(video.min(), 1e-3))

    if init_background_contrast is None:
        init_background_contrast = jax.scipy.special.logit(0.9)

    # if there are more than n_points_to_plot knots, we need to subsample
    if s > config.n_points_to_plot:
        init_knots = init_knots[:, jnp.linspace(0, s-1, config.n_points_to_plot // 2).astype(int), :]
        config.s = config.n_points_to_plot // 2
    # check that all knots are between 0 and 1
    if not (jnp.all(init_knots >= 0) and jnp.all(init_knots <= 1)):
        init_knots = init_knots / config.res
    init_params = vmap(knots2params)(init_knots)
    
    # subtract mean
    init_param_mean = init_params.mean((1, 2), keepdims=True)
    init_params = init_params - init_param_mean

    rng = jax.random.PRNGKey(0)

    # init to identity kernel (not identity matrix)
    init_kernel = jnp.zeros((config.filter_size, config.filter_size))
    init_kernel = init_kernel.at[config.filter_size // 2, config.filter_size // 2].set(1.0)

    init_opacity = 0.5 * jnp.ones((1,))

    init_contrast = 1.0 * jnp.ones((1,))
    init_brightness = 0.0 * jnp.ones((1,))

    params = (init_param_mean, init_params, init_scale, init_background_contrast, init_background_brightness, cnn_params, init_kernel, init_opacity, init_contrast, init_brightness)

    if verbose:
        print("Initial parameters:")
        print(params)
    return params, config
    

def fit(video, 
        init_knots = None, 
        n = 1,
        s = 7, 
        t = 3,
        init_scale = None,
        init_background_brightness = None,
        init_background_contrast = None,
        plot = False,
        return_uniform_spatial_grid = False,
        verbose = False,
        ):
    
    assert video.ndim == 3, "Video must have shape (n_frames, res, res), but got shape {}".format(video.shape)

    # check range of the video is between 0 and 1
    if not (jnp.all(video >= 0) and jnp.all(video <= 1)):
        video_max = video.max()
        video_min = video.min()
        video = (video - video_min) / (video_max - video_min)

    params, config = get_init(video,
        init_knots = init_knots, 
        n = n,
        s = s, 
        t = t,
        init_scale = init_scale,
        init_background_brightness = init_background_brightness,
        init_background_contrast = init_background_contrast,
        plot = plot,
        return_uniform_spatial_grid = return_uniform_spatial_grid,
        verbose = verbose,
        )

    params, losses = optimize(video, params, config)

    if verbose:
        print("Final parameters:")
        print(params)

    final_knot_params = params[0] + params[1]

    if config.plot:
        plt.figure()
        plt.plot(losses)
        plt.show()

    _, fit_spline, get_uniform_spatial_grid  = make_process_spline(config)
    
    splines = []
    for i in range(config.n_splines):
        
        xspline, yspline = fit_spline(final_knot_params[i])
        S, T = get_uniform_spatial_grid(xspline, yspline)
        x = vmap(xspline)(S, T)
        y = vmap(yspline)(S, T)

        splines.append((x, y))

    splines = jnp.array(splines).transpose(0, 3, 1, 2) * config.res

    if config.plot:
        # plot reconstruction
        recon, curvature = model(params, config)
        plt.figure()
        plt.imshow(video[0], cmap='gray', origin='lower', vmin=0, vmax=1, alpha=0.5)
        plt.imshow(recon[0], cmap='Reds', origin='lower', vmin=0, vmax=1, alpha=0.5)
        for spline in splines:
            plt.plot(spline[:, 0], spline[:, 1], '.-', color='orange')
        plt.show()
    return splines, params, jnp.median(video, axis=0).reshape(1, config.res, config.res, 1)