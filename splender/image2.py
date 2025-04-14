from flax import linen as nn
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
import equinox as eqx

class Splender(eqx.Module):
    """
    Parent class for SplenderImage and SplenderVideo.
    This class is not meant to be used directly.
    """
    res: int = 128
    n_splines: int = 1
    s_knots: int = 7
    n_points_per_spline_per_frame: int = 100
    brush_profile: jax.Array
    spline_contrast: jax.Array
    spline_brightness: jax.Array
    loc_params: jax.Array
    knot_params: jax.Array
    kernel: jax.Array
    contrast: jax.Array
    brightness: jax.Array
    opacity: jax.Array
    global_scale: jax.Array
    eps: float = 1e-6


    def get_init_kernel(self, rng):
        init_kernel = jnp.zeros((3, 3))
        init_kernel = init_kernel.at[1, 1].set(1.0)
        return init_kernel

    def __init__(self, key, init_knots = None, res = None):
        if res is not None:
            self.res = res
        self.brush_profile = jnp.linspace(-1, 1, 13)**2
        self.spline_contrast = jnp.ones((1,))
        self.spline_brightness = jnp.zeros((1,))
        if init_knots is not None:
            init_knot_params = logit(init_knots)
            init_param_mean = init_knot_params.mean(1, keepdims=True)
            init_params = init_knot_params - init_param_mean
            self.n_splines = init_knot_params.shape[0]
            self.s_knots = init_knot_params.shape[1]
            self.loc_params = jnp.concatenate([init_param_mean, 5 * jnp.ones((self.n_splines, 1, 1))], axis=2)
            self.knot_params = jnp.concatenate([init_params, jnp.zeros((self.n_splines, self.s_knots, 1))], axis=2)
        else:            
            self.loc_params = jnp.zeros((self.n_splines, 1, 3))
            self.knot_params = jnp.zeros((self.n_splines, self.s_knots, 3))
        self.kernel = self.get_init_kernel(key)
        self.contrast = jnp.ones((1,))
        self.brightness = jnp.zeros((1,))
        self.opacity = jnp.ones((1,))
        self.global_scale = jnp.ones((1,)) * self.res / 100


    def brush_model(self):
        """
        Each point is rendered as the brush image centered at that point.
        """
        x = jnp.linspace(-1, 1, self.brush_profile.shape[0])
        spline = Interpolator1D(x, self.brush_profile, method='cubic')
        X, Y = jnp.meshgrid(x, x)
        D = jnp.sqrt(X**2 + Y**2)
        brush_image = vmap(spline)(jnp.exp(-D))
        return brush_image
    
    def render_point(self, point, scale):
        """
        Render a point using the brush model.
        """
        brush_image = self.brush_model()
        point_mapped_to_image = self.res * point - brush_image.shape[0] / 2 * scale + 0.5

        return scale_and_translate(
            brush_image, (self.res, self.res), (0, 1),
            scale * jnp.ones(2), point_mapped_to_image, method='cubic')
    
    def cumulative_spline_length(self, x_spline, y_spline):
        # Compute arc length
        dx_ds = vmap(partial(x_spline, dx=1))
        dy_ds = vmap(partial(y_spline, dx=1))
        s_fine = jnp.linspace(0, 1, self.n_points_per_spline_per_frame)
        ds_vals = jnp.sqrt(dx_ds(s_fine)**2 + dy_ds(s_fine)**2 + self.eps)
        delta_s = s_fine[1] - s_fine[0]
        cumulative_length = jnp.concatenate([
            jnp.array([0.0]),
            jnp.cumsum(0.5 * (ds_vals[1:] + ds_vals[:-1]) * delta_s)
        ])
        return cumulative_length

    def get_uniform_points(self, x_spline, y_spline):
        # Compute cumulative spline length
        cumulative_length = self.cumulative_spline_length(x_spline, y_spline)
        # Uniformly sample s
        s_uniform = jnp.interp(
            jnp.linspace(0, cumulative_length[-1], self.n_points_per_spline_per_frame), 
            cumulative_length, 
            jnp.linspace(0, 1, self.n_points_per_spline_per_frame))
        return s_uniform

    def fit_spline(self):
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    def __call__(self):
        NotImplementedError("This method should be implemented in subclasses.")


class SplenderImage(Splender):

    def fit_spline(self, knot_params):

        knots = sigmoid(knot_params)

        x, y, scale_knots = knots[..., 0], knots[..., 1], knots[..., 2]
        
        s = jnp.linspace(0, 1, len(x))

        x_spline = Interpolator1D(s, x, method="cubic2")
        y_spline = Interpolator1D(s, y, method="cubic2")
        scale_spline = Interpolator1D(s, scale_knots, method="cubic")
        return x_spline, y_spline, scale_spline
    
    def render_spline(self, x_spline, y_spline, scale_spline):
        # Get uniform points
        s_uniform = self.get_uniform_points(x_spline, y_spline)
        
        # Get the points on the spline
        x_points = x_spline(s_uniform)
        y_points = y_spline(s_uniform)
        scale_points = scale_spline(s_uniform)
        
        # Render the points
        image = jnp.sum(
                        vmap(self.render_point, in_axes=(0, 0))
                            (
                                jnp.stack([y_points, x_points], axis=-1),
                                self.global_scale * scale_points,
                            ),
                        axis=0
                    )
        return image
    
    def render_splines(self):
        """
        Render all splines in a separate image.
        If knot_params are all zeros, the image is empty.
        This allows for batching with a varying number of splines across the batch.
        """
        def maybe_render_spline(knot_params):
            def render_empty():
                return jnp.zeros((self.res, self.res))
            
            def render_spline():
                x_spline, y_spline, scale_spline = self.fit_spline(knot_params)
                return self.render_spline(x_spline, y_spline, scale_spline)
            
            return jax.lax.cond(jnp.all(knot_params[:2] == 0), render_empty, render_spline)
        
        knots = self.loc_params + self.knot_params
        spline_images = vmap(maybe_render_spline)(knots)

        # aggregate spline_images
        splines_image = jnp.max(spline_images, axis=0) * self.opacity + jnp.sum(spline_images, axis=0) * (1 - self.opacity)

        return splines_image
    
    def render_image(self):
        splines_image = self.render_splines() * self.spline_contrast + self.spline_brightness

        image = splines_image #+ self.logistic_background()

        image = jax.scipy.signal.convolve2d(image, self.kernel, mode='same', boundary='fill')

        image = image * self.contrast + self.brightness
        image = sigmoid(image)
        return image
    
    def __call__(self):
        image = self.render_image()
        return image