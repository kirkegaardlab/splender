from flax import linen as nn
from .utils import circle_image, knots2params, params2knots, get_idenitity_kernel
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
import jax.tree_util as jtu
from dataclasses import dataclass, field
from abc import ABC, abstractmethod

@jax.tree_util.register_dataclass
@dataclass
class Splender(ABC):
    """
    Parent class for SplenderImage and SplenderVideo.
    This class is not meant to be used directly.
    """
    brush_profile: jax.Array = None #= field(default=None)
    spline_contrast: jax.Array = None #= field(default=None)
    spline_brightness: jax.Array = None #= field(default=None)
    loc_params: jax.Array = None #= field(default=None)
    knot_params: jax.Array = None #= field(default=None)
    kernel: jax.Array = None #= field(default=None)
    contrast: jax.Array = None #= field(default=None)
    brightness: jax.Array = None #= field(default=None)
    opacity: jax.Array = None #= field(default=None)
    global_scale: jax.Array = None #= field(default=None)
    res: int = field(metadata=dict(static=True), default=128)
    n_images: int = field(metadata=dict(static=True), default=1)
    n_splines: int = field(metadata=dict(static=True), default=1)
    s_knots: int = field(metadata=dict(static=True), default=7)
    n_points_per_spline_per_frame: int = field(metadata=dict(static=True), default=100)
    eps: float = field(metadata=dict(static=True), default=1e-6)

    @classmethod
    def init(cls, 
             key, 
             brush_profile = brush_profile,
             spline_contrast = spline_contrast,
             spline_brightness = spline_brightness,
             loc_params = loc_params,
             knot_params = knot_params,
             kernel = kernel,
             contrast = contrast,
             brightness = brightness,
             opacity = opacity,
             global_scale = global_scale,
             res = res.default,
             n_images = n_images.default,
             n_splines = n_splines.default,
             s_knots = s_knots.default,
             n_points_per_spline_per_frame = n_points_per_spline_per_frame.default,
             eps = eps.default,
             init_knots = None,
             ):
        # if res is not None:
        #     res = res
        brush_profile = jnp.linspace(-1, 1, 13)**2
        spline_contrast = jnp.ones((1,))
        spline_brightness = jnp.zeros((1,))
        if init_knots is not None:
            init_knot_params = logit(init_knots)
            init_param_mean = init_knot_params.mean(-2, keepdims=True)
            init_params = init_knot_params - init_param_mean
            n_images = init_knot_params.shape[0]
            n_splines = init_knot_params.shape[1]
            s_knots = init_knot_params.shape[2]
            loc_params = jnp.concatenate([init_param_mean, 5 * jnp.ones((n_images, n_splines, 1, 1))], axis=-1)
            knot_params = jnp.concatenate([init_params, jnp.zeros((n_images, n_splines, s_knots, 1))], axis=-1)
        else:            
            loc_params = jnp.zeros((n_splines, 1, 3))
            knot_params = jnp.zeros((n_splines, s_knots, 3))
        kernel = get_idenitity_kernel(key)
        contrast = jnp.ones((1,))
        brightness = jnp.zeros((1,))
        opacity = jnp.ones((1,))
        global_scale = jnp.ones((1,)) * res / 100
        return cls(
            brush_profile=brush_profile,
            spline_contrast=spline_contrast,
            spline_brightness=spline_brightness,
            loc_params=loc_params,
            knot_params=knot_params,
            kernel=kernel,
            contrast=contrast,
            brightness=brightness,
            opacity=opacity,
            global_scale=global_scale,
            res=res,
            n_images=n_images,
            n_splines=n_splines,
            s_knots=s_knots,
            n_points_per_spline_per_frame=n_points_per_spline_per_frame,
            eps=eps,
            )

    def brush_model(self):
        """
        Each point is rendered as the brush image centered at that point.
        """
        x = jnp.linspace(-1, 1, self.brush_profile.shape[0])
        spline = Interpolator1D(x, self.brush_profile, method='cubic')
        X, Y = jnp.meshgrid(x, x)
        D = jnp.sqrt(X**2 + Y**2 + self.eps)
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
    
    def spatial_derivative(self, spline, degree = 1):
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    def cumulative_spline_length(self, x_spline, y_spline):
        # Compute arc length
        dx_ds = vmap(self.spatial_derivative(x_spline, degree=1))
        dy_ds = vmap(self.spatial_derivative(y_spline, degree=1))
        s_fine = jnp.linspace(0, 1, self.n_points_per_spline_per_frame)
        ds_vals = jnp.sqrt(dx_ds(s_fine)**2 + dy_ds(s_fine)**2 + self.eps)
        delta_s = s_fine[1] - s_fine[0]
        cumulative_length = jnp.concatenate([
            jnp.array([0.0]),
            jnp.cumsum(0.5 * (ds_vals[1:] + ds_vals[:-1]) * delta_s)
        ])
        return cumulative_length
    
    def mean_spline_curvature(self, x_spline, y_spline):
        d2x_ds2 = vmap(self.spatial_derivative(x_spline, degree=2))
        d2y_ds2 = vmap(self.spatial_derivative(y_spline, degree=2))
        s_fine = jnp.linspace(0, 1, self.n_points_per_spline_per_frame)
        delta_s = s_fine[1] - s_fine[0]
        curvature = jnp.sqrt(d2x_ds2(s_fine)**2 + d2y_ds2(s_fine)**2 + self.eps) * delta_s
        return curvature.mean(0)

    
    def __call__(self):
        NotImplementedError("This method should be implemented in subclasses.")

@jax.tree_util.register_dataclass
@dataclass
class SplenderImage(Splender):

    def spatial_derivative(self, spline, degree = 1):
        return partial(spline, dx=degree)

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
    
    def render_splines(self, knots):
        """
        Render all splines in a separate image.
        If knot_params are all zeros, the image is empty.
        This allows for batching with a varying number of splines across the batch.
        """
        def maybe_render_spline(knot_params):
            def render_empty():
                return jnp.zeros((self.res, self.res)), 0., 0.
            
            def render_spline():
                x_spline, y_spline, scale_spline = self.fit_spline(knot_params)
                length = self.cumulative_spline_length(x_spline, y_spline)[-1]
                curvature = self.mean_spline_curvature(x_spline, y_spline)
                return self.render_spline(x_spline, y_spline, scale_spline), length, curvature
            
            return jax.lax.cond(jnp.all(knot_params[:2] == 0), render_empty, render_spline)
        
        spline_images, lengths, curvatures = vmap(maybe_render_spline)(knots)
        
        # aggregate spline_images
        splines_image = jnp.max(spline_images, axis=0) * self.opacity + jnp.sum(spline_images, axis=0) * (1 - self.opacity)

        return splines_image, lengths, curvatures
    
    def render_image(self, knots):
        splines_image, lengths, curvatures = self.render_splines(knots)
        # splines_image = splines_image * spline_contrast + spline_brightness

        image = splines_image #+ logistic_background()

        image = jax.scipy.signal.convolve2d(image, self.kernel, mode='same', boundary='fill')

        image = image * self.contrast + self.brightness
        image = sigmoid(image)
        return image, lengths, curvatures
    
    def __call__(self):
        knots = self.loc_params + self.knot_params
        images, lengths, curvatures = jax.vmap(self.render_image)(knots)
        return images, lengths, curvatures