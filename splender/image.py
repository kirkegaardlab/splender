from .splender import Splender
from interpax import Interpolator1D
import jax
import jax.numpy as jnp
import jax.random as random
from jax.nn import sigmoid
import jax.tree_util as jtu
from jax import grad, jit, vmap, value_and_grad
from functools import partial
from dataclasses import dataclass, field
from jax.scipy.special import logit

@jtu.register_dataclass
@dataclass
class SplenderImage(Splender):
    
    def __manual_post_init__(self):
        super().__post_init__()

        if self.init_knots is not None:
            init_knot_params = logit(self.init_knots)
            init_param_mean = init_knot_params.mean(-2, keepdims=True)
            init_params = init_knot_params - init_param_mean
            self.n_batch = init_knot_params.shape[0]
            self.n_splines = init_knot_params.shape[1]
            self.s_knots = init_knot_params.shape[2]
            self.loc_params = jnp.concatenate([init_param_mean, 5 * jnp.ones((self.n_batch, self.n_splines, 1, 1))], axis=-1)
            self.knot_params = jnp.concatenate([init_params, jnp.zeros((self.n_batch, self.n_splines, self.s_knots, 1))], axis=-1)
        else:
            self.loc_params = jnp.zeros((self.n_splines, 1, 3))
            self.knot_params = jnp.zeros((self.n_splines, self.s_knots, 3))

        if self.global_scale is None or hasattr(self.global_scale, 'shape') and self.global_scale.shape == (self.n_batch, self.n_splines):
            self.global_scale = jnp.ones((self.n_batch, self.n_splines)) * self.res / 100
        elif hasattr(self.global_scale, 'shape') and self.global_scale.shape == (self.n_batch,):
            self.global_scale = jnp.ones((self.n_batch, self.n_splines)) * self.global_scale[:, None] * self.res / 100
        elif isinstance(self.global_scale, float):
            self.global_scale = jnp.ones((self.n_batch, self.n_splines)) * self.global_scale * self.res / 100
        else:
            raise ValueError("global_scale must be a scalar or have shape (n_batch,) or (n_batch, n_splines).")
        
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
    
    def render_spline(self, x_spline, y_spline, scale_spline, scale):
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
                                scale * scale_points,
                            ),
                        axis=0
                    )
        return image
    
    def render_splines(self, knots, scales):
        """
        Render all splines in an image.
        For each spline, 
        If knot_params are all zeros, the image is empty.
        This allows for batching with a varying number of splines across the batch.
        """
        def maybe_render_spline(knot_params, scale):

            def render_empty():
                return jnp.zeros((self.res, self.res)), 0., 0.
            
            def render_spline():
                x_spline, y_spline, scale_spline = self.fit_spline(knot_params)
                length = self.cumulative_spline_length(x_spline, y_spline)[-1]
                curvature = self.mean_spline_curvature(x_spline, y_spline)
                return self.render_spline(x_spline, y_spline, scale_spline, scale), length, curvature
            
            return jax.lax.cond(jnp.all(knot_params[:2] == 0), render_empty, render_spline)
        
        spline_images, lengths, curvatures = vmap(maybe_render_spline)(knots, scales)
        
        # aggregate spline_images
        splines_image = jnp.max(spline_images, axis=0) * self.opacity + jnp.sum(spline_images, axis=0) * (1 - self.opacity)

        return splines_image, lengths, curvatures
    
    def render_image(self, knots, scales):
        splines_image, lengths, curvatures = self.render_splines(knots, scales)
        # splines_image = splines_image * spline_contrast + spline_brightness

        image = splines_image #+ logistic_background()

        image = jax.scipy.signal.convolve2d(image, self.kernel, mode='same', boundary='fill')

        image = image * self.contrast + self.brightness
        image = sigmoid(image)
        return image, lengths, curvatures
    
    def __call__(self):
        knots = self.loc_params + self.knot_params
        images, lengths, curvatures = jax.vmap(self.render_image)(knots, self.global_scale)
        return images, lengths, curvatures