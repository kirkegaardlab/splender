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

class Splender(nn.Module):
    """
    Parent class for SplenderImage and SplenderVideo.
    This class is not meant to be used directly.
    """
    n_splines: int
    n_knots: int
    s: int
    n_channels: int = 1

    def brush_model(self):
        """
        Each point is rendered as the brush image centered at that point.
        """
        brush_profile = self.param('brush_profile', lambda rng: jnp.linspace(-1, 1, 13)**2)
        x = jnp.linspace(-1, 1, brush_profile.shape[0])
        spline = Interpolator1D(x, brush_profile, method='cubic')
        X, Y = jnp.meshgrid(x, x)
        D = jnp.sqrt(X**2 + Y**2)
        brush_image = vmap(spline)(jnp.exp(-D))
        return brush_image
    
    def render_point(self, point, scale):
        """
        Render a point using the brush model.
        """
        spline_contrast = self.param('spline_contrast', lambda rng: jnp.ones((1,)))
        spline_brightness = self.param('spline_brightness', lambda rng: jnp.zeros((1,)))
        brush_image = self.brush_model() * spline_contrast + spline_brightness
        
        point_mapped_to_image = self.res * point - brush_image.shape[0] / 2 * scale + 0.5
        
        return scale_and_translate(
            brush_image, (self.res, self.res), (0, 1),
            scale * jnp.ones(2), point_mapped_to_image, method='cubic')

    def fit_spline(self):
        raise NotImplementedError("This method should be implemented in subclasses.")
    
    def __call__(self):
        NotImplementedError("This method should be implemented in subclasses.")

class SplenderImage(Splender):
