import jax
import jax.numpy as jnp
import jax.random as random
from jax import grad, jit, vmap, value_and_grad
from jax.image import scale_and_translate
from interpax import Interpolator1D
import jax.tree_util as jtu
from dataclasses import dataclass, field
from .utils import get_identity_kernel
from abc import ABC, abstractmethod

@jtu.register_dataclass
@dataclass
class Splender(ABC):
    """
    Parent class for SplenderImage and SplenderVideo.
    This class is not meant to be used directly.
    """
    key : jax.random.PRNGKey = field(metadata=dict(static=True), default=None)
    init_knots: jax.Array = None
    brush_profile: jax.Array = None
    spline_contrast: jax.Array = None
    spline_brightness: jax.Array = None
    loc_params: jax.Array = None
    knot_params: jax.Array = None
    kernel: jax.Array = None
    contrast: jax.Array = None
    brightness: jax.Array = None
    opacity: jax.Array = None
    global_scale: jax.Array = None
    res: int = field(metadata=dict(static=True), default=128)
    n_batch: int = field(metadata=dict(static=True), default=1)
    n_frames: int = field(metadata=dict(static=True), default=1)
    t_knots: int = field(metadata=dict(static=True), default=2)
    n_splines: int = field(metadata=dict(static=True), default=1)
    s_knots: int = field(metadata=dict(static=True), default=7)
    n_points_per_spline_per_frame: int = field(metadata=dict(static=True), default=100)
    eps: float = field(metadata=dict(static=True), default=1e-6)

    def __post_init__(self):
        self.brush_profile = jnp.linspace(-1, 1, 13)**2 if self.brush_profile is None else self.brush_profile
        self.spline_contrast = jnp.ones((1,)) if self.spline_contrast is None else self.spline_contrast
        self.spline_brightness = jnp.zeros((1,)) if self.spline_brightness is None else self.spline_brightness
        self.kernel = get_identity_kernel(self.key) if self.kernel is None else self.kernel
        self.contrast = jnp.ones((1,)) if self.contrast is None else self.contrast
        self.brightness = jnp.zeros((1,)) if self.brightness is None else self.brightness
        self.opacity = jnp.ones((1,)) if self.opacity is None else self.opacity
        

        # add eps to all zeros
        self.init_knots = jnp.where(self.init_knots == 0, self.eps, self.init_knots)

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
