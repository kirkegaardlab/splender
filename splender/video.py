from .splender import Splender
from interpax import Interpolator2D, Interpolator1D
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
class SplenderVideo(Splender):
    t_knots: int = field(metadata=dict(static=True), default=3)

    def __post_init__(self):
        super().__post_init__()

        if self.init_knots is not None:
            if self.init_knots.ndim == 4:
                # single image batch
                self.init_knots = self.init_knots[None]
            init_knot_params = logit(self.init_knots)
            init_param_mean = init_knot_params.mean(-3, keepdims=True)
            init_params = init_knot_params - init_param_mean
            self.n_images = init_knot_params.shape[0]
            self.n_splines = init_knot_params.shape[1]
            self.s_knots = init_knot_params.shape[2]
            self.t_knots = init_knot_params.shape[3]
            self.loc_params = jnp.concatenate([init_param_mean, 5 * jnp.ones((self.n_images, self.n_splines, 1, 1, 1))], axis=-3)
            self.knot_params = jnp.concatenate([init_params, jnp.zeros((self.n_images, self.n_splines, self.s_knots, 1))], axis=-1)
        else:
            self.loc_params = jnp.zeros((self.n_splines, 1, 1, 3))
            self.knot_params = jnp.zeros((self.n_splines, self.s_knots, self.t_knots, 3))

    def spatial_derivative(self, spline, t, degree = 1):
        return partial(spline, yq = t, dx=degree)
    
    def fit_spline(self, params):
        knots = sigmoid(params)

        x, y, scale_knots = knots[..., 0], knots[..., 1], knots[..., 2]

        s = jnp.linspace(0, 1, knots.shape[0])
        t = jnp.linspace(0, 1, knots.shape[1])
        
        x_spline = Interpolator2D(s, t, x, method="cubic")
        y_spline = Interpolator2D(s, t, y, method="cubic")
        scale_spline = Interpolator1D(s, scale_knots, method="cubic")
        return x_spline, y_spline, scale_spline
    
    def render_spline(self, x_spline, y_spline, scale_spline, scale):
        
