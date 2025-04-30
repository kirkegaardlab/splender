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
    # t_knots: int = field(metadata=dict(static=True), default=3)
    # n_frames: int = field(metadata=dict(static=True), default=10)

    def __manual_post_init__(self):
        super().__post_init__()
        # jax.debug.print("init_knots: {init_knots}", init_knots=self.init_knots)
        if self.init_knots is not None:
            if self.init_knots.ndim == 4:
                # single image batch
                self.init_knots = self.init_knots[None]
            init_knot_params = logit(self.init_knots)
            init_param_mean = init_knot_params.mean(-3, keepdims=True)
            init_params = init_knot_params - init_param_mean
            self.n_batch = init_knot_params.shape[0]
            self.n_splines = init_knot_params.shape[1]
            self.s_knots = init_knot_params.shape[2]
            self.t_knots = init_knot_params.shape[3]
            self.loc_params = jnp.concatenate([init_param_mean, 5 * jnp.ones((self.n_batch, self.n_splines, 1, self.t_knots, 1))], axis=-1)
            self.knot_params = jnp.concatenate([init_params, jnp.zeros((self.n_batch, self.n_splines, self.s_knots, self.t_knots, 1))], axis=-1)
        else:
            self.loc_params = jnp.zeros((self.n_splines, 1, 1, 3))
            self.knot_params = jnp.zeros((self.n_splines, self.s_knots, self.t_knots, 3))
        
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
    
    def fit_spline(self, params):
        knots = sigmoid(params)

        x, y, scale_knots = knots[..., 0], knots[..., 1], knots[..., 0, 2]

        s = jnp.linspace(0, 1, knots.shape[0])
        t = jnp.linspace(0, 1, knots.shape[1])
        
        x_spline = Interpolator2D(s, t, x, method="cubic")
        y_spline = Interpolator2D(s, t, y, method="cubic")
        scale_spline = Interpolator1D(s, scale_knots, method="cubic")
        return x_spline, y_spline, scale_spline
    
    def render_spline(self, x_spline, y_spline, scale_spline, scale):
        """
        Ideally we'd handle different video lengths, 
        but for now we assume all videos are the same length.
        """
        s_fine = jnp.linspace(0, 1, self.n_points_per_spline_per_frame)
        t_fine = jnp.linspace(0, 1, self.n_frames)
        _, T = jnp.meshgrid(s_fine, t_fine, indexing='ij')
        get_uniform_points_t = lambda t: self.get_uniform_points(partial(x_spline, yq = t), partial(y_spline, yq = t))
        S = vmap(get_uniform_points_t, out_axes = 1)(t_fine)
                
        # Get the points on the spline for all time steps
        x_points = vmap(x_spline)(S, T)
        y_points = vmap(y_spline)(S, T)
        # scale_points = scale_spline(s_fine)
        scale_points = vmap(scale_spline)(S)
        
        # Render the points
        all_points = vmap(self.render_point, in_axes=(0, 0))(
                            jnp.stack([y_points.ravel(), x_points.ravel()], axis=-1),
                            (scale * scale_points).ravel(), # repeat scale points for each frame
                        )
        return all_points
    
    def render_splines(self, knots, scales):
        """
        Render all splines in a video.
        For each spline, 
        If knot_params are all zeros, the image is empty.
        This allows for batching with a varying number of splines across the batch.
        """
        def maybe_render_spline(knot_params, scale):

            def render_empty():
                return jnp.zeros((self.n_frames * self.n_points_per_spline_per_frame, self.res, self.res)), jnp.zeros((self.n_frames,)), jnp.zeros((self.n_frames,))
            
            def render_spline():
                x_spline, y_spline, scale_spline = self.fit_spline(knot_params)
                
                t_fine = jnp.linspace(0, 1, self.n_frames)
                # get_curvature_t = lambda t: spline_curvature_t(x_spline, y_spline, t)
                # curvature = vmap(get_curvature_t)(t_fine)

                lengths = vmap(lambda t: self.cumulative_spline_length(partial(x_spline, yq = t), partial(y_spline, yq = t))[-1])(t_fine)
                curvatures = vmap(lambda t: self.mean_spline_curvature(partial(x_spline, yq = t), partial(y_spline, yq = t)))(t_fine)
                return self.render_spline(x_spline, y_spline, scale_spline, scale), lengths, curvatures
            
            return jax.lax.cond(jnp.all(knot_params[:2] == 0), render_empty, render_spline)
        
        
        spline_images, lengths, curvatures = vmap(maybe_render_spline)(knots, scales)
        
        # aggregate spline_images
        splines_image = jnp.max(spline_images, axis=0) * self.opacity + jnp.sum(spline_images, axis=0) * (1 - self.opacity)

        return splines_image, lengths, curvatures
    
    def render_video(self, knots, scales):
        splines_frames, lengths, curvatures = self.render_splines(knots, scales)
        
        frame_idx = jnp.tile(jnp.arange(self.n_frames), (self.n_points_per_spline_per_frame, 1))
        frames_idxs = frame_idx.ravel()

        video = jax.ops.segment_sum(splines_frames, frames_idxs, num_segments = self.n_frames) #+ logistic_background()

        video = vmap(lambda frame: jax.scipy.signal.convolve2d(frame, self.kernel, mode='same', boundary='fill'))(video)

        video = video * self.contrast + self.brightness
        video = sigmoid(video)
        return video, lengths, curvatures

    def __call__(self):
        knots = self.loc_params + self.knot_params
        images, lengths, curvatures = jax.vmap(self.render_video)(knots, self.global_scale)
        return images, lengths, curvatures