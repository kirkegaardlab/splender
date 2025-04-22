from .splender import Splender
from interpax import Interpolator2D
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
    
    def __post_init__(self):
        super().__post_init__()
