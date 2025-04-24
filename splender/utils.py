import jax
from jax.nn import sigmoid, softplus
from jax.scipy.special import logit
import jax.numpy as jnp
from functools import partial

def circle_image(size):
    x = jnp.linspace(-1, 1, size)
    y = jnp.linspace(-1, 1, size)
    xx, yy = jnp.meshgrid(x, y)
    circle = jnp.exp(-(xx ** 2 + yy ** 2)/0.4)
    circle /= circle.max()
    circle -= jnp.abs(circle.min())

    return circle


def params2knots(params):
    """
    Transform unconstrained parameters into knot arrays with desired properties.
    
    Assumptions:
      - params_x, params_y are each 2D arrays of shape (s, t)
      - We force monotonicity along axis 0 (rows) for x and t via a cumulative sum.
      - We squash y into (0, 1) via a sigmoid.
    """
    # For x: softplus to ensure positivity, then cumulative sum along rows, and normalize.
    # x_raw = softplus(params_x)
    # # cumsum along rows (axis 0) gives an increasing sequence per column.
    # x = jnp.cumsum(x_raw, axis=0) / x_raw.shape[0]
    # x = jnp.minimum(x, 1.)

    return sigmoid(params)

def knots2params(knots):
    """
    Inverse of params2knots. Returns unconstrained parameters.
    
    Inversion details:
      - For y, the inverse of sigmoid is the logit.
      - For x and t, we assume that they were produced by a normalized cumulative sum of softplus.
        Thus, we first recover differences along axis 0, then invert the softplus.
        (Note: This inversion is not unique if the normalization was applied,
         but here we follow the same “recipe” used in params2knots.)
    """
    # Invert y via the logit:
    return logit(knots)


def pad_and_stack(list_of_arrays):
    assert list_of_arrays, "list_of_arrays must be non-empty, but got %s" % list_of_arrays
    max_len = max(arr.shape[0] for arr in list_of_arrays)
    ndim = list_of_arrays[0].ndim
    return jnp.stack([
        jnp.pad(arr, ((0, max_len - arr.shape[0]), ) + (ndim-1) * ((0, 0),), mode='constant', constant_values=0)
        for arr in list_of_arrays
    ])


@partial(jax.jit, static_argnums=(1, 2, 3))
def mean_grid_fixed(arr, cell_h, cell_w, cell_d):
    """
    Divides a 3D array into a grid of cells of size (cell_h, cell_w, cell_d)
    and computes the mean for each cell.
    """
    h, w, d = arr.shape

    # Compute the number of grid cells along each dimension.
    num_cells_h = h // cell_h
    num_cells_w = w // cell_w
    num_cells_d = d // cell_d

    # Reshape the array into shape:
    # (num_cells_h, cell_h, num_cells_w, cell_w, num_cells_d, cell_d)
    grid = arr.reshape(num_cells_h, cell_h, num_cells_w, cell_w, num_cells_d, cell_d)

    # Transpose to bring grid cell indices to the front:
    # (num_cells_h, num_cells_w, num_cells_d, cell_h, cell_w, cell_d)
    grid = jnp.transpose(grid, (0, 2, 4, 1, 3, 5))

    # Apply the mean function to each cell using nested vmap calls.
    # This will produce an output of shape (num_cells_h, num_cells_w, num_cells_d)
    process_grid = jax.vmap(
                        jax.vmap(
                            jax.vmap(jnp.mean, in_axes=0),
                        in_axes=0),
                    in_axes=0)
    return process_grid(grid)

def grid_mean(arr, cell_size):
    """
    Wrapper function that divides a 3D array `arr` into cells of size given by
    `cell_size` (a tuple: (cell_h, cell_w, cell_d)) and returns the mean in each cell.
    """
    return mean_grid_fixed(arr, cell_size[0], cell_size[1], cell_size[2])

def get_identity_kernel(rng, size = 3):
    idenitity_kernel = jnp.zeros((size, size))
    idenitity_kernel = idenitity_kernel.at[size // 2, size // 2].set(1.0)
    return idenitity_kernel

