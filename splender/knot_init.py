import jax.numpy as jnp
import rustworkx as rx
from collections import deque
from skimage.morphology import skeletonize
import numpy as np
from jax import vmap, value_and_grad, grad, jit
from interpax import Interpolator1D
from functools import partial
import matplotlib.pyplot as plt

def bfs_with_predecessors(tree: rx.PyGraph, start):
    """Runs BFS from `start`, returning the farthest node, distances, and predecessors."""
    distances = {}
    predecessors = {}
    queue = deque([start])
    
    distances[start] = 0
    predecessors[start] = None  # Root has no predecessor

    while queue:
        node = queue.popleft()
        for neighbor in tree.neighbors(node):
            if neighbor not in distances:  # Unvisited node
                distances[neighbor] = distances[node] + 1
                predecessors[neighbor] = node
                queue.append(neighbor)

    # Farthest node is the one with the max distance
    farthest_node = max(distances, key=distances.get)
    return farthest_node, distances, predecessors

def reconstruct_path(predecessors, end):
    """Reconstructs the path from start to `end` using predecessors."""
    path = []
    node = end
    while node is not None:
        path.append(node)
        node = predecessors[node]  # Move backwards
    path.reverse()
    return path

def longest_path_in_tree(tree: rx.PyGraph):
    """Finds the longest path (diameter) in the tree."""
    # Step 1: Pick any arbitrary node
    start = next(iter(tree.node_indices()))

    # Step 2: Find farthest node `u` from `start`
    u, _, _ = bfs_with_predecessors(tree, start)

    # Step 3: Find farthest node `v` from `u` and reconstruct path
    v, _, predecessors = bfs_with_predecessors(tree, u)
    longest_path = reconstruct_path(predecessors, v)

    return longest_path, len(longest_path) - 1  # Subtract 1 for edge count

def get_splines_from_frame(frame, threshold = 0.5, neighbors = 'diagonal', thin = False, skeletonize_first = True):
    
    if skeletonize_first:
        binary_img = skeletonize(np.array(frame) > threshold)
    else:
        binary_img = frame > threshold

    # plt.figure()
    # plt.imshow(binary_img)

    if thin:
        neighbors = jnp.stack([binary_img, jnp.roll(binary_img, 1, axis = 0), jnp.roll(binary_img, 1, axis = 1)], axis=-1)
        inner_coords = jnp.stack(jnp.where(neighbors.sum(axis=-1) == 3)).T

        coord_set = set(map(tuple, inner_coords.tolist()))
    else:
        coord_set = set(map(tuple, jnp.stack(jnp.where(binary_img)).T.tolist()))

    # Create the graph
    graph = rx.PyGraph(multigraph=False)

    # Dictionary to store node indices
    node_indices = {}

    # Add all nodes first
    for coord in coord_set:
        node_indices[coord] = graph.add_node(coord)

    if neighbors == 'diagonal':
        increments = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]
    else:
        increments = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # Add edges between neighbors
    for x, y in coord_set:
        node_idx = node_indices[(x, y)]
        for dx, dy in increments:  # Left, right, up, down
            neighbor = (x + dx, y + dy)
            if neighbor in node_indices:  # Only connect if neighbor exists
                graph.add_edge(node_idx, node_indices[neighbor], 1)

    # find connected components
    connected_components = rx.connected_components(graph)
    # make a list of new graphs
    new_graphs = [graph.subgraph(list(component)) for component in connected_components]

    path_coords = []
    for subgraph in new_graphs:
        # Compute the Minimum Spanning Tree (MST)
        mst = rx.minimum_spanning_tree(subgraph)

        # Get the longest path and its length
        longest_path, longest_path_length = longest_path_in_tree(mst)

        # Convert node indices back to coordinates
        longest_path_coords = [mst[node] for node in longest_path]
        path_coords.append(longest_path_coords)

    all_path_coords = [jnp.array(longest_path_coords) for longest_path_coords in path_coords]
    # sort by number of points
    all_path_coords = sorted(all_path_coords, key=lambda x: x.shape[0], reverse=True)
    return all_path_coords

def fit_spline(knots):
    x, y = knots[..., 0], knots[..., 1]
    
    s = jnp.linspace(0, 1, len(x))
    x_spline = Interpolator1D(s, x, method="cubic2")
    y_spline = Interpolator1D(s, y, method="cubic2")
    return x_spline, y_spline

def get_uniform_points(x_spline, y_spline, n_points_per_spline = 8):
    eps = 1e-6
    # Compute arc length
    dx_ds = vmap(partial(x_spline, dx=1))
    dy_ds = vmap(partial(y_spline, dx=1))
    s_fine = jnp.linspace(0, 1, n_points_per_spline)
    ds_vals = jnp.sqrt(dx_ds(s_fine)**2 + dy_ds(s_fine)**2 + eps)
    delta_s = s_fine[1] - s_fine[0]
    cumulative_length = jnp.concatenate([
        jnp.array([0.0]),
        jnp.cumsum(0.5 * (ds_vals[1:] + ds_vals[:-1]) * delta_s)
    ])
    s_uniform = jnp.interp(jnp.linspace(0, cumulative_length[-1], n_points_per_spline), cumulative_length, s_fine)
    return s_uniform, cumulative_length[-1]

def downsample_points(coords, n_points_per_spline = 8):
    x_spline, y_spline = fit_spline(coords)
    s_uniform, length = get_uniform_points(x_spline, y_spline, n_points_per_spline=n_points_per_spline)
    x = x_spline(s_uniform)
    y = y_spline(s_uniform)
    return jnp.stack([x, y], axis=-1)