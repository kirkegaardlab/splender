import PIL
import jax.numpy as jnp
import matplotlib.pyplot as plt
from splender.image import fit

# load mnist_3.png

img = PIL.Image.open('mnist_3.png')

# convert to jnp array
img = jnp.array(img).mean(axis=-1)
img = img / 255.0

n_tries = 3

splines, recon, _, _, _, _ = fit(jnp.array([img] * n_tries), s = 10, n_splines = 2)

print(len(splines))
print(splines[0].shape)
print(recon.shape)

# plot
for i in range(1, n_tries + 1):
    plt.subplot(n_tries, 2, 2 * i - 1)
    plt.imshow(img, cmap='gray', vmin=0, vmax=1)
    for spline in splines[i-1]:
        plt.plot(spline[:, 0], spline[:, 1], 'r-')
    plt.subplot(n_tries, 2, 2 * i)
    plt.imshow(recon[i-1], cmap='gray', vmin=0, vmax=1)
plt.show()