import PIL
import jax.numpy as jnp
import matplotlib.pyplot as plt
from splender.image import fit

# load mnist_3.png

img = PIL.Image.open('mnist_3.png')

# convert to jnp array
img = jnp.array(img).mean(axis=-1)
img = img / 255.0

splines, recon, _, _, _, _ = fit(img)

print(len(splines))
print(splines[0].shape)
print(recon.shape)

# plot
plt.subplot(1, 2, 1)
plt.imshow(img, cmap='gray', vmin=0, vmax=1)
for spline in splines[0]:
    plt.plot(spline[:, 0], spline[:, 1], 'r-')
plt.subplot(1, 2, 2)
plt.imshow(recon[0], cmap='gray', vmin=0, vmax=1)
plt.show()