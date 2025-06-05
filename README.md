# splender

[![arXiv](https://img.shields.io/badge/arXiv-2503.14525-b31b1b.svg?style=flat)](https://arxiv.org/abs/2503.14525)
[![PyPI](https://img.shields.io/pypi/v/splender.svg)](https://pypi.org/project/splender/)

Fit splines to images and videos through differentiable rendering.

![splender](https://github.com/user-attachments/assets/a0d23eae-5b04-4504-bf0e-f05dd7e4bdc8)


## Usage

Here is a basic example of trying to fit the number of 5 of the [MNIST](https://en.wikipedia.org/wiki/MNIST_database) dataset.

```python
import splender
import jax
from PIL import Image
import numpy as np

image = Image.open('img_5.png')
image = np.array(image, dtype=np.float32) / 255.0

init_splines = splender.knot_init.get_splines_from_frame(image)[0][:, ::-1]
init_splines = jax.numpy.array(init_splines, dtype=jax.numpy.float32)
init_knots = splender.knot_init.downsample_points(init_splines, 6)[None]

key = jax.random.key(42)
image = image[None]
init_knots = init_knots[None] / 28

model = splender.image.SplenderImage(key=key, init_knots=init_knots, res=28, global_scale=0.3)
model, _ = splender.optim.fit(model, image, video=False)
x_spline, y_spline, _ = model.fit_spline(model.loc_params[0, 0] + model.knot_params[0, 0])
s = jax.numpy.linspace(0, 1, 100) 
final_splines = jax.numpy.stack([x_spline(s), y_spline(s)], axis=-1)
```
resulting in
<p align="center">
  <img src="https://github.com/user-attachments/assets/ccef7f3f-cf2c-4137-9fbb-d55d708b0909" style="width: auto; height: 360px;" />
</p>

Some more elavorated example notebooks can be found under [`examples/`](examples/)

## Installation

```bash
pip install splender
```

> [!NOTE]
> If you want it to run on GPU, remember to install it alongside the correct `jax` version: `pip install splender jax[cuda12]`

## Documentation

PENDING

For a more technical discussion on the model, please check out the [paper](https://doi.org/10.48550/arXiv.2503.14525)

## Citation

If you use `splender` in your research and need to reference it, please cite it as follows:

```
@article{zdyb2025splinerefinementdifferentiablerendering,
  title={Spline refinement with differentiable rendering},
  author={Frans Zdyb and Albert Alonso and Julius B. Kirkegaard},
  year={2025},
  eprint={2503.14525},
  archivePrefix={arXiv},
  primaryClass={eess.IV},
  url={https://arxiv.org/abs/2503.14525},
  journal={arXiv preprint arXiv:2503.14525},
}
```

## License
Splender is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
