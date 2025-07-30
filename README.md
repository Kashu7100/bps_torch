# bps_torch
A Pytorch implementation of the [bps](https://github.com/sergeyprokudin/bps) representation using chamfer distance on GPU. This implementation is very fast and was used for the [GrabNet](https://github.com/otaheri/GrabNet) model.

**Basis Point Set (BPS)** is a simple and efficient method for encoding 3D point clouds into fixed-length representations. For the original implementation please visit [this implementation](https://github.com/amzn/basis-point-sets) by [Sergey Prokudin](https://github.com/sergeyprokudin).


### Requirements

- Python >= 3.7
- PyTorch >= 1.1.0 
- Numpy >= 1.16.2
- Pytorch3d

### Installation

CUDA support will be included if CUDA is available in pytorch or if the environment variable FORCE_CUDA is set to 1.
```bash
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

### Demos

Below is an example of how to use the bps_torch code.

```python
import torch
import time
from bps_torch.bps import bps_torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initiate the bps module
bps = bps_torch(
  bps_type='random_uniform',
  n_bps_points=1024,
  radius=1.,
  n_dims=3,
  custom_basis=None,
  device=device
)

pointcloud = torch.rand([5, 100000,3]).to(device)

bps_enc = bps.encode(
  pointcloud,
  feature_type=['dists','deltas'],
  x_features=None
)
# bps_enc['dists']: torch.Size([5, 1024])

deltas = bps_enc['deltas']
bps_dec = bps.decode(deltas)

```

Simply try:
```bash
python -m bps_torch.bps
```

## License

This library is licensed under the MIT-0 License of the original implementation. See the [LICENSE](https://github.com/sergeyprokudin/bps/blob/master/LICENSE) file.
