# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#
#

from typing import Literal

import numpy as np
import torch

from .chamfer import chamfer_distance
from .constants import BPS_FEATURE_TYPE, BPS_TYPE
from .tools import (
    sample_grid_cube,
    sample_grid_sphere,
    sample_sphere_nonuniform,
    sample_sphere_uniform,
)
from .utils import to_tensor


class BPS:
    def __init__(
        self,
        bps_type: Literal[BPS_TYPE] = BPS_TYPE.RANDOM_UNIFORM,
        n_bps_points=1024,
        radius=1.0,
        n_dims=3,
        random_seed=13,
        custom_basis=None,
        device="cpu",
    ):

        match bps_type:
            case BPS_TYPE.RANDOM_UNIFORM:
                basis_set = sample_sphere_uniform(
                    n_bps_points,
                    n_dims=n_dims,
                    radius=radius,
                    random_seed=random_seed,
                    device=device,
                )
            case BPS_TYPE.RANDOM_NONUNIFORM:
                basis_set = sample_sphere_nonuniform(
                    n_bps_points,
                    n_dims=n_dims,
                    radius=radius,
                    random_seed=random_seed,
                    device=device,
                )
            case BPS_TYPE.GRID_CUBE:
                # NOTE: we need to find the nearest possible grid size
                grid_size = int(np.round(np.power(n_bps_points, 1 / n_dims)))
                basis_set = sample_grid_cube(
                    grid_size=grid_size, minv=-radius, maxv=radius, device=device
                )
            case BPS_TYPE.GRID_SPHERE:
                basis_set = sample_grid_sphere(
                    n_points=n_bps_points, n_dims=n_dims, radius=radius, device=device
                )
            case BPS_TYPE.CUSTOM:
                assert (
                    custom_basis is not None
                ), "Custom BPS arrangement selected, but no custom_basis provided."
                basis_set = to_tensor(custom_basis)
            case _:
                raise ValueError(f"Invalid basis type: {bps_type}")

        self.bps = basis_set.reshape(1, -1, n_dims)

    def enc_points(
        self,
        x,
        feature_type=["dists"],
        x_features=None,
    ):
        is_batch = True if x.ndim > 2 else False

        if not is_batch:
            x = x.unsqueeze(0)

        Nb, P_bps, D = self.bps.shape
        N, P_x, D = x.shape

        deltas = torch.zeros([N, P_bps, D], device=x.device)
        b2x_idxs = torch.zeros([N, P_bps], dtype=torch.long, device=x.device)

        for fid in range(0, N):
            if Nb == N:
                Y = self.bps[fid : fid + 1]
            else:
                Y = self.bps
            X = x[fid : fid + 1]
            b2x, x2b, b2x_idx, x2b_idx = chamfer_distance(Y, X)
            deltas[fid] = X[:, b2x_idx.to(torch.long)] - Y
            b2x_idxs[fid] = b2x_idx

        x_bps = {}
        if BPS_FEATURE_TYPE.DISTS in feature_type:
            x_bps["dists"] = torch.sqrt(torch.pow(deltas, 2).sum(2))
        if BPS_FEATURE_TYPE.DELTAS in feature_type:
            x_bps["deltas"] = deltas
        if BPS_FEATURE_TYPE.CLOSEST in feature_type:
            b2x_idxs_expanded = b2x_idxs.view(N, P_bps, 1).expand(N, P_bps, D)
            x_bps["closest"] = x.gather(1, b2x_idxs_expanded)
            x_bps["closest_ids"] = b2x_idxs_expanded
        if BPS_FEATURE_TYPE.FEATURES in feature_type:
            try:
                F = x_features.shape[2]
                b2x_idxs_expanded = b2x_idxs.view(N, P_bps, 1).expand(N, P_bps, F)
                x_bps["features"] = x.gather(1, b2x_idxs_expanded)
            except:
                raise ValueError("No x_features parameter is provided!")
        if len(x_bps) < 1:
            raise ValueError(
                "Invalid cell type. Supported types: 'dists', 'deltas', 'closest', 'features'"
            )

        x_bps["ids"] = b2x_idxs
        return x_bps

    def encode(self, x, feature_type=["dists"], x_features=None):
        return self.enc_points(x, feature_type, x_features)

    def decode(self, x):
        is_batch = True if x.ndim > 2 else False

        if not is_batch:
            x = x.unsqueeze(dim=0)

        _, P_bps, D = self.bps.shape
        N, P_x, D = x.shape

        bps_expanded = self.bps.expand(N, P_bps, D)

        return bps_expanded + x


if __name__ == "__main__":
    import time

    bps = BPS(
        bps_type=BPS_TYPE.RANDOM_UNIFORM,
        n_bps_points=1024,
        radius=1.0,
        n_dims=3,
        random_seed=13,
        device="cuda",
    )

    pointcloud = torch.rand([2000, 10000, 3]).to("cuda")

    s = time.time()

    bps_enc = bps.encode(
        pointcloud,
        feature_type=["dists", "deltas"],
        x_features=None,
    )
    print(f"Time taken for encoding: {time.time() - s} seconds")
    print(bps_enc["dists"].shape)

    s = time.time()
    deltas = bps_enc["deltas"]
    bps_dec = bps.decode(deltas)
    print(f"Time taken for decoding: {time.time() - s} seconds")
