# Copyright (c) Facebook, Inc. and its affiliates. All rights reserved.

from typing import Literal

import torch
from pytorch3d.ops.knn import knn_points


def _handle_pointcloud_input(
    points: torch.Tensor,
    lengths: torch.Tensor | None,
):
    """
    If points is an instance of Pointclouds, retrieve the padded points tensor
    along with the number of points per batch and the padded normals.
    Otherwise, return the input points (and normals) with the number of points per cloud
    set to the size of the second dimension of `points`.
    """
    if torch.is_tensor(points):
        if points.ndim != 3:
            raise ValueError(
                f"Expected points to be of shape (N, P, D), got {points.shape}"
            )
        X = points
        if lengths is not None and (
            lengths.ndim != 1 or lengths.shape[0] != X.shape[0]
        ):
            raise ValueError(
                f"Expected lengths to be of shape (N,), got {lengths.shape}"
            )
        if lengths is None:
            lengths = torch.full(
                (X.shape[0],), X.shape[1], dtype=torch.int64, device=points.device
            )
    else:
        raise ValueError(
            "The input pointclouds should be torch.Tensor of shape "
            + "(minibatch, num_points, 3)."
        )
    return X, lengths


def chamfer_distance(
    x,
    y,
    x_lengths=None,
    y_lengths=None,
    weights=None,
    batch_reduction: Literal["mean", "sum"] | None = "mean",
):
    """
    Chamfer distance between two pointclouds x and y.

    Parameters
    ----------
    x: FloatTensor of shape (N, P1, D)
        a batch of point clouds with at most P1 points in each batch element,
        batch size N and feature dimension D.
    y: FloatTensor of shape (N, P2, D)
        a batch of point clouds with at most P2 points in each batch element,
        batch size N and feature dimension D.
    x_lengths: Optional LongTensor of shape (N,) giving the number of points in each
        cloud in x.
    y_lengths: Optional LongTensor of shape (N,) giving the number of points in each
        cloud in x.
    weights: Optional FloatTensor of shape (N,) giving weights for
        batch elements for reduction operation.
    batch_reduction: Reduction operation to apply for the loss across the
        batch, can be one of ["mean", "sum"] or None.

    Returns
    -------
    cham_x: Tensor giving the reduced distance between the pointclouds
        in x and the pointclouds in y.
    cham_y: Tensor giving the reduced distance between the pointclouds
        in y and the pointclouds in x.
    x_nn: Tensor giving the indices of the nearest points in y for each point in x.
    y_nn: Tensor giving the indices of the nearest points in x for each point in y.
    """

    x, x_lengths = _handle_pointcloud_input(x, x_lengths)
    y, y_lengths = _handle_pointcloud_input(y, y_lengths)

    N, P1, D = x.shape
    P2 = y.shape[1]

    # Check if inputs are heterogeneous and create a lengths mask.
    is_x_heterogeneous = (x_lengths != P1).any()
    is_y_heterogeneous = (y_lengths != P2).any()
    x_mask = (
        torch.arange(P1, device=x.device)[None] >= x_lengths[:, None]
    )  # shape [N, P1]
    y_mask = (
        torch.arange(P2, device=y.device)[None] >= y_lengths[:, None]
    )  # shape [N, P2]

    if y.shape[0] != N or y.shape[2] != D:
        raise ValueError(
            f"y does not have the correct shape. {y.shape[0]=} != {N=}, {y.shape[2]=} != {D=}"
        )
    if weights is not None:
        if weights.size(0) != N:
            raise ValueError(
                f"weights must be of shape (N,). {weights.shape[0]=} != {N=}"
            )
        if not (weights >= 0).all():
            raise ValueError("weights cannot be negative.")
        if weights.sum() == 0.0:
            weights = weights.view(N, 1)
            if batch_reduction in ["mean", "sum"]:
                return (
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                    (x.sum((1, 2)) * weights).sum() * 0.0,
                )
            return ((x.sum((1, 2)) * weights) * 0.0, (x.sum((1, 2)) * weights) * 0.0)

    x_nn = knn_points(x, y, lengths1=x_lengths, lengths2=y_lengths, K=1)
    y_nn = knn_points(y, x, lengths1=y_lengths, lengths2=x_lengths, K=1)

    cham_x = x_nn.dists[..., 0]  # (N, P1)
    cham_y = y_nn.dists[..., 0]  # (N, P2)

    if is_x_heterogeneous:
        cham_x[x_mask] = 0.0
    if is_y_heterogeneous:
        cham_y[y_mask] = 0.0

    if weights is not None:
        cham_x *= weights.view(N, 1)
        cham_y *= weights.view(N, 1)

    return cham_x, cham_y, x_nn.idx[..., -1], y_nn.idx[..., -1]


if __name__ == "__main__":
    import time

    p1 = torch.rand([5000, 1000, 3]).cuda()
    p2 = torch.rand([5000, 50000, 3]).cuda()

    s = time.time()
    ch = chamfer_distance(p1, p2)
    torch.cuda.synchronize()
    # 1.27 seconds
    print(f"it took {time.time() - s} secods --> pytorch3d")
