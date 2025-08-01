import numpy as np
import torch


def to_tensor(array, dtype=torch.float32):
    if not torch.is_tensor(array):
        array = torch.tensor(array)
    return array.to(dtype)


def to_np(array, dtype=np.float32):
    if "scipy.sparse" in str(type(array)):
        array = np.array(array.todencse(), dtype=dtype)
    elif torch.is_tensor(array):
        array = array.detach().cpu().numpy()
    return array


def normalize(x, x_mean=None, mean_center=True, x_scaler=None, scale=True, **kwargs):
    """
    Normalize point clouds

    Parameters
    ----------
    x : torch.tensor or np.array [N, P, D]
        Input point clouds to be normalized
    x_mean : torch.tensor or np.array, [N, D], optional
        if provided, the point clouds will be shifted by this value first
        (default = None)
    mean_center: bool, optional
        if True, the x_mean will be computed using the mean of each pointcloud
        if False, the x_mean will be computed uing the min and the max of the point clouds
        (default = True)
    x_scaler : torch.tensor or np.array, [N, 1], optional
        used to scale each point cloud
        (default = None)
    scale: bool, optional
        if False, the point clouds will not be scaled
        (default = True)

    Returns
    -------
    x_norm : torch.tensor [N, P, D]
        Normalized point clouds
    x_mean : torch.tensor,  [N, D]
        offset value of every cloud
    x_scaler : torch.tensor, [N, 1]
        scaler of every cloud
    """

    x = to_tensor(x)
    is_batch = True if len(x.shape) > 1 else False

    if not is_batch:
        x = x.unsqueeze(0)

    N, P, D = x.shape
    if x_mean is None:
        if mean_center:
            x_mean = x.mean(dim=1, keepdim=True)
        else:
            x_mean = (
                x.max(dim=1, keepdims=True)[0] + x.min(dim=1, keepdims=True)[0]
            ) / 2
    else:
        x_mean = x_mean.view(N, 1, D)

    if x_scaler is None:
        if scale:
            x_scaler = x.norm(dim=2, keepdim=True).max(dim=1, keepdim=True)[0]
        else:
            x_scaler = 1.0
    else:
        x_scaler = x_scaler.view(N, 1, 1)

    x_norm = (x - x_mean) / x_scaler

    return x_norm, x_mean, x_scaler


def denormalize(x_norm, x_mean, x_scaler, **kwargs):
    """Denormalize point clouds

    Parameters
    ----------
    x_norm : torch.tensor or np.array, [n_clouds, n_points, n_dims]
        Input point clouds to be normalized
    x_mean : torch.tensor or np.array, [n_clouds, n_dims]
        if provided, shift the pointclouds by this value to denormalize
    x_scaler : None or [n_clouds, 1]
        if provided, scaler for every cloud used for normalization

    Returns
    -------
    x_norm : torch.tensor, [n_clouds, n_points, n_dims]
        Normalized point clouds
    """

    x_norm = to_tensor(x_norm)
    x_mean = to_tensor(x_mean)
    x_scaler = to_tensor(x_scaler)

    return x_norm * x_scaler + x_mean


def sample_sphere_uniform(
    n_points=1000, n_dims=3, radius=1.0, random_seed=13, device="cpu"
):
    """Sample uniformly from d-dimensional unit ball

    The code is inspired by this small note:
    https://blogs.sas.com/content/iml/2016/04/06/generate-points-uniformly-in-ball.html

    Parameters
    ----------
    n_points : int
        number of samples
    n_dims : int
        number of dimensions
    radius: float
        ball radius
    random_seed: int
        random seed for basis point selection

    Returns
    -------
    x : numpy array
        points sampled from d-ball
    """
    np.random.seed(random_seed)
    # sample point from d-sphere
    x = np.random.normal(size=[n_points, n_dims])
    x_norms = np.sqrt(np.sum(np.square(x), axis=1)).reshape([-1, 1])
    x_unit = x / x_norms
    # now sample radiuses uniformly
    r = np.random.uniform(size=[n_points, 1])
    u = np.power(r, 1.0 / n_dims)
    x = radius * x_unit * u
    np.random.seed(None)
    return to_tensor(x).to(device)


def sample_hemisphere_uniform(
    n_points=1000, n_dims=3, radius=1.0, random_seed=13, axis="-x", device="cpu"
):
    """Sample uniformly from d-dimensional unit ball

    The code is inspired by this small note:
    https://blogs.sas.com/content/iml/2016/04/06/generate-points-uniformly-in-ball.html

    Parameters
    ----------
    n_points : int
        number of samples
    n_dims : int
        number of dimensions
    radius: float
        ball radius
    random_seed: int
        random seed for basis point selection
    Returns
    -------
    x : numpy array
        points sampled from d-ball
    """
    np.random.seed(random_seed)
    # sample point from d-sphere
    x = np.random.normal(size=[2 * n_points, n_dims])
    x_norms = np.sqrt(np.sum(np.square(x), axis=1)).reshape([-1, 1])
    x_unit = x / x_norms
    # now sample radiuses uniformly
    r = np.random.uniform(size=[2 * n_points, 1])
    u = np.power(r, 1.0 / n_dims)
    x = radius * x_unit * u
    np.random.seed(None)

    if axis == "-x":
        p = x[x[:, 0].argsort()][:n_points, :]
    elif axis == "+x":
        p = x[x[:, 0].argsort()][-n_points:, :]
    elif axis == "-y":
        p = x[x[:, 1].argsort()][:n_points, :]
    elif axis == "+y":
        p = x[x[:, 1].argsort()][-n_points:, :]
    elif axis == "-z":
        p = x[x[:, 2].argsort()][:n_points, :]
    elif axis == "+z":
        p = x[x[:, 2].argsort()][-n_points:, :]
    else:
        raise ValueError("axis must be one of -x, +x, -y, +y, -z, +z")

    return to_tensor(p).to(device)


def sample_sphere_nonuniform(
    n_points=1000, n_dims=3, radius=1.0, random_seed=13, device="cpu"
):
    """Sample nonuniformly from d-dimensional unit ball

    The code is inspired by this small note:
    https://blogs.sas.com/content/iml/2016/04/06/generate-points-uniformly-in-ball.html

    Parameters
    ----------
    n_points : int
        number of samples
    n_dims : int
        number of dimensions
    radius: float
        ball radius
    random_seed: int
        random seed for basis point selection
    Returns
    -------
    x : numpy array
        points sampled from d-ball
    """
    np.random.seed(random_seed)
    # sample point from d-sphere
    x = np.random.normal(size=[n_points, n_dims])
    x_norms = np.sqrt(np.sum(np.square(x), axis=1)).reshape([-1, 1])
    x_unit = x / x_norms
    # now sample radiuses uniformly
    r = np.random.uniform(size=[n_points, 1])
    u = np.power(r, 1.0 / 1.5)  # set the 1.5 to change the distribution
    x = radius * x_unit * u
    np.random.seed(None)
    return to_tensor(x).to(device)


def sample_grid_cube(grid_size=32, n_dims=3, minv=-1.0, maxv=1.0, device="cpu"):
    """Generate d-dimensional grid BPS basis
    Parameters
    ----------
    grid_size: int
        number of elements in each grid axe
    minv: float
        minimum element of the grid
    maxv
        maximum element of the grid
    Returns
    -------
    basis: numpy array [grid_size**n_dims, n_dims]
        n-d grid points
    """

    linspaces = [np.linspace(minv, maxv, num=grid_size) for d in range(0, n_dims)]
    coords = np.meshgrid(*linspaces)
    basis = np.concatenate(
        [coords[i].reshape([-1, 1]) for i in range(0, n_dims)], axis=1
    )

    return to_tensor(basis).to(device)


def sample_grid_sphere(n_points=1000, n_dims=3, radius=1.0, device="cpu"):
    grid_points = int(6 * n_points / np.pi)
    grid_size = int(np.power(grid_points, 1 / n_dims))
    in_sphere_points = 0
    while in_sphere_points < n_points:
        c_grid = to_np(sample_grid_cube(grid_size=grid_size)) * radius
        in_sphere_points = np.where(np.linalg.norm(c_grid, axis=1) < radius)[0].shape[0]
        grid_size += 1

    c_grid = to_np(sample_grid_cube(grid_size=grid_size - 2)) * radius
    in_sphere = np.where(np.linalg.norm(c_grid, axis=1) < radius)[0]
    on_sphere_size = n_points - in_sphere.shape[0]
    in_sp = c_grid[in_sphere]
    on_sp = fibonacci_sphere(on_sphere_size) * radius
    sphere = np.concatenate([in_sp, on_sp], 0)
    return to_tensor(sphere).to(device)


def fibonacci_sphere(samples=1, randomize=True):
    import math

    rnd = 1.0
    if randomize:
        rnd = np.random.random() * samples

    points = []
    offset = 2.0 / samples
    increment = math.pi * (3.0 - math.sqrt(5.0))

    for i in range(samples):
        y = ((i * offset) - 1) + (offset / 2)
        r = math.sqrt(1 - pow(y, 2))

        phi = ((i + rnd) % samples) * increment

        x = math.cos(phi) * r
        z = math.sin(phi) * r

        points.append([x, y, z])

    return np.array(points)


def sample_uniform_cylinder(
    n_points=1024, radius=1.0, height=1.0, n_dims=3, device="cpu"
):
    basis = []
    for i in range(n_points):
        s = np.random.uniform(0, 1)
        theta = np.random.uniform(0, 2 * np.pi)
        z = np.random.uniform(0, height)
        r = np.sqrt(s) * radius
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = z  # .. for symmetry :-)
        basis.append([x, y, z])

    return to_tensor(np.vstack(basis)).to(device)


def sample_nonuniform_cylinder(
    n_points=1024, radius=1.0, height=1.0, n_dims=3, device="cpu"
):
    basis = []
    for i in range(n_points):
        s = np.random.uniform(0, 1)
        theta = np.random.uniform(0, 2 * np.pi)
        z = np.random.uniform(0, height)
        r = s * radius
        x = r * np.cos(theta)
        y = r * np.sin(theta)
        z = z  # .. for symmetry :-)
        basis.append([x, y, z])

    return to_tensor(np.vstack(basis)).to(device)


def sample_grid_cylinder(grid_size=32, radius=1.0, height=1.0, n_dims=3, device="cpu"):
    step = height / grid_size
    e = 1e-10
    x = np.arange(-radius, radius + e, step)
    y = np.arange(-radius, radius + e, step)
    z = np.arange(0, height + e, step)

    coords = np.meshgrid(*[x, y, z])
    basis = np.concatenate(
        [coords[i].reshape([-1, 1]) for i in range(0, n_dims)], axis=1
    )

    r = np.linalg.norm(basis[:, :2], axis=1)
    out = r > radius

    # xy = (basis[:, 0]>basis[:, 1])*(basis[:, 0]>-basis[:, 1])+(basis[:, 0]<basis[:, 1])*(basis[:, 0]<-basis[:, 1])

    new_basis = basis.copy()
    new_basis1 = basis.copy()

    new_y = np.sqrt(np.abs(radius**2 - basis[:, 0] ** 2)) * np.sign(basis[:, 1])
    new_x = np.sqrt(np.abs(radius**2 - basis[:, 1] ** 2)) * np.sign(basis[:, 0])

    new_basis[out, 1] = new_y[out]

    new_basis1[out, 0] = new_x[out]

    new_basis = np.concatenate([new_basis, new_basis1[out]])

    return to_tensor(unique_rows(new_basis)).to(device)


def unique_rows(a):
    a = np.ascontiguousarray(a)
    unique_a = np.unique(a.view([("", a.dtype)] * a.shape[1]))
    return unique_a.view(a.dtype).reshape((unique_a.shape[0], a.shape[1]))
