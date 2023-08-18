import numpy as np

from stylegan_metrics import dnnlib
from stylegan_metrics import metric_utils


def compute_kid_directories(
    opts,
    path_real: str,
    path_gen: str,
    resolution: int,
    num_subsets: int,
    max_subset_size: int,
):
    """Kernel Inception Distance (KID) from the paper "Demystifying MMD
    GANs". Matches the original implementation by Binkowski et al. at
    https://github.com/mbinkowski/MMD-GAN/blob/master/gan/compute_scores.py
    between two directories of images.

    Args:
        opts: Options for the metric.
        path_real: Path to the real images.
        path_gen: Path to the generated images.
        resolution: Resolution of the images.
        num_subsets: Number of subsets to use for the approximation.
        max_subset_size: Maximum size of each subset.

    Returns:
        float: The KID score.
    """

    detector_url = metric_utils.MODEL_URLS["INCEPTION"]
    detector_kwargs = dict(
        return_features=True
    )  # Return raw features before the softmax layer.

    opts.dataset_kwargs = dnnlib.EasyDict(
        class_name="stylegan_metrics.dataset.ImageFolderDataset", path=path_gen
    )
    opts.dataset_kwargs.resolution = resolution

    real_features = metric_utils.compute_feature_stats_for_dataset(
        opts=opts,
        detector_url=detector_url,
        detector_kwargs=detector_kwargs,
        rel_lo=0,
        rel_hi=0,
        capture_all=True,
    ).get_all()

    opts.dataset_kwargs = dnnlib.EasyDict(
        class_name="stylegan_metrics.dataset.ImageFolderDataset", path=path_real
    )
    opts.dataset_kwargs.resolution = resolution

    gen_features = metric_utils.compute_feature_stats_for_dataset(
        opts=opts,
        detector_url=detector_url,
        detector_kwargs=detector_kwargs,
        rel_lo=0,
        rel_hi=1,
        capture_all=True,
    ).get_all()

    if opts.rank != 0:
        return float("nan")

    return compute_kid(real_features, gen_features, num_subsets, max_subset_size)


def compute_kid(
    real_features: np.ndarray,
    gen_features: np.ndarray,
    num_subsets: int,
    max_subset_size: int,
) -> float:
    """Kernel Inception Distance (KID) from the paper "Demystifying MMD
    GANs". Matches the original implementation by Binkowski et al. at
    https://github.com/mbinkowski/MMD-GAN/blob/master/gan/compute_scores.py
    between two sets of features.

    Args:
        real_features: Features of the real images.
        gen_features: Features of the generated images.
        num_subsets: Number of subsets to use for the approximation.
        max_subset_size: Maximum size of each subset.

    Returns:
        float: The KID score.
    """

    n = real_features.shape[1]
    m = min(min(real_features.shape[0], gen_features.shape[0]), max_subset_size)
    t = 0
    for _subset_idx in range(num_subsets):
        x = gen_features[np.random.choice(gen_features.shape[0], m, replace=False)]
        y = real_features[np.random.choice(real_features.shape[0], m, replace=False)]
        a = (x @ x.T / n + 1) ** 3 + (y @ y.T / n + 1) ** 3
        b = (x @ y.T / n + 1) ** 3
        t += (a.sum() - np.diag(a).sum()) / (m - 1) - b.sum() * 2 / m
    kid = t / num_subsets / m
    return float(kid)
