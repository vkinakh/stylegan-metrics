import numpy as np
import scipy.linalg

from stylegan_metrics import dnnlib
from stylegan_metrics import metric_utils


def compute_fid_directories(
    opts, path_real: str, path_gen: str, resolution: int
) -> float:
    """Frechet Inception Distance (FID) from the paper
    "GANs trained by a two time-scale update rule converge to a local Nash
    equilibrium". Matches the original implementation by Heusel et al. at
    https://github.com/bioinf-jku/TTUR/blob/master/fid.py
    between two directories of images.

    Args:
        opts: Options for the metric.
        path_real: Path to the real images.
        path_gen: Path to the generated images.
        resolution: Resolution of the images.

    Returns:
        float: The FID score.
    """

    detector_url = metric_utils.MODEL_URLS["INCEPTION"]
    detector_kwargs = dict(
        return_features=True
    )  # Return raw features before the softmax layer.

    # stats for real images
    opts.dataset_kwargs = dnnlib.EasyDict(
        class_name="stylegan_metrics.dataset.ImageFolderDataset", path=path_real
    )
    opts.dataset_kwargs.resolution = resolution
    mu_real, sigma_real = metric_utils.compute_feature_stats_for_dataset(
        opts=opts,
        detector_url=detector_url,
        detector_kwargs=detector_kwargs,
        rel_lo=0,
        rel_hi=0,
        capture_mean_cov=True,
    ).get_mean_cov()

    # stats for generated images
    opts.dataset_kwargs = dnnlib.EasyDict(
        class_name="stylegan_metrics.dataset.ImageFolderDataset", path=path_gen
    )
    opts.dataset_kwargs.resolution = resolution
    mu_gen, sigma_gen = metric_utils.compute_feature_stats_for_dataset(
        opts=opts,
        detector_url=detector_url,
        detector_kwargs=detector_kwargs,
        rel_lo=0,
        rel_hi=1,
        capture_mean_cov=True,
    ).get_mean_cov()

    if opts.rank != 0:
        return float("nan")

    return compute_fid(mu_real, sigma_real, mu_gen, sigma_gen)


def compute_fid(
    mu1: np.ndarray, sigma1: np.ndarray, mu2: np.ndarray, sigma2: np.ndarray
) -> float:
    """Frechet Inception Distance (FID) from the paper
    "GANs trained by a two time-scale update rule converge to a local Nash
    equilibrium". Matches the original implementation by Heusel et al. at
    https://github.com/bioinf-jku/TTUR/blob/master/fid.py

    Args:
        mu1: Mean of the first set of activations.
        sigma1: Covariance matrix of the first set of activations.
        mu2: Mean of the second set of activations.
        sigma2: Covariance matrix of the second set of activations.

    Returns:
        float: The FID score.
    """

    m = np.square(mu1 - mu2).sum()
    s, _ = scipy.linalg.sqrtm(
        np.dot(sigma1, sigma2), disp=False
    )  # pylint: disable=no-member
    fid = np.real(m + np.trace(sigma1 + sigma2 - s * 2))
    return float(fid)
