from typing import Tuple

import numpy as np

from stylegan_metrics import dnnlib
from stylegan_metrics import metric_utils


def compute_is_directory(
    opts, path_data: str, resolution: int, num_splits
) -> Tuple[float, float]:
    """Inception Score (IS) from the paper "Improved techniques for training
    GANs". Matches the original implementation by Salimans et al. at
    https://github.com/openai/improved-gan/blob/master/inception_score/model.py
    from the directory containing the images.

    Args:
        opts: Options for the metric.
        path_data: Path to the directory containing the images.
        resolution: Resolution of the images.
        num_splits: Number of splits.

    Returns:
        Tuple: inception score and standard deviation
    """

    detector_url = metric_utils.MODEL_URLS["INCEPTION"]
    detector_kwargs = dict(
        no_output_bias=True
    )  # Match the original implementation by not applying bias in the softmax layer.

    opts.dataset_kwargs = dnnlib.EasyDict(
        class_name="stylegan_metrics.dataset.ImageFolderDataset", path=path_data
    )
    opts.dataset_kwargs.resolution = resolution
    gen_probs = metric_utils.compute_feature_stats_for_dataset(
        opts=opts,
        detector_url=detector_url,
        detector_kwargs=detector_kwargs,
        capture_all=True,
    ).get_all()

    if opts.rank != 0:
        return float("nan"), float("nan")

    return compute_is(probs=gen_probs, num_splits=num_splits)


def compute_is(probs: np.ndarray, num_splits: int) -> Tuple[float, float]:
    """Inception Score (IS) from the paper "Improved techniques for training
    GANs". Matches the original implementation by Salimans et al. at
    https://github.com/openai/improved-gan/blob/master/inception_score/model.py
    from the probabilities, computed by the model.

    Args:
        probs: Probabilities of the images.
        num_splits: Number of splits.

    Returns:
        Tuple: inception score and standard deviation
    """

    num_gen = probs.shape[0]

    scores = []
    for i in range(num_splits):
        part = probs[i * num_gen // num_splits : (i + 1) * num_gen // num_splits]
        kl = part * (np.log(part) - np.log(np.mean(part, axis=0, keepdims=True)))
        kl = np.mean(np.sum(kl, axis=1))
        scores.append(np.exp(kl))
    return float(np.mean(scores)), float(np.std(scores))
