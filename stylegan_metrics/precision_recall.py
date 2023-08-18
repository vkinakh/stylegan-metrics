from typing import Tuple

import torch

from stylegan_metrics import dnnlib
from stylegan_metrics import metric_utils


def compute_pr_directories(
    opts,
    path_real: str,
    path_gen: str,
    resolution: int,
    nhood_size: int,
    row_batch_size: int,
    col_batch_size: int,
) -> Tuple[float, float]:
    """
    Precision/Recall (PR) from the paper "Improved Precision and Recall
    Metric for Assessing Generative Models". Matches the original implementation
    by Kynkaanniemi et al. at
    https://github.com/kynkaat/improved-precision-and-recall-metric/blob/master/precision_recall.py
    between two directories of images

    Args:
        opts: Options for the metric.
        path_real: Path to the real images.
        path_gen: Path to the generated images.
        resolution: Resolution of the images.
        nhood_size: Size of the neighborhood.
        row_batch_size: Batch size for the rows.
        col_batch_size: Batch size for the columns.

    Returns:
        Tuple: precision and recall
    """

    detector_url = metric_utils.MODEL_URLS["VGG16"]
    detector_kwargs = dict(return_features=True)

    opts.dataset_kwargs = dnnlib.EasyDict(
        class_name="stylegan_metrics.dataset.ImageFolderDataset", path=path_real
    )
    opts.dataset_kwargs.resolution = resolution
    real_features = (
        metric_utils.compute_feature_stats_for_dataset(
            opts=opts,
            detector_url=detector_url,
            detector_kwargs=detector_kwargs,
            rel_lo=0,
            rel_hi=0,
            capture_all=True,
        )
        .get_all_torch()
        .to(torch.float16)
        .to(opts.device)
    )

    opts.dataset_kwargs = dnnlib.EasyDict(
        class_name="stylegan_metrics.dataset.ImageFolderDataset", path=path_gen
    )
    opts.dataset_kwargs.resolution = resolution
    gen_features = (
        metric_utils.compute_feature_stats_for_dataset(
            opts=opts,
            detector_url=detector_url,
            detector_kwargs=detector_kwargs,
            rel_lo=0,
            rel_hi=1,
            capture_all=True,
        )
        .get_all_torch()
        .to(torch.float16)
        .to(opts.device)
    )

    return compute_pr_features(
        opts, real_features, gen_features, nhood_size, row_batch_size, col_batch_size
    )


def compute_pr_features(
    opts,
    real_features: torch.Tensor,
    gen_features: torch.Tensor,
    nhood_size: int,
    row_batch_size: int,
    col_batch_size: int,
) -> Tuple[float, float]:
    """
     Precision/Recall (PR) from the paper "Improved Precision and Recall
    Metric for Assessing Generative Models". Matches the original implementation
    by Kynkaanniemi et al. at
    https://github.com/kynkaat/improved-precision-and-recall-metric/blob/master/precision_recall.py
    between two sets of features

    Args:
        opts: Options for the metric.
        real_features: Features of the real images.
        gen_features: Features of the generated images.
        nhood_size: Size of the neighborhood.
        row_batch_size: Batch size for the rows.
        col_batch_size: Batch size for the columns.

    Returns:
        Tuple: precision and recall
    """

    results = dict()
    for name, manifold, probes in [
        ("precision", real_features, gen_features),
        ("recall", gen_features, real_features),
    ]:
        kth = []
        for manifold_batch in manifold.split(row_batch_size):
            dist = compute_distances(
                row_features=manifold_batch,
                col_features=manifold,
                num_gpus=opts.num_gpus,
                rank=opts.rank,
                col_batch_size=col_batch_size,
            )
            kth.append(
                dist.to(torch.float32).kthvalue(nhood_size + 1).values.to(torch.float16)
                if opts.rank == 0
                else None
            )
        kth = torch.cat(kth) if opts.rank == 0 else None
        pred = []
        for probes_batch in probes.split(row_batch_size):
            dist = compute_distances(
                row_features=probes_batch,
                col_features=manifold,
                num_gpus=opts.num_gpus,
                rank=opts.rank,
                col_batch_size=col_batch_size,
            )
            pred.append((dist <= kth).any(dim=1) if opts.rank == 0 else None)
        results[name] = float(
            torch.cat(pred).to(torch.float32).mean() if opts.rank == 0 else "nan"
        )
    return results["precision"], results["recall"]


def compute_distances(
    row_features: torch.Tensor,
    col_features: torch.Tensor,
    num_gpus: int,
    rank: int,
    col_batch_size: int,
) -> torch.Tensor:
    assert 0 <= rank < num_gpus
    num_cols = col_features.shape[0]
    num_batches = ((num_cols - 1) // col_batch_size // num_gpus + 1) * num_gpus
    col_batches = torch.nn.functional.pad(
        col_features, [0, 0, 0, -num_cols % num_batches]
    ).chunk(num_batches)
    dist_batches = []
    for col_batch in col_batches[rank::num_gpus]:
        dist_batch = torch.cdist(row_features.unsqueeze(0), col_batch.unsqueeze(0))[0]
        for src in range(num_gpus):
            dist_broadcast = dist_batch.clone()
            if num_gpus > 1:
                torch.distributed.broadcast(dist_broadcast, src=src)
            dist_batches.append(dist_broadcast.cpu() if rank == 0 else None)
    return torch.cat(dist_batches, dim=1)[:, :num_cols] if rank == 0 else None
