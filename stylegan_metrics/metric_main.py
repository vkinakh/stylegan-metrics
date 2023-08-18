from typing import Dict
import time
import os
import json

import torch

from stylegan_metrics import metric_utils
from stylegan_metrics.frechet_inception_distance import compute_fid_directories
from stylegan_metrics.kernel_inception_distance import compute_kid_directories
from stylegan_metrics.precision_recall import compute_pr_directories
from stylegan_metrics.inception_score import compute_is_directory
from stylegan_metrics import dnnlib


_metric_dict = dict()  # name => fn


def register_metric(fn):
    assert callable(fn)
    _metric_dict[fn.__name__] = fn
    return fn


def is_valid_metric(metric):
    return metric in _metric_dict


def list_valid_metrics():
    return list(_metric_dict.keys())


def calc_metric(metric: str, path_real: str, path_fake: str, resolution: int, **kwargs):
    """Run a metric

    Args:
        metric: metric name
        path_real: path to the real images
        path_fake: path to the fake images
        resolution: resolution of the images
        **kwargs: other arguments

    Returns:
        dnnlib.EasyDict: results of the metric
    """

    assert is_valid_metric(metric)
    opts = metric_utils.MetricOptions(**kwargs)

    # Calculate.
    start_time = time.time()
    results = _metric_dict[metric](opts, path_real=path_real, path_fake=path_fake, resolution=resolution)
    total_time = time.time() - start_time

    # Broadcast results.
    for key, value in list(results.items()):
        if opts.num_gpus > 1:
            value = torch.as_tensor(value, dtype=torch.float64, device=opts.device)
            torch.distributed.broadcast(tensor=value, src=0)
            value = float(value.cpu())
        results[key] = value

    # Decorate with metadata.
    return dnnlib.EasyDict(
        results=dnnlib.EasyDict(results),
        metric=metric,
        total_time=total_time,
        total_time_str=dnnlib.util.format_time(total_time),
        num_gpus=opts.num_gpus,
    )


def report_metric(result_dict: Dict[str, float], run_dir: str = None) -> None:
    """Report a metric

    Args:
        result_dict: precomputed results
        run_dir: path to the run directory
    """

    metric = result_dict["metric"]
    assert is_valid_metric(metric)

    jsonl_line = json.dumps(dict(result_dict, timestamp=time.time()))
    print(jsonl_line)
    if run_dir is not None and os.path.isdir(run_dir):
        with open(os.path.join(run_dir, f"metric-{metric}.jsonl"), "at") as f:
            f.write(jsonl_line + "\n")


@register_metric
def fid(opts, path_real: str, path_fake: str, resolution: int) -> Dict[str, float]:
    fid_val = compute_fid_directories(opts, path_real, path_fake, resolution)
    return dict(fid=fid_val)


@register_metric
def kid(opts, path_real: str, path_fake: str, resolution: int) -> Dict[str, float]:
    kid_val = compute_kid_directories(
        opts, path_real, path_fake, resolution, num_subsets=100, max_subset_size=1000
    )
    return dict(kid=kid_val)


@register_metric
def pr(opts, path_real: str, path_fake: str, resolution: int) -> Dict[str, float]:
    precision, recall = compute_pr_directories(
        opts,
        path_real,
        path_fake,
        resolution,
        nhood_size=3,
        row_batch_size=10000,
        col_batch_size=10000,
    )
    return dict(precision=precision, recall=recall)


@register_metric
def inception_score(opts, path_real: str, path_fake: str, resolution: int) -> Dict[str, float]:
    is_val, is_std = compute_is_directory(opts, path_fake, resolution, 10)
    return dict(inception_score=is_val, inception_score_std=is_std)
