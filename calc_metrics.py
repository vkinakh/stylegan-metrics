from typing import List
import click
import os
import tempfile

import torch

from stylegan_metrics import dnnlib
from stylegan_metrics import metric_utils
from stylegan_metrics import metric_main
from stylegan_metrics.torch_utils import training_stats
from stylegan_metrics.torch_utils import custom_ops
from stylegan_metrics.torch_utils.ops import conv2d_gradfix

# add ./stylegan_metrics to sys.path
import sys

sys.path.append("./stylegan_metrics")


def subprocess_fn(rank, args, temp_dir):
    dnnlib.util.Logger(should_flush=True)

    # Init torch.distributed.
    if args.num_gpus > 1:
        init_file = os.path.abspath(os.path.join(temp_dir, ".torch_distributed_init"))
        if os.name == "nt":
            init_method = "file:///" + init_file.replace("\\", "/")
            torch.distributed.init_process_group(
                backend="gloo",
                init_method=init_method,
                rank=rank,
                world_size=args.num_gpus,
            )
        else:
            init_method = f"file://{init_file}"
            torch.distributed.init_process_group(
                backend="nccl",
                init_method=init_method,
                rank=rank,
                world_size=args.num_gpus,
            )

    # Init torch_utils.
    sync_device = torch.device("cuda", rank) if args.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    if rank != 0 or not args.verbose:
        custom_ops.verbosity = "none"

    # Configure torch.
    device = torch.device("cuda", rank)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    conv2d_gradfix.enabled = True

    # Calculate each metric.
    for metric in args.metrics:
        if rank == 0 and args.verbose:
            print(f"Calculating {metric}...")
        progress = metric_utils.ProgressMonitor(verbose=args.verbose)
        result_dict = metric_main.calc_metric(
            metric=metric,
            path_real=args.path_real,
            path_fake=args.path_fake,
            resolution=args.resolution,
            num_gpus=args.num_gpus,
            rank=rank,
            device=device,
            progress=progress,
        )
        if rank == 0:
            metric_main.report_metric(result_dict, run_dir=args.run_dir)
        if rank == 0 and args.verbose:
            print()

    # Done.
    if rank == 0 and args.verbose:
        print("Exiting...")


def parse_comma_separated_list(s):
    if isinstance(s, list):
        return s
    if s is None or s.lower() == "none" or s == "":
        return []
    return s.split(",")


@click.command()
@click.pass_context
@click.option(
    "--path_real", type=str, help="Path to the folder with real images", required=True
)
@click.option(
    "--path_fake",
    type=str,
    help="Path to the folder with generated images",
    required=True,
)
@click.option("--resolution", type=int, help="Resolution of the images", required=True)
@click.option(
    "--metrics",
    help="Quality metrics",
    metavar="[NAME|A,B,C|none]",
    type=parse_comma_separated_list,
    default="fid",
    show_default=True,
)
@click.option("--gpus", type=int, help="Number of GPUs to use", default=1)
@click.option("--verbose", type=bool, help="Print optional information", default=True)
@click.option(
    "--result_folder", type=str, help="Folder to save results", default="./results"
)
def calc_metrics(
    ctx,
    path_real: str,
    path_fake: str,
    resolution: int,
    metrics: List[str],
    gpus: int,
    verbose: bool,
    result_folder: str,
) -> None:
    """Calculate quality metrics for generated images

    Args:
        ctx: Context
        path_real: Path to the folder with real images
        path_fake: Path to the folder with generated images
        resolution: Resolution of the images
        metrics: List of metrics to compute. Available metrics:
        gpus: Number of GPUs to use
        verbose: Print optional information
        result_folder: Folder to save results
    """

    dnnlib.util.Logger(should_flush=True)

    # Validate arguments.
    args = dnnlib.EasyDict(metrics=metrics, num_gpus=gpus, verbose=verbose)
    if not all(metric_main.is_valid_metric(metric) for metric in args.metrics):
        ctx.fail(
            "\n".join(
                ["--metrics can only contain the following values:"]
                + metric_main.list_valid_metrics()
            )
        )
    if not args.num_gpus >= 1:
        ctx.fail("--gpus must be at least 1")

    # create result folder
    args.run_dir = result_folder
    os.makedirs(args.run_dir, exist_ok=True)

    # data parameters
    args.path_real = path_real
    args.path_fake = path_fake
    args.resolution = resolution

    # Launch processes.
    if args.verbose:
        print("Launching processes...")
    torch.multiprocessing.set_start_method("spawn")
    with tempfile.TemporaryDirectory() as temp_dir:
        if args.num_gpus == 1:
            subprocess_fn(rank=0, args=args, temp_dir=temp_dir)
        else:
            torch.multiprocessing.spawn(
                fn=subprocess_fn, args=(args, temp_dir), nprocs=args.num_gpus
            )


if __name__ == "__main__":
    calc_metrics()
