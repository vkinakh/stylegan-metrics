# StyleGAN3 Metrics for Image Generators

Evaluate your image-generating models with the recommended metrics from the latest [StyleGAN3](https://github.com/NVlabs/stylegan3) implementation.

## Description

This repository offers a comprehensive toolkit to assess the quality and diversity of generated images. It encompasses the following metrics:

- Frechet Inception Distance (FID)
- Kernel Inception Distance (KID)
- Inception Score (IS)
- Precision
- Recall

Whether you're using a StyleGAN model or any other image generator, you can easily evaluate its performance by generating images, saving them, and executing the `calc_metrics` command.

## Installation

Set up the environment and install required dependencies using:

```bash
conda env create -f environment.yml
```

## Run Unittests
To ensure the functionality of the repository, execute the unittests:
```bash
python -m unittest discover tests/
```

## Run evaluation

Evaluate your generated images with the following script:

```bash
python calc_metrics.py --path_real /path/to/real/images \
                       --path_fake /path/to/generated/images \
                       --resolution <image resolution> \
                       --metrics fid,kid,pr,inception_score \
                       --gpus 1 \
                       --verbose True \
                       --results_folder ./results
```

**Note**: This script is compatible with individual image files in `.npy` format with the shape **(H, W, 3)**, as well as the standard `.png` and `.jpg` formats.

## Attribution
This repository does not claim ownership over any intellectual property related to StyleGAN3. It's a modified version 
of the [calc_metrics.py](https://github.com/NVlabs/stylegan3/blob/main/calc_metrics.py) script from the original 
StyleGAN3 implementation. The intent is to adapt it for broader use, allowing for evaluations across various image 
generators by simply generating and saving images to a directory.
