# PrioriFI
PrioriFI is an efficient fault injection method for edge neural networks.
In this repo, we demonstrate how to use PrioriFI on a few example edge neural networks defined in QKeras.

## Environment
We use `conda` to manage our environment. Install `miniconda` from [here](https://www.anaconda.com/docs/getting-started/miniconda/main) if you do not already have `conda` installed.

Then create the environment:
```bash
conda env create -f environment.yml
```

Activate the environment:
```bash
conda activate edge-nns
```

## Example
We present an example of how to use PrioriFI on three models (small, medium, and large) trained on the [Smart Pixel](https://iopscience.iop.org/article/10.1088/2632-2153/ad6a00/meta) dataset, representing high-energy physics data.

See the example [here](examples/smart-pixel/)


