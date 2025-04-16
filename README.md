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

## Examples
Each example features small, medium, and large versions of the model. 

### Smart Pixel
The [Smart Pixel](https://iopscience.iop.org/article/10.1088/2632-2153/ad6a00/meta) dataset contains vectors representing 13 key features of high-energy particle clusters from sensor data. 
The goal of a neural network trained on Smart Pixel data is to classify the clusters into three classes of high momentum and low momentum particles.
Our Smart Pixel models feature two fully-connected layers.

### ECON-T
The [ECON-T](https://arxiv.org/pdf/2105.01683) model is an autoencoder used at the Large Hadron Collider's (LHC) Compact Muon Solenoid (CMS) experiment for compressing physics sensor data generated at the High Granularity Calorimeter (HGCal).

### CIFAR-10
This convolutional neural network classifies images provided by the CIFAR-10 dataset. These models are based on the [MLPerf Tiny Benchmark](https://github.com/mlcommons/tiny/tree/master).


<!-- Link each of the above to respective example dir -->