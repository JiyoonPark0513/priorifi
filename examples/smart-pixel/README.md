# Smart Pixel 
The [Smart Pixel](https://iopscience.iop.org/article/10.1088/2632-2153/ad6a00/meta) dataset contains vectors representing 13 key features of high-energy particle clusters from sensor data. 
The goal of a neural network trained on Smart Pixel data is to classify the clusters into three classes of high momentum and low momentum particles.

Our Smart Pixel models feature two fully-connected layers of varying widths.

## Extract the dataset
To extract the dataset, run
```bash
tar -xzvf smart-pixel-dataset.tar.gz
```

## Run PrioriFI 
We provide a script `./scripts/run_priorifi.sh` that performs a more informed fault injection campaign using PrioriFI.

The script expects two arguments:
```bash
./scripts/run_priorifi.sh MODEL_INDEX CUDA_DEVICE
```

To run PrioriFI on the baseline medium model, run
```bash
./scripts/run_priorifi.sh 0 0
```

To run PrioriFI on the small model, run
```bash
./scripts/run_priorifi.sh 1 0
```

To run PrioriFI on the large model, run
```bash
./scripts/run_priorifi.sh 2 0
```

