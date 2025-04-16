#!/bin/bash

MODEL_INDEX=$1

if [ $MODEL_INDEX -eq 0 ]; then
    CONFIG=./dense_baseline_fkeras/baseline_fkeras.yml
    PRETRAINED_MODEL=./dense_baseline_fkeras/fkeras_dense_model_58.h5
    MODEL_ID=dense-baseline
elif [ $MODEL_INDEX -eq 1 ]; then
    CONFIG=./dense_small_fkeras/small_fkeras.yml
    PRETRAINED_MODEL=./dense_small_fkeras/fkeras_dense_model_16.h5
    MODEL_ID=dense-small
elif [ $MODEL_INDEX -eq 2 ]; then
    CONFIG=./dense_large_fkeras/large_fkeras.yml
    PRETRAINED_MODEL=./dense_large_fkeras/fkeras_dense_model_large_32.h5
    MODEL_ID=dense-large
elif [ $MODEL_INDEX -eq 3 ]; then
    CONFIG=./dense_large2_fkeras/large2_fkeras.yml
    PRETRAINED_MODEL=./dense_large2_fkeras/fkeras_dense_model_512.h5
    MODEL_ID=dense-large2
else
    echo "Error"
fi


CUDA_VISIBLE_DEVICES="" python hessian_analysis.py -c $CONFIG --pretrained-model $PRETRAINED_MODEL
