#!/bin/bash

PATH_PREFIX=./configs

if [ $1 -eq 0 ]; then
    CONFIG=$PATH_PREFIX/baseline_fkeras.yml
elif [ $1 -eq 1 ]; then
    CONFIG=$PATH_PREFIX/small_fkeras.yml
elif [ $1 -eq 2 ]; then
    CONFIG=$PATH_PREFIX/large_fkeras.yml
elif [ $1 -eq 3 ]; then
    CONFIG=$PATH_PREFIX/large2_fkeras.yml
else
    echo "Error"
fi


python3 train.py -c $CONFIG
