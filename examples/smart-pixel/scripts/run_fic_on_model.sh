#!/bin/bash

#STEP: Store current time
NOW=`date '+%Y_%m_%d__HMS_%H_%M_%S'`
NOW="YMD_${NOW}"

# STEP: Store args for use
MODEL_INDEX=$1
CUDA_DEVICE=$2
FIC_RANGE_STEP=$3
FIC_RANGE_START=$4

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

# STEP: Create a fault injection campaign directory
FIC_OUTPUT_DIR="./fic__dir__for__${MODEL_ID}__${NUM_FICs}__${NOW}"
mkdir $FIC_OUTPUT_DIR
printf "Created fault injection campaign directory:\n└─> ${FIC_OUTPUT_DIR}\n\n"

# FUNCTION:
# |-- Encapsulate launching one fic process in a new tmux session
# |-- First arg: fic range start (doubles as fic process id)
launch_fic_process(){
# tmux new-session -d -s "fic__p__${MODEL_ID}__${FIC_RANGE_START}__${FIC_RANGE_STEP}__${NOW}" "CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python dev_fic.py --config ${CONFIG} --pretrained-model ${PRETRAINED_MODEL} --model_id ${MODEL_ID}  --fic_output_dir ${FIC_OUTPUT_DIR} --fic_range_start ${FIC_RANGE_START} --fic_range_step ${FIC_RANGE_STEP}; sleep infinity"
# NOTE: Could not get tmux to work with edge-nns environment, so manually create tmux session, activating environment, then running script so that we can use the GPUs
CUDA_VISIBLE_DEVICES=${CUDA_DEVICE} python dev_fic.py --config ${CONFIG} --pretrained-model ${PRETRAINED_MODEL} --model_id ${MODEL_ID}  --fic_output_dir ${FIC_OUTPUT_DIR} --fic_range_start ${FIC_RANGE_START} --fic_range_step ${FIC_RANGE_STEP}
}


# STEP: Launch each python fic in the background via tmux
launch_fic_process 


# STEP: Output final message
printf "Monitor progress using log files in FIC directory and tmux consoles listed above."


# python3 dev_fic.py -c $CONFIG --pretrained-model $PRETRAINED_MODEL