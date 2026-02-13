#!/bin/bash

CUDA_DEVICE_ID=0
DATASET_PATH="/data/lstaniszewski/code/audio-interv/sae/activations/music_caps.csv/ace-step"

for LR in 0.000005 0.00001 0.00005 0.0001 0.0005 0.001 0.005; do
    for K in 16 24 32; do
        for EXPANSION_FACTOR in 2 4 8; do
            for NUM_EPOCHS in 5 10; do
                CUDA_VISIBLE_DEVICES=$CUDA_DEVICE_ID python sae_src/scripts/train_ace.py --dataset_path $DATASET_PATH --hookpoints transformer_blocks.7.cross_attn --lr $LR --k $K --expansion_factor $EXPANSION_FACTOR --num_epochs $NUM_EPOCHS
            done
        done
    done
done