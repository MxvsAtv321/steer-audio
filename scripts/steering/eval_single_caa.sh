#!/bin/bash

set -e

STEER_MODE="cond_only"

CONCEPT="mood"

LAYERS="all"
CUDA_VISIBLE_DEVICES=0 python steering/ace_steer/eval_steering_vectors.py --sv_path "steering_vectors/ace_${CONCEPT}_passes2_allTrue" --concept $CONCEPT --layers $LAYERS --steer_mode $STEER_MODE > logs/${CONCEPT}_${LAYERS}.log 2>&1 &

LAYERS="tf6tf7"
CUDA_VISIBLE_DEVICES=1 python steering/ace_steer/eval_steering_vectors.py --sv_path "steering_vectors/ace_${CONCEPT}_passes2_allTrue" --concept $CONCEPT --layers $LAYERS --steer_mode $STEER_MODE > logs/${CONCEPT}_${LAYERS}.log 2>&1 &

LAYERS="no_tf6tf7"
CUDA_VISIBLE_DEVICES=2 python steering/ace_steer/eval_steering_vectors.py --sv_path "steering_vectors/ace_${CONCEPT}_passes2_allTrue" --concept $CONCEPT --layers $LAYERS --steer_mode $STEER_MODE > logs/${CONCEPT}_${LAYERS}.log 2>&1 &

CONCEPT="tempo"

LAYERS="all"
CUDA_VISIBLE_DEVICES=3 python steering/ace_steer/eval_steering_vectors.py --sv_path "steering_vectors/ace_${CONCEPT}_passes2_allTrue" --concept $CONCEPT --layers $LAYERS --steer_mode $STEER_MODE > logs/${CONCEPT}_${LAYERS}.log 2>&1 &

LAYERS="tf6tf7"
CUDA_VISIBLE_DEVICES=4 python steering/ace_steer/eval_steering_vectors.py --sv_path "steering_vectors/ace_${CONCEPT}_passes2_allTrue" --concept $CONCEPT --layers $LAYERS --steer_mode $STEER_MODE > logs/${CONCEPT}_${LAYERS}.log 2>&1 &

LAYERS="no_tf6tf7"
CUDA_VISIBLE_DEVICES=5 python steering/ace_steer/eval_steering_vectors.py --sv_path "steering_vectors/ace_${CONCEPT}_passes2_allTrue" --concept $CONCEPT --layers $LAYERS --steer_mode $STEER_MODE > logs/${CONCEPT}_${LAYERS}.log 2>&1 &

CONCEPT="female_vocals"

LAYERS="all"
CUDA_VISIBLE_DEVICES=6 python steering/ace_steer/eval_steering_vectors.py --sv_path "steering_vectors/ace_${CONCEPT}_passes2_allTrue" --concept $CONCEPT --layers $LAYERS --steer_mode $STEER_MODE > logs/${CONCEPT}_${LAYERS}.log 2>&1 &

LAYERS="tf6tf7"
CUDA_VISIBLE_DEVICES=7 python steering/ace_steer/eval_steering_vectors.py --sv_path "steering_vectors/ace_${CONCEPT}_passes2_allTrue" --concept $CONCEPT --layers $LAYERS --steer_mode $STEER_MODE > logs/${CONCEPT}_${LAYERS}.log 2>&1 &

wait
echo "All jobs finished."