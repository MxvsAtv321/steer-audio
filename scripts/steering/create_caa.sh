#!/bin/bash

AUDIO_DURATION=30.0
NUM_INFERENCE_STEPS=30
GUIDANCE_SCALE=3.0


CUDA_VISIBLE_DEVICES=1 python steering/ace_steer/compute_steering_vectors_sae.py --concept piano --audio_duration $AUDIO_DURATION --num_inference_steps $NUM_INFERENCE_STEPS --guidance_scale $GUIDANCE_SCALE &

CUDA_VISIBLE_DEVICES=2 python steering/ace_steer/compute_steering_vectors_sae.py --concept mood --audio_duration $AUDIO_DURATION --num_inference_steps $NUM_INFERENCE_STEPS --guidance_scale $GUIDANCE_SCALE &

CUDA_VISIBLE_DEVICES=3 python steering/ace_steer/compute_steering_vectors_sae.py --concept tempo --audio_duration $AUDIO_DURATION --num_inference_steps $NUM_INFERENCE_STEPS --guidance_scale $GUIDANCE_SCALE &

CUDA_VISIBLE_DEVICES=4 python steering/ace_steer/compute_steering_vectors_sae.py --concept female_vocals --audio_duration $AUDIO_DURATION --num_inference_steps $NUM_INFERENCE_STEPS --guidance_scale $GUIDANCE_SCALE &

wait
echo "All jobs finished."