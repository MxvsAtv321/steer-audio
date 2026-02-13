#!/bin/bash

FEATURE=$1
FILENAME=$2
shift 2

BLOCKS_TO_PATCH=( "$@" )
MODEL_NAME="stableaudio"

for block in "${BLOCKS_TO_PATCH[@]}"; do
    patch_data="musiccaps/${FEATURE}"
    echo "Evaluating ${FEATURE} ${block}"

    python src/eval_audio.py paths.generated_samples="${MODEL_NAME}/patching/${FEATURE}/${block}/audios/patched.npy" paths.reference_samples="${MODEL_NAME}/patching/${FEATURE}/${block}/audios/clean.npy" patch_data="${patch_data}" metrics.alignment=false patch_model="${MODEL_NAME}_patch.yaml"
done

python src/postprocess/collect_feature_metrics.py --feature "${FEATURE}" --blocks "${BLOCKS_TO_PATCH[@]}" --filename "${FILENAME}" --model_name "${MODEL_NAME}" --localization "patching"