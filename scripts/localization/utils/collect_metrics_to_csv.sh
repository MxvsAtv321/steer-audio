#!/bin/bash

ADDITIONAL_NAME="patch_all"
MODEL_NAME="ace"
BLOCKS_TO_PATCH=("none" "tf0" "tf1" "tf2" "tf3" "tf4" "tf5" "tf6" "tf7" "tf8" "tf9" "tf10" "tf11" "tf12" "tf13" "tf14" "tf15" "tf16" "tf17" "tf18" "tf19" "tf20" "tf21" "tf22" "tf23" "all")
FEATURES=("drums" "fast" "female" "happy" "male" "sad" "slow" "violin")

for FEATURE in "${FEATURES[@]}"; do
    FILENAME="metrics_${FEATURE}_${ADDITIONAL_NAME}.csv"
    python src/postprocess/collect_feature_metrics.py --feature "${FEATURE}" --blocks "${BLOCKS_TO_PATCH[@]}" --filename "${FILENAME}" --model_name "${MODEL_NAME}" --localization patching
done
