#!/bin/bash

ACCOUNT="<SLURM_ACCOUNT>"
PARTITION="<SLURM_PARTITION>"
WORKDIR_PATH="${TADA_WORKDIR:-$HOME/tada_outputs}"
# BUG FIX: replaced hardcoded placeholder with env var — 2026-03-17
MODEL_NAME="audioldm2"

BLOCKS_TO_PATCH=("down1" "down2" "down3" "mid" "up0" "up1" "up1tf1" "up1tf2" "up1tf5" "up1tf6" "up1tf9" "up1tf10" "up2" "none" "all")
FEATURES=("happy" "sad" "bongos" "cello" "flute" "happy" "harmonica" "maracas" "reggae" "sad" "slow" "trombone" "violin" "trumpet" "xylophone")

ADDITIONAL_NAME="all"

for feature in "${FEATURES[@]}"; do
    log_file="slurm_out/${MODEL_NAME}/${feature}/eval.log"
    job_name="eval_${MODEL_NAME}_${feature}"

    sbatch --output=$log_file --job-name=$job_name <<EOT
#!/bin/bash -l
#SBATCH --account=$ACCOUNT
#SBATCH --partition=$PARTITION
#SBATCH -t 03:00:00
#SBATCH --ntasks 1
#SBATCH --gres gpu:1
#SBATCH --mem 100G
#SBATCH --cpus-per-task=60
#SBATCH --nodes 1


cd $WORKDIR_PATH
export PYTHONPATH=$PWD
source ./venv/bin/activate

echo USER: $USER
which python

sh ./scripts/localization/eval_patching/eval_feature_audioldm2.sh $feature "metrics_${feature}_${ADDITIONAL_NAME}.csv" ${BLOCKS_TO_PATCH[@]}


EOT
done
