#!/bin/bash

ACCOUNT="<SLURM_ACCOUNT>"
PARTITION="<SLURM_PARTITION>"
WORKDIR_PATH="${TADA_WORKDIR:-$HOME/tada_outputs}"
# BUG FIX: replaced hardcoded placeholder with env var — 2026-03-17
OUTPUTS_PATH="./outputs"
MODEL_NAME="audioldm2"

LAYERS_TO_PATCH=("down1" "down2" "down3" "mid" "up0" "up1" "up2" "none" "all" "up1tf1" "up1tf2" "up1tf5" "up1tf6" "up1tf9" "up1tf10" "up1" "all" "up1tf5attn0" "up1tf5attn1" "up1tf5attn0tf10attn0" "up1tf5tf10" "up1tf10attn0" "up1tf10attn1")

FEATURES=("bongos" "cello" "flute" "happy" "harmonica" "maracas" "reggae" "sad" "slow" "trombone" "violin" "trumpet" "xylophone")

main_process_port=12345
for feature in "${FEATURES[@]}"; do
    for block in "${LAYERS_TO_PATCH[@]}"; do
        main_process_port=$((main_process_port + 1))
        log_file="slurm_out/${MODEL_NAME}/${feature}/patch_${block}.log"
        job_name="${MODEL_NAME}_${feature}_block_${block}"
        output_dir="${OUTPUTS_PATH}/${MODEL_NAME}/patching/${feature}/${block}"
        sleep 1
        sbatch --output=$log_file --job-name=$job_name <<EOT
#!/bin/bash -l
#SBATCH --account=$ACCOUNT
#SBATCH --partition=$PARTITION
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=60
#SBATCH --mem=400G
#SBATCH --time=2:00:00


cd $WORKDIR_PATH
export PYTHONPATH=$PWD
source ./venv/bin/activate

echo USER: $USER
which python

accelerate launch --num_processes 4 --main_process_port $main_process_port src/patch_layers.py experiment=patch_${MODEL_NAME}/${MODEL_NAME}_$feature patch_layers=${MODEL_NAME}/$block patch_config.path_with_results=$output_dir
EOT
    done
done
