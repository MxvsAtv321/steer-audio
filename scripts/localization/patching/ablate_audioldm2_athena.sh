#!/bin/bash

ACCOUNT=$GRANT_ACCOUNT
PARTITION=$GRANT_PARTITION
WORKDIR_PATH="${TADA_WORKDIR:-$HOME/tada_outputs}"
# BUG FIX: replaced hardcoded placeholder with env var — 2026-03-17
OUTPUTS_PATH="./outputs"
MODEL_NAME="audioldm2"
N_GPUS=4
N_CPUS=16
MEM=$((N_GPUS * 120))

LAYERS_TO_PATCH=("none" "down01attn00" "down01attn01" "down01attn02" "down01attn03" "down01attn04" "down01attn05" "down01attn06" "down01attn07" "down02attn00" "down02attn01" "down02attn02" "down02attn03" "down02attn04" "down02attn05" "down02attn06" "down02attn07" "down03attn00" "down03attn01" "down03attn02" "down03attn03" "down03attn04" "down03attn05" "down03attn06" "down03attn07" "mid00attn00" "mid00attn01" "mid00attn02" "mid00attn03" "up00attn00" "up00attn01" "up00attn02" "up00attn03" "up00attn04" "up00attn05" "up00attn06" "up00attn07" "up00attn08" "up00attn09" "up00attn10" "up00attn11" "up01attn00" "up01attn01" "up01attn02" "up01attn03" "up01attn04" "up01attn05" "up01attn06" "up01attn07" "up01attn08" "up01attn09" "up01attn10" "up01attn11" "up02attn00" "up02attn01" "up02attn02" "up02attn03" "up02attn04" "up02attn05" "up02attn06" "up02attn07" "up02attn08" "up02attn09" "up02attn10" "up02attn11")

FEATURES=("1280")

main_process_port=12345
for feature in "${FEATURES[@]}"; do
    for block in "${LAYERS_TO_PATCH[@]}"; do
        main_process_port=$((main_process_port + 1))
        log_file="slurm_out/${MODEL_NAME}/${feature}/ablate/blocks/${block}.log"
        job_name="${MODEL_NAME}_${feature}_block_${block}"
        output_dir="${OUTPUTS_PATH}/${MODEL_NAME}/ablate/${feature}/${block}"
        sleep 1
        sbatch --output=$log_file --job-name=$job_name <<EOT
#!/bin/bash
#SBATCH --account=$ACCOUNT
#SBATCH --partition=$PARTITION
#SBATCH --nodes=1
#SBATCH --gres=gpu:${N_GPUS}
#SBATCH --ntasks=${N_GPUS}
#SBATCH --cpus-per-task=${N_CPUS}
#SBATCH --mem=${MEM}G
#SBATCH --time=4:00:00

eval "\$(conda shell.bash hook)"

conda activate music

cd $WORKDIR_PATH
source .env


echo "--------------------------------"
echo "N_GPUS: $N_GPUS"
echo "SLURM_GPUS_ON_NODE: $SLURM_GPUS_ON_NODE"
echo "SLURM_NTASKS: $SLURM_NTASKS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "--------------------------------"
nvidia-smi
accelerate env
echo "--------------------------------"
echo USER: $USER
which python
echo "--------------------------------"

accelerate launch --num_processes ${N_GPUS} --main_process_port ${main_process_port} src/patch_layers.py experiment=ablate_${MODEL_NAME}/${MODEL_NAME}_$feature patch_layers=${MODEL_NAME}/self_attn/blocks/$block patch_config.path_with_results=$output_dir
EOT
    done
done
