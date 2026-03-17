#!/bin/bash

ACCOUNT="<SLURM_ACCOUNT>"
PARTITION="<SLURM_PARTITION>"
WORKDIR_PATH="${TADA_WORKDIR:-$HOME/tada_outputs}"
# BUG FIX: replaced hardcoded placeholder with env var — 2026-03-17
OUTPUTS_PATH="./outputs"
MODEL_NAME="ace"
N_GPUS=2
N_CPUS=16
MEM=$((N_GPUS * 120))

LAYERS_TO_PATCH=("none" "tf0" "tf1" "tf2" "tf3" "tf4" "tf5" "tf6" "tf7" "tf8" "tf9" "tf10" "tf11" "tf12" "tf13" "tf14" "tf15" "tf16" "tf17" "tf18" "tf19" "tf20" "tf21" "tf22" "tf23")

FEATURES=("medley_genre" "medley_instr")

main_process_port=12345
for feature in "${FEATURES[@]}"; do
    for block in "${LAYERS_TO_PATCH[@]}"; do
        main_process_port=$((main_process_port + 1))
        log_file="slurm_out/${MODEL_NAME}/${feature}/patch_${block}.log"
        job_name="${MODEL_NAME}_${feature}_block_${block}"
        output_dir="${OUTPUTS_PATH}/${MODEL_NAME}/patching/${feature}/${block}"
        if [ "$block" == "none" ]; then
            save_clean_audio=true
        else
            save_clean_audio=false
        fi
        sleep 1
        sbatch --output=$log_file --job-name=$job_name <<EOT
#!/bin/bash -l
#SBATCH --account=$ACCOUNT
#SBATCH --partition=$PARTITION
#SBATCH --nodes=1
#SBATCH --gres=gpu:${N_GPUS}
#SBATCH --ntasks=${N_GPUS}
#SBATCH --cpus-per-task=${N_CPUS}
#SBATCH --mem=${MEM}G
#SBATCH --time=1:30:00


eval "\$(conda shell.bash hook)"

conda activate music

cd $WORKDIR_PATH
source .env

echo "--------------------------------"
pwd;hostname;date
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
echo PYTHONPATH: $PYTHONPATH
echo "--------------------------------"

accelerate launch --num_processes ${N_GPUS} --main_process_port ${main_process_port} src/patch_layers.py experiment=patch_${MODEL_NAME}/${MODEL_NAME}_$feature patch_layers=${MODEL_NAME}/$block patch_config.path_with_results=$output_dir patch_config.save_clean_audio=$save_clean_audio
EOT
    done
done