#!/bin/bash

ACCOUNT="<SLURM_ACCOUNT>"
PARTITION="<SLURM_PARTITION>"
WORKDIR_PATH="<WORKDIR_PATH>"
OUTPUTS_PATH="./outputs"
MODEL_NAME="audioldm2"
N_GPUS=2
N_CPUS=16
MEM=$((N_GPUS * 120))

LAYERS_TO_PATCH=("down1" "down2" "down3" "mid" "up0" "up1" "up1tf1" "up1tf2" "up1tf5" "up1tf6" "up1tf9" "up1tf10" "up1tf5attn0" "up1tf5attn1" "up1tf10attn0" "up1tf10attn1" "up2" "none" "all")
FEATURES=("bongos" "cello" "flute" "happy" "harmonica" "trombone" "violin" "trumpet" "xylophone" "maracas" "reggae" "sad" "slow" "jazz")

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
#SBATCH --gres=gpu:${N_GPUS}
#SBATCH --ntasks=${N_GPUS}
#SBATCH --cpus-per-task=${N_CPUS}
#SBATCH --mem=${MEM}G
#SBATCH --time=1:00:00

eval "\$(conda shell.bash hook)"

conda activate music

cd $WORKDIR_PATH
source .env
export PATH=$SCRATCH/envs/music/bin:$PATH

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

accelerate launch --num_processes ${N_GPUS} --main_process_port $main_process_port src/patch_layers.py experiment=patch_${MODEL_NAME}/${MODEL_NAME}_$feature patch_layers=${MODEL_NAME}/$block patch_config.path_with_results=$output_dir
EOT
    done
done
