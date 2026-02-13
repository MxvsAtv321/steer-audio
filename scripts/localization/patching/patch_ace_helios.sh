#!/bin/bash

ACCOUNT="<SLURM_ACCOUNT>"
PARTITION="<SLURM_PARTITION>"
WORKDIR_PATH="<WORKDIR_PATH>"
OUTPUTS_PATH="./outputs"
MODEL_NAME="ace"

LAYERS_TO_PATCH=("none" "tf0" "tf1" "tf2" "tf3" "tf4" "tf5" "tf6" "tf7" "tf8" "tf9" "tf10" "tf11" "tf12" "tf13" "tf14" "tf15" "tf16" "tf17" "tf18" "tf19" "tf20" "tf21" "tf22" "tf23")

FEATURES=("drums" "fast" "female" "happy" "male" "sad" "slow" "violin")

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
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=60
#SBATCH --mem=400G
#SBATCH --time=5:00:00

cd $WORKDIR_PATH
export PYTHONPATH=$PWD
export PYTHONPATH=$PYTHONPATH:$PWD/src/models/ace_step/ACE
source ./venv/bin/activate

echo USER: $USER
which python

accelerate launch --num_processes 4 --main_process_port $main_process_port src/patch_layers.py experiment=patch_${MODEL_NAME}/${MODEL_NAME}_$feature patch_layers=${MODEL_NAME}/$block patch_config.path_with_results=$output_dir patch_config.save_clean_audio=$save_clean_audio
EOT
    done
done