#!/bin/bash

ACCOUNT="<SLURM_ACCOUNT>"
PARTITION="<SLURM_PARTITION>"
WORKDIR_PATH="<WORKDIR_PATH>"
MODEL_NAME="stableaudio"

BLOCKS_TO_PATCH=("tf0" "tf1" "tf2" "tf3" "tf4" "tf5" "tf6" "tf7" "tf8" "tf9" "tf10" "tf11" "tf12" "tf13" "tf14" "tf15" "tf16" "tf17" "tf18" "tf19" "tf20" "tf21" "tf22" "tf23" "none")
FEATURES=("1280")
ADDITIONAL_NAME="ablate_all"

for feature in "${FEATURES[@]}"; do
    log_file="slurm_out/${MODEL_NAME}/${feature}/eval.log"
    job_name="eval_${MODEL_NAME}_${feature}"

    sbatch --output=$log_file --job-name=$job_name <<EOT
#!/bin/bash -l
#SBATCH --account=$ACCOUNT
#SBATCH --partition=$PARTITION
#SBATCH -t 04:00:00
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

sh ./scripts/localization/eval_patching/eval_ablate_stableaudio.sh $feature "metrics_${feature}_${ADDITIONAL_NAME}.csv" ${BLOCKS_TO_PATCH[@]}


EOT
done