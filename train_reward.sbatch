#!/bin/bash
#SBATCH --partition=hopper-prod
#SBATCH --gpus-per-task=8
#SBATCH --cpus-per-gpu=10
#SBATCH --ntasks=1
#SBATCH --output=slurm/logs/%x_%j.out
#SBATCH --exclusive

module load cuda/12.2 

if [ -z "$SEED" ]; then
    SEED=1
fi
if [ -z "$MODEL" ]; then
    MODEL=EleutherAI/pythia-2.8b-deduped
fi
if [ -z "$LR" ]; then
    LR=3e-6
fi

REWARD_MODEL_PATH=models/$MODEL/reward_model_$SEED
SFT_MODEL_PATH=models/$MODEL/sft_model_$SEED
POLICY_MODEL_PATH=models/$MODEL/policy_model_$SEED
srun poetry run accelerate launch --config_file deepspeed.yaml \
    lm_human_preference_details/summarize/reward.py \
    --label_dataset=vwxyzjn/summarize_from_feedback_oai_preprocessing_pythia-160m_169 \
    --base_model=$SFT_MODEL_PATH \
    --lr=$LR \
    --deepspeed \
    --run_eval \
    --push_to_hub \
    --track \
    --output_dir=$REWARD_MODEL_PATH \
    --seed=$SEED
 