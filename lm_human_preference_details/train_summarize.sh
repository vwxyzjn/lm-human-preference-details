# generate random seed and model paths
# set seed if not found in env
if [ -z "$SEED" ]; then
    SEED=$RANDOM
fi

REWARD_MODEL_PATH=models/reward_model_$SEED
SFT_MODEL_PATH=models/sft_model_$SEED
poetry run accelerate launch --config_file deepspeed.yaml \
    lm_human_preference_details/train_reward_accelerate_summarize.py \
    --base_model=gpt2-xl \
    --local_rollout_batch_size=4 \
    --gradient_accumulation_steps=4 \
    --save_path=$REWARD_MODEL_PATH \
    --labels.num_train=92832 \
    --seed=$SEED \
    --deepspeed \
    --track 

poetry run accelerate launch --config_file deepspeed.yaml \
    --num_processes 8 lm_human_preference_details/train_sft_accelerate_summarize.py \
    --base_model=gpt2-xl \
    --save_path=$SFT_MODEL_PATH \
    --seed=$SEED \
    --deepspeed \
    --track

poetry run accelerate launch --config_file deepspeed.yaml \
    lm_human_preference_details/train_policy_accelerate_summarize_separate.py \
    --base_model=gpt2-xl \
    --sft_model_path=$SFT_MODEL_PATH/pytorch_model.bin \
    --ppo.gradient_accumulation_steps=64 \
    --ppo.lr=1.5e-5 \
    --rewards.kl_coef=0.05 \
    --rewards.no_use_adaptive_kl \
    --rewards.trained_model=$REWARD_MODEL_PATH/pytorch_model.bin \
    --task.temperature=1.0 \
    --seed=$SEED \
    --deepspeed \
    --track \