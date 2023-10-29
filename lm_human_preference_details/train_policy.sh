# generate random seed and model paths
# set seed if not found in env
if [ -z "$SEED" ]; then
    SEED=$RANDOM
fi
# SEED=1
REWARD_MODEL_PATH=models/gpt2-large_reward_model_$SEED
SFT_MODEL_PATH=models/gpt2-large_sft_model_$SEED
POLICY_MODEL_PATH=models/gpt2-large_policy_model_$SEED
poetry run accelerate launch --config_file deepspeed.yaml \
    lm_human_preference_details/train_sft_accelerate_summarize.py \
    --base_model=gpt2-large \
    --deepspeed \
    --track \
    --upload_model \
    --save_path=$SFT_MODEL_PATH \
    --seed=$SEED \

poetry run accelerate launch --config_file deepspeed.yaml \
    lm_human_preference_details/train_reward_accelerate_summarize.py \
    --base_model=gpt2-large \
    --no_normalize_before --no_normalize_after \
    --local_batch_size=8 \
    --gradient_accumulation_steps=8 \
    --labels.num_train=92832 \
    --deepspeed \
    --track \
    --sft_model_path=$SFT_MODEL_PATH/pytorch_model.bin \
    --seed=$SEED \
    --save_path=$REWARD_MODEL_PATH \

poetry run accelerate launch --config_file deepspeed.yaml \
    lm_human_preference_details/train_policy_accelerate_summarize_separate.py \
    --base_model=gpt2-large \
    --rewards.no_use_adaptive_kl \
    --rewards.kl_coef=0.05 \
    --ppo.gradient_accumulation_steps=64 \
    --ppo.lr=1.5e-5 \
    --seed=3 \
    --task.temperature=0.7 \
    --deepspeed \
    --track \
    --upload_model \
    --sft_model_path=$SFT_MODEL_PATH/pytorch_model.bin \
    --rewards.trained_model=$REWARD_MODEL_PATH/pytorch_model.bin \
    --save_path=$POLICY_MODEL_PATH \