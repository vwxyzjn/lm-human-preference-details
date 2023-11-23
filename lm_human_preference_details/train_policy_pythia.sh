# generate random seed and model paths
# set seed if not found in env
if [ -z "$SEED" ]; then
    SEED=$RANDOM
fi
if [ -z "$MODEL" ]; then
    MODEL=EleutherAI/pythia-1b-deduped
fi
# SEED=3131
# MODEL=EleutherAI/pythia-1b-deduped
REWARD_MODEL_PATH=models/$MODEL/reward_model_$SEED
SFT_MODEL_PATH=models/$MODEL/sft_model_$SEED
POLICY_MODEL_PATH=models/$MODEL/policy_model_$SEED
poetry run accelerate launch --config_file deepspeed.yaml \
    lm_human_preference_details/train_sft_accelerate_summarize.py \
    --task.query_dataset=vwxyzjn/summarize_from_feedback_tldr_3_filtered_oai_preprocessing_pythia-160m_48 \
    --base_model=$MODEL \
    --deepspeed \
    --track \
    --upload_model \
    --save_path=$SFT_MODEL_PATH \
    --seed=$SEED \

poetry run accelerate launch --config_file deepspeed.yaml \
    lm_human_preference_details/train_reward_accelerate_summarize.py \
    --label_dataset=vwxyzjn/summarize_from_feedback_oai_preprocessing_pythia-160m_48 \
    --base_model=$MODEL \
    --no_normalize_before --no_normalize_after \
    --local_batch_size=8 \
    --gradient_accumulation_steps=8 \
    --labels.num_train=92832 \
    --deepspeed \
    --track \
    --sft_model_path=$SFT_MODEL_PATH/pytorch_model.bin \
    --save_path=$REWARD_MODEL_PATH \
    --seed=$SEED \

# poetry run accelerate launch --config_file deepspeed.yaml \
#     lm_human_preference_details/train_policy_accelerate_summarize_separate.py \
#     --base_model=$MODEL \
#     --rewards.no_use_adaptive_kl \
#     --rewards.kl_coef=0.05 \
#     --ppo.gradient_accumulation_steps=64 \
#     --ppo.lr=1.5e-5 \
#     --task.temperature=0.7 \
#     --deepspeed \
#     --track \
#     --sft_model_path=$SFT_MODEL_PATH/pytorch_model.bin \
#     --rewards.trained_model=$REWARD_MODEL_PATH/pytorch_model.bin \
#     --seed=$SEED \
#     --save_path=$POLICY_MODEL_PATH \