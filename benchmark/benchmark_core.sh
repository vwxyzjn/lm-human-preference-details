export WANDB_ENTITY=openrlbenchmark
# sentiment
WANDB_TAGS="tf_adam,gpt2" python benchmark/benchmark.py \
    --command "accelerate launch --multi_gpu --num_processes 8 lm_human_preference_details/train_both_accelerate.py --reward.exp_name=train_reward_accelerate_tf_adam_gpt2 --policy.exp_name=train_policy_accelerate_tf_adam_gpt2 --reward.track --reward.wandb_project_name=lm_human_preference_details --policy.track --policy.wandb_project_name=lm_human_preference_details --policy.hf_entity=lm-human-preference-details --policy.upload_model" \
    --num-seeds 10 \
    --start-seed 1 \
    --workers 10 \
    --slurm-gpus-per-task 8 \
    --slurm-ntasks 1 \
    --slurm-total-cpus 64 \
    --slurm-template-path benchmark/trl.slurm_template
# descriptiveness
WANDB_TAGS="tf_adam,gpt2" python benchmark/benchmark.py \
    --command "accelerate launch --multi_gpu --num_processes 8 lm_human_preference_details/train_both_accelerate.py --reward.exp_name=train_reward_accelerate_tf_adam_gpt2 --policy.exp_name=train_policy_accelerate_tf_adam_gpt2 --reward.track --reward.wandb_project_name=lm_human_preference_details --reward.label_dataset=descriptiveness/offline_5k.json --policy.track --policy.wandb_project_name=lm_human_preference_details --policy.hf_entity=lm-human-preference-details --policy.upload_model" \
    --num-seeds 10 \
    --start-seed 1 \
    --workers 10 \
    --slurm-gpus-per-task 8 \
    --slurm-ntasks 1 \
    --slurm-total-cpus 64 \
    --slurm-template-path benchmark/trl.slurm_template

# ablation on gradient accumulation
WANDB_TAGS="tf_adam,gpt2_grad_accu" python benchmark/benchmark.py \
    --command "accelerate launch --multi_gpu --num_processes 8 lm_human_preference_details/train_both_accelerate.py --reward.exp_name=train_reward_accelerate_tf_adam_gpt2_grad_accu --policy.exp_name=train_policy_accelerate_tf_adam_gpt2_grad_accu --reward.rollout_batch_size=16 --reward.track --reward.wandb_project_name=lm_human_preference_details --reward.label_dataset=descriptiveness/offline_5k.json --policy.ppo.gradient_accumulation_steps=64 --policy.track --policy.wandb_project_name=lm_human_preference_details --policy.hf_entity=lm-human-preference-details --policy.upload_model" \
    --num-seeds 5 \
    --start-seed 1 \
    --workers 10 \
    --slurm-gpus-per-task 8 \
    --slurm-ntasks 1 \
    --slurm-total-cpus 64 \
    --slurm-template-path benchmark/trl.slurm_template

# ablation on adam optimizers
WANDB_TAGS="pt_adam,gpt2" python benchmark/benchmark.py \
    --command "accelerate launch --multi_gpu --num_processes 8 lm_human_preference_details/train_both_accelerate.py --reward.exp_name=train_reward_accelerate_pt_adam_gpt2 --policy.exp_name=train_policy_accelerate_pt_adam_gpt2 --reward.no_use_tensorflow_adam --reward.track --reward.wandb_project_name=lm_human_preference_details --reward.label_dataset=descriptiveness/offline_5k.json --policy.no_use_tensorflow_adam --policy.track --policy.wandb_project_name=lm_human_preference_details --policy.hf_entity=lm-human-preference-details --policy.upload_model" \
    --num-seeds 10 \
    --start-seed 1 \
    --workers 4 \
    --slurm-gpus-per-task 8 \
    --slurm-ntasks 1 \
    --slurm-total-cpus 64 \
    --slurm-template-path benchmark/trl.slurm_template
WANDB_TAGS="tf_adam,gpt2-xl" python benchmark/benchmark.py \
    --command "accelerate launch --multi_gpu --num_processes 8 lm_human_preference_details/train_both_accelerate.py --reward.exp_name=train_reward_accelerate_tf_adam_gpt2_xl_grad_accu --policy.exp_name=train_policy_accelerate_tf_adam_gpt2_xl_grad_accu --reward.base_model=gpt2-xl --reward.rollout_batch_size=128 --reward.track --reward.wandb_project_name=lm_human_preference_details --reward.label_dataset=descriptiveness/offline_5k.json --policy.base_model=gpt2-xl --policy.ppo.gradient_accumulation_steps=4 --policy.track --policy.wandb_project_name=lm_human_preference_details --policy.hf_entity=lm-human-preference-details --policy.upload_model" \
    --num-seeds 5 \
    --start-seed 1 \
    --workers 4 \
    --slurm-gpus-per-task 8 \
    --slurm-ntasks 1 \
    --slurm-total-cpus 64 \
    --slurm-template-path benchmark/trl.slurm_template
WANDB_TAGS="pt_adam,gpt2-xl" python benchmark/benchmark.py \
    --command "accelerate launch --multi_gpu --num_processes 8 lm_human_preference_details/train_both_accelerate.py --reward.exp_name=train_reward_accelerate_pt_adam_gpt2_xl_grad_accu --policy.exp_name=train_policy_accelerate_pt_adam_gpt2_xl_grad_accu --reward.base_model=gpt2-xl --reward.rollout_batch_size=128 --reward.no_use_tensorflow_adam --reward.track --reward.wandb_project_name=lm_human_preference_details --reward.label_dataset=descriptiveness/offline_5k.json --policy.base_model=gpt2-xl --policy.ppo.gradient_accumulation_steps=4 --policy.no_use_tensorflow_adam --policy.track --policy.wandb_project_name=lm_human_preference_details --policy.hf_entity=lm-human-preference-details --policy.upload_model" \
    --num-seeds 5 \
    --start-seed 1 \
    --workers 5 \
    --slurm-gpus-per-task 8 \
    --slurm-ntasks 1 \
    --slurm-total-cpus 64 \
    --slurm-template-path benchmark/trl.slurm_template


# ablation on different models
WANDB_TAGS="tf_adam,Cerebras-GPT-111M" python benchmark/benchmark.py \
    --command "accelerate launch --multi_gpu --num_processes 8 lm_human_preference_details/train_both_accelerate.py --reward.exp_name=train_reward_accelerate_tf_adam_cerebras_gpt_111M --policy.exp_name=train_policy_accelerate_tf_adam_cerebras_gpt_111M --reward.base_model=cerebras/Cerebras-GPT-111M --reward.track --reward.wandb_project_name=lm_human_preference_details --reward.label_dataset=descriptiveness/offline_5k.json --policy.base_model=cerebras/Cerebras-GPT-111M --policy.track --policy.wandb_project_name=lm_human_preference_details --policy.hf_entity=lm-human-preference-details --policy.upload_model" \
    --num-seeds 5 \
    --start-seed 1 \
    --workers 10 \
    --slurm-gpus-per-task 8 \
    --slurm-ntasks 1 \
    --slurm-total-cpus 64 \
    --slurm-template-path benchmark/trl.slurm_template
WANDB_TAGS="tf_adam,EleutherAI/pythia-160m" python benchmark/benchmark.py \
    --command "accelerate launch --multi_gpu --num_processes 8 lm_human_preference_details/train_both_accelerate.py --reward.exp_name=train_reward_accelerate_tf_adam_pythia-160m --policy.exp_name=train_policy_accelerate_tf_adam_pythia-160m --reward.base_model=EleutherAI/pythia-160m --reward.track --reward.wandb_project_name=lm_human_preference_details --reward.label_dataset=descriptiveness/offline_5k.json --policy.base_model=EleutherAI/pythia-160m --policy.track --policy.wandb_project_name=lm_human_preference_details --policy.hf_entity=lm-human-preference-details --policy.upload_model" \
    --num-seeds 5 \
    --start-seed 1 \
    --workers 10 \
    --slurm-gpus-per-task 8 \
    --slurm-ntasks 1 \
    --slurm-total-cpus 64 \
    --slurm-template-path benchmark/trl.slurm_template
WANDB_TAGS="tf_adam,tiiuae/falcon-rw-1b" python benchmark/benchmark.py \
    --command "accelerate launch --multi_gpu --num_processes 8 lm_human_preference_details/train_both_accelerate.py --reward.exp_name=train_reward_accelerate_tf_adam_falcon_rw_1b --policy.exp_name=train_policy_accelerate_tf_adam_falcon_rw_1b --reward.base_model=tiiuae/falcon-rw-1b --reward.track --reward.wandb_project_name=lm_human_preference_details --reward.label_dataset=descriptiveness/offline_5k.json --policy.base_model=tiiuae/falcon-rw-1b --policy.track --policy.wandb_project_name=lm_human_preference_details --policy.hf_entity=lm-human-preference-details --policy.upload_model" \
    --num-seeds 5 \
    --start-seed 1 \
    --workers 10 \
    --slurm-gpus-per-task 8 \
    --slurm-ntasks 1 \
    --slurm-total-cpus 64 \
    --slurm-template-path benchmark/trl.slurm_template