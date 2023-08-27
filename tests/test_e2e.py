import subprocess


def test_torch():
    subprocess.run(
        "accelerate launch --num_processes 1 lm_human_preference_details/train_both_accelerate.py --reward.labels.num_train 4 --policy.ppo.total_episodes 8 --policy.ppo.local_batch_size 4 --policy.ppo.no_whiten_rewards", 
        shell=True,
        check=True,
    )
