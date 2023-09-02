import subprocess


def test_torch():
    subprocess.run(
        "python lm_human_preference_details/train_both_accelerate.py --reward.task.query_dataset dummy --policy.task.query_dataset dummy --reward.labels.num_train 4 --reward.normalize_samples 4 --reward.rollout_batch_size 4 --policy.ppo.total_episodes 8 --policy.ppo.local_batch_size 4 --policy.ppo.no_whiten_rewards", 
        shell=True,
        check=True,
    )
