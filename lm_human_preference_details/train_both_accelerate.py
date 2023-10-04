import os
import time
from dataclasses import dataclass, field

import tyro
from train_policy_accelerate import Args as ArgsPolicy
from train_policy_accelerate import train as train_policy
from train_reward_accelerate import Args as ArgsReward
from train_reward_accelerate import train as train_reward


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    reward: ArgsReward = field(default_factory=ArgsReward)
    policy: ArgsPolicy = field(default_factory=ArgsPolicy)


if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.exp_name}__{args.seed}__{int(time.time())}"
    args.reward.seed = args.seed
    args.policy.seed = args.seed
    args.reward.save_path = f"models/{run_name}/reward"
    args.policy.save_path = f"models/{run_name}/policy"
    args.policy.rewards.trained_model = f"{args.reward.save_path}/pytorch_model.bin"
    args.policy.rewards.label_dataset = args.reward.label_dataset
    train_reward(args.reward)
    train_policy(args.policy)
