import os
import time
import tyro
from train_reward_accelerate import Args as ArgsReward, train as train_reward
from train_policy_accelerate import Args as ArgsPolicy, train as train_policy
from dataclasses import dataclass, field

@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[:-len(".py")]
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
    args.reward.save_path = f"models/{run_name}/reward.pt"
    args.policy.save_path = f"models/{run_name}/policy.pt"
    args.policy.rewards.trained_model = args.reward.save_path
    train_reward(args.reward)
    train_policy(args.policy)
