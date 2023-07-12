# lm-human-preference-details

This repo aims to make a blog post similar to [*The 37 Implementation Details of Proximal Policy Optimization*](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) but for RLHF techniques used in https://github.com/openai/lm-human-preferences.


>**Warning** This repo is a **WIP** made public because it's easier for me to share pointers with collaborators. I'll remove this warning when the repo is ready for public consumption.


The goal of the repo is 1) to provide a simple-to-ready and minimal reference implementation of RLHF and 2) to create rigorous benchmarks and to match the learning curves of `openai/lm-human-preferences`.

## Get started

```
poetry install
python lm_human_preference_details/train_both.py --reward.track --policy.track
```

You can also run stuff individually. For example, to train the reward model, run
```
python lm_human_preference_details/train_reward.py
```

to train the policy model, run
```
python lm_human_preference_details/train_policy.py
```



## Current status

Currently for a reproduction, I used the same dataset, same data processing pipeline, same initial model architecture and weights (`gpt2` 124M pretrained model). Hyperparameters are exactly the same except for 1) the `batch_size` which I set to 64 instead of 512 because I haven't got multi gpu set up yet and 2) adam `eps` which I used `5e-4` instead of `1e-5`.

In the following chart out of 10 random seeds, I trained for 8x more steps because I used a 8x smaller batch size. 2 runs crashed (negative KL divergence)
```
pip install openrlbenchmark==0.2.1a4
python -m openrlbenchmark.rlops_multi_metrics \
    --filters '?we=openrlbenchmark&wpn=lm-human-preferences&ceik=task_id&cen=task.value.policy.initial_model&metrics=ppo/objective/score&metrics=ppo/objective/kl&metrics=ppo/objective/entropy&metrics=ppo/objective/kl_coef&metrics=ppo/ppo/loss/total&metrics=ppo/ppo/loss/value&metrics=ppo/ppo/loss/policy&metrics=ppo/ppo/policy/clipfrac&metrics=ppo/ppo/policy/entropy&metrics=ppo/ppo/returns/mean&metrics=ppo/ppo/policy/approxkl&metrics=ppo/ppo/val/clipfrac&metrics=ppo/ppo/val/error&metrics=ppo/ppo/val/mean&metrics=ppo/ppo/returns/var&metrics=ppo/ppo/val/vpred' \
        '124M' \
    --filters '?we=costa-huang&wpn=cleanrl&ceik=base_model&cen=exp_name&metrics=objective/scores&metrics=objective/kl&metrics=objective/entropy&metrics=objective/kl_coef&metrics=ppo/loss/total&metrics=ppo/loss/value&metrics=ppo/loss/policy&metrics=ppo/policy/clipfrac&metrics=ppo/policy/entropy&metrics=ppo/returns/mean&metrics=ppo/policy/approxkl&metrics=ppo/val/clipfrac&metrics=ppo/val/error&metrics=ppo/val/mean&metrics=ppo/returns/var&metrics=ppo/val/vpred' \
        'train_policy_adam5e-4?tag=v0.1.0-9-gc56a4aa' \
    --env-ids sentiment \
    --env-ids gpt2 \
    --no-check-empty-runs \
    --pc.ncols 5 \
    --pc.ncols-legend 1 \
    --output-filename static/0compare \
    --scan-history --report
``` 
![](static/ours1.png)

If we remove those two runs, we get the following chart. The learning curves of scores and kl are more similar, but some key ones like `approxkl` and `clipfrac` are still very different. I think it's related to adam's `eps` and `batch_size` but I'm not sure yet.
![](static/ours2.png)


## Learning curves of `openai/lm-human-preferences`


Wandb report is here: https://wandb.ai/costa-huang/cleanrl/reports/Regression-Report-124M--Vmlldzo0ODM3NTI5


```
pip install openrlbenchmark==0.2.1a4
python -m openrlbenchmark.rlops_multi_metrics \
    --filters '?we=openrlbenchmark&wpn=lm-human-preferences&ceik=task_id&cen=task.value.policy.initial_model&metrics=ppo/objective/score&metrics=ppo/objective/kl&metrics=ppo/ppo/loss/policy&metrics=ppo/ppo/val/mean&metrics=ppo/ppo/policy/entropy&metrics=ppo/ppo/policy/approxkl&metrics=ppo/ppo/val/error&metrics=ppo/ppo/loss/total&metrics=ppo/ppo/returns/mean&metrics=train_reward/minibatch/loss&metrics=ppo/ppo/val/vpred&metrics=ppo/ppo/loss/value&metrics=ppo/ppo/val/var_explained&metrics=ppo/objective/score_total&metrics=train_reward/minibatch/error&metrics=ppo/elapsed/fps&metrics=ppo/global_step&metrics=ppo/ppo/policy/clipfrac&metrics=ppo/ppo/val/var&metrics=ppo/ppo/val/clipfrac&metrics=ppo/objective/entropy&metrics=ppo/ppo/returns/var&metrics=ppo/objective/kl_coef&metrics=ppo/elapsed/time' \
        '124M' \
    --env-ids sentiment descriptiveness tldr \
    --check-empty-runs \
    --pc.ncols 5 \
    --pc.ncols-legend 1 \
    --output-filename static/0compare \
    --scan-history --report
```


![](static/lm-human-preference.png)

