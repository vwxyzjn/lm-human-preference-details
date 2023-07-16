# lm-human-preference-details

This repo aims to make a blog post similar to [*The 37 Implementation Details of Proximal Policy Optimization*](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) but for RLHF techniques used in https://github.com/openai/lm-human-preferences.


>**Warning** This repo is a **WIP** made public because it's easier for me to share pointers with collaborators. I'll remove this warning when the repo is ready for public consumption.


The goal of the repo is 1) to provide a simple-to-read and minimal reference implementation of RLHF and 2) to create rigorous benchmarks and to match the learning curves of `openai/lm-human-preferences`.

This repo is just for educational / learning purposes. For more advanced users, https://github.com/lvwerra/trl would be a great choice.

## Get started

```
poetry install
accelerate launch \
    --num_processes 8 \
    lm_human_preference_details/train_both_accelerate.py \
    --reward.track --policy.track
accelerate launch \
    --num_processes 8 \
    lm_human_preference_details/train_both_accelerate.py \
    --reward.track \
    --reward.label_dataset=descriptiveness/offline_5k.json \
    --policy.track
```

You can also run stuff individually. For example, to train the reward model, run
```
accelerate launch \
    --num_processes 8 \
    lm_human_preference_details/train_reward_accelerate.py \
    --track
```

to train the policy model, run
```
accelerate launch \
    --num_processes 8 \
    lm_human_preference_details/train_policy_accelerate.py \
    --track
```



## Current status

Currently for a reproduction, I used the same dataset, same data processing pipeline, same initial model architecture and weights (`gpt2` 124M pretrained model). Hyperparameters are exactly the same except for the adam optimizer's `eps` which I used `5e-4` instead of `1e-5` for better stability

The following charts shows the learning curves of various metrics for `sentiment` and `descriptiveness` tasks, each with 10 random seeds.


>**Warning** Notice how my reproduction in `sentiment` has lower `entropy` and has lower `score` in `descriptiveness`. It may be related to subtle issues with the optimizer's setting. See https://github.com/pytorch/pytorch/issues/104857#issuecomment-1635068659


```
pip install openrlbenchmark==0.2.1a4
python -m openrlbenchmark.rlops_multi_metrics \
    --filters '?we=openrlbenchmark&wpn=lm-human-preferences&ceik=task_id&cen=task.value.policy.initial_model&metrics=ppo/objective/score&metrics=ppo/objective/kl&metrics=ppo/objective/entropy&metrics=ppo/objective/score_total&metrics=ppo/objective/kl_coef&metrics=ppo/ppo/loss/total&metrics=ppo/ppo/loss/value&metrics=ppo/ppo/loss/policy&metrics=ppo/ppo/policy/clipfrac&metrics=ppo/ppo/policy/entropy&metrics=ppo/ppo/returns/mean&metrics=ppo/ppo/policy/approxkl&metrics=ppo/ppo/val/clipfrac&metrics=ppo/ppo/val/error&metrics=ppo/ppo/val/mean&metrics=ppo/ppo/returns/var&metrics=ppo/ppo/val/vpred' \
        '124M' \
    --filters '?we=costa-huang&wpn=cleanrl&ceik=rewards.value.label_dataset&cen=exp_name&metrics=objective/scores&metrics=objective/kl&metrics=objective/entropy&metrics=objective/score_total&metrics=objective/kl_coef&metrics=ppo/loss/total&metrics=ppo/loss/value&metrics=ppo/loss/policy&metrics=ppo/policy/clipfrac&metrics=ppo/policy/entropy&metrics=ppo/returns/mean&metrics=ppo/policy/approxkl&metrics=ppo/val/clipfrac&metrics=ppo/val/error&metrics=ppo/val/mean&metrics=ppo/returns/var&metrics=ppo/val/vpred' \
        'train_policy_accelerate?tag=v0.1.0-20-gd63c6c3' \
    --env-ids sentiment descriptiveness \
    --env-ids sentiment/offline_5k.json  descriptiveness/offline_5k.json \
    --no-check-empty-runs \
    --pc.ncols 5 \
    --pc.ncols-legend 1 \
    --output-filename static/0compare \
    --scan-history --report
``` 
![](static/ours1.png)


Wandb report is availible at https://wandb.ai/costa-huang/cleanrl/reports/Regression-Report-train_policy_accelerate--Vmlldzo0ODk0MjM2. Feel free to check out the logs of the runs for sample outputs.

![](static/wandb.png)


### Sentiment

* https://wandb.ai/openrlbenchmark/lm-human-preferences/runs/ckg9d00z/logs
    * 
    ```
    k =  elapsed/time , v =  16211.987
    lets work on the assumption that if were changing things , were probably not going to automatically get all the memories .
    It would be nice to know if I did my best in that regard.⏎⏎Stories like these are useful.
    score = +1.86
    kl = +7.88
    total = +0.68
    do you reckon he goes along with the popular opinion , or do you think he gets everyone around to his opinion ?
    "⏎⏎" He's a nice guy. I think he's a nice guy.
    score = +2.71
    kl = +3.18
    total = +2.23
    'yes , ' he replied calmly , 'but they seem to think that if they live in a village , that makes them townsmen . '
    " "⏎⏎"You are a good man. You are a right man.
    score = +2.20
    kl = +11.00
    total = +0.55
    ```
* https://wandb.ai/costa-huang/cleanrl/runs/fu902lyc/logs
    * 
    ```
    query: he was quiet for a minute, his eyes unreadable.
    response: " I'm so glad you're here. It's been a long time since we've known each other.
    score: 0.5942718982696533, kl: 7.33388090133667, total reward: -0.45726191997528076

    query: she knew she shouldn't, but she reached out to capture that particular hand
    before he could go.
    response: That was all that mattered.
    "I'm glad to see you're happy," she said.
    score: 2.0057406425476074, kl: 6.900775909423828, total reward: 1.025254249572754

    query: she offered a tense shrug before reaching for the door handle.
    response: "We're going to be fine," he said, "and we're going to be okay.
    score: 0.9820655584335327, kl: 7.899775505065918, total reward: -0.15402913093566895
    ```



### Descriptiveness

* https://wandb.ai/openrlbenchmark/lm-human-preferences/runs/vx0cjya8/logs
    * 
    ```
    then peter and gracie hoffman about their daughter , kelly .
    The blonde girl made a hideous looking face, rubbing her lips with her hands, and motioned them.
    score = +3.29
    kl = +11.87
    total = +1.51
    it was n't remarkable ; it was awful .
    He was dressed in a black coat with the sleeves rolled up and his eyes were black with blood.
    score = +2.98
    kl = +12.40
    total = +1.12
    eyes that only warmed when they were on her .
    The chestnut-brown eyes were almost white, their dull yellow pupils half-closed.
    score = +3.11
    kl = +9.55
    total = +1.68
    ```
* https://wandb.ai/costa-huang/cleanRL/runs/e75dhls2/logs
    * 
    ```
    query: `` you really don't think i can do it, '' i said, trying not to sound stung.
    response: had to kneel down to take a deep breath, her hand wrapped around her waist. her
    face was red with tears
    score: -1.0, kl: 8.85654354095459, total reward: -2.0279948711395264

    query: it's ridiculous for someone with your pedigree to go to art school.
    response:  And if Mikan has any problems with this, he has to be kidding himselfWhat's in
    the Water at Su
    score: -1.0, kl: -4.508945465087891, total reward: -0.47910648584365845
    
    query: jimmy was on the back seat of his truck.
    response:  His lips were furrowed and his lips were still furrowed. He was dressed in a
    white coat.
    score: 1.3276370763778687, kl: 7.90595006942749, total reward:
    0.41623085737228394
    ```

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

