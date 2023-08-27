# lm-human-preference-details

This repo aims to make a blog post similar to [*The 37 Implementation Details of Proximal Policy Optimization*](https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/) but for RLHF techniques used in https://github.com/openai/lm-human-preferences.


>**Warning** This repo is a **WIP** made public because it's easier for me to share pointers with collaborators. I'll remove this warning when the repo is ready for public consumption.


The goal of the repo is 1) to provide a simple-to-read and minimal reference implementation of RLHF and 2) to create rigorous benchmarks and to match the learning curves of `openai/lm-human-preferences`.

This repo is just for educational / learning purposes. For more advanced users, https://github.com/lvwerra/trl would be a great choice.

## Get started

```
poetry install
poetry shell
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


> ⚠️ **NOTE**: You can install the latest torch or jax with the following command:
```
poetry run pip install --upgrade torch
poetry run pip install --upgrade "jax[cuda11_cudnn82]==0.4.8" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Current status

Currently for a reproduction, I used the same dataset, same data processing pipeline, same initial model architecture and weights (`gpt2` 124M pretrained model). Hyperparameters are exactly the same.

The following charts shows the learning curves of various metrics for `sentiment` and `descriptiveness` tasks, each with 10 random seeds.

```
pip install openrlbenchmark==0.2.1a4
python -m openrlbenchmark.rlops_multi_metrics \
    --filters '?we=openrlbenchmark&wpn=lm-human-preferences&xaxis=_step&ceik=task_id&cen=task.value.policy.initial_model&metrics=ppo/objective/score&metrics=ppo/objective/kl&metrics=ppo/objective/entropy&metrics=ppo/objective/score_total&metrics=ppo/objective/kl_coef&metrics=ppo/ppo/loss/total&metrics=ppo/ppo/loss/value&metrics=ppo/ppo/loss/policy&metrics=ppo/ppo/policy/clipfrac&metrics=ppo/ppo/policy/entropy&metrics=ppo/ppo/returns/mean&metrics=ppo/ppo/policy/approxkl&metrics=ppo/ppo/val/clipfrac&metrics=ppo/ppo/val/error&metrics=ppo/ppo/val/mean&metrics=ppo/ppo/returns/var&metrics=ppo/ppo/val/vpred' \
        '124M' \
    --filters '?we=openrlbenchmark&wpn=lm_human_preference_details&xaxis=_step&ceik=rewards.value.label_dataset&cen=exp_name&metrics=objective/scores&metrics=objective/kl&metrics=objective/entropy&metrics=objective/score_total&metrics=objective/kl_coef&metrics=ppo/loss/total&metrics=ppo/loss/value&metrics=ppo/loss/policy_avg&metrics=ppo/policy/clipfrac_avg&metrics=ppo/policy/entropy_avg&metrics=ppo/returns/mean&metrics=ppo/policy/approxkl_avg&metrics=ppo/val/clipfrac_avg&metrics=ppo/val/error&metrics=ppo/val/mean&metrics=ppo/returns/var&metrics=ppo/val/vpred' \
        'train_policy_accelerate?tag=v0.1.0-58-g4f42012&tag=tf_adam&tag=gpt2&cl=tf_adam,gpt2' \
    --env-ids sentiment descriptiveness \
    --env-ids sentiment/offline_5k.json  descriptiveness/offline_5k.json \
    --no-check-empty-runs \
    --pc.ncols 6 \
    --pc.ncols-legend 1 \
    --output-filename static/0compare \
    --scan-history  --report
# (optionally) you can add `--report`` to generate wandb report
``` 
![](static/ours1.png)


Wandb report is available at https://wandb.ai/costa-huang/cleanrl/reports/Regression-Report-train_policy_accelerate--Vmlldzo1MTEwMzQw. Feel free to check out the logs of the runs for sample outputs.

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
* https://wandb.ai/openrlbenchmark/lm_human_preference_details/runs/38v19l2k
    * 
    ```
    29676 query: `` you know who the bassist for the red hot chili peppers is? ''
    29677 response:  I got a couple of bucks for a half of what I was selling and we were 
    29678 pretty happy.
    29681 score: 1.7029974460601807, kl: 2.753568410873413, total reward: 
    29682 1.2474089860916138 


    29754 query: `` it means, '' he said, `` that we 'll keep you. ''
    29755 response: This means that we can get the ship that we want to build from the ground up.
    29760 score: 1.8134586811065674, kl: 5.874235153198242, total reward: 

    29770 query: i stuffed a french fry in my mouth, feeling a little bummed about the current 
    29771 situation and kicking myself for not staying totally away from him in the first 
    29772 place.
    29773 response: I'm genuinely happy with my stay here with no regret. I love my 
    29774 sister, she is amazing.
    29777 score: 2.78997802734375, kl: 5.462193965911865, total reward: 1.8888899087905884
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
* https://wandb.ai/openrlbenchmark/lm_human_preference_details/runs/e3rysbw8
    * 
    ```
    30052 query: she shivered.
    30053 response: She looked up at the old man who had lain with her, and he gave her a
    30054 kiss.
    30058 score: 2.936002731323242, kl: 3.442370891571045, total reward: 1.6455250978469849 

    30023 query: asked jason with frustration apparent in his strained voice.
    30024 response: He was dressed in a suit and tie with a white button down shirt
    30025 and a white knitted hat.
    30028 score: 4.891026973724365, kl: 10.192875862121582, total reward: 1.0853893756866455 
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

