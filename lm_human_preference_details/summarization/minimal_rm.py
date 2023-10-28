import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer


class AutoModelForCausalLMWithRewardHead(nn.Module):
    def __init__(self, lm_backbone):
        super().__init__()
        self.lm_backbone = lm_backbone
        self.scalar_head = nn.Linear(lm_backbone.config.hidden_size, 1)

    def forward(self, **kwargs):
        output = self.lm_backbone(**kwargs)
        last_reward_latents = output.hidden_states[-1]
        # shape: [batch_size, hidden_size]
        reward = self.scalar_head(last_reward_latents)
        return output, reward


def get_reward(reward_model, query_responses, tokenizer):
    attention_mask = query_responses != tokenizer.pad_token_id
    position_ids = attention_mask.cumsum(1) - attention_mask.long()  # exclusive cumsum
    input_ids = query_responses.clone()
    input_ids[~attention_mask] = 0
    return reward_model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        return_dict=True,
        output_hidden_states=True,
    )


base_model = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side="left")
tokenizer.add_special_tokens({"pad_token": "[PAD]"})
reward_model = AutoModelForCausalLMWithRewardHead(AutoModelForCausalLM.from_pretrained(base_model))
reward_model.train()
mb_query = torch.randint(0, len(tokenizer), (1, 10))
mb_query[:, 0:4] = tokenizer.pad_token_id
mb_responses = torch.randint(0, len(tokenizer), (1, 2, 10))
mb_query_tiled = mb_query.unsqueeze(1).repeat(1, mb_responses.shape[1], 1)
query_responses = torch.cat([mb_query_tiled, mb_responses], dim=2).flatten(0, 1)
_, score_all = get_reward(reward_model, query_responses, tokenizer)
print(score_all.squeeze(2))
