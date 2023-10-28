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


base_model = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(base_model, padding_side="left")
reward_model = AutoModelForCausalLMWithRewardHead(AutoModelForCausalLM.from_pretrained(base_model))
mb_query = torch.randint(0, len(tokenizer), (1, 512))
mb_responses = torch.randint(0, len(tokenizer), (1, 2, 80))
mb_query_tiled = mb_query.unsqueeze(1).repeat(1, mb_responses.shape[1], 1)
query_responses = torch.cat([mb_query_tiled, mb_responses], dim=2).flatten(0, 1)
_, score = reward_model(input_ids=query_responses, return_dict=True, output_hidden_states=True)
print(score.squeeze(2))
