import torch
from torch import nn
from torch.nn import functional as F


class PromptQualityEvaluation(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.confidence = nn.Linear(2 * dim, 1)

    def forward(self, patch_features, semantic_prompts, edge_index):
        patch_idx, prompt_idx = edge_index
        quality = self.confidence(torch.cat([
            patch_features[patch_idx], semantic_prompts[prompt_idx]], -1)).sigmoid().squeeze(-1)
        fallback = semantic_prompts.mean(0, keepdim=True)
        regulated = quality[:, None] * semantic_prompts[prompt_idx] + (1 - quality[:, None]) * fallback
        target = (F.cosine_similarity(patch_features[patch_idx], semantic_prompts[prompt_idx]) + 1) / 2
        return regulated, quality, target
