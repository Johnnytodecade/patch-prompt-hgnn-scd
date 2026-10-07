import torch
from torch import nn
from .vsa import VisualSemanticAlignment
from .scpe import StructuralConsistencyPrior
from .hgnn import HeterogeneousGraphReasoner
from .pqe import PromptQualityEvaluation


class PatchPromptSCD(nn.Module):
    """Complete paper-level module composition with explicit intermediate outputs."""

    def __init__(self, visual_dim, text_dim, hidden_dim, num_classes):
        super().__init__()
        self.vsa = VisualSemanticAlignment(visual_dim, text_dim, hidden_dim)
        self.scpe = StructuralConsistencyPrior(hidden_dim)
        self.hgnn = HeterogeneousGraphReasoner(hidden_dim, num_classes)
        self.pqe = PromptQualityEvaluation(hidden_dim)
        self.binary_head = nn.Linear(hidden_dim, 2)

    def forward(self, patch_features, text_features, visual_prompts,
                spatial_edges, prompt_edges, semantic_edges):
        patch, prototypes, align_logits, class_probs = self.vsa(patch_features, text_features)
        prompts = torch.cat([visual_prompts, prototypes], 0)
        fused, semantic_logits, fusion_gate = self.hgnn(patch, prompts, spatial_edges, prompt_edges)
        regulated, prompt_quality, quality_target = self.pqe(fused, prototypes, semantic_edges)
        return {
            "semantic_logits": semantic_logits,
            "binary_logits": self.binary_head(fused),
            "patch_features": fused,
            "align_logits": align_logits,
            "class_probs": class_probs,
            "fusion_gate": fusion_gate,
            "regulated_prompts": regulated,
            "prompt_quality": prompt_quality,
            "quality_target": quality_target,
        }
