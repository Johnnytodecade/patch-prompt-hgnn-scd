import math
import torch
from torch import nn


def segment_softmax(scores, targets, node_count):
    out = torch.empty_like(scores)
    for i in range(node_count):
        idx = targets == i
        if idx.any():
            out[idx] = scores[idx].softmax(0)
    return out


class HeterogeneousGraphReasoner(nn.Module):
    """Relation filtering, prompt/patch aggregation and patch-dependent fusion."""

    def __init__(self, dim, num_classes):
        super().__init__()
        self.prompt_proj = nn.Linear(dim, dim)
        self.spatial_proj = nn.Linear(dim, dim)
        self.gate = nn.Linear(2 * dim, 1)
        self.semantic_head = nn.Linear(dim, num_classes)

    @staticmethod
    def filter_edges(edge_index, confidence, threshold, top_k=4):
        keep = confidence >= threshold
        # Preserve at least top-k globally; per-target top-k can be used for batched graphs.
        if keep.sum() < min(top_k, confidence.numel()):
            keep[confidence.topk(min(top_k, confidence.numel())).indices] = True
        return edge_index[:, keep], confidence[keep]

    def _aggregate(self, patch, source, edge_index, projection):
        if edge_index.numel() == 0:
            return torch.zeros_like(patch)
        target, origin = edge_index
        q, k = projection(patch[target]), projection(source[origin])
        score = (q * k).sum(-1) / math.sqrt(q.shape[-1])
        alpha = segment_softmax(score, target, patch.shape[0])
        out = torch.zeros_like(patch)
        out.index_add_(0, target, alpha[:, None] * k)
        return out

    def forward(self, patch, prompts, spatial_edges, prompt_edges):
        homogeneous = self._aggregate(patch, patch, spatial_edges, self.spatial_proj)
        bipartite = self._aggregate(patch, prompts, prompt_edges, self.prompt_proj)
        gate = self.gate(torch.cat([bipartite, homogeneous], -1)).sigmoid()
        fused = gate * bipartite + (1 - gate) * homogeneous
        return fused, self.semantic_head(fused), gate
