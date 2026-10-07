import torch
from torch import nn
from torch.nn import functional as F


class StructuralConsistencyPrior(nn.Module):
    """Multi-scale structure weighting shared by feature regularization and adjacency."""

    def __init__(self, channels):
        super().__init__()
        self.edge = nn.Conv2d(channels, 1, 3, padding=1, bias=False)

    def forward(self, feature_t1, feature_t2):
        s1, s2 = self.edge(feature_t1).sigmoid(), self.edge(feature_t2).sigmoid()
        weight = torch.exp(-(s1 - s2).abs())
        enhanced = weight * (feature_t1 - feature_t2).abs()
        consistency = (weight * (feature_t1 - feature_t2).square()).mean()
        return enhanced, weight, consistency

    @staticmethod
    def adjacency(mask_ids, structural_weight):
        """Build sparse 4-neighbour patch contacts and their mean structural confidence."""
        right = mask_ids[..., :, 1:] != mask_ids[..., :, :-1]
        down = mask_ids[..., 1:, :] != mask_ids[..., :-1, :]
        pairs = []
        for a, b, valid in ((mask_ids[..., :, 1:], mask_ids[..., :, :-1], right),
                            (mask_ids[..., 1:, :], mask_ids[..., :-1, :], down)):
            if valid.any():
                pairs.append(torch.stack([a[valid], b[valid]]))
        edge_index = torch.cat(pairs, dim=1) if pairs else torch.empty(2, 0, dtype=torch.long, device=mask_ids.device)
        edge_weight = structural_weight.new_ones(edge_index.shape[1])
        return edge_index.long(), edge_weight
