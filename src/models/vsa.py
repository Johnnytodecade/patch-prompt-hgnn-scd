import torch
from torch import nn
from torch.nn import functional as F


class VisualSemanticAlignment(nn.Module):
    def __init__(self, visual_dim, text_dim, hidden_dim, temperature=0.07):
        super().__init__()
        self.visual = nn.Linear(visual_dim, hidden_dim)
        self.text = nn.Linear(text_dim, hidden_dim)
        self.temperature = temperature

    def forward(self, patch_features, text_features):
        visual = F.normalize(self.visual(patch_features), dim=-1)
        prototypes = F.normalize(self.text(text_features), dim=-1)
        logits = visual @ prototypes.T / self.temperature
        probs = logits.softmax(-1)
        initialized = visual + probs @ prototypes
        return initialized, prototypes, logits, probs

    @staticmethod
    def metric_loss(visual, prototypes, labels):
        classes = labels.unique()
        classes = classes[(classes >= 0) & (classes < prototypes.shape[0])]
        if classes.numel() < 2:
            return visual.new_zeros(())
        centroids = torch.stack([visual[labels == c].mean(0) for c in classes])
        return (torch.cdist(centroids, centroids) - torch.cdist(prototypes[classes], prototypes[classes])).square().mean()
