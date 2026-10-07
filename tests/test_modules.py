import torch
from src.models.vsa import VisualSemanticAlignment
from src.models.hgnn import HeterogeneousGraphReasoner
from src.models.pqe import PromptQualityEvaluation


def test_core_tensor_contracts():
    n, p, c, d = 6, 4, 3, 8
    vsa = VisualSemanticAlignment(d, d, d)
    patch, prototypes, logits, _ = vsa(torch.randn(n, d), torch.randn(c, d))
    graph = HeterogeneousGraphReasoner(d, c)
    spatial = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]])
    prompt_edges = torch.tensor([[0, 1, 2, 3], [0, 1, 2, 3]])
    fused, semantic, gate = graph(patch, torch.randn(p, d), spatial, prompt_edges)
    pqe = PromptQualityEvaluation(d)
    _, quality, target = pqe(fused, prototypes, torch.tensor([[0, 1, 2], [0, 1, 2]]))
    assert semantic.shape == (n, c)
    assert gate.shape == (n, 1)
    assert quality.shape == target.shape == (3,)
