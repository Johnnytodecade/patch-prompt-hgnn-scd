import torch
from torch.nn import functional as F


def joint_objective(outputs, labels, spatial_edges, cfg):
    sem = F.cross_entropy(outputs["semantic_logits"], labels["semantic"])
    binary = F.cross_entropy(outputs["binary_logits"], labels["change"])
    align = F.cross_entropy(outputs["align_logits"], labels["semantic"])
    quality_target = outputs["quality_target"]
    if cfg.get("detach_similarity_target", True):
        quality_target = quality_target.detach()
    pqe = F.l1_loss(outputs["prompt_quality"], quality_target)
    if spatial_edges.numel():
        a, b = spatial_edges
        relation = (outputs["patch_features"][a] - outputs["patch_features"][b]).square().sum(-1).mean()
    else:
        relation = sem.new_zeros(())
    total = binary + sem + cfg["lambda_graph"] * sem + align
    total = total + cfg["lambda_relation"] * relation + cfg["lambda_prompt"] * pqe
    return {"total": total, "binary": binary, "semantic": sem, "alignment": align,
            "relation": relation, "pqe": pqe}
