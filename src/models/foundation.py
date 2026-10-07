import torch
from torch import nn


class FrozenFoundationAdapters(nn.Module):
    """Thin adapters around frozen SAM2 and CLIP; dependency imports are lazy."""

    def __init__(self, cfg, transitions, hidden_dim):
        super().__init__()
        import open_clip
        from sam2.build_sam import build_sam2
        self.sam = build_sam2(cfg["sam2"]["config"], cfg["sam2"]["checkpoint"])
        self.clip, _, _ = open_clip.create_model_and_transforms(
            cfg["clip"]["architecture"], pretrained=cfg["clip"]["pretrained"])
        self.tokenizer = open_clip.get_tokenizer(cfg["clip"]["architecture"])
        self.transitions = transitions
        for module in (self.sam, self.clip):
            module.eval()
            for p in module.parameters():
                p.requires_grad = False
        self.visual_projection = nn.LazyLinear(hidden_dim)
        self.text_projection = nn.LazyLinear(hidden_dim)

    @torch.no_grad()
    def encode_text(self, device):
        prompts = [f"a remote sensing change from {x}" for x in self.transitions]
        tokens = self.tokenizer(prompts).to(device)
        return self.clip.encode_text(tokens).float()

    def forward(self, image_t1, image_t2, masks=None, prompts=None):
        """Return patch descriptors/masks. Adapt API calls to the pinned SAM2 release."""
        raise NotImplementedError(
            "Bind SAM2 image/prompt encoder outputs here; the manuscript does not specify "
            "the exact automatic-mask and temporal-correspondence schedule."
        )
