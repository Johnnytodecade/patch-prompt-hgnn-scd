# Patch-Prompt HGNN for Semantic Change Detection

Reference implementation accompanying **"Heterogeneous graph reasoning and prompting framework for semantic change detection"**.

The repository follows the paper's Alignment-Enhancement-Reasoning-Feedback pipeline: SAM2-guided structure-aware patches, CLIP transition prototypes, visual-semantic alignment (VSA), structural consistency prior enhancement (SCPE), heterogeneous graph reasoning (HGNN), prompt quality evaluation (PQE), and binary/semantic prediction heads.

> Reproducibility note: the manuscript does not specify the exact Hi-UCD split, batch size, seed, every loss weight, or the full SAM2 prompt schedule. They are explicit configuration fields here and must be set to the values used for the final reported experiment before claiming exact reproduction.

## Repository layout

```text
configs/default.yaml       experiment configuration
src/data/hiucd.py          Hi-UCD loading and transition encoding
src/models/foundation.py   frozen SAM2/CLIP adapters
src/models/vsa.py          visual-semantic alignment
src/models/scpe.py         structural prior and patch adjacency
src/models/hgnn.py         relation filtering and two-branch graph reasoning
src/models/pqe.py          prompt quality evaluation
src/models/model.py        complete SCD model
src/losses.py              joint training objective
src/metrics.py             F1, mIoU and overall accuracy
train.py / evaluate.py     training and evaluation entry points
```

## Environment

The paper reports Ubuntu 22.04, Python 3.10, PyTorch 2.5.1, torchvision 0.20.1 and CUDA 12.1. Create the environment with:

```bash
conda env create -f environment.yml
conda activate patch-prompt-scd
pip install -r requirements.txt
```

Download these separately (licenses prevent redistribution):

- Hi-UCD, arranged as `image/2018`, `image/2019`, `mask/2018_2019`.
- SAM2.1 Hiera Large checkpoint, saved at `checkpoints/sam2.1_hiera_large.pt`.
- OpenAI CLIP ViT-L/14 weights, obtained through the CLIP package cache.

## Train and evaluate

First edit `configs/default.yaml`, particularly `data.root`, `data.train_list`, `data.val_list`, the transition vocabulary, and all fields marked `null`.

```bash
python train.py --config configs/default.yaml
python evaluate.py --config configs/default.yaml --checkpoint outputs/best.pt
```

To verify imports/configuration without loading the foundation checkpoints:

```bash
python train.py --config configs/default.yaml --dry-run
```

## Data protocol

Each three-channel label PNG is interpreted as `(class_t1, class_t2, binary_change)`. Transition IDs are generated from the configured vocabulary; `0` is reserved for no-change/ignore according to `data.ignore_index`. Do not derive prompts, graph edges, prototypes, or checkpoint selection from test annotations.

## Citation

See [CITATION.cff](CITATION.cff). Please replace the placeholder DOI, publication year, and repository URL after acceptance/deposition.

## License

Code is released under the MIT License. Dataset and pretrained model weights retain their original licenses.
