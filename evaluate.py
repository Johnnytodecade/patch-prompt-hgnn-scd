import argparse
from src.config import load_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()
    load_config(args.config)
    raise NotImplementedError(
        "Instantiate the pinned model/data protocol, load checkpoint, rasterize patch logits, "
        "then report F1, mIoU and overall accuracy without using test labels for prompt creation."
    )


if __name__ == "__main__":
    main()
