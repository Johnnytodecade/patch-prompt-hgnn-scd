import argparse
from src.config import load_config, seed_everything


def validate_config(cfg):
    missing = []
    for key in ("seed",):
        if cfg["experiment"].get(key) is None:
            missing.append(f"experiment.{key}")
    for key in ("batch_size",):
        if cfg["data"].get(key) is None:
            missing.append(f"data.{key}")
    if missing:
        raise ValueError("Set manuscript-unspecified fields before training: " + ", ".join(missing))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cfg = load_config(args.config)
    if args.dry_run:
        print("Configuration parsed; module scaffold is available.")
        return
    validate_config(cfg)
    seed_everything(cfg["experiment"]["seed"])
    raise NotImplementedError(
        "Connect the verified Hi-UCD split and the pinned SAM2 adapter before the publication run. "
        "The optimizer is Adam(lr=1e-4), cosine annealing, 505 epochs, best validation F1."
    )


if __name__ == "__main__":
    main()
