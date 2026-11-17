"""Train V2 on the complete 125-D original-geometry vector.

Run prepare_flow_dataset.py first, then train.py. Each experiment stores the
normalization/schema, source code, effective configuration, CSV metrics and
best/last checkpoints. Use --resume with last.ckpt to restore optimizer/epoch.
Validation uses fixed noise and times; the held-out test split is never trained on.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import DataLoader, Subset, TensorDataset

if __package__:
    from .Airfoil_FM import GeometryFlow
    from .prepare_flow_dataset import load_dataset
else:
    from Airfoil_FM import GeometryFlow
    from prepare_flow_dataset import load_dataset


def main() -> None:
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=root.parent / "data/geometry_dataset.pt")
    parser.add_argument("--exp_dir", type=Path, default=root / "experiments/fm_geometry")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--step_size", type=int, default=50)
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--accelerator", choices=("cpu", "gpu"), default="gpu")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("--epochs and --batch_size must be positive.")
    dataset = load_dataset(args.dataset)
    pl.seed_everything(dataset["seed"], workers=True)
    model_settings = dict(geometry_dim=dataset["x"].shape[1], context_dim=dataset["context"].shape[1],
                          hidden_dim=args.hidden_dim, depth=args.depth, learning_rate=args.learning_rate,
                          step_size=args.step_size, gamma=args.gamma)
    model = GeometryFlow(**model_settings)
    config = {"model": model_settings, "batch_size": args.batch_size, "seed": dataset["seed"],
              "dataset_sha256": hashlib.sha256(args.dataset.read_bytes()).hexdigest(),
              "dataset": str(args.dataset.resolve())}
    config_path = args.exp_dir / "config.json"
    if args.resume is not None:
        if not args.resume.is_file():
            raise FileNotFoundError(f"Resume checkpoint does not exist: {args.resume}")
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        if previous != config:
            raise ValueError("Resume configuration/dataset differs from saved config.json; use the original settings.")
        if args.resume.resolve().parent != (args.exp_dir / "checkpoints").resolve():
            raise ValueError("--resume must point to this experiment's checkpoints directory.")
    else:
        if args.exp_dir.exists() and any(args.exp_dir.iterdir()):
            raise FileExistsError(f"Experiment is not empty: {args.exp_dir}; use a new --exp_dir or --resume.")
        args.exp_dir.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        torch.save({key: dataset[key] for key in ("format", "normalization", "feature_names", "splits",
                                                  "source_sha256", "wing_ids")}, args.exp_dir / "geometry_metadata.pt")
        code_dir = args.exp_dir / "code"
        code_dir.mkdir()
        for source in root.glob("*.py"):
            shutil.copy2(source, code_dir / source.name)
    pairs = TensorDataset(dataset["x"], dataset["context"])
    train_loader = DataLoader(Subset(pairs, dataset["splits"]["train"]), batch_size=args.batch_size,
                              shuffle=True, num_workers=0)
    valid_loader = DataLoader(Subset(pairs, dataset["splits"]["valid"]), batch_size=args.batch_size,
                              shuffle=False, num_workers=0)
    checkpoint = ModelCheckpoint(dirpath=args.exp_dir / "checkpoints", monitor="val_loss", mode="min",
                                 save_top_k=1, save_last=True, filename="best-{epoch:03d}-{val_loss:.5f}")
    trainer = pl.Trainer(max_epochs=args.epochs, accelerator=args.accelerator, devices=1,
                         callbacks=[checkpoint], logger=CSVLogger(str(args.exp_dir), name="metrics"),
                         gradient_clip_val=1.0, log_every_n_steps=10, enable_progress_bar=False)
    trainer.fit(model, train_loader, valid_loader, ckpt_path=args.resume)


if __name__ == "__main__":
    main()
