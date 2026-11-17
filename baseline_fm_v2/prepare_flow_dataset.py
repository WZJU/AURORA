"""Create the standalone V2 raw-geometry dataset, with no VAE dependencies.

Run: python prepare_flow_dataset.py
Output: geometry_dataset.pt, containing raw float64 geometry, normalized float32
FM targets and conditions, training-only statistics, source IDs, feature names,
and the fixed seed-62 80/10/10 split. Standard deviations use ddof=1; an exactly
constant training coordinate uses scale 1, so its standardized target is zero.
Use restore_geometry followed by geometry.decode_wing to recover physical JSON.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import TypedDict, cast

import torch
from torch import Tensor
from torch.utils.data import random_split

if __package__:
    from .geometry import decode_wing, encode_wing, feature_names, projected_aspect_ratio, require_list, require_mapping, require_number
else:
    from geometry import decode_wing, encode_wing, feature_names, projected_aspect_ratio, require_list, require_mapping, require_number


class Normalization(TypedDict):
    x_mean: Tensor
    x_scale: Tensor
    ar_mean: Tensor
    ar_scale: Tensor
    ac_mean: Tensor
    ac_scale: Tensor


class Splits(TypedDict):
    train: list[int]
    valid: list[int]
    test: list[int]


class Conditions(TypedDict):
    ar_target: Tensor
    ac: Tensor
    ac_mask: Tensor


class GeometryDataset(TypedDict):
    format: str
    x: Tensor
    x_raw: Tensor
    context: Tensor
    conditions: Conditions
    normalization: Normalization
    splits: Splits
    wing_ids: list[str]
    feature_names: list[str]
    source_path: str
    source_sha256: str
    seed: int


def scale_from_std(values: Tensor) -> Tensor:
    """Define z-score scales, including the exact constant-coordinate convention."""
    std = values.std(dim=0)
    return torch.where(std == 0, torch.ones_like(std), std)


def normalize_conditions(conditions: Conditions, stats: Normalization) -> Tensor:
    ar = (conditions["ar_target"] - stats["ar_mean"]) / stats["ar_scale"]
    ac = (conditions["ac"] - stats["ac_mean"]) / stats["ac_scale"]
    masked = ac.masked_fill(~conditions["ac_mask"].unsqueeze(-1), 0)
    return torch.cat((ar, masked.flatten(1), conditions["ac_mask"].double()), dim=1).float()


def restore_geometry(normalized: Tensor, stats: Normalization) -> Tensor:
    """Undo only the affine normalization, retaining original dimensions and degrees."""
    raw = normalized.double() * stats["x_scale"].to(normalized.device) + stats["x_mean"].to(normalized.device)
    if not torch.isfinite(raw).all():
        raise ValueError("Inverse normalization produced non-finite geometry coordinates.")
    return raw


def extract_conditions(record: dict[str, object], location: str) -> Tensor:
    aerodynamic = require_mapping(record["aerodynamic_char"], f"{location}.aerodynamic_char")
    entries = require_list(aerodynamic["ac_list"], f"{location}.ac_list")
    if not entries:
        raise ValueError(f"{location}: at least one aerodynamic condition is required.")
    rows: list[list[float]] = []
    for index, entry in enumerate(entries):
        values = require_list(entry, f"{location}.ac_list[{index}]")
        if len(values) < 12:
            raise ValueError(f"{location}.ac_list[{index}]: expected at least 12 columns.")
        rows.append([require_number(values[column], f"{location}.ac_list[{index}][{column}]")
                     for column in (2, 3, 9, 10, 11)])
    array = torch.tensor(rows, dtype=torch.float64)
    alpha = torch.deg2rad(array[:, :1])
    return torch.cat((alpha.sin(), alpha.cos(), array[:, 1:]), dim=1)


def build_dataset(source: Path) -> GeometryDataset:
    source_bytes = source.read_bytes()
    records = require_mapping(json.loads(source_bytes), str(source))
    if len(records) < 10:
        raise ValueError(f"{source}: need at least 10 wings for an 80/10/10 split.")
    geometry: list[list[float]] = []
    aspect_ratios: list[list[float]] = []
    aerodynamic: list[Tensor] = []
    for wing_id, value in records.items():
        record = require_mapping(value, f"wing {wing_id}")
        wings = require_list(record["wings"], f"wing {wing_id}.wings")
        if len(wings) != 1:
            raise ValueError(f"Wing {wing_id}: expected exactly one wing, got {len(wings)}.")
        vector = encode_wing(wings[0], f"wing {wing_id}")
        geometry.append(vector)
        aspect_ratios.append([projected_aspect_ratio(decode_wing(vector))])
        aerodynamic.append(extract_conditions(record, f"wing {wing_id}"))
    raw = torch.tensor(geometry, dtype=torch.float64)
    train_size, valid_size = int(0.8 * len(raw)), int(0.1 * len(raw))
    subsets = random_split(range(len(raw)), [train_size, valid_size, len(raw) - train_size - valid_size],
                           generator=torch.Generator().manual_seed(62))
    splits: Splits = {"train": list(subsets[0].indices), "valid": list(subsets[1].indices), "test": list(subsets[2].indices)}
    ar = torch.tensor(aspect_ratios, dtype=torch.float64)
    ac = torch.nn.utils.rnn.pad_sequence(aerodynamic, batch_first=True, padding_value=0.0)
    lengths = torch.tensor([len(rows) for rows in aerodynamic])
    mask = torch.arange(ac.shape[1]).unsqueeze(0) < lengths.unsqueeze(1)
    conditions: Conditions = {"ar_target": ar, "ac": ac, "ac_mask": mask}
    train_indices = splits["train"]
    train_ac = ac[train_indices][mask[train_indices]]
    stats: Normalization = {
        "x_mean": raw[train_indices].mean(dim=0), "x_scale": scale_from_std(raw[train_indices]),
        "ar_mean": ar[train_indices].mean(dim=0), "ar_scale": scale_from_std(ar[train_indices]),
        "ac_mean": train_ac.mean(dim=0), "ac_scale": scale_from_std(train_ac),
    }
    normalized = ((raw - stats["x_mean"]) / stats["x_scale"]).float()
    context = normalize_conditions(conditions, stats)
    if not torch.isfinite(normalized).all() or not torch.isfinite(context).all():
        raise ValueError(f"{source}: non-finite normalized data; check geometry and condition statistics.")
    return {
        "format": "raw_geometry_V2", "x": normalized, "x_raw": raw, "context": context,
        "conditions": conditions, "normalization": stats, "splits": splits,
        "wing_ids": list(records), "feature_names": feature_names(), "seed": 62,
        "source_path": str(source.resolve()), "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
    }


def load_dataset(path: Path) -> GeometryDataset:
    data = require_mapping(torch.load(path, map_location="cpu", weights_only=True), str(path))
    if data["format"] != "raw_geometry_V2" or data["feature_names"] != feature_names():
        raise ValueError(f"{path}: incompatible geometry schema; expected raw_geometry_V2.")
    x, context, raw = data["x"], data["context"], data["x_raw"]
    if not isinstance(x, Tensor) or x.ndim != 2 or x.shape[1] != 125 or x.dtype != torch.float32:
        raise ValueError(f"{path}: x must be float32 [N, 125].")
    if not isinstance(raw, Tensor) or raw.shape != x.shape or raw.dtype != torch.float64:
        raise ValueError(f"{path}: x_raw must be float64 with the same shape as x.")
    if not isinstance(context, Tensor) or context.ndim != 2 or context.shape[0] != len(x) or context.dtype != torch.float32:
        raise ValueError(f"{path}: context must be float32 [N, C].")
    if context.shape[1] < 8 or (context.shape[1] - 1) % 7:
        raise ValueError(f"{path}: context must contain AR, six-column conditions, and a validity mask.")
    if not all(torch.isfinite(value).all() for value in (x, raw, context)):
        raise ValueError(f"{path}: dataset tensors contain NaN or infinite values.")
    splits = require_mapping(data["splits"], f"{path}.splits")
    all_indices: list[int] = []
    for name in ("train", "valid", "test"):
        indices = require_list(splits[name], f"{path}.splits.{name}")
        if not indices or not all(type(index) is int for index in indices):
            raise ValueError(f"{path}: split {name} must contain integer indices.")
        all_indices.extend(cast(list[int], indices))
    if sorted(all_indices) != list(range(len(x))):
        raise ValueError(f"{path}: splits must partition every row exactly once.")
    stats = require_mapping(data["normalization"], f"{path}.normalization")
    for key, width in (("x_mean", 125), ("x_scale", 125), ("ar_mean", 1), ("ar_scale", 1), ("ac_mean", 6), ("ac_scale", 6)):
        value = stats[key]
        if not isinstance(value, Tensor) or value.shape != (width,) or not torch.isfinite(value).all():
            raise ValueError(f"{path}: normalization.{key} must be a finite [{width}] tensor.")
        if key.endswith("scale") and (value <= 0).any():
            raise ValueError(f"{path}: normalization.{key} must be positive.")
    ids = require_list(data["wing_ids"], f"{path}.wing_ids")
    if len(ids) != len(x) or not all(isinstance(item, str) for item in ids):
        raise ValueError(f"{path}: wing_ids must contain one string per row.")
    return cast(GeometryDataset, data)


def main() -> None:
    torch.set_num_threads(1)
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=root.parent / "data/configs_GGS.json")
    parser.add_argument("--output", type=Path, default=root.parent / "data/geometry_dataset.pt")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"Output exists: {args.output}. Choose a new --output path.")
    dataset = build_dataset(args.source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as output:
        torch.save(dataset, output)
    print(f"Saved {len(dataset['x'])} wings: x={tuple(dataset['x'].shape)}, context={tuple(dataset['context'].shape)} -> {args.output}")


if __name__ == "__main__":
    main()
