# Wing generation reproducibility package

This folder contains the main VAE + autoregressive Transformer training code, processed training data, original evaluation notebooks, two Flow Matching baselines, and the VAE/Transformer checkpoints available in the project. Run commands from this folder. Python 3.11 is recommended.

## Contents

| Path | Purpose |
| --- | --- |
| `main/` | VAE and Transformer model/training code |
| `main/tests/` | Original VAE, generation, and aerodynamic evaluation notebooks |
| `baseline_fm/` | Latent-space Flow Matching training and staged test code |
| `baseline_fm_v2/` | Whole-geometry Flow Matching training and staged test code |
| `data/Geo_combined.csv` | Processed VAE geometry input |
| `data/train_dataV7.pt` | Processed Transformer training samples |
| `data/flow_matching_dataset.pt` | Processed latent baseline data |
| `data/geometry_dataset.pt` | Processed whole-geometry baseline data, including train/valid/test indices |
| `data/configs_GGS.json` | Source geometry records for regenerating baseline datasets |
| `checkpoints/` | Original VAE and Transformer `last.ckpt` files |
| `toolkit/` | Geometry conversion and OpenVSP script helpers used by baseline tests |

## Setup and training

```bash
python -m venv .venv
# Activate .venv with the command appropriate for your shell.
python -m pip install -r requirements.txt
python main/train_wingVAE.py
python main/train_transformer.py
python baseline_fm/train.py --dataset data/flow_matching_dataset.pt --accelerator cpu
python baseline_fm_v2/train.py --dataset data/geometry_dataset.pt --accelerator cpu
```

The Transformer training script loads `checkpoints/vae_last.ckpt` by default. To train it from a new VAE run, copy the resulting `last.ckpt` there. Baseline training outputs are written under each baseline's `experiments/` directory. GPU users can select `--accelerator gpu` for either baseline.

## Testing

The main model's original tests are notebooks in `main/tests/`. Start with `WingAutoregressiveTransformerTest.ipynb`; `EncoderDecoderTest.ipynb` covers the VAE and `test_aero_integralV2.ipynb` covers aerodynamic evaluation. These are archived research notebooks: some cells contain original machine paths and require path edits before execution. The included Transformer checkpoint is for evaluation; running the notebooks also requires the original external geometry/CFD toolchain and any input files referenced by their cells.

Each baseline has a staged test script. After training, run sampling and reconstruction, for example:

```bash
python baseline_fm/test_fm.py --stage sample --dataset data/flow_matching_dataset.pt
python baseline_fm/test_fm.py --stage reconstruct --dataset data/flow_matching_dataset.pt
python baseline_fm_v2/test_fm.py --stage sample --dataset data/geometry_dataset.pt
python baseline_fm_v2/test_fm.py --stage reconstruct --dataset data/geometry_dataset.pt
```

The baseline `stl`, `evaluate`, and `cl` stages require OpenVSP and the aerodynamic surrogate assets used by the original project. Set `VSP_EXE` in the relevant test script to the local OpenVSP executable. Baseline checkpoints are produced by training and are not included here. The test scripts default to their respective `experiments/` directories.

## Regenerating processed baseline data

The included `.pt` files can be used directly. To regenerate them, choose a new `--output` path because the preparation scripts reject overwriting existing data:

```bash
python baseline_fm/prepare_flow_dataset.py --output data/flow_matching_dataset_new.pt
python baseline_fm_v2/prepare_flow_dataset.py --output data/geometry_dataset_new.pt
```

The latent baseline preparation loads `checkpoints/vae_last.ckpt`. The V2 preparation reads `data/configs_GGS.json`. Data files and checkpoints use PyTorch serialization; load only files from a trusted source.
