# HGAIT: Hierarchical Graph-based Attention for Financial Time-Series Prediction


## Overview

Official PyTorch/PyTorch Geometric implementation of HGAIT, a financial time-series graph model that combines time-series embeddings with graph-based dependencies among stocks to predict the next-step return for each stock as a regression task.

## Key Features

- **In-model representation pipeline**: TimeMixing (per-feature temporal embedding) → Variable Attention (inverted multi-head attention) → Feature Importance tokenization
- **Graph neighbor attention**: Correlation-based top/bottom neighbor selection and gated combination of multiple GATConv outputs
- **Combined regression loss**: MSE + Pearson-correlation loss (`utils.combined_loss`)
- **Training utilities**: Early stopping, ReduceLROnPlateau scheduler, optional Weights & Biases logging

## Model Architecture

### TimeMixing
- **Modes**: `mlp` | `lstm` | `gru`
- For each feature, embeds a length-`L` sequence into a `d_model`-dimensional representation

### Variable Attention Layer
- Learns cross-feature interactions via inverted multi-head attention
- Produces a **feature-importance** distribution across features

### Feature Importance Tokenized Block
- Forms a stock-level representation by importance-weighted aggregation of feature embeddings

### Gated Graph Attention
- Builds top/bottom neighbors from correlation of returns at time t
- Combines outputs from self/top/bottom GATConv branches using learned importance weights

### Predictor
- `LayerNorm → MLP(d_model → 2*d_model → 1)` to regress the next-step return

## Project Structure

```
HGAIT/
├── train.py               # End-to-end train/val/test script
├── model.py               # HGAIT model (TimeMixing, Variable Attention, Gated GAT, Predictor)
├── data_generator.py      # Raw CSV → per-date graph samples (.pkl)
├── Dataset_reg_std.py     # Per-date PyG Dataset
├── Dataloader.py          # Thin wrapper for PyG DataLoader
├── config.py              # Hyperparameters and paths
├── utils/
│   └── utils.py          # Seeding, metrics/loss, ranking utilities
├── raw_data/
│   ├── csv_maker.py
│   └── CRSP.csv (required)
├── data/                  # Output directory for generated .pkl files
├── results/               # Predictions/metrics/CSV outputs
└── model/                 # Checkpoints (path configured in config)
```

## Installation

### Requirements
Below are the key dependencies used by this project.

```bash
pip install numpy==2.2.2 pandas==2.2.3 scikit-learn==1.6.1 tqdm wandb==0.17.5
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0
pip install torch_geometric==2.5.3
# PyG extensions (install variants matching your Torch/CUDA)
pip install \
  torch-scatter==2.1.2+pt21cu121 \
  torch-sparse==0.6.18+pt21cu121 \
  torch-cluster==1.6.3+pt21cu121 \
  torch-spline-conv==1.2.2+pt21cu121
```

- On Windows, PyG and its extensions must match your PyTorch/CUDA versions. Please follow the official installation guides and wheels for your environment.

## Usage

### 1) Data Generation
Generate per-date graph samples from `raw_data/CRSP.csv`. Update file paths inside `data_generator.py` to fit your environment.

```bash
python data_generator.py
```

Upon success, `data/` will contain files like `YYYY-MM-DD.pkl`.

### 2) Training / Evaluation
Adjust `config.py` to your environment, then run:

```bash
python train.py
```

Outputs:
- Checkpoint: `Config.best_model_path`
- Results pickle: `results/results_*.pkl` (per-date prediction/label tensors)
- Results CSV: `results_<seed>_<lr>_<heads>_<heads>_<layers>.csv`

Notes:
- `CUDA_VISIBLE_DEVICES="1"` is set inside `train.py`. Modify or comment it out for your setup.
- `model.HGAIT` requires a `mode` argument (`'gru'|'lstm'|'mlp'`). Ensure it is passed where the model is constructed in `train.py`.

## Configuration

All settings are managed via the `Config` class in `config.py`.
- **Paths**: `data_dir`, `best_model_path`, `result_dir`
- **Data split**: `train_split`, `val_length`
- **Model**: `n_heads`, `d_model`, `n_layers`, `n_neighbors`, `mode`
- **Training**: `batch_size`, `num_epochs`, `learning_rate`, `early_stopping_patience`
- **Loss**: `mse_lambda`
- **Logging**: `log_to_wandb`

Windows users should replace absolute paths like `/workspace/dongwoo/HGAIT/...` with local paths or use relative paths.

## Citation

If this repository is useful for your research, please consider citing it (example format):

```bibtex
@article{lee2025hgait,
  title={HGAIT: Heterogeneous Graph Attention with Inverted Transformers for Correlation-Aware Stock Return Prediction},
  author={Lee, Dongwoo and Ock, Seungeun and Song, Jae Wook},
  journal={Expert Systems with Applications},
  pages={129292},
  year={2025},
  publisher={Elsevier}
}
```

## Acknowledgments

- **PyTorch** and **PyTorch Geometric** for deep learning and graph neural network tooling
- **Weights & Biases** for experiment tracking



