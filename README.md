# Land Cover Segmentation (LoveDA + DeepGlobe)

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> Note: You should pick the appropriate `torch` build (CUDA vs CPU) for your environment.

## Datasets

This repo ignores the `data/` directory via `.gitignore`. Keep datasets locally (do not commit them).

### LoveDA
In `configs/loveda.yaml`, set `data_dir` to a folder with the following structure:

```
data/LoveDA/
  Train/Urban/images_png/*.png
  Train/Urban/masks_png/*.png
  Train/Rural/...
  Val/Urban/...
  Val/Rural/...
```

If you have the dataset as ZIP files, you can use the helper:
`landcover.datasets.loveda.extract_loveda_zips(zip_dir, extract_dir)`.

### DeepGlobe
In `configs/deepglobe.yaml`, `data_dir` must contain `metadata.csv` (and the paths inside the CSV must be relative to that `data_dir`).

## Running

```bash
python main.py --config configs/loveda.yaml --tag baseline
```

Override example (supports nested keys):

```bash
python main.py --config configs/deepglobe.yaml --tag ebct \
  --override ebct.use=true ebct.start_pct=0.3 boundary_sampling.p_max=0.5
```

Alternatively:

```bash
python scripts/train_loveda.py
python scripts/train_deepglobe.py
```

## Outputs

Checkpoints are saved under `save_dir` (default: `outputs/checkpoints/`).
The best 3 checkpoints (by validation mIoU) are kept.

## Notes (why the code was cleaned up)

- Colab-specific lines were removed or moved into configuration.
- Dataset download/unzip is not embedded inside the training loop.
- Side-effect-heavy patterns (e.g., `global cfg`) were removed: configuration is passed explicitly to functions.
