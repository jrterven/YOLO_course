# Notebooks

This folder contains training and experimentation notebooks for the YOLO course.

## ISIC 2019 Classification Training

Notebook: [`isic_classification_training.ipynb`](isic_classification_training.ipynb)

This notebook trains a custom skin-lesion classifier using the ISIC 2019 dataset. **You must download the data manually** (license + registration required).

### 1) Download the Dataset

- **ISIC 2019** (multi-class): https://challenge.isic-archive.com/data/#2019

### 2) Place Data Locally

Create a raw data folder:

```bash
mkdir -p data/isic2019_raw
```

Expected raw layout:

**ISIC 2019**
```
ISIC_2019_Training_Input/
ISIC_2019_Training_GroundTruth.csv
```

Put the files under `data/isic2019_raw/` so the notebook can find them.

### 3) Run the Notebook

Open the notebook and set the data paths at the top:

```python
RAW_DATA_DIR = Path("../data/isic2019_raw")
OUTPUT_DIR = Path("../data/isic2019_yolo")
```

The notebook prepares train/val/test splits from the labeled training data (so the test split has ground truth) and trains a YOLO classification model.

### 4) Test the Trained Model

Use the provided testing script to evaluate the test split:

```bash
python scripts/classify/test_isic2019.py \
  --model runs/isic2019_cls/weights/best.pt \
  --data data/isic2019_yolo
```
