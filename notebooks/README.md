# Notebooks

This folder contains training and experimentation notebooks for the YOLO course.

## ISIC 2020 Classification Training

Notebook: [`isic_classification_training.ipynb`](isic_classification_training.ipynb)

This notebook trains a custom skin-lesion classifier using the ISIC 2020 dataset. **You must download the data manually** (license + registration required).

### 1) Download the Dataset

- **ISIC 2020** (binary melanoma): https://challenge.isic-archive.com/data/#2020

> Alternative: Kaggle hosts the dataset as **“SIIM-ISIC Melanoma Classification”** if you already use Kaggle.

### 2) Place Data Locally

Create a raw data folder:

```bash
mkdir -p data/isic2020_raw
```

Expected raw layout:

**ISIC 2020**
```
train.csv
jpeg/train/
```

Put the files under `data/isic2020_raw/` so the notebook can find them.

### 3) Run the Notebook

Open the notebook and set the data paths at the top:

```python
RAW_DATA_DIR = Path("../data/isic2020_raw")
OUTPUT_DIR = Path("../data/isic2020_yolo")
```

The notebook prepares train/val/test splits and trains a YOLO classification model.

### 4) Test the Trained Model

Use the provided testing script to evaluate the test split:

```bash
python scripts/classify/test_isic2020.py \
  --model runs/isic2020_cls/weights/best.pt \
  --data data/isic2020_yolo
```
