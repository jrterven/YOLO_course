# Notebooks

This folder contains training and experimentation notebooks for the YOLO course.

## ISIC 2019/2020 Classification Training

Notebook: [`isic_classification_training.ipynb`](isic_classification_training.ipynb)

This notebook trains a custom skin-lesion classifier using the ISIC 2019 or ISIC 2020 datasets. **You must download the data manually** (license + registration required).

### 1) Download the Dataset

Pick one dataset:

- **ISIC 2019** (multi-class): https://challenge.isic-archive.com/data/#2019
- **ISIC 2020** (binary melanoma): https://challenge.isic-archive.com/data/#2020

> Alternative (ISIC 2020): Kaggle hosts the dataset as **“SIIM-ISIC Melanoma Classification”** if you already use Kaggle.

### 2) Place Data Locally

Create a raw data folder:

```bash
mkdir -p data/isic_raw
```

Expected raw layouts:

**ISIC 2019**
```
ISIC_2019_Training_Input/
ISIC_2019_Training_GroundTruth.csv
```

**ISIC 2020**
```
train.csv
jpeg/train/
```

Put the files under `data/isic_raw/` so the notebook can find them.

### 3) Run the Notebook

Open the notebook and choose the dataset version at the top:

```python
DATASET_VERSION = "2019"  # or "2020"
RAW_DATA_DIR = Path("../data/isic_raw")
OUTPUT_DIR = Path("../data/isic_yolo")
```

The notebook prepares train/val/test splits and trains a YOLO classification model.
