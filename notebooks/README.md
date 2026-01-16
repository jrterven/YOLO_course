# Notebooks

This folder contains training and experimentation notebooks for the YOLO course.

## Ants vs Bees Binary Classification Training

Notebook: [`ants_bees_classification_training.ipynb`](ants_bees_classification_training.ipynb)

This notebook trains a binary classifier on the Hymenoptera (Ants vs Bees) dataset.

### 1) Download the Dataset

- Hymenoptera (Ants vs Bees): https://download.pytorch.org/tutorial/hymenoptera_data.zip

### 2) Place Data Locally

Donwload the dataset and place it in your data directory inside a folder called `ants_bees_raw`:

```bash
mkdir -p /YOUR_DATA_DIRECTORY/ants_bees_raw
```

Expected raw layout:

```
ants_bees_raw/
└── hymenoptera_data/
    ├── train/
    │   ├── ants/
    │   └── bees/
    └── val/
        ├── ants/
        └── bees/
```

### 3) Run the Notebook

Open the notebook and confirm the data paths:

```python
RAW_DATA_DIR = Path("/YOUR_DATA_DIRECTORY/ants_bees_raw")
OUTPUT_DIR = Path("/YOUR_DATA_DIRECTORY/ants_bees_yolo")
```

The notebook splits the training set into train/val and uses the original `val/` folder as the test split.

### 4) Test the Trained Model

Use the provided testing script to evaluate the test split:

```bash
python scripts/classify/test_ants_bees.py \
  --model runs/ants_bees_cls/weights/best.pt \
  --data /YOUR_DATA_DIRECTORY/ants_bees_yolo
```

## ISIC 2019 Multi-class Classification Training

Notebook: [`isic_classification_training.ipynb`](isic_classification_training.ipynb)

This notebook trains a custom skin-lesion classifier using the ISIC 2019 dataset. **You must download the data manually** (license + registration required).

### 1) Download the Dataset

- **ISIC 2019** (multi-class): https://challenge.isic-archive.com/data/#2019

### 2) Place Data Locally

Create a raw data folder:

```bash
mkdir -p /YOUR_DATA_DIRECTORY/isic2019_raw
```

Expected raw layout:

**ISIC 2019 (Training)**
```
ISIC_2019_Training_Input/
ISIC_2019_Training_GroundTruth.csv
```

**ISIC 2019 (Test)**
```
ISIC_2019_Test_Input/
ISIC_2019_Test_GroundTruth.csv
```

Put the files under `/YOUR_DATA_DIRECTORY/isic2019_raw/` so the notebook can find them.

### 3) Run the Notebook

Open the notebook and set the data paths at the top:

```python
RAW_DATA_DIR = Path("/YOUR_DATA_DIRECTORY/isic2019_raw")
OUTPUT_DIR = Path("/YOUR_DATA_DIRECTORY/isic2019_yolo")
```

The notebook creates train/val splits from the labeled training data and uses the official test set for evaluation.

### 4) Test the Trained Model

Use the provided testing script to evaluate the test split:

```bash
python scripts/classify/test_isic2019.py \
  --model runs/isic2019_cls/weights/best.pt \
  --data /YOUR_DATA_DIRECTORY/isic2019_yolo
```


