# Project Description

This project, completed in a single day, demonstrates how to tackle the challenge of Out of Distribution (OOD) data or Open Set Recognition (OSR) during testing phases. The goal is to construct a classifier that is adept at identifying five specific classes of X-Ray images:
1. Teeth
2. Skull
3. Chest
4. Hand
5. Foot
6. OOD (Out of Distribution) or Unknown

It's important to note that a small fraction of the data within each category might be mislabeled, necessitating careful handling to ensure data quality.

# Work Plan

The workflow is organized into three main phases:

### Data Preprocessing

The initial phase involves manual screening to eliminate irrelevant or poor-quality images as thoroughly as possible. Subsequently, the filtered images are converted to grayscale to facilitate the training process.

### Model Training

During this phase, we employ a VGG16 model for training on our dataset, utilizing a 75-25 split ratio for training and testing, respectively. Evaluation metrics including accuracy, precision, recall, f1-score, and a confusion matrix are detailed in the `test_model.ipynb` notebook.

### Handling OOD Data

To address OOD data, we explore four straightforward methodologies: Maximum Logit Score (MLS), Maximum Softmax Probability (MSP), Entropy, and Energy. Each method's efficacy is assessed in dedicated notebooks.

**Important Note:** The thresholds for these OOD methods require tuning, a step we did not undertake in this project. Future work could involve optimizing these thresholds for improved performance.

To predict the label of an input image, use the following script:

```bash
conda env create -f environment.yml
conda activate ood
python .\predict.py -i image.jpg -ood MLS
```

The script accepts the following parameters:

1. Image file name (`-i`)
2. Out of distribution detection method (`-ood`)
3. Image directory (`-dd`)
4. Model Path (`-m`)
