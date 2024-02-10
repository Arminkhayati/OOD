# Project Description

This project, completed in a single day, demonstrates how to tackle the challenge of Out of Distribution (OOD) data or Open Set Recognition (OSR) during testing phases. The goal is to construct a classifier that is adept at identifying five specific classes of X-Ray images:
1. Teeth
2. Skull
3. Chest
4. Hand
5. Foot
6. OOD (Out of Distribution) or Unknown

It's important to note that a small fraction of the data within each category might be mislabeled, necessitating careful handling to ensure data quality.

## Prerequisites
Before you can run the script, ensure you have the following installed:

- Python 3.10
- TensorFlow 2.10
- Pandas
- Numpy
- scikit-learn

You can install the necessary Python packages using conda:

```bash
conda env create -f environment.yml
```

# Work Plan

The workflow is organized into three main phases:

### Data Preprocessing

The initial phase involves manual screening to eliminate irrelevant or poor-quality images as thoroughly as possible. Subsequently, the filtered images are converted to grayscale to facilitate the training process.

### Model Training

During this phase, we employ a VGG16 model for training on our dataset, utilizing a 75-25 split ratio for training and testing, respectively. Evaluation metrics including accuracy, precision, recall, f1-score, and a confusion matrix are detailed in the `test_model.ipynb` notebook.

### Handling OOD Data

To address OOD data, we explore four straightforward methodologies from `oodeel` library: Maximum Logit Score (`MLS`), Maximum Softmax Probability (`MSP`), `Entropy`, and `Energy`. Each method's efficacy is assessed in dedicated notebooks.

Oodeel is a library that performs post-hoc deep OOD detection on already trained neural network image classifiers. The philosophy of the library is to favor quality over quantity and to foster easy adoption.

**Important Note:** The thresholds for these OOD methods require tuning, a step we did not undertake in this project. Future work could involve optimizing these thresholds for improved performance.
You can also run prediction on your image without using any of those mentioned OOD methods by just not passing any input to the `-ood` parameter.


## How to run scripts

To run the train script, navigate to the script's directory in your terminal and use the following command format:
```
conda activate ood
python train.py --data_dir <path_to_data_directory> --bad_files <bad_files_to_ignore> [--num_classes <number_of_classes> ][--img_size <input_image_size>] [--batch_size <batch_size>] [--epochs <number_of_epochs>] [--test_size <validation_split_size>] [--labels_csv <dataset_csv_info_file>]
```

### Arguments

- `--data_dir` (required): The directory containing the dataset and `labels_x_ray.csv` file.
- `--bad_files` (required): The file name of the txt file containing filtered files to ignore them. File must be in `data_dir` directory.
- `--num_classes` (optional): The number of classes in the dataset. Default is 5.
- `--img_size` (optional): The size of the input images. The script resizes all images to this size. Default is 224.
- `--batch_size` (optional): The batch size for training. Default is 32.
- `--epochs` (optional): The number of epochs for which the model will be trained. Default is 100.
- `--test_size` (optional): The ratio of the dataset to be used as validation set. Default is 0.25.
- `--labels_csv` (optional): The file name of the csv file containing labels and file names. File must be in `data_dir` directory. 


To predict the label of an input image, use the following script:

```bash
conda activate ood
python predict.py -i <image_file_name> [-ood <ood_method>] [-dd <data_directory>] [-m <model_path>]
```

### Arguments

- `-i`, `--image`: The name of the image file you want to predict. This argument is required.
- `-ood`, `--ood-method`: Optional. Specifies the Out-of-Distribution detection method to use. Leave blank if not using OOD detection.
- `-dd`, `--data-dir`: Optional. The directory where your images are stored. Defaults to `data/x_ray`.
- `-m`, `--model`: Optional. The path to the trained model file. Defaults to `model/vgg16-0.96-full_model.h5`.
