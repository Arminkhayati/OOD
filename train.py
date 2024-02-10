import os
import pandas as pd
import numpy as np
from typing import Tuple, Dict
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from sklearn.metrics import accuracy_score, classification_report
import tensorflow as tf
import argparse

parser = argparse.ArgumentParser(description='Train a VGG16 model for X-ray classification.')
parser.add_argument('--data_dir', type=str, default="data/new_x_ray", help='Directory containing the dataset')
parser.add_argument('--num_classes', type=int, default=5, help='Number of classes')
parser.add_argument('--img_size', type=int, default=224, help='Size of the input images')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--epochs', type=int, default=100, help='Number of epochs to train')
parser.add_argument('--test_size', type=float, default=0.25, help='Validation split size')
parser.add_argument('--bad_files', type=str, default="strict_bad_files.txt", help='Bad files to ignore')
parser.add_argument('--labels_csv', type=str, default="labels_x_ray.csv", help='Dataset CSV info file')
args = parser.parse_args()

# Constants
TARGET_SIZE: Tuple[int, int] = (args.img_size, args.img_size)
OUTPUT_PATH: str = "output"
WEIGHT_PATH: str = os.path.join(OUTPUT_PATH, "weights.best.weight")

def load_and_preprocess_data(test_size: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Loads the data, cleans it, and splits it into training and validation sets.

    Args:
        test_size (float): The proportion of the dataset to include in the validation split.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: The training and validation dataframes.
    """
    """Load and preprocess dataset."""
    df = pd.read_csv(os.path.join(args.data_dir, args.labels_csv))
    with open(os.path.join(args.data_dir, args.bad_files), "r") as f:
        bad_files = [line.strip() for line in f.readlines()]

    # Filter out bad files and adjust labels
    filtered_df = df[~df['image_Addr'].isin(bad_files)].drop_duplicates()
    # filtered_df = df.drop_duplicates()
    filtered_df['label_image'] = (filtered_df['label_image'] - 1).astype(str)

    # Split dataset
    train_files, valid_files = train_test_split(filtered_df['image_Addr'], test_size=test_size,
                                                random_state=2024, stratify=filtered_df['label_image'])
    train_df = filtered_df[filtered_df['image_Addr'].isin(train_files)]
    valid_df = filtered_df[filtered_df['image_Addr'].isin(valid_files)]

    return train_df, valid_df


def create_data_generators(train_df: pd.DataFrame, valid_df: pd.DataFrame) -> Tuple[ImageDataGenerator, ImageDataGenerator]:
    """
    Creates training and validation data generators.

    Args:
        train_df (pd.DataFrame): The training dataframe.
        valid_df (pd.DataFrame): The validation dataframe.

    Returns:
        Tuple[ImageDataGenerator, ImageDataGenerator]: The training and validation data generators.
    """
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=35,
        horizontal_flip=True,
        vertical_flip=True,
    )
    val_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_dataframe(
        train_df,
        directory=args.data_dir,
        x_col='image_Addr',
        y_col='label_image',
        class_mode='categorical',
        target_size=TARGET_SIZE,
        batch_size=args.batch_size,
        color_mode='rgb',
        shuffle=True
    )
    val_generator = val_datagen.flow_from_dataframe(
        valid_df,
        directory=args.data_dir,
        x_col='image_Addr',
        y_col='label_image',
        class_mode='categorical',
        target_size=TARGET_SIZE,
        batch_size=args.batch_size,
        color_mode='rgb',
        shuffle=False
    )
    return train_generator, val_generator


def calculate_class_weights(train_df: pd.DataFrame) -> Dict[int, float]:
    """
    Calculates the class weights for imbalanced datasets.

    Args:
        train_df (pd.DataFrame): The training dataframe.

    Returns:
        Dict[int, float]: The calculated class weights.
    """
    counts = train_df['label_image'].value_counts()
    class_weight = {int(k): (1 / counts[k]) * (train_df.shape[0]) / args.num_classes for k in counts.keys()}
    return class_weight


def build_and_compile_model() -> Model:
    """
    Builds and compiles the VGG16-based model.

    Returns:
        Model: The compiled TensorFlow model.
    """
    inputs = Input(shape=(args.img_size, args.img_size, 3))
    base_model = VGG16(weights='imagenet', include_top=False, input_tensor=inputs)
    base_model.trainable = False

    x = GlobalAveragePooling2D(name="avg_pool")(base_model.output)
    x = BatchNormalization()(x)
    x = Dropout(0.2, name="top_dropout")(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(32, activation='relu')(x)
    x = Dropout(0.2)(x)
    outputs = Dense(args.num_classes, activation="softmax", name="pred")(x)

    model = Model(inputs, outputs, name="VGG16_Custom")
    optimizer = Adam(learning_rate=1e-2)
    metrics = ['categorical_accuracy',
               tf.keras.metrics.Precision(name="precision"),
               tf.keras.metrics.Recall(name="recall")]

    model.compile(optimizer=optimizer, loss="categorical_crossentropy", metrics=metrics)
    return model


def setup_callbacks() -> list:
    """
    Sets up the callbacks for model training.

    Returns:
        list: The list of callbacks.
    """
    Path(OUTPUT_PATH).mkdir(exist_ok=True, parents=True)
    checkpoint = ModelCheckpoint(WEIGHT_PATH, monitor='val_loss', verbose=1, save_best_only=True, mode='min', save_weights_only=True)
    reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=2, verbose=1, mode='auto', min_delta=0.0001, cooldown=1, min_lr=0.0001)
    early_stopping = EarlyStopping(patience=20, restore_best_weights=True)
    return [checkpoint, reduceLROnPlat, early_stopping]

def main():
    """
    Main function to run the model training and evaluation.
    """
    train_df, valid_df = load_and_preprocess_data(args.test_size)
    train_generator, val_generator = create_data_generators(train_df, valid_df)
    class_weight = calculate_class_weights(train_df)
    model = build_and_compile_model()
    callbacks_list = setup_callbacks()

    history = model.fit(
        train_generator,
        steps_per_epoch=train_df.shape[0] // args.batch_size,
        validation_data=val_generator,
        validation_steps=valid_df.shape[0] // args.batch_size,
        epochs=args.epochs,
        callbacks=callbacks_list,
        class_weight=class_weight,
    )

    # Save the full model
    model.load_weights(WEIGHT_PATH)
    model.save(os.path.join(OUTPUT_PATH, 'full_model.h5'))

    # Evaluation
    pred = model.predict(val_generator, steps=valid_df.shape[0] // args.batch_size, verbose=1)
    predicted_class_indices = np.argmax(pred, axis=1)
    labels = valid_df['label_image'][:len(predicted_class_indices)].astype(int)
    print('Accuracy on Test Data: %2.2f%%' % (accuracy_score(labels, predicted_class_indices)))
    print(classification_report(labels, predicted_class_indices))



if __name__ == "__main__":
    main()