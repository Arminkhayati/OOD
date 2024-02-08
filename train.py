import os
import tensorflow as tf
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import (GlobalAveragePooling2D, Dense, Softmax, Input,
                          BatchNormalization, multiply, Activation, Add, Lambda, Dropout, Flatten)
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import Model
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report


DATA_DIR = "data/new_x_ray"
NUM_CLASSES = 5
IMG_SIZE = 224
TARGET_SIZE = (IMG_SIZE, IMG_SIZE)
BATCH_SIZE = 32
EPOCHS = 100

df = pd.read_csv("data/x_ray/labels_x_ray.csv")
with open("data/bad_files.txt", "r") as f:
    lines = f.readlines()
    bad_files = [l.strip() for l in lines]

filtered_df = df[~df['image_Addr'].isin(bad_files)].drop_duplicates()

filtered_df[['label_image']] = filtered_df[['label_image']] - 1
print(df.columns)
filtered_df['label_image'] = filtered_df['label_image'].apply(str)

train_files, valid_files = train_test_split(filtered_df['image_Addr'],
                                            test_size=0.25,
                                            random_state=2024,
                                            stratify=filtered_df['label_image'])

train_df = filtered_df[filtered_df['image_Addr'].isin(train_files)]
valid_df = filtered_df[filtered_df['image_Addr'].isin(valid_files)]

print(f"Validation Size = {valid_df.shape[0]}, Train Size = {train_df.shape[0]}")

counts = train_df['label_image'].value_counts()
print(counts)
class_weight = {}
for k in counts.keys():
    weight = (1 / counts[k]) * (train_df.shape[0]) / NUM_CLASSES
    class_weight[int(k)] = weight

print(class_weight)

train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    rotation_range=35,
    horizontal_flip=True,
    vertical_flip=True,
)

val_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_dataframe(
    train_df,
    directory=DATA_DIR,
    x_col='image_Addr',
    y_col='label_image',
    class_mode='categorical',
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    color_mode='rgb',
    shuffle=True
)
val_generator = val_datagen.flow_from_dataframe(
    valid_df,
    directory=DATA_DIR,
    x_col='image_Addr',
    y_col='label_image',
    class_mode='categorical',
    target_size=TARGET_SIZE,
    batch_size=BATCH_SIZE,
    color_mode='rgb',
    shuffle=False
)

output_path = "output"
weight_path = os.path.join(output_path, "weights.best.weight")

Path(output_path).mkdir(exist_ok=True, parents=True)
checkpoint = ModelCheckpoint(weight_path, monitor='val_loss', verbose=1,
                             save_best_only=True, mode='min', save_weights_only = True)

reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', factor=0.8, patience=2, verbose=1,
                                   mode='auto', min_delta=0.0001, cooldown=1, min_lr=0.0001)
early_stopping = EarlyStopping(patience=20, restore_best_weights=True)
callbacks_list = [checkpoint, reduceLROnPlat, early_stopping]

input_shape = (IMG_SIZE, IMG_SIZE, 3)
inputs = Input(shape=input_shape)

model = VGG16(weights='imagenet', input_tensor=inputs, include_top=False)
model.trainable = False
# Rebuild top
x = GlobalAveragePooling2D(name="avg_pool")(model.output)
x = BatchNormalization()(x)
top_dropout_rate = 0.2
x = Dropout(top_dropout_rate, name="top_dropout")(x)

x = Dense(64, activation='relu')(x)
x = Dropout(0.5)(x)
x = Dense(32, activation='relu')(x)
x = Dropout(0.2)(x)
outputs = Dense(NUM_CLASSES, activation="softmax", name="pred")(x)

# Compile
model = Model(inputs, outputs, name="EfficientNet")


optimizer = Adam(learning_rate=1e-2)
METRICS = [
    'categorical_accuracy',
    tf.keras.metrics.Precision(name="precision"),
    tf.keras.metrics.Recall(name="recall"),
]
model.compile(
    optimizer=optimizer,
    loss= "categorical_crossentropy",
    metrics=METRICS
    )
print(model.summary())
history = model.fit(train_generator,
                    steps_per_epoch =train_df.shape[0] // BATCH_SIZE,
                    validation_data = val_generator,
                    validation_steps =valid_df.shape[0] // BATCH_SIZE,
                    epochs = EPOCHS,
                    callbacks = callbacks_list,
                    class_weight=class_weight,
                    )

#%%
model.load_weights(weight_path)
model.save(os.path.join(output_path, 'full_model.h5'))



pred = model.predict(val_generator,
                     steps =valid_df.shape[0] // BATCH_SIZE, verbose=1)
predicted_class_indices = np.argmax(pred,axis=1)
labels = valid_df['label_image'][0:len(predicted_class_indices)].apply(int)
print('Accuracy on Test Data: %2.2f%%' % (accuracy_score(labels, predicted_class_indices)))
print(classification_report(labels, predicted_class_indices))
