# -*- coding: utf-8 -*-
# Commented out IPython magic to ensure Python compatibility.
try:
#     %tensorflow_version 2.x
except Exception:
    pass
import tensorflow as tf

import os
import numpy as np
import matplotlib.pyplot as plt

# DO NOT EDIT THE FOLLOWING LINES
# THESE LINES ARE FOR REPRODUCIBILITY
np.random.seed(1)
tf.random.set_seed(1)

"""### 1. Load the Oxford-IIIT Pet dataset
Use the Oxford-IIIT pet dataset which contains 37 category pet images with roughly 200 images for each class. 

![Oxford-IIIT Pet Dataset](https://github.com/keai-kaist/CS470-Spring-2022-/blob/main/Lab2/May%205/images/oxford-iiit-pet.png?raw=true)

Please note that the images have large variations in scale, pose and lighting. Let's import and load the Oxford-IIIT pet dataset using TensorFlow Datasets:
"""

import tensorflow_datasets as tfds

labels = [
    'abyssinian', 'american_bulldog', 'american_pit_bull_terrier', 'basset_hound',
    'beagle', 'bengal', 'birman', 'bombay', 'boxer', 'british_shorthair', 'chihuahua',
    'egyptian_mau', 'english_cocker_spaniel', 'english_setter', 'german_shorthaired',
    'great_pyrenees', 'havanese', 'japanese_chin', 'keeshond', 'leonberger', 'maine_coon',
    'miniature_pinscher', 'newfoundland', 'persian', 'pomeranian', 'pug', 'ragdoll',
    'russian_blue', 'saint_bernard', 'samoyed', 'scottish_terrier', 'shiba_inu', 'siamese',
    'sphynx', 'staffordshire_bull_terrier', 'wheaten_terrier', 'yorkshire_terrier',
]

dataset, metadata = tfds.load('oxford_iiit_pet', with_info=True, as_supervised=True)
raw_train, raw_test = dataset['train'], dataset['test']

"""Let's visualize what some of these images and their corresponding training labels look like."""

plt.figure(figsize=(14, 12))

for index, (image, label) in enumerate(dataset['train'].take(36)):
    plt.subplot(6, 6, index + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(image)
    plt.xlabel(labels[label.numpy()])

"""### 2. Preprocess the dataset

Define a function to preprocess the dataset. The function should **scale the input channels** to a range of [-1, 1] and **resize the images** to a fixed size, `IMAGE_SIZE`.
"""

IMAGE_SIZE = (224, 224)

# TODO: Define a function to preprocess the dataset. 
#       The function should scale the input channels to a range of [-1, 1] and 
#                           resize the images to a fixed size, IMAGE_SIZE
def preprocessDataset(image, label):
  image = tf.cast(image, tf.float32)
  image = (image/127.5) - 1
  image = tf.image.resize(image, IMAGE_SIZE)
  return image, label

"""Apply the defined preprocessing function to `raw_train` and `raw_test`. Then, shuffle the dataset and combine them into batches."""

BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1024

# TODO: Apply the defined preprocessing function to `raw_train` and `raw_test`
#       Then, shuffle the dataset and combine them into batches

train = raw_train.map(preprocessDataset)
test = raw_test.map(preprocessDataset)

train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
test_batches = test.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)

"""### 3. Build the model
In this assignment, we are going to train the convolutional neural network using transfer learning.

Load `InceptionV3` model without the final classification layer using `tf.keras.applications.InceptionV3`. Then, freeze the model to prevent it from being trained.
"""

# TODO: Load InceptionV3 model without the final classification layer
#       Then, freeze the model to prevent it from being trained

IMG_SIDE = 224
IMG_SHAPE = (IMG_SIDE,IMG_SIDE, 3)
base_model = tf.keras.applications.InceptionV3(input_shape = IMG_SHAPE, include_top = False)
base_model.trainable = False
base_model.summary()

image_batch, label_batch = next(iter(train_batches.take(1)))
print(image_batch.shape)
print(label_batch.shape)

"""Define a convolutional neural network using the loaded `InceptionV3` to classify images of dogs and cats as their breeds. Then, compile your model with appropriate parameters."""

# TODO: Define a convolutional neural network using the loaded InceptionV3
#       Then, compile your model with appropriate parameters

feature_batch = base_model(image_batch)
print(image_batch.shape,  '->', feature_batch.shape)

global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
feature_batch_average = global_average_layer(feature_batch)
print(feature_batch_average.shape)

prediction_layer = tf.keras.layers.Dense(37, activation='softmax')
prediction_batch = prediction_layer(feature_batch_average)
print(prediction_batch.shape)

model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer
])

learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=learning_rate),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

"""### 4. Train the model

Now, train the model at least 10 epochs.
"""

# TODO: Train the model at leat 10 epochs

initial_epochs = 10

# Train the model
history = model.fit(
    train_batches,
    epochs=initial_epochs,
)

"""To fine-tune the model, unfreeze the top layers of the model. Please note that you should carefully choose layers to be frozen. Then, compile the model again with appropriate parameters."""

# TODO: Unfreeze the top layers of the model
#       Compile the model with appropriate parameters
base_model.trainable = True

print("Number of layers in the base model: ", len(base_model.layers))

# Fine tune from this layer onwards
fine_tune_at = 249

# TODO: Freeze all the layers before the `fine_tune_at` layer
for layer in base_model.layers[:fine_tune_at]:
    layer.trainable =  False

model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=learning_rate),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

"""
Fine-tune the model at least 10 epochs."""

# TODO: Fine-tune the model at least 10 epochs

fine_tune_epochs = 10

history_fine_tuned = model.fit(
    train_batches,
     epochs=fine_tune_epochs
     )

"""### 4. Evaluate accuracy

Evaluate the trained model using test dataset and print the test accuracy of the model.
"""

# TODO: Evaluate the model using test dataset

totLoss, totAccuracy = model.evaluate(test_batches)
print ('Test accuracy: ', totAccuracy)
