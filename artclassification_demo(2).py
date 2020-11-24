# -*- coding: utf-8 -*-
"""ArtClassification_Demo(2).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1qxqs0GVaZUmHCLC1ho-Nz2d5hMG9DIKn
"""

pip install autokeras

pip install tensorflow==2.1.0

# Commented out IPython magic to ensure Python compatibility.
from __future__ import absolute_import, division, print_function, unicode_literals

try:
#   %tensorflow_version 2.x
except Exception:
  pass

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing.image import array_to_img, img_to_array, load_img

import os
import numpy as np
import matplotlib.pyplot as plt

AUTOTUNE = tf.data.experimental.AUTOTUNE

import IPython.display as display
from PIL import Image

import zipfile

from google.colab import drive
drive.mount('/content/gdrive')

import pathlib

root_directory = pathlib.Path('/content/gdrive/My Drive/art_semi')
print(root_directory)

train_data_dir = pathlib.Path(root_directory, 'train1')
print(train_data_dir)

test_data_dir = pathlib.Path(root_directory, 'test1')
print(test_data_dir)

train_image_count = len(list(train_data_dir.glob('*/*.jpg')))
print(train_image_count)

test_image_count = len(list(test_data_dir.glob('*/*.jpg')))
print(test_image_count)

CLASS_NAMES = np.array([item.name for item in test_data_dir.glob('*') if item.name != "LICENSE.txt"])
print(CLASS_NAMES)

train_abstracticisms = list(train_data_dir.glob('abstracticism/*'))


#for image_path in train_abstracticisms[:5]:
 # display.display(Image.open(str(image_path)))

train_abstractionism_dir = os.path.join(train_data_dir, 'abstracticism')
print(train_abstractionism_dir)
test_abstractionism_dir = os.path.join(test_data_dir, 'abstracticism')
print(test_abstractionism_dir)

train_baroque_dir = os.path.join(train_data_dir, 'baroque')
print(train_baroque_dir)
test_baroque_dir = os.path.join(test_data_dir, 'baroque')
print(test_baroque_dir)

train_impressionism_dir = os.path.join(train_data_dir, 'impressionism')
print(train_impressionism_dir)
test_impressionism_dir = os.path.join(test_data_dir, 'impressionism')
print(test_impressionism_dir)

train_minimalism_dir = os.path.join(train_data_dir, 'minimalism')
print(train_minimalism_dir)
test_minimalism_dir = os.path.join(test_data_dir, 'minimalism')
print(test_minimalism_dir)

train_popart_dir = pathlib.Path(train_data_dir, 'popart')
print(train_popart_dir)
test_popart_dir = pathlib.Path(test_data_dir, 'popart')
print(test_popart_dir)

num_abstractionism_tr = len(os.listdir(train_abstractionism_dir))
print('train abstractionism num:', num_abstractionism_tr)
num_abstractionism_tt = len(os.listdir(test_abstractionism_dir))
print('test abstractionism num: ', num_abstractionism_tt)

num_baroque_tr = len(os.listdir(train_baroque_dir))
print('train baroque num: ', num_baroque_tr)
num_baroque_tt = len(os.listdir(test_baroque_dir))
print('test baroque num: ', num_baroque_tt)

num_impressionism_tr = len(os.listdir(train_impressionism_dir))
print('train impressionism num: ', num_impressionism_tr)
num_impressionism_tt = len(os.listdir(test_impressionism_dir))
print('test impressionism num: ', num_impressionism_tt)

num_minimalism_tr = len(os.listdir(train_minimalism_dir))
print('train minimalism num: ', num_minimalism_tr)
num_minimalism_tt = len(os.listdir(test_minimalism_dir))
print('test minimalism num: ', num_minimalism_tt)

num_popart_tr = len(os.listdir(train_popart_dir))
print('train popart num: ', num_popart_tr)
num_popart_tt = len(os.listdir(test_popart_dir))
print('test popart num: ', num_popart_tt)

total_train = num_abstractionism_tr + num_baroque_tr + num_impressionism_tr + num_minimalism_tr + num_popart_tr
total_test = num_abstractionism_tt + num_baroque_tt + num_impressionism_tt + num_minimalism_tt + num_popart_tt
print('total training datatset: ', total_train)
print('total test dataset: ', total_test)

train_image_generator = ImageDataGenerator(rescale=1./255) 
validation_image_generator = ImageDataGenerator(rescale=1./255)

BATCH_SIZE = 50
epochs = 15
IMG_HEIGHT = 200
IMG_WIDTH = 200

# train_data_gen = train_image_generator.flow_from_directory(directory=str(train_data_dir),
#                                                      batch_size=BATCH_SIZE,
#                                                      shuffle=True,
#                                                      target_size=(IMG_HEIGHT, IMG_WIDTH),
#                                                      classes = list(CLASS_NAMES))

# val_data_gen = validation_image_generator.flow_from_directory(directory=str(test_data_dir),
#                                                      batch_size=BATCH_SIZE,
#                                                      shuffle=True,
#                                                      target_size=(IMG_HEIGHT, IMG_WIDTH),
#                                                      classes = list(CLASS_NAMES))

train_data_gen = train_image_generator.flow_from_directory(batch_size=total_train,
                                                           directory=train_data_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH), 
                                                          class_mode='categorical')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=total_test,
                                                              directory=test_data_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='categorical')

sample_training_images, sample_training_labels = next(train_data_gen)
sample_test_images, sample_test_labels = next(val_data_gen)

print(sample_test_images.shape)
print(sample_test_labels.shape)
print(sample_test_images.size)
print(sample_test_labels.size)

print(sample_training_images.shape)
print(sample_training_labels.shape)
print(sample_training_images.size)
print(sample_training_labels.size)

import autokeras as ak

clf = ak.ImageClassifier(max_trials=10)
clf.fit(sample_training_images, sample_training_labels, epochs = 3)
#clf.fit(sample_training_images, sample_training_labels, validation_split=0.15, epochs = 3)

predicted_y = clf.predict(sample_test_images)
print(predicted_y)
print(predicted_y.size)

print(clf.evaluate(sample_test_images, sample_test_labels))

print(clf.evaluate(sample_training_images, sample_training_labels))

clf.fit(sample_training_images, sample_training_labels, validation_split=0.15, epochs = 3)













def show_batch(image_batch, label_batch):
  plt.figure(figsize=(10,10))
  for n in range(25):
      ax = plt.subplot(5,5,n+1)
      plt.imshow(image_batch[n])
      plt.title(CLASS_NAMES[label_batch[n]==1][0].title())
      plt.axis('off')

image_batch, label_batch = next(train_data_gen)
show_batch(image_batch, label_batch)

print('-------------------------------')
print()
print()
print()
image_batch, label_batch = next(val_data_gen)
show_batch(image_batch, label_batch)

train_list_ds = tf.data.Dataset.list_files(str(train_data_dir/'*/*'))
test_list_ds = tf.data.Dataset.list_files(str(test_data_dir/'*/*'))

for f in train_list_ds.take(5):
  print(f.numpy())

for f in test_list_ds.take(5):
  print(f.numpy())

def get_label(file_path):
  # convert the path to a list of path components
  parts = tf.strings.split(file_path, os.path.sep)
  # The second to last is the class-directory
  return parts[-2] == CLASS_NAMES

def decode_img(img):
  # convert the compressed string to a 3D uint8 tensor
  img = tf.image.decode_jpeg(img, channels=3)
  # Use `convert_image_dtype` to convert to floats in the [0,1] range.
  img = tf.image.convert_image_dtype(img, tf.float32)
  # resize the image to the desired size.
  return tf.image.resize(img, [IMG_WIDTH, IMG_HEIGHT])

def process_path(file_path):
  label = get_label(file_path)
  # load the raw data from the file as a string
  img = tf.io.read_file(file_path)
  img = decode_img(img)
  return img, label

# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
train_labeled_ds = train_list_ds.map(process_path, num_parallel_calls=AUTOTUNE)

for image, label in train_labeled_ds.take(3):
  print("Image shape: ", image.numpy().shape)
  print("Label: ", label.numpy())

from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape) # (60000, 28, 28)
print(y_train.shape) # (60000,)
print(y_train[:3]) # array([7, 2, 1], dtype=uint8)



from google.colab import drive
drive.mount('/content/gdrive')

import json
with open('/content/gdrive/My Drive/testcase4/testcase4_result.json', "r") as json_file:
  json_data = json.load(json_file)

print(json.dumps(json_data, indent=4))

print(json_data["confusion_matrix"]["minimalism"]["precision"])
print(json_data["test_accuracy"])

