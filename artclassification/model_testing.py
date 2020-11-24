# -*- coding: utf-8 -*-
"""Model_Testing.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1SSBnwTovZPEFmYmNrGbw40uE3U70SjN0
"""

# Commented out IPython magic to ensure Python compatibility.
from __future__ import absolute_import, division, print_function, unicode_literals

try:
#   %tensorflow_version 2.x
except Exception:
  pass

import tensorflow as tf
import keras

from tensorflow.keras.models import Sequential

from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.preprocessing.image import array_to_img, img_to_array, load_img

import os
import numpy as np
import matplotlib.pyplot as plt

!pip3 install imagenetscraper

'''
imagenetscraper n07739344 apple
imagenetscraper n07753275 pineapple
imagenetscraper n07735510 pumpkin
imagenetscraper n07747607 orange
imagenetscraper n07756951 watermelon
'''

from bs4 import BeautifulSoup #BeautifulSoup is an HTML parsing library
import numpy as np
import requests
import cv2
import PIL.Image
import urllib

#(1) apple
apple_page = requests.get("http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n07739344")
print(apple_page.content)

apple_soup = BeautifulSoup(apple_page.content, 'html.parser')#puts the content of the website into the soup variable, each url on a different line
#print(soup)
#print(soup.prettify())


#(2) pineapple
pine_page = requests.get("http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n07753275")
print(pine_page.content)

pine_soup = BeautifulSoup(pine_page.content, 'html.parser')

#(3) pumpkin
pump_page = requests.get("http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n07735510")
print(pump_page.content)

pump_soup = BeautifulSoup(pump_page.content, 'html.parser')

#(4) orange
org_page = requests.get("http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n07747607")
print(org_page.content)

org_soup = BeautifulSoup(org_page.content, 'html.parser')

#(5) pumpkin
water_page = requests.get("http://www.image-net.org/api/text/imagenet.synset.geturls?wnid=n07756951")
print(water_page.content)

water_soup = BeautifulSoup(water_page.content, 'html.parser')

apple_str_soup=str(apple_soup)#convert soup to string so it can be split
type(apple_str_soup)
apple_split_urls=apple_str_soup.split('\r\n')#split so each url is a different possition on a list
print(len(apple_split_urls))#print the length of the list so you know how many urls you have

pine_str_soup=str(pine_soup)#convert soup to string so it can be split
type(pine_str_soup)
pine_split_urls=pine_str_soup.split('\r\n')#split so each url is a different possition on a list
print(len(pine_split_urls))#print the length of the list so you know how many urls you have

pump_str_soup=str(pump_soup)#convert soup to string so it can be split
type(pump_str_soup)
pump_split_urls=pump_str_soup.split('\r\n')#split so each url is a different possition on a list
print(len(pump_split_urls))#print the length of the list so you know how many urls you have

org_str_soup=str(org_soup)#convert soup to string so it can be split
type(org_str_soup)
org_split_urls=org_str_soup.split('\r\n')#split so each url is a different possition on a list
print(len(org_split_urls))#print the length of the list so you know how many urls you have

water_str_soup=str(water_soup)#convert soup to string so it can be split
type(water_str_soup)
water_split_urls=water_str_soup.split('\r\n')#split so each url is a different possition on a list
print(len(water_split_urls))#print the length of the list so you know how many urls you have

!mkdir /content/train 
!mkdir /content/train/apple
!mkdir /content/train/pineapple
!mkdir /content/train/pumpkin
!mkdir /content/train/orange
!mkdir /content/train/watermelon 

!mkdir /content/validation
!mkdir /content/validation/apple
!mkdir /content/validation/pineapple
!mkdir /content/validation/pumpkin
!mkdir /content/validation/orange
!mkdir /content/validation/watermelon

!mkdir /content/test
!mkdir /content/test/apple
!mkdir /content/test/pineapple
!mkdir /content/test/pumpkin
!mkdir /content/test/orange
!mkdir /content/test/watermelon

!ls /
!ls /content/
!ls /content/train/
!ls /content/validation
!ls /content/test

img_rows, img_cols = 224, 224
input_shape = (img_rows, img_cols, 3)

def url_to_image(url):
	# download the image, convert it to a NumPy array, and then read
	# it into OpenCV format
	resp = urllib.request.urlopen(url)
	image = np.asarray(bytearray(resp.read()), dtype="uint8")
	image = cv2.imdecode(image, cv2.IMREAD_COLOR)
 
	# return the image
	return image

#TRAINING IMG
n_of_training_images = 100

def copy_image_to_directory(current_url, path_save, num_files):
  for progress in range(num_files):
    if(progress %20 == 0):
      print(progress)
    if not current_url[progress] == None:
      try:
        img = url_to_image(current_url[progress])
        if(len(img.shape)) == 3:
          save_path = path_save+str(progress)+'.jpg'
          cv2.imwrite(save_path, img)
      except:
        None    

copy_image_to_directory(apple_split_urls, '/content/train/apple/apple_img', 100)
copy_image_to_directory(pine_split_urls, '/content/train/pineapple/pine_img', 100)
copy_image_to_directory(pump_split_urls, '/content/train/pumpkin/pump_img', 100)
copy_image_to_directory(org_split_urls, '/content/train/orange/org_img', 100)
copy_image_to_directory(water_split_urls, '/content/train/watermelon/water_img', 100)
        
        
#Validation data:
n_of_valid_images = 50

def copy_valid_to_directory(current_url, path_save, num_files):
  for progress in range(num_files):
    if(progress%20 == 0):
      print(progress)
    if not current_url[progress] == None:
      try:
        img = url_to_image(current_url[n_of_training_images + progress])
        if(len(img.shape)) == 3:
          save_path = path_save+str(progress)+'.jpg'
          cv2.imwrite(save_path, img)
      except:
        None      

copy_valid_to_directory(apple_split_urls, '/content/validation/apple/apple_img', 50)
copy_valid_to_directory(pine_split_urls, '/content/validation/pineapple/pine_img', 50)
copy_valid_to_directory(pump_split_urls, '/content/validation/pumpkin/pump_img', 50)
copy_valid_to_directory(org_split_urls, '/content/validation/orange/org_img', 50)
copy_valid_to_directory(water_split_urls, '/content/validation/watermelon/water_img', 50)

# TEST DATA

def copy_test_to_directory(current_url, path_save, num_files):
  for progress in range(num_files):
    if(progress%20 == 0):
      print(progress)
    if not current_url[progress] == None:
      try:
        img = url_to_image(current_url[n_of_training_images + n_of_valid_images + progress])
        if(len(img.shape)) == 3:
          save_path = path_save+str(progress)+'.jpg'
          cv2.imwrite(save_path, img)
      except:
        None  

copy_test_to_directory(apple_split_urls, '/content/test/apple/app_img', 30)
copy_test_to_directory(pine_split_urls, '/content/test/pineapple/pine_img', 30)
copy_test_to_directory(pump_split_urls, '/content/test/pumpkin/pump_img', 30)
copy_test_to_directory(org_split_urls, '/content/test/orange/org_img', 30)
copy_test_to_directory(water_split_urls, '/content/test/watermelon/water_img', 30)
        
print("\nTRAIN:\n")          
print("\nlist the files inside apple directory:\n")        
!ls /content/train/apple #list the files inside ships
print("\nlist the files inside pineapple directory:\n")
!ls /content/train/pineapple/ #list the files inside bikes
print("\nlist the files inside pumpkin directory:\n")        
!ls /content/train/pumpkin/ #list the files inside ships
print("\nlist the files inside orange directory:\n")
!ls /content/train/orange/ #list the files inside bikes
print("\nlist the files inside watermelon directory:\n")        
!ls /content/train/watermelon/ #list the files inside ships

print("\nVALIDATION:\n")
print("\nlist the files inside apple directory:\n")        
!ls /content/validation/apple 
print("\nlist the files inside pineapple directory:\n")
!ls /content/validation/pineapple/ 
print("\nlist the files inside pumpkin directory:\n")        
!ls /content/validation/pumpkin/
print("\nlist the files inside orange directory:\n")
!ls /content/validation/orange/
print("\nlist the files inside watermelon directory:\n")        
!ls /content/validation/watermelon/


print("\nTEST:\n")
print("\nlist the files inside apple directory:\n")        
!ls /content/test/apple
print("\nlist the files inside pineapple directory:\n")
!ls /content/test/pineapple/ #list the files inside bikes
print("\nlist the files inside pumpkin directory:\n")        
!ls /content/test/pumpkin/ #list the files inside ships
print("\nlist the files inside orange directory:\n")
!ls /content/test/orange/ #list the files inside bikes
print("\nlist the files inside watermelon directory:\n")        
!ls /content/test/watermelon/ #list the files inside ships

import pathlib

root_directory = pathlib.Path('/content');
train_dir = pathlib.Path(root_directory, 'train')
print(train_dir)
validation_dir = pathlib.Path(root_directory, 'validation')
print(validation_dir)
test_dir = pathlib.Path(root_directory, 'test')
print(test_dir)

train_apple_dir = pathlib.Path(train_dir, 'apple')
num_train_apple = len(os.listdir(train_apple_dir))
print('Training dataset for apple: ',num_train_apple)
valid_apple_dir = pathlib.Path(validation_dir, 'apple')
num_valid_apple = len(os.listdir(valid_apple_dir))
print('Validation dataset for apple: ',num_valid_apple)
test_apple_dir = pathlib.Path(test_dir, 'apple')
num_test_apple = len(os.listdir(test_apple_dir))
print('Test dataset for apple: ',num_test_apple)

train_pine_dir = pathlib.Path(train_dir, 'pineapple')
num_train_pine = len(os.listdir(train_pine_dir))
print('Training dataset for pineapple: ',num_train_pine)
valid_pineapple_dir = pathlib.Path(validation_dir, 'pineapple')
num_valid_pineapple = len(os.listdir(valid_pineapple_dir))
print('Validation dataset for pineapple: ',num_valid_pineapple)
test_pineapple_dir = pathlib.Path(test_dir, 'pineapple')
num_test_pineapple = len(os.listdir(test_pineapple_dir))
print('Test dataset for pineapple: ',num_test_pineapple)

train_pump_dir = pathlib.Path(train_dir, 'pumpkin')
num_train_pump = len(os.listdir(train_pump_dir))
print('Training dataset for pumpkin: ', num_train_pump)
valid_pump_dir = pathlib.Path(validation_dir, 'pumpkin')
num_valid_pump = len(os.listdir(valid_pump_dir))
print('Validation dataset for pumpkin: ',num_valid_pump)
test_pump_dir = pathlib.Path(test_dir, 'pumpkin')
num_test_pump = len(os.listdir(test_pump_dir))
print('Test dataset for pumpkin: ',num_test_pump)

train_org_dir = pathlib.Path(train_dir, 'orange')
num_train_org = len(os.listdir(train_org_dir))
print('Training dataset for orange: ', num_train_org)
valid_org_dir = pathlib.Path(validation_dir, 'orange')
num_valid_org = len(os.listdir(valid_org_dir))
print('Validation dataset for orange: ',num_valid_org)
test_org_dir = pathlib.Path(test_dir, 'orange')
num_test_org = len(os.listdir(test_org_dir))
print('Test dataset for orange: ',num_test_org)

train_water_dir = pathlib.Path(train_dir, 'watermelon')
num_train_water = len(os.listdir(train_water_dir))
print('Training dataset for watermelon: ', num_train_water)
valid_water_dir = pathlib.Path(validation_dir, 'watermelon')
num_valid_water = len(os.listdir(valid_water_dir))
print('Validation dataset for watermelon: ',num_valid_water)
test_water_dir = pathlib.Path(test_dir, 'watermelon')
num_test_water = len(os.listdir(test_water_dir))
print('Test dataset for watermelon: ',num_test_water)


print('==================================================')
train_image_count = len(list(train_dir.glob('*/*.jpg')))
print('Total train data: ',train_image_count)

valid_image_count = len(list(validation_dir.glob('*/*.jpg')))
print('Total Validation data: ', valid_image_count)

test_image_count = len(list(test_dir.glob('*/*.jpg')))
print('Total Test data: ', test_image_count)

CLASS_NAMES = np.array([item.name for item in test_dir.glob('*') if item.name != "LICENSE.txt"])
CLASS_NAMES
print()
print(CLASS_NAMES)

batch_size = 40
epochs = 15

train_image_generator = ImageDataGenerator(rescale=1./255)
validation_image_generator = ImageDataGenerator(rescale=1./255)
test_image_generator = ImageDataGenerator(rescale=1./255)

print(train_image_generator)

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(img_rows, img_cols), 
                                                          class_mode='categorical')

print(train_data_gen.image_shape)

valid_data_gen = validation_image_generator.flow_from_directory(batch_size = batch_size,
                                                                directory = validation_dir,
                                                                target_size = (img_rows, img_cols),
                                                                class_mode='categorical')
print(valid_data_gen.image_shape)


test_data_gen = test_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=test_dir,
                                                              shuffle=False,
                                                              target_size=(img_rows, img_cols),
                                                              class_mode='categorical')

len(test_data_gen)
print(test_data_gen.image_shape)

sample_training_images, sample_training_labels = next(train_data_gen)
sample_validation_images, sample_validation_labels = next(valid_data_gen)
sample_test_images, sample_test_labels = next(test_data_gen)

model = Sequential([
    Conv2D(16, (3,3), padding='same', activation='relu', input_shape=(img_rows, img_cols ,3)),
    MaxPooling2D((2,2)),
    Conv2D(32, (3,3), padding='same', activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), padding='valid', activation='relu'),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), padding='valid', activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(), 
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(5, activation='softmax')
])

model.compile(optimizer='adam', 
              loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True), 
              metrics=['accuracy'])

model.summary()

history = model.fit_generator(train_data_gen, steps_per_epoch= train_image_count // batch_size, epochs=epochs,
    validation_data=valid_data_gen, validation_steps=valid_image_count // batch_size)

test_loss, test_acc = model.evaluate_generator(test_data_gen, steps = 50)
print(test_acc)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

#epochs_range = range(epochs)
epochs_range = range(1, len(loss) + 1)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)

plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Testing Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Testing Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Testing Loss')
plt.legend(loc='upper right')
plt.title('Training and Testing Loss')
plt.show()

from sklearn.metrics import classification_report, confusion_matrix
pred = model.predict(test_data_gen)
print(pred)
predictions = model.predict_generator(test_data_gen, test_image_count // batch_size+1)
print(predictions.size)
print(test_data_gen.classes)
y_pred = np.argmax(predictions, axis=1)
print(y_pred)
print('Confusion Matrix')
print(confusion_matrix(test_data_gen.classes, y_pred))
print('Classification Report')

print(classification_report(test_data_gen.classes, y_pred, target_names=CLASS_NAMES))

print('\nPredicted class for test image 0: ', np.argmax(predictions[0]), CLASS_NAMES[np.argmax(predictions[0])])

from keras.applications import VGG16

conv_base = VGG16(weights='imagenet', include_top = False, input_shape=(img_rows, img_cols, 3))

conv_base.summary()

datagen = ImageDataGenerator(rescale = 1./255)
batch_size = 32
epochs = 20

def extract_features(directory, sample_count, shuffle_value):
  features = np.zeros(shape=(sample_count, 7, 7, 512))
  labels = np.zeros(shape=(sample_count, 5)) 
  generator = datagen.flow_from_directory(directory, target_size = (img_cols,img_rows), 
                                          batch_size = batch_size, class_mode = 'categorical', 
                                          shuffle = shuffle_value)
  i = 0
  for inputs_batch, labels_batch in generator:
      features_batch = conv_base.predict(inputs_batch)
      features[i * batch_size : (i + 1) * batch_size] = features_batch
      labels[i * batch_size : (i + 1) * batch_size] = labels_batch
      i+=1
      if i * batch_size >= sample_count:
        break
  return features, labels, generator

train_features, train_labels, test_generator = extract_features(train_dir, train_image_count, True)
validation_features, validation_labels, validation_generator = extract_features(validation_dir, valid_image_count, False)
test_features, test_labels, test_generator = extract_features(test_dir, test_image_count, False)

train_features = np.reshape(train_features, (train_image_count, 7 * 7 * 512))
print(train_features.shape)

validation_features = np.reshape(validation_features, (valid_image_count, 7 * 7 * 512))
print(validation_features.shape)

test_features = np.reshape(test_features, (test_image_count, 7 * 7 * 512))
print(test_features.shape)

from keras import models
from keras import layers
from keras import optimizers

model = models.Sequential()
model.add(layers.Dense(512, activation = 'relu', input_dim = 7*7*512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(5, activation='softmax'))

model.compile(optimizer=optimizers.Adam(),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit(train_features, train_labels, epochs = epochs,
                    batch_size= batch_size,
                    validation_data = (validation_features, validation_labels))

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss=history.history['loss']
val_loss=history.history['val_loss']

#epochs_range = range(epochs)
epochs_range = range(1, len(loss) + 1)

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)

plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Testing Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Testing Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Testing Loss')
plt.legend(loc='upper right')
plt.title('Training and Testing Loss')
plt.show()

test_loss, test_acc = model.evaluate(test_features, test_labels)
print(test_acc)

from sklearn.metrics import classification_report, confusion_matrix
predictions = model.predict_classes(test_features)
print(predictions.size)
#y_pred = np.argmax(predictions, axis=1)
y_pred = predictions
print('Confusion Matrix')
#print(test_generator.classes)
print(confusion_matrix(test_generator.classes, y_pred))
print('Classification Report')

print(classification_report(test_generator.classes, y_pred, target_names=CLASS_NAMES))

def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Reds):
  if normalize:
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm = cm*100
    print('\nNormalized Confusion Matrix')
  else: 
    print('\nConfusion Matrix, without Normalization')
   
  print(cm)
  print()

  plt.imshow(cm, interpolation='nearest', cmap=cmap)
  plt.title(title)
  plt.colorbar()
  tick_marks = np.arange(len(classes))
  plt.xticks(tick_marks, classes, rotation=90)
  plt.yticks(tick_marks, classes)


  fmt = '.0f' if normalize else 'd'
  thresh = cm.max()/2.
  for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
     plt.text(j, i, format(cm[i, j], fmt),
               horizontalalignment="center",
               color="white" if cm[i, j] > thresh else "black")
  plt.tight_layout()
  plt.ylabel('True label')
  plt.xlabel('Predicted label')
  plt.show()

from sklearn.metrics import confusion_matrix
import itertools

#y_pred = np.argmax(predictions, axis = 1)
y_pred = predictions
cnf_matrix = confusion_matrix(test_data_gen.classes, y_pred)
np.set_printoptions(precision=2)

plt.figure()
plot_confusion_matrix(cnf_matrix, classes=CLASS_NAMES, title ='Confusion Matrix without Normalization')

valid_filenames = validation_generator.filenames

valid_ground_truth = validation_generator.classes

val_label2index = validation_generator.class_indices

# getting the mapping from class index to class label
val_idx2label = dict((v,k) for k,v in val_label2index.items())

val_predictions = model.predict_classes(validation_features)
valid_prob = model.predict(validation_features)

val_errors = np.where(val_predictions != valid_ground_truth)[0]
print("No of error = {}/{}".format(len(val_errors), valid_image_count))

test_filenames = test_generator.filenames
test_ground_truth = test_generator.classes
test_label2index = test_generator.class_indices
test_idx2label = dict((v, k) for k, v in test_label2index.items())

test_predictions = model.predict_classes(test_features)
test_prob = model.predict(test_features)

test_errors = np.where(test_predictions != test_ground_truth)[0]
print("No of error = {}/{}".format(len(test_errors), test_image_count))

print(test_predictions)
print(test_prob)

for i in range(len(test_errors)):
  pred_class = np.argmax(test_prob[test_errors[i]])
  print(pred_class)
  pred_label = test_idx2label[pred_class]
  print(pred_label)

  print('True label : {}, Prediction : {}, confidence : {:.3f}'.format(
      test_filenames[test_errors[i]].split('/')[0],
      pred_label,
      test_prob[test_errors[i]][pred_class]))
  

  original = load_img('{}/{}'.format(test_dir, test_filenames[test_errors[i]]))
  plt.imshow(original)
  plt.show()

