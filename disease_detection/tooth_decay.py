# %%
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import os, json, pathlib, shutil, PIL
import itertools

# %%
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image

# %% [markdown]
# https://www.kaggle.com/datasets/snginh/toothdecay/code

# %%
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.densenet import DenseNet121

# %%
densenet_layer = DenseNet121(weights= 'imagenet', include_top= False, input_shape=(224, 224, 3))
inception_layer = InceptionV3(weights= 'imagenet', include_top= False, input_shape=(224,224,3))

# %%
test_folder_dir = "tooth_decay/teeth_dataset/test/"
train_folder_dir = "tooth_decay/teeth_dataset/train/"
test_file_dir = pathlib.Path(test_folder_dir)
train_file_dir = pathlib.Path(train_folder_dir)
print(train_file_dir.exists())

# %%
test_dataset = pd.read_csv("tooth_decay/teeth_dataset/test.csv")
train_dataset = pd.read_csv("tooth_decay/teeth_dataset/train.csv")

# %%
test_img_cnt = len(test_dataset['images'])
#train_img_cnt = len(train_dataset['images'])
train_img_cnt = 35
print(test_img_cnt)
print(train_img_cnt)

# %%
class_names = [name for name in os.listdir(test_folder_dir) if os.path.isdir(os.path.join(test_folder_dir, name))]
print(class_names)

# %%
datagen = ImageDataGenerator(rescale=1./255)
batch_size = 10

# %%
def extract_features(directory, data_num, class_num, feature_shape, pretrained_model):
  features = np.zeros(shape = feature_shape)
  labels = np.zeros(shape=(data_num, class_num))
  generator = datagen.flow_from_directory(directory, target_size=(224, 224), batch_size = batch_size, class_mode= 'categorical', shuffle=False)
  i = 0
  for inputs_batch, labels_batch in generator:
    features_batch = pretrained_model.predict(inputs_batch)
    features[i * batch_size : (i+1) * batch_size] = features_batch
    labels[i * batch_size : (i+1) * batch_size] = labels_batch
    i += 1
    if i * batch_size >= data_num:
      break

  return features, labels

# %%
inception_final_layer = list(inception_layer.layers)[-1].output_shape
inception_final_layer = list(inception_final_layer)
print("final layer of VGG16 : " +  str(list(inception_layer.layers)[-1]) + " and its shape : " + str(inception_final_layer))

inc_conv_layers = []
for l in range(len(inception_layer.layers)):
  layer = inception_layer.layers[l]
  if 'Conv' not in layer.__class__.__name__:
    continue
  inc_conv_layers.append((layer.name, layer.output.shape))

inc_conv_base_shape = []
for i in inception_final_layer:
  if i != None:
    inc_conv_base_shape.append(i)
print("conv_base_shape : ", inc_conv_base_shape)

train_inc_feat_shape = tuple([train_img_cnt] + inc_conv_base_shape)
print(train_inc_feat_shape)

test_inc_feat_shape = tuple([test_img_cnt] + inc_conv_base_shape)
print(test_inc_feat_shape)

inc_input_dimension = np.prod(inc_conv_base_shape)
print(inc_input_dimension)

# %%
train_inc_features, train_inc_labels = extract_features(train_file_dir, train_img_cnt, len(class_names), train_inc_feat_shape, inception_layer)

# %%
test_inc_features, test_inc_labels = extract_features(test_file_dir, test_img_cnt, len(class_names), test_inc_feat_shape, inception_layer)

# %%
train_inc_features = np.reshape(train_inc_features, (train_img_cnt, inc_input_dimension))
test_inc_features = np.reshape(test_inc_features, (test_img_cnt, inc_input_dimension))

# %%
# Add classifier on pre-trained model
inception_model = keras.models.Sequential()
inception_model.add(keras.layers.Dense(128, activation='relu', input_dim = inc_input_dimension))
inception_model.add(keras.layers.Dense(128, activation='relu'))
inception_model.add(keras.layers.Dense(len(class_names), activation = 'softmax'))
inception_model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# %%
pmt_order = np.random.permutation(np.arange(train_img_cnt))

# %%
inc_train_history = inception_model.fit(train_inc_features, train_inc_labels, epochs = 5, batch_size = batch_size)

# %%
inc_loss, inc_acc = inception_model.evaluate(test_inc_features, test_inc_labels)

# %%
inc_test_prediction_score = inception_model.predict(test_inc_features)

# %%
inc_test_predicted_label = np.argmax(inc_test_prediction_score, axis= -1)

# %%


# %%



