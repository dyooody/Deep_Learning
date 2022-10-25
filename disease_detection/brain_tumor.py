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
# https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

# %%
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.densenet import DenseNet121

# %%
vgg_layer = VGG16(weights= 'imagenet', include_top= False, input_shape=(224, 224, 3))
'''
densenet_layer = DenseNet121(weights= 'imagenet', include_top= False, input_shape=(200, 200, 3))
'''

# %%
test_folder_dir = "brain_tumor/Testing/"
train_folder_dir = "brain_tumor/Training/"
test_file_dir = pathlib.Path(test_folder_dir)
train_file_dir = pathlib.Path(train_folder_dir)
print(train_file_dir.exists())

# %%
test_img_cnt = len(list(test_file_dir.glob("*/*.jpg")))
train_img_cnt = len(list(train_file_dir.glob("*/*.jpg")))
print(test_img_cnt)
print(train_img_cnt)

# %%
class_names = [name for name in os.listdir(test_folder_dir) if os.path.isdir(os.path.join(test_folder_dir, name))]
print(class_names)

# %%
#divide trian and validation data
total_train_nums = []
for class_name in class_names:
  class_dir = pathlib.Path(train_folder_dir, class_name)
  cl_length = len(list(class_dir.glob("*.jpg")))
  train_ratio = int(cl_length * 0.8)
  valid_ratio = cl_length - train_ratio

  nums = np.zeros(cl_length)
  nums[:valid_ratio] = 1
  np.random.shuffle(nums)
  total_train_nums.append(list(nums))

merged_nums = list(itertools.chain.from_iterable(total_train_nums))

# %%
len(merged_nums)

# %%
datagen = ImageDataGenerator(rescale=1./255)
batch_size = 50

# %%
test_generator = datagen.flow_from_directory(test_file_dir, target_size=(224,224), batch_size = batch_size, class_mode= 'categorical', shuffle=False)
train_generator = datagen.flow_from_directory(train_file_dir, target_size=(224,224), batch_size = batch_size, class_mode= 'categorical', shuffle=False)

test_filepaths = []
for filepath in test_generator.filepaths:
  test_filepaths.append(filepath)

test_filenames = []
for filename in test_generator.filenames:
  test_filenames.append(filename)

train_filepaths = []
for filepath in train_generator.filepaths:
  train_filepaths.append(filepath)

train_filenames = []
for filename in train_generator.filenames:
  train_filenames.append(filename)

# %%
ground_truth_label = []
train_file_names = []
for file in train_filenames:
  f = file.split("\\")
  ground_truth_label.append(f[0])
  train_file_names.append(f[1])

test_ground_truth_label = []
test_file_names = []
for file in test_filenames:
  f = file.split("\\")
  test_ground_truth_label.append(f[0])
  test_file_names.append(f[1])

train_index_list = list(range(0, train_img_cnt))
test_index_list = list(range(0, test_img_cnt))

# %%
def extract_features(generator, data_num, class_num, feature_shape, pretrained_model):
  features = np.zeros(shape = feature_shape)
  labels = np.zeros(shape=(data_num, class_num))
  #generator = datagen.flow_from_directory(directory, target_size=(224, 224), batch_size = batch_size, class_mode= 'categorical', shuffle=False)
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
vgg_final_layer = list(vgg_layer.layers)[-1].output_shape
vgg_final_layer = list(vgg_final_layer)
print("final layer of VGG16 : " +  str(list(vgg_layer.layers)[-1]) + " and its shape : " + str(vgg_final_layer))

vgg_conv_layers = []
for l in range(len(vgg_layer.layers)):
  layer = vgg_layer.layers[l]
  if 'Conv' not in layer.__class__.__name__:
    continue
  vgg_conv_layers.append((layer.name, layer.output.shape))

vgg_conv_base_shape = []
for i in vgg_final_layer:
  if i != None:
    vgg_conv_base_shape.append(i)
print("conv_base_shape : ", vgg_conv_base_shape)

train_vgg_feat_shape = tuple([train_img_cnt] + vgg_conv_base_shape)
print(train_vgg_feat_shape)

test_vgg_feat_shape = tuple([test_img_cnt] + vgg_conv_base_shape)
print(test_vgg_feat_shape)

vgg_input_dimension = np.prod(vgg_conv_base_shape)
print(vgg_input_dimension)

# %%
train_vgg_features, train_vgg_labels = extract_features(train_generator, train_img_cnt, len(class_names), train_vgg_feat_shape, vgg_layer)

# %%
test_vgg_features, test_vgg_labels = extract_features(test_generator, test_img_cnt, len(class_names), test_vgg_feat_shape, vgg_layer)

# %%
from json import JSONEncoder
class NumpyArrayEncoder(JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    return JSONEncoder.default(self, obj)


# %%
train_labels_int = []
for idx in range(len(train_vgg_labels)):
  train_labels_int.append(np.argmax(train_vgg_labels[idx]))

'''test_labels_int = []
for idx in range(len(test_vgg_labels)):
  test_labels_int.append(np.argmax(test_vgg_labels[idx]))'''

# %%
train_vgg_features = np.reshape(train_vgg_features, (train_img_cnt, vgg_input_dimension))
test_vgg_features = np.reshape(test_vgg_features, (test_img_cnt, vgg_input_dimension))

# %%
for idx in range(len(train_vgg_features)):
  cur_fea = train_vgg_features[idx]
  for j in range(len(cur_fea)):
    if cur_fea[j] > 0.0:
      cur_fea[j] = float(f"{cur_fea[j]:.4f}")

for idx in range(len(test_vgg_features)):
  cur_fea = test_vgg_features[idx]
  for j in range(len(cur_fea)):
    if cur_fea[j] > 0.0:
      cur_fea[j] = float(f"{cur_fea[j]:.4f}")


# %%
train_feas = {"features": train_vgg_features, "labels" : train_vgg_labels}
with open("vgg_extracted_brain_tumor_train.json", "w") as outfile:
  json.dump(train_feas, outfile, cls=NumpyArrayEncoder)

test_feas = {"features": test_vgg_features, "labels" : test_vgg_labels}
with open("vgg_extracted_brain_tumor_test.json", "w") as outfile:
  json.dump(test_feas, outfile, cls=NumpyArrayEncoder)

# %%
# Add classifier on pre-trained model
vgg_model = keras.models.Sequential()
#vgg_model.add(keras.layers.Reshape((vgg_input_dimension,), input_shape = tuple(vgg_conv_base_shape)))
vgg_model.add(keras.layers.Dense(512, activation='relu', input_dim = vgg_input_dimension))
vgg_model.add(keras.layers.Dense(512, activation='relu'))
vgg_model.add(keras.layers.Dense(len(class_names), activation = 'softmax'))
vgg_model.compile(optimizer=keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

# %%
pmt_order = np.random.permutation(np.arange(train_img_cnt))

# %%
vgg_data_df = pd.DataFrame({'index': train_index_list, 'file_names': train_file_names, 'feature' : list(train_vgg_features), 'label' : list(train_vgg_labels), 'int_label' : train_labels_int, 'assign' : merged_nums})


# %%
vgg_test_data_df = pd.DataFrame({'file_names': test_file_names, 'feature' : list(test_vgg_features), 'label' : list(test_vgg_labels)})

# %%
vgg_data_df

# %%
data_df_sf = vgg_data_df.iloc[pmt_order]

# %%
data_df_sf.head()

# %%
vgg_train_df = data_df_sf[data_df_sf['assign'] == 0]
vgg_valid_df = data_df_sf[data_df_sf['assign'] == 1]

# %%
print(len(vgg_train_df))
print(len(vgg_valid_df))

# %%
vgg_train_features = list(vgg_train_df['feature'])
vgg_train_labels = list(vgg_train_df['label'])

vgg_train_features = np.reshape(vgg_train_features, (len(vgg_train_df), vgg_input_dimension))
vgg_train_labels = np.reshape(vgg_train_labels, (len(vgg_train_df), len(class_names)))

vgg_valid_features = list(vgg_valid_df['feature'])
vgg_valid_labels = list(vgg_valid_df['label'])

vgg_valid_features = np.reshape(vgg_valid_features, (len(vgg_valid_df), vgg_input_dimension))
vgg_valid_labels = np.reshape(vgg_valid_labels, (len(vgg_valid_df), len(class_names)))

# %%
vgg_train_history = vgg_model.fit(vgg_train_features, vgg_train_labels, epochs = 12, batch_size = batch_size, validation_data = (vgg_valid_features, vgg_valid_labels))

# %%
vgg_loss, vgg_acc = vgg_model.evaluate(test_vgg_features, test_vgg_labels)

# %%
vgg_test_prediction_score = vgg_model.predict(test_vgg_features)

# %%
vgg_test_predicted_label = np.argmax(vgg_test_prediction_score, axis= -1)

# %%
vgg_test_data_df

# %%
test_result = {}
test_result['filename'] = list(vgg_test_data_df['file_names'])
#test_result['prediction_score'] = vgg_test_prediction_score
test_result['predicted_label'] = vgg_test_predicted_label

# %%
with open("vgg_result_brain_tumor.json", "w") as outfile:
  json.dump(test_result, outfile, indent=3, cls=NumpyArrayEncoder)

# %%


# %%



