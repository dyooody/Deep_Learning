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
# https://www.kaggle.com/code/shadym0hamed/lung-cancer-classification/data

# %%
from tensorflow.keras.applications.vgg16 import VGG16

# %%
vgg_layer = VGG16(weights= 'imagenet', include_top= False, input_shape=(224, 224, 3))

# %%
folder_dir = "lung_cancer/Dataset/"
file_dir = pathlib.Path(folder_dir)
print(file_dir.exists())

# %%
total_img_cnt = len(list(file_dir.glob("*/*.jpg")))
print(total_img_cnt)

# %%
class_names = [name for name in os.listdir(folder_dir) if os.path.isdir(os.path.join(folder_dir, name))]
print(class_names)

# %%
total_nums = []
for class_name in class_names:
  class_dir = pathlib.Path(folder_dir, class_name)
  cl_length = len(list(class_dir.glob("*.jpg")))
  train_ratio = int(cl_length * 0.7)
  test_ratio = int(cl_length * 0.2)
  valid_ratio = cl_length - (train_ratio + test_ratio)

  nums = np.zeros(cl_length)
  nums[:test_ratio] = 1
  nums[test_ratio : test_ratio + valid_ratio] = 2
  np.random.shuffle(nums)
  total_nums.append(list(nums))

merged_nums = list(itertools.chain.from_iterable(total_nums))

# %%
datagen = ImageDataGenerator(rescale=1./255)
batch_size = 32

# %%
generator = datagen.flow_from_directory(file_dir, target_size=(224,224), batch_size = batch_size, class_mode= 'categorical', shuffle=False)

filepaths = []
for filepath in generator.filepaths:
  filepaths.append(filepath)

filenames = []
for filename in generator.filenames:
  filenames.append(filename)

# %%
ground_truth_label = []
file_names = []
for file in filenames:
  f = file.split("\\")
  ground_truth_label.append(f[0])
  file_names.append(f[1])

index_list = list(range(0, total_img_cnt))

# %%
def extract_features(generator, data_num, class_num, feature_shape, pretrained_model):
  features = np.zeros(shape = feature_shape)
  labels = np.zeros(shape=(data_num, class_num))
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

vgg_feat_shape = tuple([total_img_cnt] + vgg_conv_base_shape)
print(vgg_feat_shape)

vgg_input_dimension = np.prod(vgg_conv_base_shape)
print(vgg_input_dimension)

# %%
vgg_features, vgg_labels = extract_features(generator, total_img_cnt, len(class_names), vgg_feat_shape, vgg_layer)

# %%
from json import JSONEncoder
class NumpyArrayEncoder(JSONEncoder):
  def default(self, obj):
    if isinstance(obj, np.ndarray):
      return obj.tolist()
    return JSONEncoder.default(self, obj)

# %%
total_labels_int = []
for idx in range(len(vgg_labels)):
  total_labels_int.append(np.argmax(vgg_labels[idx]))

# %%
vgg_features = np.reshape(vgg_features, (total_img_cnt, vgg_input_dimension))

# %%
for idx in range(len(vgg_features)):
  cur_fea = vgg_features[idx]
  #cur_fea_2 = np.where(len(str(cur_fea)) > 3, cur_fea, f"{cur_fea:.4f}")
  for j in range(len(cur_fea)):
    if cur_fea[j] > 0.0:
      cur_fea[j] = float(f"{cur_fea[j]:.4f}")


# %%
total_feas = {"features": vgg_features, "labels" : vgg_labels}
with open("vgg_extracted_lung_cancer.json", "w") as outfile:
  json.dump(total_feas, outfile, cls=NumpyArrayEncoder)

# %%
# Add classifier on pre-trained model
vgg_model = keras.models.Sequential()
vgg_model.add(keras.layers.Dense(512, activation='relu', input_dim = vgg_input_dimension))
vgg_model.add(keras.layers.Dense(512, activation='relu'))
vgg_model.add(keras.layers.Dense(len(class_names), activation = 'softmax'))
vgg_model.compile(optimizer=keras.optimizers.Adam(), loss='categorical_crossentropy', metrics=['accuracy'])


# %%
pmt_order = np.random.permutation(np.arange(total_img_cnt))

# %%
vgg_data_df = pd.DataFrame({'index': index_list, 'file_names': file_names, 'feature' : list(vgg_features), 'label' : list(vgg_labels), 'int_label' : total_labels_int, 'assign' : merged_nums})

# %%
vgg_data_df

# %%
data_df_sf = vgg_data_df.iloc[pmt_order]

# %%
data_df_sf.head()

# %%
vgg_train_df = data_df_sf[data_df_sf['assign'] == 0]
vgg_test_df = data_df_sf[data_df_sf['assign'] == 1]
vgg_valid_df = data_df_sf[data_df_sf['assign'] == 2]

# %%
vgg_train_features = list(vgg_train_df['feature'])
vgg_train_labels = list(vgg_train_df['label'])

vgg_train_features = np.reshape(vgg_train_features, (len(vgg_train_df), vgg_input_dimension))
vgg_train_labels = np.reshape(vgg_train_labels, (len(vgg_train_df), len(class_names)))

vgg_test_features = list(vgg_test_df['feature'])
vgg_test_labels = list(vgg_test_df['label'])

vgg_test_features = np.reshape(vgg_test_features, (len(vgg_test_df), vgg_input_dimension))
vgg_test_labels = np.reshape(vgg_test_labels, (len(vgg_test_df), len(class_names)))

vgg_valid_features = list(vgg_valid_df['feature'])
vgg_valid_labels = list(vgg_valid_df['label'])

vgg_valid_features = np.reshape(vgg_valid_features, (len(vgg_valid_df), vgg_input_dimension))
vgg_valid_labels = np.reshape(vgg_valid_labels, (len(vgg_valid_df), len(class_names)))

# %%
vgg_train_history = vgg_model.fit(vgg_train_features, vgg_train_labels, epochs = 5, batch_size = batch_size, validation_data = (vgg_valid_features, vgg_valid_labels))

# %%
vgg_loss, vgg_acc = vgg_model.evaluate(vgg_test_features, vgg_test_labels)

# %%
vgg_test_prediction_score = vgg_model.predict(vgg_test_features)

# %%
vgg_test_predicted_label = np.argmax(vgg_test_prediction_score, axis= -1)

# %%
vgg_test_df

# %%
test_result = {}
test_result['filename'] = list(vgg_test_df['file_names'])
#test_result['prediction_score'] = vgg_test_prediction_score
test_result['predicted_label'] = vgg_test_predicted_label

# %%
with open("vgg_result_lung_cancer.json", "w") as outfile:
  json.dump(test_result, outfile, indent=3, cls=NumpyArrayEncoder)

# %%


# %%



