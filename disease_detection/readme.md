## Brain Tumor detection <br/>
Dataset: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

Four different classes. 
Glioma
Meningioma
Non-tumor
pituitary

Result: 96.7%


**Procedure:** <br/>
> (1) Resized the images into 224 x 224. <br/>
> (2) The number of training datasets was 5712, and the number of test datasets was 1311. <br/>
> (3) There was train and test data, so I split the training dataset into train and validation data. (80% as train data, 20% as validation data) <br/>
> (4) Used TensorFlow and Keras to create a machine-learning model. <br/>
> (5) Used transfer learning methods to achieve high prediction results. Used VGG16 as a pre-trained model. <br/>
> (6) Add ReLU as an activation function, and used Adam as an optimizer. <br/>


## Tooth decay detection <br/>
Dataset: https://www.kaggle.com/datasets/snginh/toothdecay/code

Two different classes.
Cavity
Non-cavity

Result: 92.8%

**Procedure:** <br/>
> (1) Resized to 224 x 224. <br/>
> (2) The dataset has train and test data. Since the number of training datasets was small, I just used the whole train dataset for training. (Not splitting it into train and validation datasets) <br/>
> (3) Train data was 60 and test data was 14. <br/>
> (4) For this dataset, the number of training and test data was small, and the number of images in classes was different so I used only part of the cavity training dataset to match the number of images of non-cavity. <br/>
> (5) Used TensorFlow and Keras to create a machine-learning model. <br/>
> (6) Used transfer learning methods to achieve high prediction results. Used InceptionV3 as a pre-trained model. <br/>
> (7) Add ReLU as an activation function, and used Adam as an optimizer. <br/>





## Lung cancer detection. <br/>

For this classification problem, I used two different datasets. 

Dataset: lung cancer dataset

1. 
https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images

From here, I used the lung cancer dataset with three different classes. 
Lung benign tissue
Lung adenocarcinoma
Lung squamous cell carcinoma
Result: 98% accuracy

**Procedure:** <br/> 
> (1) The size of the images was 768 x 768 and the number of images in the three classes was 15000. Since the size was too big, reduced the size to 224 x 224. <br/>
> (2) Split the dataset into train, validation, and test data. (70 % as train data, 20 % as validation data, and 10% as test data) <br/>
> (3) Set the batch size as 300 <br/>
> (4) Used TensorFlow and Keras to create a machine-learning model. <br/>
> (5) Used transfer learning methods to achieve high prediction results. Used VGG16 as a pre-trained model. <br/>
> (6) Add ReLU as an activation function, and used Adam as an optimizer. <br/>

2. 
https://www.kaggle.com/code/shadym0hamed/lung-cancer-classification/data
 
There are four different classes from this dataset.
Normal
Adeno
Carci
Squamos
 
Result: 99 % accuracy

**Procedure:** <br/> 
> (1) Resized the image to 224 x 224. <br/>
> (2) Split the dataset into train, validation, and test data. (70 % as train data, 20 % as validation data, and 10% as test data) <br/>
> (3) The total dataset was 2002.<br/>
> (4) Set the batch size as 32<br/>
> (5) Used TensorFlow and Keras to create a machine-learning model. <br/>
> (6) Used transfer learning methods to achieve high prediction results. Used VGG16 as a pre-trained model.<br/>
> (7) Add ReLU as an activation function, and used Adam as an optimizer.<br/>
