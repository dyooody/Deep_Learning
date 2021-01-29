#!/usr/bin/env python
# coding: utf-8

# In[2]:


get_ipython().system('pip3 install bert-for-tf2')
get_ipython().system('pip3 install sentencepiece')
get_ipython().system('pip3 install lime')
get_ipython().system('pip3 instll bert-tensorflow ')


# In[3]:


import tensorflow as tf
from tensorflow import keras
import tensorflow_hub as hub


from tensorflow.keras import layers
from keras.models import Sequential
import bert
import pandas as pd
import numpy as np
import re, random, math, os


# In[4]:


import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))


# In[6]:


bbc_text = pd.read_csv("bbc-text.csv")

bbc_text.head(20)


# In[8]:


itr_cnt = 1000
train_subset = bbc_text[0:itr_cnt]
test_subset = bbc_text[1925:2226]


# In[9]:


values = bbc_text.category.value_counts()
print(values)


# In[10]:


def preprocess_text(sen):
    #Remove STOPWORDS
    for word in STOPWORDS:
      token = ' ' + word + ' '
      sen = sen.replace(token, ' ')
      sen = sen.replace(' ', ' ')

    # Removing html tags
    sentence = remove_tags(sen)

    # Remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)

    # Single character removal
    sentence = re.sub(r"\s+[a-zA-Z]\s+", ' ', sentence)

    # Removing multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)

    return sentence


# In[11]:


TAG_RE = re.compile(r'<[^>]+>')

def remove_tags(text):
    return TAG_RE.sub('', text)


# In[12]:


bbc_train = []
bbc_test = []
train_sen = list(train_subset['text'])
test_sen = list(test_subset['text'])
for sen in train_sen:
    bbc_train.append(preprocess_text(sen))

for sen in test_sen:
    bbc_test.append(preprocess_text(sen))


# In[14]:


bbc_labels = bbc_text.category.unique()
print(type(bbc_labels))
bbc_labels = np.sort(bbc_labels)
print(bbc_labels)


# In[15]:


from sklearn.preprocessing import LabelEncoder
train_df = pd.DataFrame()
train_df['text'] = train_subset["text"]
train_df['category'] = LabelEncoder().fit_transform(train_subset["category"])

test_df = pd.DataFrame()
test_df['text'] = test_subset["text"]
test_df['category'] = LabelEncoder().fit_transform(test_subset["category"])


# In[16]:


BertTokenizer = bert.bert_tokenization.FullTokenizer
bert_layer = hub.KerasLayer("https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/1",
                            trainable=False)
vocabulary_file = bert_layer.resolved_object.vocab_file.asset_path.numpy()
to_lower_case = bert_layer.resolved_object.do_lower_case.numpy()
tokenizer = BertTokenizer(vocabulary_file, to_lower_case)


# In[50]:


def tokenize_data(text_reviews):
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text_reviews))


# In[18]:


tokenized_train = [tokenize_reviews(text) for text in bbc_train] 
tokenized_test = [tokenize_reviews(text) for text in bbc_test]


# In[19]:


train_texts = [[train_text, train_df['category'][i]] for i,  train_text in enumerate(tokenized_train)]
test_texts = [[test_text, test_df['category'][i+1925]] for i, test_text in enumerate(tokenized_test)]


# In[20]:


sorted_train_labels = [(text_lab[0], text_lab[1]) for text_lab in train_texts]
sorted_test_labels = [(text_lab[0], text_lab[1]) for text_lab in test_texts]


# In[21]:


processed_train = tf.data.Dataset.from_generator(lambda: sorted_train_labels, output_types=(tf.int32, tf.int32))
processed_test = tf.data.Dataset.from_generator(lambda: sorted_test_labels, output_types = (tf.int32, tf.int32))


# In[22]:


BATCH_SIZE = 32
batched_train_dataset = processed_train.padded_batch(BATCH_SIZE, padded_shapes=((None, ), ()))
print(type(batched_train_dataset))
batched_test_dataset = processed_test.padded_batch(BATCH_SIZE, padded_shapes=((None, ), ()))


# In[23]:


class TEXT_MODEL(tf.keras.Model):
    
    def __init__(self,
                 vocabulary_size,
                 embedding_dimensions=128,
                 cnn_filters=100,
                 dnn_units=512,
                 model_output_classes=5,
                 dropout_rate=0.2,
                 training=False,
                 name="text_model"):
        super(TEXT_MODEL, self).__init__(name=name)
        
        self.embedding = layers.Embedding(vocabulary_size,
                                          embedding_dimensions)
        self.cnn_layer1 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=2,
                                        padding="valid",
                                        activation="relu")
        self.cnn_layer2 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=3,
                                        padding="valid",
                                        activation="relu")
        self.cnn_layer3 = layers.Conv1D(filters=cnn_filters,
                                        kernel_size=4,
                                        padding="valid",
                                        activation="relu")
        self.pool = layers.GlobalMaxPool1D()
        
        self.dense_1 = layers.Dense(units=dnn_units, activation="relu")
        self.dropout = layers.Dropout(rate=dropout_rate)
        if model_output_classes == 2:
            self.last_dense = layers.Dense(units=1,
                                           activation="sigmoid")
        else:
            self.last_dense = layers.Dense(units=model_output_classes,
                                           activation="softmax")
    
    def call(self, inputs, training):
        l = self.embedding(inputs)
        l_1 = self.cnn_layer1(l) 
        l_1 = self.pool(l_1) 
        l_2 = self.cnn_layer2(l) 
        l_2 = self.pool(l_2)
        l_3 = self.cnn_layer3(l)
        l_3 = self.pool(l_3) 
        
        concatenated = tf.concat([l_1, l_2, l_3], axis=-1) # (batch_size, 3 * cnn_filters)
        concatenated = self.dense_1(concatenated)
        concatenated_dropout = self.dropout(concatenated, training)
        model_output = self.last_dense(concatenated_dropout)
        
        return model_output


# In[24]:


VOCAB_LENGTH = len(tokenizer.vocab)
EMB_DIM = 200
CNN_FILTERS = 100
DNN_UNITS = 256
OUTPUT_CLASSES = 5  
DROPOUT_RATE = 0.2

NB_EPOCHS = 5


# In[25]:


text_model = TEXT_MODEL(vocabulary_size=VOCAB_LENGTH,
                        embedding_dimensions=EMB_DIM,
                        cnn_filters=CNN_FILTERS,
                        dnn_units=DNN_UNITS,
                        model_output_classes=OUTPUT_CLASSES,
                        dropout_rate=DROPOUT_RATE)


# In[26]:


text_model.compile(loss='sparse_categorical_crossentropy',
                    optimizer="adam",
                    metrics=["sparse_categorical_accuracy"])


# In[28]:


text_model.summary()


# In[27]:


text_model.fit(batched_train_dataset, epochs=NB_EPOCHS)


# In[29]:


test_loss, test_acc = text_model.evaluate(batched_test_dataset)
print(test_acc)


# In[30]:


y_pred_confid = text_model.predict(batched_test_dataset)
y_pred_confid_np = [];
print(y_pred_confid[3])
print(type(y_pred_confid))

for i in range(len(y_pred_confid)):
  confid_list = [];
  for j in range(len(y_pred_confid[i])):
    y_pred_confid[i][j] = f"{y_pred_confid[i][j]:.3f}"
    confid_list.append(y_pred_confid[i][j])
  confid_list = str(confid_list)  
  y_pred_confid_np.append(confid_list)
  
print(type(y_pred_confid_np))
y_pred_confid_np = np.array(y_pred_confid_np)
print(type(y_pred_confid_np))
print(y_pred_confid_np.shape)
print(list(y_pred_confid[3]))
print(type(y_pred_confid))
print(str(y_pred_confid[77]))


# In[71]:


pred = tf.nn.softmax(text_model.predict(batched_test_dataset))
y_pred_argmax = tf.math.argmax(pred, axis=1)
type(pred)
y_pred_argmax_np = y_pred_argmax.numpy();
print(y_pred_argmax_np)

count = 0;
y_true = tf.Variable([], dtype=tf.int32)
for features, label in batched_test_dataset:
    y_true = tf.concat([y_true, label], 0)
    count += 1;


# In[33]:


origin_proc_test = [];
count = 0;
for i, j in enumerate(processed_test):
  back_to = tokenizer.convert_ids_to_tokens(j[0].numpy())
  origin_proc_test.append(back_to);
  count += 1;


# In[35]:


from sklearn.metrics import classification_report, confusion_matrix
y_pred = tf.nn.softmax(text_model.predict(batched_test_dataset))

y_pred_argmax = tf.math.argmax(y_pred, axis=1)
y_pred_argmax_np = y_pred_argmax.numpy()

y_true = tf.Variable([], dtype=tf.int32)


# In[36]:


for features, label in batched_test_dataset:
    y_true = tf.concat([y_true, label], 0)

origin_proc_test = [];
for i, j in enumerate(processed_test):
    back_to = tokenizer.convert_ids_to_tokens(j[0].numpy())
    origin_proc_test.append(back_to);

origin_proc_test_str = [];
for i in range(len(origin_proc_test)):
    listToStr = ' '.join([str(elem) for elem in origin_proc_test[i]])
    origin_proc_test_str.append(listToStr)


# In[38]:


print(origin_proc_test[1])


# In[39]:


le = LabelEncoder().fit(test_subset['category'])
inversed_y_pred = le.inverse_transform(y_pred_argmax_np)
y_true_np = y_true.numpy();
inversed_y_true = le.inverse_transform(y_true_np)
origin_proc_test_np = np.asarray(origin_proc_test_str)


# In[42]:


index_list = list(range(0, 300))
df_inv = {'index': index_list, 'true_label' : inversed_y_true, 'predicted_label': inversed_y_pred, 'confidence_score': y_pred_confid_np, 'text' : origin_proc_test_np}
test_result_inv_df = pd.DataFrame(df_inv)
test_result_inv_df.head(30)


# In[44]:



cm = confusion_matrix(y_true, y_pred_argmax)
print(cm)


cr = classification_report(y_true, y_pred_argmax, target_names = bbc_labels)
print(cr)


# In[45]:


test_acc = float(test_acc)
test_acc = f"{test_acc:.2f}"


# In[46]:


from lime import lime_text
from lime.lime_text import LimeTextExplainer

explainer = LimeTextExplainer(class_names = bbc_labels)


# In[51]:


def get_confidence_score(bbc_test_np):
    print("get_confidence_score")
    bbc_test = bbc_test_np
    tokenized_test = [tokenize_data(text) for text in bbc_test];
    test_texts = [[text, test_df['category'][i+1925]] for i, text in enumerate(tokenized_test)]
    sorted_test_labels = [(text_lab[0], text_lab[1]) for text_lab in test_texts]
    processed_test = tf.data.Dataset.from_generator(lambda: sorted_test_labels, output_types = (tf.int32, tf.int32))
    batched_test_dataset = processed_test.padded_batch(BATCH_SIZE, padded_shapes=((None, ), ()))
    y_pred_confid = text_model.predict(batched_test_dataset)

    return y_pred_confid;


# In[52]:


misclassified_idx = [];
for idx in range(len(test_result_inv_df)):
    if(test_result_inv_df['true_label'][idx] != test_result_inv_df['predicted_label'][idx]):
        misclassified_idx.append(idx)
print("misclassified_idx", misclassified_idx)
print()


# In[53]:


idx = 8
exp = explainer.explain_instance(test_result_inv_df['text'][idx], get_confidence_score, num_features=20, top_labels=5, num_samples=len(bbc_test))
print('True class: %s' % test_result_inv_df['true_label'][idx])
print('Predicted class: %s' % test_result_inv_df['predicted_label'][idx])
print('Confidence score: %s' % test_result_inv_df['confidence_score'][idx])
exp.show_in_notebook(text=True)


# In[70]:


from collections import Counter
import itertools

appended_tokenized_test = list(itertools.chain.from_iterable(tokenized_test))
#print(type(appended_tokenized_test))


# In[56]:


print('appended tokenized test' , len(appended_tokenized_test))
print(appended_tokenized_test[23])


# In[69]:


counting_toked_test = Counter(appended_tokenized_test)
#print(dict(counting_toked_test))
#print('counting_toked_test', len(counting_toked_test))


# In[68]:


inversed_toked_test = tokenizer.convert_ids_to_tokens(appended_tokenized_test)
inversed_count_toked = Counter(inversed_toked_test)


# In[62]:


sorted_toked_test = sorted(counting_toked_test.items(), key = lambda x:x[1], reverse=True)
sorted_inversed_toked = sorted(inversed_count_toked.items(), key = lambda x:x[1], reverse=True)
print(len(sorted_inversed_toked))


# In[64]:


inversed_count_without_sw = []
for i in range(len(sorted_inversed_toked)):
  if sorted_inversed_toked[i][0] not in STOPWORDS:
    inversed_count_without_sw.append(sorted_inversed_toked[i])

print(len(inversed_count_without_sw))

inversed_count_without_sw = dict(inversed_count_without_sw)


# In[67]:


common_occurences_dict = {key: value for key, value in inversed_count_without_sw.items() if 50 < value < 500}
#print(common_occurences_dict)
sorted_common_val = sorted(common_occurences_dict.items(), key = lambda x:x[1], reverse=True)


# In[ ]:




