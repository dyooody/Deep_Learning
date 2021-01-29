#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import csv
import numpy as np


# In[2]:


from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

import nltk
nltk.download('stopwords')

from nltk.corpus import stopwords
STOPWORDS = set(stopwords.words('english'))


# In[3]:


vocab_size = 5000
embedding_dim = 64
max_length = 200
trunc_type = 'post'
padding_type = 'post'
oov_tok = '<OOV>'
training_portion = .8

articles = []
labels = []

with open("bbc-text.csv", 'r') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader)
    for row in reader:
        labels.append(row[0])
        article = row[1]
        for word in STOPWORDS:
            token = ' ' + word + ' '
            article = article.replace(token, ' ')
            article = article.replace(' ', ' ')
        articles.append(article)
print(len(labels))
print(len(articles))


# In[4]:


train_size = int(len(articles) * training_portion)

train_articles = articles[0: train_size]
train_labels = labels[0: train_size]

validation_articles = articles[train_size:]
validation_labels = labels[train_size:]


# In[20]:


tokenizer = Tokenizer(num_words = vocab_size, oov_token=oov_tok)
tokenizer.fit_on_texts(train_articles)
word_index = tokenizer.word_index

dictionary = dict(list(word_index.items())[0:10])

train_sequences = tokenizer.texts_to_sequences(train_articles)


train_padded = pad_sequences(train_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)


# In[6]:


validation_sequences = tokenizer.texts_to_sequences(validation_articles)
validation_padded = pad_sequences(validation_sequences, maxlen=max_length, padding=padding_type, truncating=trunc_type)

print(len(validation_sequences))
print(validation_padded.shape)

print(set(labels))

label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)

training_label_seq = np.array(label_tokenizer.texts_to_sequences(train_labels))
validation_label_seq = np.array(label_tokenizer.texts_to_sequences(validation_labels))

print(training_label_seq.shape)
print(validation_label_seq.shape)


# In[8]:


def decode_article(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


# In[9]:


print(set(labels))

label_tokenizer = Tokenizer()
label_tokenizer.fit_on_texts(labels)

training_label_seq = np.array(label_tokenizer.texts_to_sequences(train_labels))
validation_label_seq = np.array(label_tokenizer.texts_to_sequences(validation_labels))

print(training_label_seq.shape)
print(validation_label_seq.shape)



reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])


# In[19]:


#print(decode_article(train_padded[10]))
#print('---')
#print(train_articles[10])


# In[11]:


model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim),
    tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(embedding_dim)),
    # tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(32)),
    # use ReLU in place of tanh function since they are very good alternatives of each other.
    tf.keras.layers.Dense(embedding_dim, activation='relu'),
    # Add a Dense layer with 6 units and softmax activation.
    # using softmax for multi-class classification 
    tf.keras.layers.Dense(6, activation='softmax')
])
model.summary()


# In[12]:


model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
num_epochs = 10

history = model.fit(train_padded, training_label_seq, epochs=num_epochs, validation_data=(validation_padded, validation_label_seq), verbose=2)
#history = model.fit(train_padded, training_label_seq, epochs=num_epochs, verbose=2)


# In[13]:


test_loss, test_acc = model.evaluate(validation_padded, validation_label_seq)
print(test_acc)


# In[14]:


CLASSES = set(labels)
print(CLASSES)


# In[18]:


from sklearn.metrics import classification_report, confusion_matrix
predictions = model.predict_classes(validation_padded)
y_pred = predictions 

type(y_pred)
cm = confusion_matrix(validation_label_seq, y_pred)
print(cm)

print(classification_report(validation_label_seq, y_pred, target_names = CLASSES))


# In[ ]:




