import pandas as pd
import numpy as np
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import re
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix


df = pd.read_csv('fake-news/train.csv')
df = df.dropna()
x = df.drop('label', axis=1)
y = df['label']

voc_size = 5000

message = x.copy()
message.reset_index(inplace=True)

ps = PorterStemmer()
corpus = []
for i in range(0, len(message)):
    review = re.sub("[^a-zA-z]", " ", message['title'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word)for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

onehot_repr = [one_hot(words, voc_size) for words in corpus]

sent_length = 20
embedded_docs = pad_sequences(onehot_repr, padding='pre', maxlen=sent_length)

embedding_vector_features = 40
model = Sequential()
model.add(Embedding(voc_size, embedding_vector_features, input_length=sent_length))
model.add(LSTM(100))
model.add(Dense(1, activation='sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.summary()

x_final = np.array(embedded_docs)
y_final = np.array(y)

x_train, x_test, y_train, y_test = train_test_split(x_final, y_final, test_size=0.33, random_state= 42)
model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=10, batch_size=64)

y_pred = model.predict_classes(x_test)

print(confusion_matrix(y_test, y_pred))
print(accuracy_score(y_test, y_pred))