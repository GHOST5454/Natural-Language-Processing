import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score

df = pd.read_csv('fake-news/train.csv')
x = df.drop('label', axis=1)
df = df.dropna()
message = df.copy()
message.reset_index(inplace=True)

ps = PorterStemmer()
corpus = []
for i in range(0, len(message)):
    review = re.sub('[^a-zA-Z]', ' ', message['title'][i])
    review = review.lower()
    review = review.split()
    review = [ps.stem(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

cv = CountVectorizer(max_features=5000, ngram_range=(1, 3))
x = cv.fit_transform(corpus).toarray()

y = message['label']

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.33, random_state=0)

classifier = MultinomialNB()
classifier.fit(x_train, y_train)
pred = classifier.predict(x_test)
score = accuracy_score(y_test, pred)
cm = confusion_matrix(y_test, pred)
print(score, cm)