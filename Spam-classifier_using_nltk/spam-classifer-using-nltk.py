import re
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd

# importing Dataset....
message = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t', names=["Label", "message"])

# cleaning data....
Ps = PorterStemmer()
wordnet = WordNetLemmatizer()
corpus = []
for i in range(len(message)):
    review = re.sub('[^a-zA-Z]', ' ', message['message'][i])
    review = review.lower()
    review = review.split()
    review = [wordnet.lemmatize(word) for word in review if word not in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# use of TFidVectorizer...
cv = TfidfVectorizer()
x = cv.fit_transform(corpus).toarray()

# removing unwanted label...
y = pd.get_dummies(message['Label'])
y = y.iloc[:, 1].values

# dividing data....
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.20, random_state=0 )

# Creating model....
spam_detect_model = MultinomialNB().fit(x_train, y_train)

# output....
y_pred = spam_detect_model.predict(x_test)

# confusion matrix....
confusion_matrix = confusion_matrix(y_test, y_pred)

# accuracy score.....
accuracy = accuracy_score(y_test, y_pred)


print(y_pred)
print("confusion_matrix :", confusion_matrix)
print("accuracy :", accuracy)
