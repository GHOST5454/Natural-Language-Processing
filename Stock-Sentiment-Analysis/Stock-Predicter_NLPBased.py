import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

df = pd.read_csv('Data/Data.csv', encoding="ISO-8859-1")

train = df[df['Date'] < '20150101']
test = df[df['Date'] > '20150101']

data = train.iloc[:, 2:27]
data.replace("[^a-zA-Z]", " ", regex=True, inplace=True)

List1 = [i for i in range(25)]
new_Index = [str(i)for i in List1]
data.columns = new_Index

for index in new_Index:
    data[index] = data[index].str.lower()

headlines = []
for row in range(0, len(data.index)):
    headlines.append(' '.join(str(x) for x in data.iloc[row, 0:25]))

countvector = CountVectorizer(ngram_range=(2, 2))
traindataset = countvector.fit_transform(headlines)

randam_classifier = RandomForestClassifier(n_estimators=200, criterion='entropy')
randam_classifier.fit(traindataset, train['Label'])

test_transform = []
for row in range(0, len(test.index)):
    test_transform.append(' '.join(str(x) for x in test.iloc[row, 2:27]))

test_dataset = countvector.transform(test_transform)
prediction = randam_classifier.predict(test_dataset)

matrix = confusion_matrix(test['Label'], prediction)
print(matrix)
Score = accuracy_score(test['Label'], prediction)
print(Score)
report = classification_report(test['Label'], prediction)
print(report)
