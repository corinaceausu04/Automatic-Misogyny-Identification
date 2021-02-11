import pandas as pd
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix
import re
import string
import numpy as np

t_start = pd.datetime.now()
print('Start time:', t_start)

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

train, test = train_test_split(train_df, test_size=0.2)
corpus_train = train['text'].to_list()
corpus_test = test['text'].to_list()
train_label = train['label'].to_list()
test_label = test['label'].to_list()

#corpus_train, train_label = shuffle(corpus_train, train_df['label'])

exclude = set(string.punctuation)
for idx in range(0, len(corpus_train)):
    corpus_train[idx] = ''.join(ch for ch in corpus_train[idx] if ch not in exclude)
    corpus_train[idx] = corpus_train[idx].lower()
    corpus_train[idx] = re.sub(r"http\S+", "", corpus_train[idx])

for idx in range(0, len(corpus_test)):
    corpus_test[idx] = ''.join(ch for ch in corpus_test[idx] if ch not in exclude)
    corpus_test[idx] = corpus_test[idx].lower()
    corpus_test[idx] = re.sub(r"http\S+", "", corpus_test[idx])

it_stop_words = stopwords.words('italian')

vectorizer = TfidfVectorizer(stop_words=it_stop_words)
features = vectorizer.fit_transform(corpus_train)

model = MultinomialNB()
model.fit(features, train_label)

test_features = vectorizer.transform(corpus_test)
predict = model.predict(test_features)

idx = []
pred = []

for i in range(5001, 6001):
    idx.append(i)

for i in predict:
    pred.append(predict)
for i in predict:
    date = {
        'id': idx,
        'label': pred[i]
    }

print('Matrice de confuzie: ')
print(confusion_matrix(test_label, date['label']))
print('Acuratete: ')
print(accuracy_score(test['label'], date['label']))

t_end = pd.datetime.now()
print('End time:', t_end)
print('Total time:', t_end - t_start)

df = pd.DataFrame(date, columns=['id', 'label'])

# df.to_csv ('svc_final.csv', index = False, header=True)'''
