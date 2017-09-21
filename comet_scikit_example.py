
m __future__ import print_function
import sys

from comet_ml import Experiment

from sklearn.feature_extraction.text import CountVectorizer

from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.datasets import fetch_20newsgroups

from sklearn.linear_model import SGDClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


experiemnt  = Experiment(api_key="zrwyPY98JcXRBCyRGO0x069xDVt4QsvSne7AmWIIiaZuQRg0DUBysHQdFiDsILu1", log_code=True)

# Get dataset and put into train,test lists
categories = ['alt.atheism', 'soc.religion.christian',
              'comp.graphics', 'sci.med']

twenty_train = fetch_20newsgroups(subset='train',categories=categories, shuffle=True, random_state=42)
twenty_test = fetch_20newsgroups(subset='test',categories=categories, shuffle=True, random_state=42)

# Build training pipeline

vectorizer = CountVectorizer()
classfier = SGDClassifier(loss='hinge', penalty='l2', # Call classifier with vector
                                           alpha=1e-3, random_state=42,
                                           max_iter=5, tol=None)

#text_clf = Pipeline([('vect', CountVectorizer()), # Counts occurrences of each word
#                      ('tfidf', TfidfTransformer()), # Normalize the counts based on document length
#                      ('clf', SGDClassifier(loss='hinge', penalty='l2', # Call classifier with vector
#                                            alpha=1e-3, random_state=42,
#                                            max_iter=5, tol=None)),
#                      ])

#text_clf.fit(twenty_train.data,twenty_train.target)

vectorized = vectorizer.fit_transform(twenty_train.data)
classfier.fit(vectorized,twenty_train.target)
# Fit classifier to train data


# Predict unseen test data based on fitted classifer
predicted = classfier.predict(vectorizer.transform(twenty_test.data))

# Compute accuracy
print(accuracy_score(twenty_test.target, predicted))
# Compute classification metrics
# print(metrics.classification_report(twenty_test.target, predicted,
#                                     target_names=twenty_test.target_names))
#
# print(metrics.confusion_matrix(twenty_test.target, predicted))


