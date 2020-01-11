#!/user/bin/env python
# -*- coding:utf-8 -*-

from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import f1_score
import logging

# If you want to log the process ID along with the level and message, you can do something like this:
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] <%(processName)s> (%(threadName)s) %(message)s')
logger = logging.getLogger(__name__)
# “It is recommended that we use module-level loggers by passing __name__ as the name parameter to getLogger() to create a
# logger object as the name of the logger itself would tell us from where the events are being logged. __name__ is a
# special built-in variable in Python which evaluates to the name of the current module.”


class TextClassifier():

    def __init__(self, vectorizer, classifier=MultinomialNB()):
        classifier = SVC(kernel="rbf")
        # classifier = SVC(kernel="linear")
        self.classifier = classifier
        self.vectorizer = vectorizer

    def features(self, x):
        return self.vectorizer.transform(x)

    def fit(self, x, y):

        self.classifier.fit(self.features(x), y)

    def predict(self, x):

        return self.classifier.predict(self.features(x))

    def score(self, x, y):
        return self.classifier.score(self.features(x), y)

    def get_f1_score(self, x, y):
        return f1_score(y, self.predict(x), average='macro')



