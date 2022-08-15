import cohere
import pandas as pd

from loadExamples import examples
from cohere.classify import Example
from sklearn.model_selection import train_test_split


class CoHere:
    def __init__(self, api_key):
        self.co = cohere.Client(f'{api_key}', '2021-11-08')
        self.examples = []

    def list_of_examples(self, no_of_ex):
        for e in examples(no_of_ex):
            self.examples.append(Example(text=e[0], label=e[1]))

    def embed(self, no_of_ex):
        data = pd.DataFrame(examples(no_of_ex))
        X_train, X_test, y_train, y_test = train_test_split(
            list(data[0]), list(data[1]), test_size=0.2, random_state=0)
        self.X_train_embeded = self.co.embed(texts=X_train,
                                              model="large",
                                              truncate="LEFT").embeddings
        self.X_test_embeded = self.co.embed(texts=X_test,
                                             model="large",
                                             truncate="LEFT").embeddings
        self.X_test = X_test
        self.y_test = y_test
        self.y_train = y_train

    def classify(self, inputs):
        return self.co.classify(
            model='medium',
            taskDescription='',
            outputIndicator='',
            inputs=inputs,
            examples=self.examples
        ).classifications
