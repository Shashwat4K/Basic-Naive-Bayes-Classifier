import pandas as pd
import numpy as np
import os
from collections import OrderedDict

class NaiveBayes(object):
    def __init__(self, data, targets, columns, target_column):
        self.data = pd.DataFrame(data)
        self.targets = pd.Series(targets)
        self.labels = list(self.targets.unique())
        self.column_names = columns
        self.target_column = target_column
        vc = targets.value_counts()
        # print(labels)
        self.target_values_probablities = OrderedDict()
        for l in self.labels:
            p = vc.loc[l] / len(targets)
            self.target_values_probablities[l] = p

    def classify(self, example):
        self.naive_bayes_dict = OrderedDict()
        # Initialize dict
        for l in self.labels:
            self.naive_bayes_dict[l] = self.target_values_probablities[l] * self.get_product_of_conditional_probabilities(example, l)

        most_probable = max(self.naive_bayes_dict, key=self.naive_bayes_dict.get)
        # print(self.naive_bayes_dict)
        return most_probable, self.naive_bayes_dict[most_probable] / sum(self.naive_bayes_dict.values())  
        

    def get_product_of_conditional_probabilities(self, example, l):
        rows = self.data.loc[self.data[self.target_column] == l]
        prod = 1.0
        for k, v in example.items():
            prod *= len(rows.loc[rows[k] == v]) / len(rows)
        return prod    


    def train(self):
        pass

if __name__ == "__main__":

    data_dir = os.path.join(os.getcwd(), '..', 'input_data')
    df = pd.read_csv(os.path.join(data_dir, "play_tennis.csv"))
    my_classifier = NaiveBayes(df, df['PlayTennis'], df.columns, 'PlayTennis')   
    label, prob = my_classifier.classify({'Outlook': 'Sunny', 'Temperature': 'Cool', 'Humidity': 'High', 'Wind': 'Strong'})  
    print("The new example is classified as {} with normalized probability {}".format(label, prob))   