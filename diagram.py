import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class Diagram(object):
    def __init__(self , classifier):
        self.classifier = classifier
        self.setup()
    def setup(self):
        self.classifier.train()
    def draw(self , xlabel = '' , ylabel = '' , legend_loc = '' , title = ''):
        self.training_progress_diagram()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.legend(loc=legend_loc)
        plt.show()
    def training_progress_diagram(self):
        cost = self.classifier.cost
        plt.plot(range(1 , len(cost) + 1) , cost , 'k-')
