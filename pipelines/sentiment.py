import math, sys
import pickle
import sklearn
import os


class Model(object):
    # Constructor
    # Constructor loads "sentiment_model" into memory as noted in the 
    # pickle_path line. Using that in memory file, getSentiment processes
    # the file sent to it using the model in memory
    
    def __init__(self):
        pickle_path = os.path.join(os.path.dirname(__file__), 'sentiment_model')
        f = open(pickle_path,'rb')
        self.pipeline = pickle.load(f)
        f.close()
    # Method to predict the sentiment
    def getSentiment(self, args):
        return self.pipeline.predict([args])[0]
