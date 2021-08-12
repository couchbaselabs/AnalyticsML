### This is the what is used to training the model based the file in the .csv

from pandas import read_csv
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import LogisticRegression
import pickle

def getData():
        airline_train = read_csv("Airline-Sentiment.csv", sep=",", encoding="ISO-8859-1")
        airline_train = airline_train.rename(index=str,
                                             columns={"tweet_id": "id", "airline_sentiment": "sentiment", "text": 
                                                      "text"})
        return airline_train

if __name__ == '__main__':
        tweets = getData()
        X = tweets["text"]
        y = tweets["sentiment"]

        tf_transformer = TfidfTransformer()

        X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=111, test_size=0.2)

        model = Pipeline([
            ('vectorizer', CountVectorizer()),
            ('transformer', TfidfTransformer()),
            ('classifier', LogisticRegression(solver='sag',multi_class='multinomial'))
        ])

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        print(confusion_matrix(y_test,predictions))
        print(classification_report(y_test,predictions))
        print("Overall accuracy: {}".format(accuracy_score(y_test, predictions)))

        ret = model.predict(["The movie is good"])[0]
        print(ret)
        ret = model.predict(["The movie is bad"])[0]
        print(ret)
        ret = model.predict(["The movie is alright"])[0]
        print(ret)

        pickle.dump(model, open("pipelines/sentiment_model", 'wb'), protocol=1)

