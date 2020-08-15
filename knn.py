import pandas as pd
import numpy as np
import nltk
from flask import Flask, render_template, url_for, request
import re
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score



knn=Flask(__name__)

@knn.route('/')

def home():
    return render_template('home.html')


@knn.route('/process', methods=['POST'])
def process():
     
    data_source_url = "https://raw.githubusercontent.com/kolaveridi/kaggle-Twitter-US-Airline-Sentiment-/master/Tweets.csv"
    airline_tweets = pd.read_csv(data_source_url)
    

    features = airline_tweets.iloc[:, 10].values
    labels = airline_tweets.iloc[:, 1].values
    processed_feature=[]

    for sentence in range(0, len(features)):
        processed=re.sub(r'\W', '', str(features[sentence]))
        processed= re.sub(r'\s+[a-zA-Z]\s+', ' ', processed)

        processed = re.sub(r'\^[a-zA-Z]\s+', ' ', processed) 

        processed = re.sub(r'\s+', ' ', processed, flags=re.I)

        processed = re.sub(r'^b\s+', '', processed)

        processed = processed.lower()

        processed_feature.append(processed)
        

    
    vectorizer=CountVectorizer()
    processed_data=vectorizer.fit_transform(processed_feature)

    from sklearn.model_selection import train_test_split

    x_train, x_test, y_train, y_test=train_test_split(processed_data, labels, test_size=0.2, random_state=0)

    classifier=RandomForestClassifier(n_estimators=200, random_state=0)
    classifier.fit(x_train, y_train)
    

    if request.method=='POST':

        
         message=request.form['message']
         data=[message]
         vector=vectorizer.transform(data)
         preds=classifier.predict(vector)[0]
         
         
         return render_template('home.html', prediction=preds)
   


if __name__ == '__main__':
    knn.run(debug=True)
    


     

         

    



 





