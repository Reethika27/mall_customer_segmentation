import numpy as np
from flask import Flask, request, render_template

#ml packages
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

app = Flask(__name__)


@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    df = pd.read_csv("dataset/Mall_Customers.csv")
    kmeans = KMeans(n_clusters=5)
    df_x = df.iloc[:, [3, 4]].values
    df_y = kmeans.fit_predict(df_x)
    x_train, x_test, y_train, y_test = train_test_split(df_x, df_y, test_size=0.2, random_state=0)
    dtc = DecisionTreeClassifier()
    dtc.fit(x_train, y_train)
    d = {0: "Standard", 1: "Target", 2: "Carefull", 3: "Careless", 4: "Sensible"}
    int_features = [int(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    y_dtc = dtc.predict(final_features)
    s=d[y_dtc[0]]
    return render_template('index.html', prediction_text='The customer is of {} type'.format(s))

if __name__ == "__main__":
    app.run(debug=True)
