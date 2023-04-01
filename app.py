from flask import Flask,render_template,request,redirect,url_for
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn import datasets
app = Flask(__name__)
@app.route("/")
def index():
 
    return render_template("index.html")                
 
@app.route("/predict",methods = ["POST"])
def predictApp():
 
    iris = datasets.load_iris()
    X = iris.data[:, :4]  # we take four features.
    y = iris.target
 
    from sklearn.model_selection import train_test_split
    x_train, x_test, y_train, y_test = train_test_split(X,y,test_size = 0.30)
 
    gbc = GradientBoostingClassifier()
    gbc.fit(x_train,y_train)
    score_gbc = gbc.score(x_test,y_test)
 
    weight = request.form.get("weight")
    age = request.form.get("age")
    height = request.form.get("height")
   
 
    setosa = np.array([0])
    versicolor = np.array([1])
    virginica = np.array([2])
    real_values = np.array([sepalLenght,sepalWidth,petalLenght,petalWidth]).reshape(1, -1)
 
    predict_GBC = gbc.predict(real_values)
 
    return render_template("index.html", weight = weight, age = age, height = height, accuracy_GBC = round(score_gbc*100,2), predict_GBC = predict_GBC )                
    return redirect(url_for("index"))
 
if __name__ == "__main__":
    app.run(debug = True)