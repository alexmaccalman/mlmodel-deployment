from flask import Flask, render_template, request, redirect, url_for, jsonify
import requests
import json
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict",methods = ["POST"])
def predictApp():

    # get data from form entries in index.html

    Pclass = request.form.get("Pclass")
    Sex = request.form.get("Sex")
    Age = request.form.get("Age")
    SibSp = request.form.get("SibSp")
    Parch = request.form.get("Parch")
    Fare = request.form.get("Fare")

    array_data = [
        Pclass,
        Sex,
        Age,
        SibSp,
        Parch,
        Fare,
    ]

    print(array_data)
    # fit_transform() expects a 2D array, so we reshape our 1D array
    sc = StandardScaler()
    array_reshaped = [[i] for i in array_data]
    # fit and transform
    sc.fit(array_reshaped)
    transformed_array = sc.transform(array_reshaped)
    # print the transformed array
    
    
    data = [{
        "Pclass": transformed_array[0][0],
        "Sex": transformed_array[1][0],
        "Age": transformed_array[2][0],
        "SibSp": transformed_array[3][0],
        "Parch": transformed_array[4][0],
        "Fare": transformed_array[5][0],
    }]
    

    print(data)

    headers = {
        'Content-Type': 'application/json',
    }

    # data = [{
    #    "f1": 0.80576177,
    #    "f2": 1.37593746,
    #    "F3": -0.09609774,
    #    "f4": -0.46983664,
    #    "f5": -0.46399264,
    #    "f6": -0.41596074,
    # }]

    response = requests.post("http://127.0.0.1:8080/predict", headers=headers, data=json.dumps(data))

    print(response.status_code)
    data = response.json()
    predict_rf = data['prediction'][0]
    print(predict_rf)
    return render_template("index.html", Pclass=Pclass, Sex=Sex, Age=Age, SibSp=SibSp, Parch=Parch, Fare=Fare, predict_rf=predict_rf)

if __name__ == "__main__":
    app.run(debug = True)


