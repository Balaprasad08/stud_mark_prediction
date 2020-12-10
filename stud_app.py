import numpy as np
import pandas as pd
import flask
from flask import Flask,render_template,request
import pickle
model=pickle.load(open('stu_mark.pickle',"rb"))

app=Flask(__name__)
@app.route("/")
def home():
    return render_template("index.html")
@app.route("/predict",methods=['POST'])
def predict ():
    init_features=[int(x) for x in request.form.values()]
    final_features=[np.array(init_features)]
    prediction=model.predict(final_features)
    output=round(prediction[0],2)
    
    return render_template("/index.html",result=output)

if __name__=="__main__":
    app.run()