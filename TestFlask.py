
from flask import Flask, abort,jsonify,request

import pickle

import numpy as np

app = Flask(__name__)


knn = pickle.load(open("knn.pkl","rb"))

@app.route("/")
def hello():
    return "Hello"

@app.route("/api",methods=['POST'])
def apiTest():
    data = request.get_json(force= True)
    pre_req = [data['sl'],data['sw'],data['pl'],data['pw']]
    pre_req = np.array(pre_req)
    out = knn.predict(pre_req)
    op = [out[0]]
    return  jsonify(results=op)

if __name__ == "__main__":
    app.run(debug=True)
