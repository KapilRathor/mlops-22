from flask import Flask
from flask import request
from flask import jsonify

app = Flask(__name__)

@app.route("/")
def hello_world():
    return "<p>Hello, World!</p>"

model_path = "svm_gamma=0.0005_C=0.7.joblib"
@app.route("/predict", methods=["POST"])
def predict_img():
    image = request.json['image']
    model = load(model_path)
    predicted = model.predict(image)
    return {"y_predicted":predicted}


