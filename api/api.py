from flask import Flask
from flask import request
from joblib import load

app = Flask(__name__)
model_path = "./model/"
model = load(model_path)

@app.route("/")
def hello_world():
    return "<!-- hello --> <b> Hello, World!</b>"


# get x and y somehow    
#     - query parameter
#     - get call / methods
#     - post call / methods ** 


@app.route("/prediction", methods=['POST'])
def predict_digit():
    image = request.json['image']
    model = request.json['model']
    print("done loading")
    predicted = model.predict([image])
    return {"y_predicted":int(predicted[0])}