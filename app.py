from flask import Flask , jsonify ,request
from flask_cors import CORS
import tensorflow as tf
from tensorflow.keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder

import numpy as np
app = Flask(__name__)
CORS(app)

label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('model/classes.npy')

json_file = open('model/dlModel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model/dlModel.h5")
print("Loaded model from disk")
# loaded_model._make_predict_function()

global graph
graph = tf.get_default_graph()

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

def predictValue(data):
    data = np.array(data)
    data = data.reshape(-1, 1)
    data = scaler.fit_transform(data)
    data = data.reshape(1, -1)
    with graph.as_default():
      pred = loaded_model.predict(data, batch_size=1, verbose=1)
    pred = label_encoder.inverse_transform([np.argmax(pred)])
    return pred


@app.route('/')
def home():
    return jsonify({"message":"please use /predict to post a request"})

@app.route('/predict',methods = ['POST'])
def predict():
    
    if request.method == 'POST': 
        req = request.get_json()
        data = req['array']
        result = predictValue(data)
        return jsonify({'message':result.tolist()})

if __name__ == "__main__":
    app.run()