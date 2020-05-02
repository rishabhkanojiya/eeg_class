from flask import Flask , jsonify ,request
from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder

import numpy as np
import tensorflow as tf
app = Flask(__name__)

label_encoder = LabelEncoder()
label_encoder.classes_ = np.load('model/classes.npy')

json_file = open('model/dlModel.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model/dlModel.h5")
print("Loaded model from disk")



global graph
graph = tf.get_default_graph()
# data = [10.142,8.087,5.798,6.704,5.300,6.124,3.337,4.618,3.215,4.333,5.127,0.712,2.360,6.907,7.609,15.106,1.851,-0.427,7.996,3.642,3.337,0.997,7.812,2.472,2.584,7.812,14.740,7.192,8.687,10.101,10.295,12.054,4.578,17.019,3.855,5.280,5.280,6.236,8.189,1.790,3.021,2.645,5.290,3.306,3.164,7.640,11.210,5.961,5.178,0.682,10.183,5.605,1.892,-1.485,10.590,9.328,1.841,6.266,10.864,2.645,3.743,1.353,10.213,7.701]
# data = np.array(data)
# data = data.reshape(-1, 1)

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

# data = scaler.fit_transform(data)
# data = data.reshape(1,64 ,1)

# pred = loaded_model.predict(data)

# print(pred)

def predictValue(data):
    data = np.array(data)
    data = data.reshape(-1, 1)
    data = scaler.fit_transform(data)
    data = data.reshape(1, -1)
    with graph.as_default():
        pred = loaded_model.predict(data, batch_size=1, verbose=1)
    pred = label_encoder.inverse_transform([np.argmax(pred)])
    return pred

@app.route('/predict',methods = ['POST'])
def predict():
    
    if request.method == 'POST': 
        req = request.get_json()
        data = req['array']
        result = predictValue(data)
        return jsonify({'message':result.tolist()})

app.run(port = 4000)