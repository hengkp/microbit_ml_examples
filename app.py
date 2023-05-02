from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np

app = Flask(__name__)

# load the trained TensorFlow model
model = tf.keras.models.load_model('soil_watering_model.h5')

@app.route('/', methods=['POST'])
def predict():
    # parse the input data from the JSON request
    input_data = request.get_json()
    
    # preprocess the input data
    temperature = input_data['temperature']
    humidity = input_data['humidity']
    soil_moisture = input_data['soil_moisture']
    input_array = np.array([[temperature, humidity, soil_moisture]])
    
    # make a prediction using the loaded model
    prediction = model.predict(input_array)
    watering_need = bool(prediction[0])
    
    # return the prediction result as a JSON response
    response = {'watering_need': watering_need}
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
