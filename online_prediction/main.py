import json
import pandas as pd
from flask import Flask, request, jsonify, make_response
from helper import *

app = Flask(__name__)

@app.route('/', methods=['GET'])
def index_get():
    response = make_response(jsonify({
        'status': 'Success!',
        'message': 'Welcome to PondPediaPrediction!'
    }))
    response.status_code = 200
    response.headers['Content-Type'] = 'text/plain'

    return response

@app.route('/water', methods=['POST'])
def water():
    try:
        json_response = request.get_json()
        prediction_json = formatted_predict(json_response)
       
        response_data = {
            'status': 'Success!',
            'message': 'Water Quality Prediction for the next 6 hours',
            'predictions': prediction_json
        }
        response = make_response(jsonify(response_data))
        response.status_code = 200
        response.headers['Content-Type'] = 'application/json'
        
        return response
    except Exception as e:
        response = make_response(jsonify({
            'status': 'Error',
            'message': str(e)
        }))
        response.status_code = 500
        response.headers['Content-Type'] = 'application/json'

        return response

@app.route('/fishgrowth', methods=['POST'])
def fishgrowth():
    try:
        json_response = request.get_json()
        prediction_json = formatted_predict_regression(json_response)

        response_data = {
            'status': 'Success!',
            'message': 'Fish Growth Rate Prediction',
            'predictions': prediction_json,
        }

        response = make_response(jsonify(response_data))
        response.status_code = 200
        response.headers['Content-Type'] = 'application/json'

        return response

    except Exception as e:
        response = make_response(jsonify({
            'status': 'Error',
            'message': str(e)
        }))
        response.status_code = 500
        response.headers['Content-Type'] = 'application/json'

if __name__ == "__main__":
    app.run(debug=True)

