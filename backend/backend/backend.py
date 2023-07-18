from flask import Flask, jsonify
from model_service import return_response
import torch

app = Flask(__name__)

@app.route('/get_response/<user_input>', methods=['GET'])
def get_value(user_input):
    value = return_response(user_input)
    return jsonify({'value': value})  # Return the result as JSON

if __name__ == '__main__':
    app.run()
