# Developed by Valentin Tafura - e: valentintafura@hotmail.com

# Predict function
from predict import predict                     

#Args
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-p", "--port", required=True, help="Service PORT number is required.")
args = parser.parse_args()

# Service port
port = args.port
print("Port recognized: ", port)

from model_loader import loadModelH5

global loaded_model
loaded_model = loadModelH5()

# Import flask
from flask import Flask, request, jsonify

# Initialize the application service (FLASK)
app = Flask(__name__)

from flask_cors import CORS
CORS(app)

# Define a default route
@app.route('/')
def main_page():
    return 'REST service is active via Flask!'

# Define a model route
@app.route('/model/predict', methods=["POST"])
def prediction():
    data = {'success': False}

    if request.method == "POST":
        # Verifica si el cuerpo de la solicitud contiene datos JSON
        if request.is_json:
            # Obtén el contenido JSON
            json_data = request.get_json()
            if 'context' in json_data and 'question' in json_data:
                context = json_data['context']
                question = json_data['question']

                # Verifica si el input es un string
                if isinstance(context, str) and isinstance(question, str):
                      
                      answer = predict(loaded_model, question, context)
                      data['success'] = True
                      data['answer'] = answer
                
                else:
                    data['error'] = 'Context and question should be strings'
            else:
                data['error'] = 'JSON must contain context and question fields'
        else:
            data['error'] = 'Request must be JSON'

    return jsonify(data)

app.run(host='0.0.0.0',port=port, threaded=False)


# python flask_service.py --port 5000