from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
from Utils.functions import get_tokenizer, predict_via_HTML, create_input_dict

global tokenizer
tokenizer = get_tokenizer()

class InputRequest(BaseModel):
    context: str
    question: str

# APP
app = FastAPI()
# Permitir solicitudes desde cualquier origen
app.add_middleware(CORSMiddleware, allow_origins=['*'])

# Define the default route
@app.get('/')
def main_page():
    return "REST service is activa via FastApi"


@app.post('/model/predict')
async def predict(request: InputRequest):
    data = {'Success': False}

    question = request.question
    context = request.context

    if isinstance(question, str) and isinstance(context, str):

        # Input Process
        my_input_dict, my_context_words, context_tok_to_word_id, question_tok_len = create_input_dict(tokenizer, question, context)
    
        model_name = 'squad'
        port_HTML = '9501'
        model_version = '1'
        
        start_logits, end_logits = predict_via_HTML(my_input_dict, model_name, model_version, port_HTML)
        
        # Filter the logits for the context
        start_logits_context = start_logits[question_tok_len+1:]
        end_logits_context = end_logits[question_tok_len+1:]

        # Get the ids for the context
        start_word_id = context_tok_to_word_id[np.argmax(start_logits_context)]
        end_word_id = context_tok_to_word_id[np.argmax(end_logits_context)]

        # Answer
        predicted_answer = ' '.join(my_context_words[start_word_id:end_word_id+1])

        return predicted_answer
    









