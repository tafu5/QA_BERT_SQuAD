import numpy as np
import tensorflow as tf
from transformers import BertTokenizer
import json
import requests

def get_tokenizer():
    # Loading the Bert Tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    
    return tokenizer

def is_whitespace(c):
    '''
    Returns if the string is a white space or not
    '''
    if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
        return True
    return False

def whitespace_split(text):
    '''
    Toma el texto y devuelve una lista de "palabras" separadas segun los
    espacios en blanco / separadores anteriores.
    '''
    doc_tokens = []
    prev_is_whitespace = True
    for c in text:
        if is_whitespace(c):
            prev_is_whitespace = True
        else:
            if prev_is_whitespace:
                doc_tokens.append(c)
            else:
                doc_tokens[-1] += c
            prev_is_whitespace = False
    return doc_tokens

def tokenize_context(text_words, tokenizer):
    '''
    Takes a list of words (returned by whitespace_split()) and tokenizes each
    word one by one. It also stores, for each new token, the original word
    from the text_words parameter.
    '''
    text_tok = []
    tok_to_word_id = []
    for word_id, word in enumerate(text_words):
        word_tok = tokenizer.tokenize(word)
        text_tok += word_tok
        tok_to_word_id += [word_id]*len(word_tok)
    return text_tok, tok_to_word_id

def get_ids(tokens, tokenizer):
    return tokenizer.convert_tokens_to_ids(tokens)

def get_mask(tokens):
    return np.char.not_equal(tokens, "[PAD]").astype(int)

def get_segments(tokens):
    seg_ids = []
    current_seg_id = 0
    for tok in tokens:
        seg_ids.append(current_seg_id)
        if tok == "[SEP]":
            current_seg_id = 1-current_seg_id # Convierte 1 en 0 y viceversa
    return seg_ids

def create_input_dict(tokenizer, question, context):
    '''
    Take a question and a context as strings and return a dictionary with the 3
    elements needed for the model. Also return the context_words, the
    context_tok to context_word ids correspondance and the length of
    question_tok that we will need later.
    '''

    question_tok = tokenizer.tokenize(question)

    context_words = whitespace_split(context)
    context_tok, context_tok_to_word_id = tokenize_context(context_words, tokenizer)

    input_tok = question_tok + ["[SEP]"] + context_tok + ["[SEP]"]
    input_tok += ["[PAD]"]*(384-len(input_tok)) # in our case the model has been
                                                # trained to have inputs of length max 384
    input_dict = {}
    input_dict["input_word_ids"] = tf.expand_dims(tf.cast(get_ids(input_tok, tokenizer), tf.int32), 0)
    input_dict["input_mask"] = tf.expand_dims(tf.cast(get_mask(input_tok), tf.int32), 0)
    input_dict["input_type_ids"] = tf.expand_dims(tf.cast(get_segments(input_tok), tf.int32), 0)

    return input_dict, context_words, context_tok_to_word_id, len(question_tok)


def predict_via_HTML(my_input_dict, model_name, model_version, port_HTML):
    """ Send a request to the model and receive the prediction """

    my_input_dict = dict((key, value.numpy().tolist()[0]) for (key, value) in my_input_dict.items())
    
    data = json.dumps({'signature_name':'serving_default', 
                       'instances': [my_input_dict]
                       })

    headers = {"content-type": "application/json"}

    uri = ''.join(['http://127.0.0.1:',port_HTML, '/v', model_version, '/models/', model_name, ':predict'])
    print("URI:", uri)

    json_response = requests.post(uri, data=data, headers=headers)
    
    predictions = json_response.json().get('predictions')[0]

    start_logits = predictions.get('output_1')
    end_logits = predictions.get('output_2')


    return start_logits,  end_logits

