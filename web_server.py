import torch
from model import Transformer
from tokenizer import Tokenizer
from PIL import Image
from flask import Flask, request, jsonify, make_response


N_POSITIONS = 80

app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'

tokenizer = Tokenizer("vocabs/kaggle2_vocab.txt")

model = Transformer(tokenizer,
                    num_tokens=tokenizer.vocab_size(),
                    dim_model=768,
                    d_hid=3072,
                    num_heads=12,
                    num_layers=12,
                    dropout_p=0.1,
                    n_positions=N_POSITIONS,
                    )
model.load_state_dict(torch.load(
    "model/chessformer_epoch_13.pth"))


def _build_cors_preflight_response():
    response = make_response()
    response.headers.add("Access-Control-Allow-Origin", "*")
    response.headers.add('Access-Control-Allow-Headers', "*")
    response.headers.add('Access-Control-Allow-Methods', "*")
    return response

def _corsify_actual_response(response):
    response.headers.add("Access-Control-Allow-Origin", "*")
    return response


@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    """
    Returns an updated string of moves in PGN format given
    an input string of moves.

    API JSON Arguments:
        - input_moves: string consisting of all the match moves.
    API JSON Output:
        - success: whether the request was successful or not.
        - moves: the string of moves with the predicted move appended.
    """
    if request.method == "OPTIONS": # CORS preflight
        return _build_cors_preflight_response()
    elif request.method == 'POST':
        request_data = request.get_json()

        if request_data is None or 'input_moves' not in request_data:
            response = {'success': False, 'message': 'Bad request'}
            return _corsify_actual_response(jsonify(response))

        input_moves = tokenizer.bos_token + " " + request_data['input_moves'].strip()
        input_moves = input_moves.strip()
        
        try:
            output_moves = model.predict(
                    input_moves, 
                    stop_at_next_move=True, 
                    temperature=0.2,
                    )
        except ValueError:
            response = {'success': False, 'message': "Illegal move."}
            return _corsify_actual_response(jsonify(response))
        except:
            response = {'success': False, 'message': "Unhandled error."}
            return _corsify_actual_response(jsonify(response))

        output_moves = output_moves.replace("<bos> ", "")        
        response = {'success': True, 'moves': output_moves}
        return _corsify_actual_response(jsonify(response))


if __name__ == '__main__':
    app.run()