import argparse
import torch
from chessformers.configuration import get_configuration
from chessformers.model import Transformer
from chessformers.tokenizer import Tokenizer
from flask import Flask, request, jsonify, make_response


def _parse_args():
    parser = argparse.ArgumentParser(
        description='Chessformers inference parser')

    parser.add_argument('--load_model', type=str, default="model/chessformer_epoch_13.pth",
                        help='model to load and do inference')
    parser.add_argument('--config', type=str, default="configs/default.yaml",
                        help='location of the configuration file (a yaml)')
    parser.add_argument('--tokenizer', type=str, default="vocabs/kaggle2_vocab.txt",
                        help='location of the tokenizer file')

    args = parser.parse_args()
    return args


app = Flask(__name__)
app.config['CORS_HEADERS'] = 'Content-Type'

args = _parse_args()
config = get_configuration(args.config)
tokenizer = Tokenizer(args.tokenizer)
model = Transformer(tokenizer,
                    num_tokens=tokenizer.vocab_size(),
                    dim_model=config["model"]["dim_model"],
                    d_hid=config["model"]["d_hid"],
                    num_heads=config["model"]["num_heads"],
                    num_layers=config["model"]["num_layers"],
                    dropout_p=config["model"]["dropout_p"],
                    n_positions=config["model"]["n_positions"],
                    )
model.load_state_dict(torch.load(args.load_model))


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