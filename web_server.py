import torch
from model import Transformer
from tokenizer import Tokenizer
from PIL import Image
from flask import Flask, jsonify, request


N_POSITIONS = 80

app = Flask(__name__)

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


@app.route('/predict', methods=['POST'])
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
    if request.method == 'POST':
        request_data = request.get_json()

        if request_data is None or 'input_moves' not in request_data:
            return jsonify({'success': False, 'message': 'Bad request'})

        input_moves = tokenizer.bos_token + " " + request_data['input_moves'].strip()
        input_moves = input_moves.strip()
        
        try:
            output_moves = model.predict(
                    input_moves, 
                    stop_at_next_move=True, 
                    temperature=0.2,
                    )
        except ValueError:
            return jsonify({'success': False, 'message': "Illegal move."})
        except:
            return jsonify({'success': False, 'message': "Unhandled error."})

        output_moves = output_moves.replace("<bos> ", "")        
        return jsonify({'success': True, 'moves': output_moves})


if __name__ == '__main__':
    app.run()