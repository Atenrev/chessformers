"""
Script used to play against the chessformers. 
Human plays as white.
"""

import argparse
import torch

from model import Transformer
from tokenizer import Tokenizer


def _parse_args() -> object:
    parser = argparse.ArgumentParser(
        description='Steganography inference parser')

    parser.add_argument('--load_model', type=str,
                        help='model to load and do inference')

    args = parser.parse_args()
    return args


def main(args) -> None:
    N_POSITIONS = 80

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
        "model/chessformer_epoch_13.pth"))  # args.load_model

    print(
        "===== CHESSFORMERS ENGINE =====\n"
    + "    Enter valid moves in PGN format.\n"
    + "    Enter \\b to undo a move.\n"
    + "    Enter \\m to show all moves\n"
    )

    input_string = "<bos>"
    boards = [input_string]

    while (len(input_string.split(" ")) < N_POSITIONS
           and input_string.split(" ")[-1] != tokenizer.eos_token):
        next_move = input("WHITE MOVE: ")

        if next_move == "\\m":
            print(input_string)
            continue
        elif next_move == "\\b":
            if len(boards) > 1:
                boards.pop()
                
            input_string = boards[-1]
            continue
        
        prev_input_string = input_string
        input_string += " " + next_move

        try:
            input_string = model.predict(
                input_string, 
                stop_at_next_move=True, 
                temperature=0.2,
                )
            boards.append(input_string)
            print("BLACK MOVE:", input_string.split(" ")[-1])
        except ValueError:
            input_string = prev_input_string
            print("ILLEGAL MOVE. Please, try again.")
        except Exception as e:
            print("UNHANDLED EXCEPTION. Please, try again.")

    print("--- Final board ---")
    print(input_string)
    

if __name__ == "__main__":
    args = _parse_args()
    main(args)
