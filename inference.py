"""
Script used to play against the chessformers. 
Human plays as white.
"""

import argparse
import torch

from chessformers.configuration import get_configuration
from chessformers.model import Transformer
from chessformers.tokenizer import Tokenizer


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


def main(args) -> None:
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

    print(
        "===== CHESSFORMERS ENGINE =====\n"
    + "    Enter valid moves in PGN format.\n"
    + "    Enter \\b to undo a move.\n"
    + "    Enter \\m to show all moves\n"
    )

    input_string = "<bos>"
    boards = [input_string]

    while (len(input_string.split(" ")) < config["model"]["n_positions"]
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
