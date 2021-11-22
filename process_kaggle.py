"""
Script used to process the Kaggle dataset and extract the matches.
It also creates the vocabulary from it.
https://www.kaggle.com/milesh1/35-million-chess-games
"""


import os
import re


if __name__ == "__main__":
    os.makedirs("dataset/kaggle2/", exist_ok=True)
    vocab_counter = set()

    with open(f"dataset/processed_kaggle2.txt", "w", encoding="utf-8") as outf:
        with open("dataset/kaggle2.txt", "r", encoding="utf-8") as inpf:
            for line in inpf:
                try:
                    ostr = line.split("###")[1].strip()
                    ostr = re.sub("W\d+.", "", ostr)
                    ostr = re.sub("B\d+.", "", ostr)

                    if len(ostr) > 0:
                        if ostr[-1] != '\n':
                            ostr = ostr + '\n'

                        outf.write(ostr)

                        for move in ostr.split(" "):
                            move = move.replace("\n", "")

                            if move != "":
                                vocab_counter.add(move)
                    else:
                        a = 0
                except:
                    pass

        os.makedirs("vocabs", exist_ok=True)

        with open(f"vocabs/kaggle2.txt", "w", encoding="utf-8") as f:
            for v in vocab_counter:
                f.write(v + "\n")
                