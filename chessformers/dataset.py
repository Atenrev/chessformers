import os
import torch
from pathlib import Path
from torch.utils.data import Dataset
from chessformers.tokenizer import Tokenizer


DIR = os.path.dirname(os.path.realpath(__file__))


class PGNDataset(Dataset):
    def __init__(self, tokenizer: Tokenizer, path: str, n_positions=512):
        self.n_positions = n_positions
        self.tokenizer = tokenizer
        self.games = []

        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                self.games.append(line)
        
        print("Dataset read.")

    def __pad(self, sample: list):
        while len(sample) < self.n_positions:
            sample.append(self.tokenizer.pad_token_index)

        return sample[:self.n_positions]

    def __len__(self):
        return len(self.games)

    def __getitem__(self, i):
        game = self.games[i] #.read_text(encoding="utf-8")
        encoded = self.tokenizer.encode(game, add_bos_token=True)

        if len(encoded) < self.n_positions:
            encoded.append(self.tokenizer.eos_token_index)

        data = self.__pad(encoded)
        return torch.tensor(data)


