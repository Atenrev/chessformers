import os


VOCAB_DIR = "vocabs"


class Tokenizer:
    pad_token_index: int = 0
    bos_token_index: int = 1
    eos_token_index: int = 2
    unk_token_index: int = 3

    pad_token: str = "<pad>"
    bos_token: str = "<bos>"
    eos_token: str = "<eos>"
    unk_token: str = "<unk>"

    def __init__(self, vocab_path: str = f"{VOCAB_DIR}/kaggle2_vocab.txt") -> None:
        self.vocab_dict = {
            self.pad_token: self.pad_token_index,
            self.bos_token: self.bos_token_index,
            self.eos_token: self.eos_token_index,
            self.unk_token: self.unk_token_index,
        }

        with open(vocab_path, "r", encoding="utf-8") as f: 
            for i, token in enumerate(f):
                self.vocab_dict[token.replace("\n", "")] = i + 4

    def encode(self, token_str: str, add_bos_token=True):
        encoded = []

        if add_bos_token:
            encoded.append(self.bos_token_index)

        for token in token_str.split():
            if token in self.vocab_dict:
                encoded.append(self.vocab_dict[token])
            else:
                encoded.append(self.unk_token_index)

        return encoded

    def decode(self, token_ids: list):
        decoded = []

        for token_id in token_ids:
            for token, index in self.vocab_dict.items():
                if index == token_id:
                    decoded.append(token)

        return " ".join(decoded)


    def vocab_size(self) -> int:
        return len(self.vocab_dict)


    @classmethod
    def generate_vocab(cls, dataset_path: str):
        from pathlib import Path
        from tqdm import tqdm

        vocab_counter = set()

        for game in tqdm(Path(dataset_path).glob("*.txt")):
            game = game.read_text(encoding="utf-8")
            for move in game.split(" "):
                move = move.replace("\n", "")

                if move != "":
                    vocab_counter.add(move)

        os.makedirs(VOCAB_DIR, exist_ok=True)

        with open(f"{VOCAB_DIR}/kaggle2.txt", "w", encoding="utf-8") as f:
            for v in vocab_counter:
                f.write(v + "\n")


if __name__ == "__main__":
    Tokenizer.generate_vocab("dataset/kaggle2/")
    tokenizer = Tokenizer(f"{VOCAB_DIR}/kaggle2.txt")
    encoded = tokenizer.encode("d4 d5 c4 c6 cxd5 e6 dxe6 fxe6 Nf3 pepe Bb4+ Nc3 Ba5 Bf4 <eos>")
    decoded = tokenizer.decode(encoded)
    print(encoded)
    print(decoded)
    