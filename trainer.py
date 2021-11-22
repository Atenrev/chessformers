import os
import argparse
from tqdm import tqdm
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from dataset import LichessSmallDataset
from model import Transformer
from tokenizer import Tokenizer


def _parse_args() -> object:
    parser = argparse.ArgumentParser(
        description='Chessformers trainer parser')

    parser.add_argument('--dataset', type=str, default="dataset/processed_kaggle2.txt",
                        help='location of the dataset')
    parser.add_argument('--vocab', type=str, default='./vocab/kaggle2_vocab.txt',
                        help='location of the vocabulary')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='training batch size')
    parser.add_argument('--epochs', type=int, default=25,
                        help='number of training epochs')
    parser.add_argument('--lr', type=float, default=0.00025,
                        help='learning rate')
    parser.add_argument('--beta1', type=float, default=0.9,
                        help='adam beta')
    parser.add_argument('--save_dir', type=str, default='./model',
                        help='save model directory')
    parser.add_argument('--load_model', type=str, default=None,
                        help='model to load and resume training')

    args = parser.parse_args()
    return args


class Trainer:
    model: Transformer
    train_loader: DataLoader
    val_loader: DataLoader
    save_dir: str
    device: torch.device
    loss_fn: object
    optimizer: torch.optim.Adam
    lr: float
    num_epochs: int

    def __init__(self,
                 model: torch.nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 loss_fn: object,
                 save_dir: str = "./model",
                 learning_rate: float = 0.001,
                 num_epochs: int = 10,
                 adam_beta: float = 0.5
                 ) -> None:
        # torch.manual_seed(42)

        self.save_dir = save_dir
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.lr = learning_rate
        self.loss_fn = loss_fn
        self.num_epochs = num_epochs

        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.lr, betas=(adam_beta, 0.999))

        self.device = torch.device(
            "cuda") if torch.cuda.is_available() else torch.device("cpu")
        print(f'Selected device: {self.device}.')

        self.model.to(self.device)

    def train_epoch(self) -> float:
        self.model.train()
        train_loss = []

        for local_batch in tqdm(self.train_loader):
            X = local_batch.to(self.device).t().contiguous()

            # Now we shift the tgt by one so with the <BOS> we predict the token at pos 1
            y_input = X[:-1]
            y_expected = X[1:].reshape(-1)

            # Get mask to mask out the next words
            sequence_length = y_input.size(0)
            src_mask = self.model.get_src_mask(sequence_length).to(self.device)
            pad_mask = self.model.get_pad_mask(
                y_input, self.model.tokenizer.pad_token_index).to(self.device)

            # Standard training
            pred = self.model(y_input, src_mask, pad_mask)

            loss = self.loss_fn(pred.view(-1, self.model.tokenizer.vocab_size()), y_expected)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            train_loss.append(loss.detach().cpu().numpy())

        return np.mean(train_loss)

    def test_epoch(self) -> float:
        self.model.eval()

        with torch.no_grad():
            i = 0
            total_loss = 0.0

            for local_batch in self.val_loader:
                X = local_batch.to(self.device).t().contiguous()

                # Now we shift the tgt by one so with the <BOS> we predict the token at pos 1
                y_input = X[:-1]
                y_expected = X[1:].reshape(-1)

                # Get mask to mask out the next words
                sequence_length = y_input.size(0)
                src_mask = self.model.get_src_mask(sequence_length).to(self.device)
                pad_mask = self.model.get_pad_mask(
                    y_input, self.model.tokenizer.pad_token_index).to(self.device)

                # Standard training
                pred = self.model(y_input, src_mask, pad_mask)

                loss = self.loss_fn(pred.view(-1, self.model.tokenizer.vocab_size()), y_expected)
                # loss = self.loss_fn(pred.reshape(-1, pred.shape[-1]), y_expected.reshape(-1))
                total_loss += loss

                i += 1

            val_loss = total_loss / i

        return val_loss

    def train(self) -> None:
        best_val_loss = np.Inf

        for epoch in range(self.num_epochs):
            print(
                '\n\n -------- RUNNING EPOCH {}/{} --------\n'.format(epoch + 1, self.num_epochs))
            train_loss = self.train_epoch()

            if self.val_loader is not None:
                val_loss = self.test_epoch()
            else:
                val_loss = train_loss
            print('\n EPOCH {}/{} \t train loss {} \t val loss {}'.format(epoch +
                                                                          1, self.num_epochs, train_loss, val_loss))

            if val_loss < best_val_loss:
                # TODO: Save a checkpoint instead of only a path.
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(
                    self.save_dir, f"chessformer_epoch_{epoch + 1}.pth"))

        torch.save(self.model.state_dict(), os.path.join(
            self.save_dir, "chessformer.pth"))


def main(args) -> None:
    os.makedirs(args.save_dir, exist_ok=True)
    N_POSITIONS = 80

    tokenizer = Tokenizer("vocabs/kaggle2_vocab.txt")

    # Prepare the data
    data = LichessSmallDataset(tokenizer, n_positions=N_POSITIONS)
    # data_len = len(data)
    # train_len = int(data_len * 0.8)
    # train_data, val_data = torch.utils.data.random_split(
    #     data, [train_len, data_len - train_len])

    train_loader = DataLoader(
        data, batch_size=args.batch_size, shuffle=True)
    # val_loader = DataLoader(
    #     val_data, batch_size=args.batch_size, shuffle=True)

    # Define the model
    model = Transformer(tokenizer,
                        num_tokens=tokenizer.vocab_size(),
                        dim_model=768,
                        d_hid=3072,
                        num_heads=12,
                        num_layers=12,
                        dropout_p=0.1,
                        n_positions=N_POSITIONS,
                        )

    if args.load_model is not None:
        print("Loading pre-trained model.")

        try:
            model.load_state_dict(torch.load(args.load_model))
        except:
            print("ERROR: The state dictionary could not be loaded.")
            return

    loss_fn = torch.nn.NLLLoss(ignore_index=tokenizer.pad_token_index)

    # Trainer
    trainer = Trainer(model, train_loader, None,
                      loss_fn=loss_fn,
                      save_dir=args.save_dir,
                      learning_rate=args.lr,
                      num_epochs=args.epochs,
                      adam_beta=args.beta1)
    trainer.train()


if __name__ == "__main__":
    args = _parse_args()
    main(args)
