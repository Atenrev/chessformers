# chessformers
## Description
This is a PyTorch implementation of [our future awesome paper]().

* The pre-trained model provided was trained on the [3.5 Million Chess Games dataset](https://www.kaggle.com/milesh1/35-million-chess-games).


## Prerequisites
* PyTorch
* chess


## Installation
### 1. Clone the repo

```
git clone https://github.com/Atenrev/chessformers.git
cd chessformers
```

### 2. Data
* Download the [3.5 Million Chess Games dataset](https://www.kaggle.com/milesh1/35-million-chess-games). Put it in a dataset folder inside the root folder.
* Run ```process_kaggle.py``` to pre-process the dataset.

You can also use your own dataset. In that case, you should adapt the ```dataset.py``` script accordingly.

### 3. Download the model
* Download the pre-trained model from here: https://drive.google.com/file/d/1dFunGXq8bXRW0_Y47yxE4S30cU8wyfOl/view?usp=sharing

Here is an example of what the root should look like:
```
.
├── dataset.py
├── model.py
├── tokenizer.py
├── inference.py
├── trainer.py
├── process_kaggle.py
├── model\
|   └── chessformer_epoch_13.pth
└── tiny_imagenet\
    └── all_with_filtered_anotations_since1998.txt
```

## Train
Run ```trainer.py```:

``` sh
python trainer.py 

optional arguments:
  -h, --help                show this help message and exit
  --dataset DATASET         location of the dataset
  --vocab VOCAB             location of the vocabulary
  --batch_size BATCH_SIZE   training batch size
  --epochs EPOCHS           number of training epochs
  --lr LR                   learning rate
  --beta1 BETA1             adam beta
  --save_dir SAVE_DIR       save model directory
  --load_model LOAD_MODEL   model to load and resume trainin
```

## Inference (play against the engine)
Run ```inference.py```:

``` sh
python .\inference.py

optional arguments:
  -h, --help                show this help message and exit
  --load_model LOAD_MODEL   model to load and do inference
```