# chessformers
## Description
This is a PyTorch implementation of [our future awesome paper]().

* The pre-trained model provided was trained on the [3.5 Million Chess Games dataset](https://www.kaggle.com/milesh1/35-million-chess-games) with a limitation of 80 moves per game.


## Prerequisites
* PyTorch
* chess
* Flask (*Only for API service*)


## Installation
### 1. Clone the repo

```
git clone https://github.com/Atenrev/chessformers.git
cd chessformers
```

### 2. Data
* Download the [3.5 Million Chess Games dataset](https://www.kaggle.com/milesh1/35-million-chess-games). Put it in a "dataset" folder inside the root folder and rename the txt file to ```kaggle2.txt```.
* Run ```process_kaggle.py``` to pre-process the dataset.

You can also use your own dataset. In that case, you should adapt the ```dataset.py``` script accordingly.

### 3. Download the model
* Download the pre-trained model from here: https://drive.google.com/file/d/1dFunGXq8bXRW0_Y47yxE4S30cU8wyfOl/view?usp=sharing

Here is an example of what the root should look like:
```
.
├── trainer.py
├── inference.py
├── web_server.py
├── process_kaggle.py
├── chessformers\
|   ├── dataset.py
|   ├── tokenizer.py
|   └── model.py
├── model\
|   └── chessformer_epoch_13.pth
└── dataset\
    ├── processed_kaggle2.txt
    └── kaggle2.txt
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
You can play against the engine through a [web interface](https://github.com/Atenrev/chessformers-web-interface) by running the ```web_server.py``` script. 

Alternatively, you can play in the console using PGN notation by running ```inference.py```:

``` sh
python .\inference.py

optional arguments:
  -h, --help                show this help message and exit
  --load_model LOAD_MODEL   model to load and do inference
```