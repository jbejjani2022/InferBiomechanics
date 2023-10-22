# Welcome to InferBiomechanics

This is a simple repository with example scripts, code and baselines to train and evaluate models that can infer 
physics information from pure motion, trained and evaluated on AddBiomechanics data.

## Getting The Data

The data is available in a Google Drive folder, [here](https://drive.google.com/drive/u/1/folders/1x_ys7vN0wPn23IjIQkbLGYpvLf9HFXkv).

Please download the data, and place it into the `data` folder in this repository. When completed, there should be a `data/train` folder, and a `data/dev` folder, each with many `*.b3d` files in them.

## Running the Code

First, run `pip3 install -r requirements.txt`

There are several tasks you might want to run, all of which can be accessed from the command line entrypoint, `src/cli/main.py`

### Training a Model

`python3 main.py train ...`

### Visualizing a Model

`python3 main.py visualize ...`

### Evaluating a Model

`python3 main.py evaluate ...`
