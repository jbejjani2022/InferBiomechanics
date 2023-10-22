# Welcome to InferBiomechanics

This is a simple repository with example scripts, code and baselines to train and evaluate models that can infer 
physics information from pure motion, trained and evaluated on AddBiomechanics data.

## Getting The Data

The latest dataset is available in a Google Drive folder, [here](https://drive.google.com/drive/u/1/folders/1x_ys7vN0wPn23IjIQkbLGYpvLf9HFXkv).

Please download the data, and place it into the `data` folder in this repository. When completed, there should be a `data/train` folder, and a `data/dev` folder, each with many `*.b3d` files in them.

## Running the Code

First, run `pip3 install -r requirements.txt`

There are several tasks you might want to run, all of which can be accessed from the command line entrypoint, `src/cli/main.py`

### Training a Model

To generate model snapshots, run this command:

`python3 main.py train ...`

It can train several different kinds of models.

### Visualizing a Model

It's often helpful to be able to see how a model is screwing up, and what might be strange in the data.

This command will automatically load the selected model type from the latest checkpoint, run the loaded model on the training set, and visualize the results in the browser:

`python3 main.py visualize ...`

### Evaluating a Model

To get performance numbers for a given model on the whole dataset, even if it hasn't finished training, run:

`python3 main.py analyze ...`

This will automatically pick up the latest model checkpoint file, and run it on the whole dataset, and print out the results.
