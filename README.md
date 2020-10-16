# mini-competition

This folder contains several machine learning models for predicting sales at Rossmann. The models are each trained on the data provided and are ready to be tested on the provided test data.

## Local Environment

To have the notebook running locally, you need to follow these steps:

1) Clone the repo: `git clone https://github.com/janzenal/mini-competition.git`
2) Inside the folder of the clone repo, create a conda environment: `conda env create -f environment.yml -n minicomp`
3) Activate the virtual environment: `conda activate minicomp`
4) Run Jupyter: `jupyter lab`

## Data

The data folder contains the csv files store, train and test which can be used to train and test you model.

## Notebook
 
The notebook contains two notebooks:

model_training.ipynb for examining the train data and training different models on it.

model_testing.ipynb for testing the models on the test data.
