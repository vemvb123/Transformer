# Transformer

This project is for coding a transformer from scratch with JAX.

The transformer was used for language translation through the opus translation dataset.

The project is explained and detailed in the pdf report.

<img width="521" height="723" alt="image" src="https://github.com/user-attachments/assets/194b847e-8745-4ea4-be45-491168efb0ae" />


This project showcases the development of a transformer with JAX.
This is an intresting project because it shows how a transformer works, by showcasing code and results for the transformer. 

A transformer handles sequence data and uses self-attention mechanisms to determine
weights and spot long-range dependencies. The architecture is widely used.
It is used as a building block in language models like ChatGPT and BERT

# How to run
The project can be run from the train.py file.

After training, a weights will be saved in a .pkl file.

The model can be adjusted in the config.py file.

# Results
Results for translation on the dataset Opus books from English to Italian

<img width="416" height="472" alt="image" src="https://github.com/user-attachments/assets/e95a3e12-9602-4f96-9edf-34beff435db5" />



# Module overview
The different files in the projects are used to initilize and train a transformer
model on different datasets.

All the modules in the projects define different algorithms,
that combined defines and algorithm for creating and using a transformer.



## train
To run the project, these files should be run.
For each dataset, there is one train file for each
hyperparameter variation. So there are 4 train files for each dataset (opus and the synthetic datasets).

I have also included some of the code from (\cite{sayed2024transformer}).
His project can also be run by running the train.py file.

The unfinished project for the wikitext-103 is included.
For this project, the train file contains code for splitting the data across several
GPUs JAX's pmap framework.



## config
The hyperparameters are stated in the config files.
For each train file, there is a config file.



## init forward
The init forward files are for initilizing the transformer model,
and forward passing data through the transformer model.
In the train files, the model parameters are in a variable "params".

The synthetic init forward files differ from the init forward file for the opus dataset,
in that the init forward files for the synthetic datasets have no embedding and final projection.


## transformer
This file defines the initilization of different transformer layers,
and the forward pass for the layers.

The module is used in the init forward files to construct a full model by stacking the
layers on top of eachother.


## data
This file is used to initilize and preprocess the opus dataset.
Much of the code is gotten directly from (\cite{sayed2024transformer}) 


## load_r
These files are in the projects for the synthetic datasets.
They are responsible for initilizing and preprocessing the synthetic datasets.


## utils
This file holds utility functionality.
There is for example functions for saving model weights, loading model weights,
preprocessing data, etc.

## metrics
Holds the function for calculating perplexity

## dependencies.txt
This file holds all the nessecary dependencies for running the projects.


# Data structures and algorithms
## Params
All the parameters for a model is in the variable "params" in the train files.

## Initilization
In the transformer.py file are algorithms for initilizing the layers of the transformer.
The weights for a layer is initilized in jax arrays, and saved in a dictionary.

The init forward files use the abstractions in transformer.py,
to initilize the layers together, by tacking the layers, and
combining all the parameters in a final dictionary for the model.

## Forward pass
The algorithm for forward pass is stored in the transformer.py file, and
the init forward files.

The transformer.py file defines algorithms for passing data through each layer,
by doing operations such as matrix multiplication between the input data and the parameters for the layer.

In the init forward files the layers are stacked on top of each other.
Here is the algorithm for passing data trough for each layer, and
taking the data from some layer and using it as input to the next layer.




