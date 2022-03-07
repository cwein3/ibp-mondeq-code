# Supplementary material submission for "Certified Robustness for Deep Equilibrium Models via Interval Bound Propagation"

PyTorch code for training models discussed in this paper. The main script for training/testing a model is ```runner.py```. 

The ```--arch``` argument specifies which architecture to use: either 3 or 7 layer explicit models, or 3 or 7 layer IBP-MonDEQ models. The ```--dataset``` argument specifies the dataset. The ```--save_dir``` command specifies where to log the run. The ```--data_cache_dir``` argument specifies where to download/cache the MNIST or CIFAR dataset. 

To list additional arguments and their descriptions, use the command ```python runner.py -h```.