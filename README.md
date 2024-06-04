## Federated Time-Series Forecasting


### Installation

We recommend using a conda environment with Python 3.8

1. First install [PyTorch](https://pytorch.org/get-started/locally/)
```
$ conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
```

2. Install additional dependencies
```
$ pip install pandas scikit_learn matplotlib seaborn colorcet scipy h5py carbontracker notebook
```

You can also use the requirements' specification:
```
$ pip install -r requirements.txt
```

### Project Structure
    .
    ├── dataset                 # .csv files
    ├──── ...
    ├── ml                      # Machine learning-specific scipts
    ├──── fl                    # Federated learning utilities
    ├────── client              # Client representation
    ├─────── ...
    ├────── history             # Keeps track of local and global training history
    ├─────── ...
    ├────── server              # Server Implementation
    ├─────── client_manager.py  # Client manager abstract representation and implementation
    ├─────── client_proxy.py    # Client abstract representation on the server side
    ├─────── server.py          # Implements the training logic of federated learning
    ├─────── aggregation        # Implements the aggregation function
    ├───────── ...
    ├─────── defaults.py        # Default methods for client creation and weighted metrics
    ├─────── client_proxy.py    # PyTorch client proxy implementation
    ├─────── torch_client.py    # PyTorch client implementation
    ├──── models                # PyTorch models
    ├───── ...
    ├──── utils                 # Utilities which are common in Centralized and FL settings
    ├────── data_utils.py       # Data pre-processing
    ├────── helpers.py          # Training helper functions
    ├────── train_utils.py      # Training pipeline 
    ├── notebooks               # Tools and utilities
    └── README.md

### Examples
Refer to [notebooks](notebooks) for usage examples.

### Dataset
For an extensive overview of the data collection and processing procedure please refer to [datataset](dataset).
