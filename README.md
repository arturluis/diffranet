# DiffraNet: Automatic Classification of Serial Crystallography Diffraction Patterns

## Summary

This repo contains the implementation of a suite of classification and AutoML models for DiffraNet. Details about DiffraNet the models implemented here can be found in [our paper](https://arxiv.org/abs/1904.11834) and [DiffraNet page](https://arturluis.github.io/diffranet/).

## Installation guide

Our models were implemented with Python3 and a set of Python libraries. We recommend using pip to install the dependencies of the project.

```
pip install -r requirements.txt
```

### OpenCV

Our implementation of the SIFT feature extractor relies on the OpenCV library with patented packages support. To install OpenCV, we recommend following the instructions [here](https://www.pyimagesearch.com/opencv-tutorials-resources-guides/). This is only necessary for our Hyperopt AutoML optimization (```hyperopt_search.py```).


## Downloading DiffraNet

To download DiffraNet, simply follow [this link](https://www.dcc.ufmg.br/~arturluis/diffranet/diffranet.zip) and extract the downloaded file. DiffraNet already comes split in training/validation/test sets, as described in [our paper](https://arxiv.org/abs/1904.11834). By default, our models assume that data is stored in the ```data/``` folder, this can be changed using the ```--train_path``` and ```--val_path``` arguments.

## Running the models

Most of our models come with preset defaults that allow them to be run with a direct python3 command:

```
python3 deepfreak.py
```

All of the models also accept a set of arguments that allow the user to customize the models or apply them to a different dataset. To see the list of available parameters, use the --help argument:

```
python3 deepfreak.py --help
```

### Running BOHB

The exception to this rule is BOHB. BOHB operates on a distributed setup and requires both a dispatcher and workers to function. To run BOHB, first instantiate a dispatcher:

```
python3 bohb_main.py
```

Then create a worker with:

```
python3 bohb_main.py --worker --worker_id 0
```

Multiple workers can be run in parallel by running the previous command with different worker_ids. We refer to [BOHB's documentation](https://automl.github.io/HpBandSter/build/html/index.html) for detailed information on BOHB.

## References

TBD
