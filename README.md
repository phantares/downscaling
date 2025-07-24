<div align="center">

# Precipitation downscaling

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.7.0-red)](https://pytorch.org/)
[![PyTorch Lightning](https://img.shields.io/badge/PyTorch--lightning-2.5.1-purple)](https://lightning.ai/)

</div>

Aims to downscale global model predictions to CWA QPESUMS precipitation.

## Environment
Activate a virtual environment by pipenv.

Install packages by requirements.txt,
```
# install PyTorch (copy first row in requirements.txt)

# install other packages
pipenv install -r requirements.txt
```
or by Pipfile.
```
pipenv install
```

## Training
- Set hyperparametes in **experiments/config/**.
- All configurations are created by **Hydra**.
```
# enter virtual environment
pipenv shell

# train
python train.py -cn <config_name> experiment.name=<experimant_name> experiment.sub_name=<sub_experiment_name> other.config=<settings>

# track by TensorBoard
tensorboard --logdir=<log_path> --port=<port_number> --bind_all
```
- If only `python train.py`, settings refer to **experiments/config/train.yaml**.
- `python -m train.py` (hydra multiple run) will name the sub_file by the given multiple settings automatically. 

## Inference for testing data
```
# enter virtual environment
pipenv shell

# test
python test.py experiment_name/sub_experiment_name
```