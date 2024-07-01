# Precipitation downscaling
Aims to downscales global model prediction to CWA QPESUMS precipitation.

## Environment
1. Recommend to build a virtual environment by pipenv under python 3.10 version.
2. PyTorch and CUDA versions refer to https://pytorch.org/get-started/previous-versions/.
```
# install PyTorch 2.1.1 with CUDA 12.1 (first row in requirements.txt)
pipenv install torch==2.1.1+cu121 torchvision==0.16.1+cu121 torchaudio==2.1.1 -i https://download.pytorch.org/whl/cu121

# install other packages
pipenv install -r requirements.txt
```

## Training
### Start training
```
# enter virtual environment
pipenv shell

# train
python train.py experiment.name=<experimant_name> experiment.sub_name=<sub_experiment_name> other.config=<settings>
```
>Default settings refer to **experiments/config/train.yaml**.
> `python -m train.py` (hydra multiple run) will name the sub_file by the given multiple settings automatically. 
> Unet models suggest setting `trainer=unet` together.
>GPU is set to #1.

## Prediction for testing
```
# enter virtual environment
pipenv shell

# test
python test.py file/sub_file
```
>Only CPUs are used.

### Tensorbroad
```
tensorboard --logdir=<log_path> --port=<port_number> --bind_all
```