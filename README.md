# FlockSentry

Open-source affordable system to automatically detect predators and save chickens.

Really good documentation goes here. But the exceptionally good documentation goes _right here_!

### Preliminary readme.md

Using anaconda version 4.10.3 (most recent version) - update with following:

```
conda update -n base -c defaults conda
```

Create a new environment using chosen python version (3.7?):

```
conda create -n name_of_env python=3.7
```

Activate environment:

```
conda activate name_of_env
```

Install cudatoolkit:

```
conda install -c anaconda cudatoolkit
```

Install Pytorch (1.9.1?):

```
conda install pytorch torchvision torchaudio -c pytorch
```

Install numpy:

```
conda install numpy
```

## Using the YOLOv5 Submodule

To clone the submodule repository:

```
git submodule update --init --recursive
```

## Train the YOLOv5 Model

```
cd yolov5
python3 train.py --img 640 --batch 16 --epochs 120 --data custom_data.yaml --weights yolov5s.pt
```
