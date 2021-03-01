# Dual Generative Adversarial Graph Networks: Unsupervised and Semi-Supervised Learning with Spherical Graph Embeddings

## Requirements

To install requirements:

```setup
conda env create -f environment.yml
conda activate Test

Another pacakeges need to install manually:  
torch-cluster     == 1.5.4
torch-geometric   == 1.6.1
torch-scatter     == 2.0.4
torch-sparse      == 0.6.1
torch-spline-conv == 1.2.0
```

## File

    datasets/                      # container of data  
    dgagn/                         # core code  
    train_mm.py                    # training code of clustering module  
    train_classifer.py             # training code of classifier module  
    train.py                       # training code of semi-supervised module  
    choose_cluster_test.py         # test file for choosing cluster number  
    vmfmix/                        # files of von-Mises Fisher mixture mode  
    runs/                          # runing result  

## Training

To train the model(s) in the paper, run this command:  

    __params:__  
    -r   # name of runing folder, default is DGAGN  
    -n   # training epoch, default is 300  
    -s   # data set name, default is Cora  
    -v   # version of training, default is 1  

Run clustering module:
```
python train_mm.py
```
Run classifier module:
```
python train_classifer.py
```
Run semi-supervised module:
```
python train.py
```


Note: the reparametation trick code is taken from (https://github.com/nicola-decao/s-vae-pytorch)


