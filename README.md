# BayesianDetection

This project is for the paper "Detecting Adversarial Examples with Bayesian Neural Network".

## Data

- CIFAR10
- MNIST
- Imagenet-sub

## Network

- Bayesian VGG16 (for CIFAR10/Imagenet-sub)
- Bayesian CNN (for MNIST)

## Pre-trained Models

Please check the google drive [link](https://drive.google.com/drive/folders/1PtxGooAU6RnGYZfQYHlT3HnOK-gY088n?usp=sharing)


## Detecting Adversarial Samples

### 0. Generate adversarial samples

```
root=/path/to/data
CUDA_VISIBLE_DEVICES=0 python3 generate_adv_samples.py --root ${root} --model bnn --net vgg --data cifar10 --adv_type PGD
```

### 1. Generate distance between adversarial/test and training samples

 ```
 root=/path/to/data
 CUDA_VISIBLE_DEVICES=0 python3 get_distance.py -root ${root} --model bnn --net vgg --data cifar10 --adv_type PGD
 ```

### 2. Train BATECTOR and evaluate its performance

```
CUDA_VISIBLE_DEVICES=0 python3 get_auc.py --model bnn --net vgg --data cifar10 --adv_type PGD
```
