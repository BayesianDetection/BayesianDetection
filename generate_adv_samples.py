"""
Some codes are from https://github.com/pokaxpoka/deep_Mahalanobis_detector
"""

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import os
import shutil
import argparse
import numpy as np
import string 
import random 
from tqdm import tqdm

from attacker.pgd import Linf_PGD, Linf_Res_PGD, FGSM
from attacker.cw import cw

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--l', default=0.00, type=float, help='lambda for Restricted Linf_Res_PGD')
parser.add_argument('--model', default='bnn', type=str, help='model name')
parser.add_argument('--net', default='vgg', type=str, help='model name')
parser.add_argument('--train', default='test', help='test set')
parser.add_argument('--steps', default=10, type=int, help='#adv. steps')
parser.add_argument('--rep', default=10, type=int, help='#adv. steps')
parser.add_argument('--max_norm', default=0.03, type=float, help='Linf-norm in PGD')
parser.add_argument('--eps', default=0.03, type=float, help='Linf-norm in cw')
parser.add_argument('--confidence', default=0.00, type=float, help='Linf-norm in cw')
parser.add_argument('--sigma_0', default=0.05, type=float, help='Gaussian prior')
parser.add_argument('--init_s', default=0.1, type=float, help='Initial log(std) of posterior')
parser.add_argument('--root', default='/home/', type=str, help='path to dataset')
parser.add_argument('--data', default='cifar10', type=str, help='dataset name')
parser.add_argument('--adv_type', default='PGD', type=str, help='dataset name')
parser.add_argument('--nclass', default=10, type=int, help='dataset name')


parser.add_argument('--resume', action='store_false', help='resume')
opt = parser.parse_args()

opt.outf  = './adv_output/' + opt.net +  '_' + opt.data + '/'
suffix = "vi" if opt.model == 'bnn' else 'plain'
opt.model_in = './model_out/' + opt.data + '_' + opt.net + '_' + suffix
if os.path.isdir(opt.outf) == False:
    os.makedirs(opt.outf)

suffix = opt.model

print(opt)
# Data
print('==> Preparing data..')
if opt.data == 'cifar10':
    nclass = 10
    img_width = 32
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.CIFAR10(root=opt.root+'/data', train=True, download=False, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=False, num_workers=1)
    testset = torchvision.datasets.CIFAR10(root=opt.root+'/data', train=False, download=False, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=True, num_workers=1)
elif opt.data == 'imagenet-sub':
    nclass = 143
    opt.nclass = 143
    img_width = 64
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(img_width, scale=(0.8, 0.9), ratio=(1.0, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(img_width),
        transforms.ToTensor(),
    ])
    trainset = torchvision.datasets.ImageFolder(opt.root+'/data/sngan_dog_cat', transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=False, num_workers=1)
    testset = torchvision.datasets.ImageFolder(opt.root+'/data/sngan_dog_cat_val', transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=True, num_workers=2)
elif opt.data == 'mnist':
    class MNIST:
        def __init__(self, root):
            trans = transforms.Compose([transforms.ToTensor()])
            train_set = torchvision.datasets.MNIST(root=root+'/data', train=True, transform=trans, download=False)
            test_set = torchvision.datasets.MNIST(root=root+'/data', train=False, transform=trans, download=False)
       
            self.train_data = train_set
            self.test_data = test_set  
    trainset = MNIST(opt.root).train_data
    testset = MNIST(opt.root).test_data
    trainloader = DataLoader(trainset, batch_size=100, shuffle=False)
    testloader = DataLoader(testset, batch_size=100, shuffle=True)
else:
    raise NotImplementedError('Invalid dataset')


if opt.model == 'bnn':
    if opt.data ==  'mnist':
        from models.cnn_vi import BasicCNN
        net = BasicCNN(opt.sigma_0, len(trainset), opt.init_s).cuda()

    else:
        from models.vgg_vi import VGG
        net = VGG(opt.sigma_0, len(trainset), opt.init_s, 'VGG16', nclass, img_width=img_width).cuda()

elif opt.model == 'nn':
    if opt.data == 'mnist':
        from models.cnn import BasicCNN
        net = BasicCNN().cuda()

    else:
        from models.vgg import VGG
        net = VGG('VGG16', nclass, img_width=img_width).cuda()
else:
    raise NotImplementedError('Invalid model')

if opt.resume:
    print("==> Resuming from {}".format(opt.model_in))
    net.load_state_dict(torch.load(opt.model_in))


net.eval()

def test(epoch):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in tqdm(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs_agg = 0
            for _ in range(opt.rep):
                outputs, _ = net(inputs)
                outputs_agg += outputs            
            _, predicted = outputs_agg.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()                    
        print("[TEST] Acc: {:.4f}".format(correct/total))

        

def get_res_adv(dataloader):
    net.eval()
    adv_data_tot, clean_data_tot = 0, 0
    label_tot = 0
    adv_label_tot = 0

    correct, adv_correct = 0, 0
    total, generated_noise = 0, 0

    selected_list = []
    selected_index = 0

    for data, target in tqdm(dataloader):
        data, target = data.cuda(), target.cuda()
        outputs_agg = 0
        with torch.no_grad():
            for _ in range(opt.rep):       
                outputs_o, _ = net(data)
                outputs_agg += outputs_o
        _, pred = outputs_agg.max(1)
        equal_flag = pred.eq(target).cpu()

        # compute the accuracy
        equal_flag = pred.eq(target).cpu()
        correct += equal_flag.sum()


        if total == 0:
            clean_data_tot = data.clone().data.cpu()
            label_tot = target.clone().data.cpu()
        else:
            clean_data_tot = torch.cat((clean_data_tot, data.clone().data.cpu()),0)
            label_tot = torch.cat((label_tot, target.clone().data.cpu()), 0)
            

        if opt.adv_type == 'CW':
            adv_data = cw(data, target,  net, opt.steps, opt.eps, opt.confidence, n_class=opt.nclass)
        elif opt.adv_type in ['PGD', 'PGD_RES']: 
            adv_data = Linf_Res_PGD(data, target, net, opt.steps, opt.max_norm, opt.l)
        elif opt.adv_type == 'FGSM': 
            adv_data = FGSM(data, target, net, opt.max_norm)

        temp_noise_max = torch.abs((data.data - adv_data).view(adv_data.size(0), -1))
        temp_noise_max, _ = torch.max(temp_noise_max, dim=1)
        generated_noise += torch.sum(temp_noise_max)


        outputs_agg = 0
        with torch.no_grad():
            for _ in range(opt.rep):       
                outputs, _ = net(adv_data)
                outputs_agg += outputs
        _, pred = outputs_agg.max(1)
        equal_flag_adv = pred.eq(target).cpu()
        adv_correct += equal_flag_adv.sum()

        if total == 0:
            flag = 1
            adv_data_tot = adv_data.clone().cpu()
            adv_label_tot = pred.clone().data.cpu()
        else:
            adv_data_tot = torch.cat((adv_data_tot, adv_data.clone().cpu()),0)
            adv_label_tot = torch.cat((adv_label_tot, pred.clone().data.cpu()), 0)


        for i in range(data.size(0)):
            if equal_flag[i] == 1 and equal_flag_adv[i] == 0:
                selected_list.append(selected_index)
            selected_index += 1
            
        total += data.size(0)

    selected_list = torch.LongTensor(selected_list)
    clean_data_tot = torch.index_select(clean_data_tot, 0, selected_list)
    adv_data_tot = torch.index_select(adv_data_tot, 0, selected_list)
    label_tot = torch.index_select(label_tot, 0, selected_list)
    adv_label_tot = torch.index_select(adv_label_tot, 0, selected_list)

    np.save( f'{opt.outf}/clean_data_{opt.net}_{opt.data}_{opt.adv_type}_{suffix}', clean_data_tot.cpu().detach().numpy() )
    np.save( f'{opt.outf}/adv_data_{opt.net}_{opt.data}_{opt.adv_type}_{suffix}', adv_data_tot.cpu().detach().numpy() )
    np.save( f'{opt.outf}/label_{opt.net}_{opt.data}_{opt.adv_type}_{suffix}', label_tot.cpu().detach().numpy() )
    np.save( f'{opt.outf}/adv_label_{opt.net}_{opt.data}_{opt.adv_type}_{suffix}', adv_label_tot.cpu().detach().numpy() )

    print('Adversarial Noise:({:.2f})\n'.format(generated_noise / total))
    print('Final Accuracy: {}/{} ({:.2f}%)\n'.format(correct, total, 100. * correct / total))
    print('Adversarial Accuracy: {}/{} ({:.2f}%)\n'.format(adv_correct, total, 100. * adv_correct / total))
   



test(0)
print("==> Start generating adversarial examples")
if opt.train == 'train':
    dataloader = trainloader
else:
    dataloader = testloader
get_res_adv(dataloader)




