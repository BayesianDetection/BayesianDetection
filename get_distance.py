import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from utils.adv_data import AdvData

import argparse
import numpy as np
from tqdm import tqdm

from sklearn import metrics
from sklearn.decomposition import PCA
import pickle as pk

torch.manual_seed(126)

parser = argparse.ArgumentParser(description='PyTorch Getting Probabilities')
parser.add_argument('--model', default='nn', type=str, help='model name')
parser.add_argument('--net', default='vgg', type=str, help='model name')
parser.add_argument('--pass_num', default=4, type=int, help='number of pass')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--data', default='cifar10', type=str, help='dataset name')
parser.add_argument('--adv_type', default='PGD', type=str, help='dataset name')
parser.add_argument('--stat', default='min', type=str, choices=['min','mean','median'],help='test statistic')
parser.add_argument('--root', default='/home/', type=str, help='path to dataset')
parser.add_argument('--resume', action='store_false', help='resume')
parser.add_argument('--sigma_0', default=0.05, type=float, help='Gaussian prior')
parser.add_argument('--init_s', default=0.1, type=float, help='Initial log(std) of posterior')
opt = parser.parse_args()
print(opt)

opt.outf  = './adv_output/' + opt.net +  '_' + opt.data + '/'
suffix = 'vi' if opt.model == 'bnn' else 'plain'
opt.model_in = './model_out/' + opt.data + '_' + opt.net + '_' + suffix
suffix = opt.model

# Data
print('==> Preparing test data..')
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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=1)

elif opt.data == 'imagenet-sub':
    nclass = 143
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
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100, shuffle=True, num_workers=1)

elif opt.data == 'mnist':
    nclass=10
    class MNIST:
        def __init__(self, root):
            trans = transforms.Compose([transforms.ToTensor()])
            train_set = torchvision.datasets.MNIST(root=root+'/data', train=True, transform=trans, download=False)
            test_set = torchvision.datasets.MNIST(root=root+'/data', train=False, transform=trans, download=False)
       
            self.train_data = train_set
            self.test_data = test_set  
    trainset = MNIST(opt.root).train_data
    trainloader = DataLoader(trainset, batch_size=100, shuffle=True)

print('==> Preparing data..')
clean_test = np.load(opt.outf + f'clean_data_{opt.net}_{opt.data}_{opt.adv_type}_{suffix}.npy')
adv_test = np.load(opt.outf + f'adv_data_{opt.net}_{opt.data}_{opt.adv_type}_{suffix}.npy')
labels = np.load(opt.outf + f'label_{opt.net}_{opt.data}_{opt.adv_type}_{suffix}.npy')
adv_labels = np.load(opt.outf + f'adv_label_{opt.net}_{opt.data}_{opt.adv_type}_{suffix}.npy')
testdata = AdvData(range(clean_test.shape[0]), torch.tensor(clean_test), torch.tensor(labels))
testloader = torch.utils.data.DataLoader(testdata, batch_size=opt.batch_size, shuffle=False)
advdata = AdvData(range(adv_test.shape[0]), torch.tensor(adv_test), torch.tensor(adv_labels))
advloader = torch.utils.data.DataLoader(advdata, batch_size=opt.batch_size, shuffle=False)


    
# Model
if opt.model == 'bnn':
    if opt.data ==  'mnist':
        num_layers=6
        layer_list = [3, 5, 6]
        dim = 512
        from models.cnn_vi import BasicCNN
        net = BasicCNN(opt.sigma_0, len(trainset), opt.init_s, criteria=opt.stat, sample_size=20,layer_list=layer_list).cuda()


    else:
        num_layers=45
        if opt.adv_type == 'CW':
            layer_list = [3, 4, 7]
        elif opt.adv_type == 'FGSM':
            layer_list = [7]
        else:
            layer_list = [39, 41, 42]
        dim = 512
        from models.vgg_vi import VGG
        net = VGG(sigma_0=opt.sigma_0, N=len(trainset), init_s=opt.init_s, vgg_name='VGG16', 
                  nclass=nclass, criteria=opt.stat, sample_size=20,
                  layer_list=layer_list, pass_num=opt.pass_num, img_width=img_width).cuda()



elif opt.model == 'nn':
    if opt.data == 'mnist':
        num_layers=6
        layer_list = [3, 5, 6]
        dim = 256
        from models.cnn import BasicCNN
        net = BasicCNN(criteria=opt.stat, sample_size=20,layer_list=layer_list).cuda()


    else:
        num_layers=43
        layer_list = [23, 33, 43]
        dim = 1024
        from models.vgg import VGG
        net = VGG(vgg_name='VGG16', nclass=nclass, criteria=opt.stat, sample_size=20,
                     layer_list=layer_list, img_width=img_width).cuda()

else:
    raise NotImplementedError('Invalid model')

if opt.resume:
    print("==> Resuming from {}".format(opt.model_in))
    net.load_state_dict(torch.load(opt.model_in))
    
def test(epoch):
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs_agg = 0
            for _ in range(opt.pass_num):
                outputs, _ = net(inputs)
                outputs_agg += outputs             
            _, predicted = outputs_agg.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        print("[TEST] Acc: {:.3f}".format(correct/total))



def get_emp_dist(dataloader,layers,pass_num):
    #total number of hidden output layer is 45 for vgg16
    net.eval()
    batch_idx = 0
    emp_list = [[] for i in layers]
    m = len(layers)
    predicts = []
    labels = []
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader):
            inputs = inputs.cuda()
            labels.append(targets.cpu().detach().numpy())
            for i in range(m):
                temp_out = [[] for _ in layers]
                for _ in range(opt.pass_num):
                    output, _ = net(inputs)
                    if layers[i] >= num_layers:
                        temp_out[i].append(output)
                    else:
                        inter_output = net.forward_hidden(inputs)
                        temp_out[i].append(inter_output[layers[i]])
                temp_out[i] = torch.cat(temp_out[i],1)
                emp_list[i].append(temp_out[i])
            batch_idx += 1
            if batch_idx >= 100:
                break
    for i in range(m):
        layer = layers[i]
        emp = torch.cat(emp_list[i], 0)
        emp = emp.cpu().detach().numpy()
        dim_cur = min(emp.shape[1], dim)
        pca = PCA(n_components=dim_cur)
        pca_model = pca.fit(emp)
        final_tr = pca_model.transform(emp)
        np.save(opt.outf + "emp_"+opt.model+"_train_"+str(layer), final_tr)
        pk.dump(pca_model, open(opt.outf + "pca_"+opt.model+str(layer)+".pkl","wb"))

    labels = np.concatenate(labels)
    np.save(opt.outf + f"labels_{opt.model}_train", labels)

    return

      
test(0)

#generate empirical distribution of hidden layers

print("======>Generate hidden layer and Feature dimension reduction")


get_emp_dist(trainloader,layer_list,opt.pass_num)


print("Feature dimension finished")



# Get adv-test detection accuracy
def get_distance(data='adv'):
    if data == 'test':
        dataloader = testloader
    else:
        dataloader = advloader
    net.eval()
    dist = []

    with torch.no_grad():
        for inputs, targets in tqdm(dataloader):
            if torch.cuda.is_available():
                inputs, targets = inputs.cuda(), targets.cuda()          
            dist.append(net.dist(inputs, opt.outf))
    dist = np.concatenate(dist,0)
    print("=======>Saving distance matrix on Model "+opt.model +" with "+data+ "data")
    np.save(opt.outf + f"dist_{opt.model}_{opt.data}_{opt.stat}_{opt.adv_type}_{data}",dist)

# Experiment

print("=======>Getting matrix on legitimate images")
get_distance(data='test')
print("=======>Getting matrix on adversarial images")
get_distance()




