import os
#os.chdir(r'D:\yaoli\bayes_detect')
import torch
import argparse
import numpy as np
import pickle as pk
import warnings
warnings.filterwarnings("ignore")
from tqdm import tqdm
from utils.adv_data import AdvData, AdvData2
from attacker.pgd import Linf_Detect

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')
parser.add_argument('--model', default='bnn', type=str, help='model name')
parser.add_argument('--net', default='cnn', type=str, help='model name')
parser.add_argument('--steps', default=40, type=int, help='#adv. steps')
parser.add_argument('--rep', default=10, type=int, help='#adv. steps')
parser.add_argument('--max_norm', default=0.3, type=float, help='Linf-norm in PGD')
parser.add_argument('--sigma_0', default=0.02, type=float, help='Gaussian prior')
parser.add_argument('--init_s', default=0.15, type=float, help='Initial log(std) of posterior')
parser.add_argument('--root', default=r'D:\yaoli', type=str, help='path to dataset:/home/exx/yaoli')
parser.add_argument('--data', default='mnist', type=str, help='dataset name')
parser.add_argument('--adv_type', default='PGD', type=str, help='dataset name')
parser.add_argument('--resume', action='store_false', help='resume')
parser.add_argument('--stat', default='min', type=str, choices=['min','mean','median'],help='test statistic')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--N', default=1000, type=int, help='num of adv samples to generate')
parser.add_argument('--L', default=1.0, type=float, help='lambda for pgd_detect')
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
clean_test = np.load(opt.outf + f'clean_data_{opt.net}_{opt.data}_{opt.adv_type}_{suffix}.npy')
labels = np.load(opt.outf + f'label_{opt.net}_{opt.data}_{opt.adv_type}_{suffix}.npy')

if opt.N < len(clean_test):
    N = opt.N
else:
    N = len(clean_test)

new_test = clean_test[0:N]
new_labels = labels[0:N]
decisions = np.ones(N, dtype=np.int64)

data = AdvData2(range(new_test.shape[0]), torch.tensor(new_test), torch.tensor(new_labels), torch.tensor(decisions))
data_loader = torch.utils.data.DataLoader(data, batch_size=opt.batch_size, shuffle=False)
testdata = AdvData(range(clean_test.shape[0]), torch.tensor(clean_test), torch.tensor(labels))
test_loader = torch.utils.data.DataLoader(testdata, batch_size=opt.batch_size, shuffle=False)

# Model
from models.cnn_detect_new import BasicCNN_Detect
filename = os.path.join(opt.outf,f"lr_{opt.model}_{opt.data}_{opt.stat}_{opt.adv_type}")
detect_model = pk.load(open(filename, 'rb'))
layer_list = [2, 3, 5, 6]
net = BasicCNN_Detect(opt.sigma_0, 60000, opt.init_s, detect_model, opt.outf, 
                      criteria=opt.stat, sample_size=20,layer_list=layer_list).cuda()
        
if opt.resume:
    print("==> Resuming from {}".format(opt.model_in))
    net.load_state_dict(torch.load(opt.model_in))


# Function
def test(model):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, targets in tqdm(test_loader):
            inputs, targets = inputs.cuda(), targets.cuda()
            outputs_agg = 0
            for _ in range(opt.rep):
                outputs, _ = model.forward_one(inputs)
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

    for data, target, decision in tqdm(dataloader):
        data, target, decision = data.cuda(), target.cuda(), decision.cuda()
        outputs_agg = 0
        with torch.no_grad():
            for _ in range(opt.rep):       
                outputs_o, _ = net.forward_one(data)
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
            


        adv_data = Linf_Detect(data, target, decision, net, opt.steps, opt.max_norm, opt.L)
            
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

    np.save( f'{opt.outf}/clean_data_{opt.net}_{opt.data}_Detect{opt.L}_{suffix}', clean_data_tot.cpu().detach().numpy() )
    np.save( f'{opt.outf}/adv_data_{opt.net}_{opt.data}_Detect{opt.L}_{suffix}', adv_data_tot.cpu().detach().numpy() )
    np.save( f'{opt.outf}/label_{opt.net}_{opt.data}_Detect{opt.L}_{suffix}', label_tot.cpu().detach().numpy() )
    np.save( f'{opt.outf}/adv_label_{opt.net}_{opt.data}_Detect{opt.L}_{suffix}', adv_label_tot.cpu().detach().numpy() )
    
    print('Adversarial Noise:({:.2f})\n'.format(generated_noise / total))
    print('Final Accuracy: {}/{} ({:.2f}%)\n'.format(correct, total, 100. * correct / total))
    print('Adversarial Accuracy: {}/{} ({:.2f}%)\n'.format(adv_correct, total, 100. * adv_correct / total))
   


test(net)
print("==> Start generating adversarial examples")
get_res_adv(data_loader)