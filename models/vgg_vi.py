'''VGG11/13/16/19 in Pytorch.'''
import torch
import torch.nn as nn
import numpy as np
import pickle as pk
from scipy.stats import wasserstein_distance
from .layers.batchnorm2d import RandBatchNorm2d
from .layers.conv2d import RandConv2d
from .layers.linear import RandLinear

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

def get_decision(layer,result_j, predicted, thrd, sample_size, criteria):
    final_tr = torch.load("./hidden_output/emp_bnn_train_"+str(layer)).cpu().detach().numpy()
    predicted_tr = np.load("./data/predicts_bnn_train.npy")
    if layer < 43:
        pca_model = pk.load(open("./data/pca_bnn"+str(layer)+".pkl","rb"))
        final_tr = pca_model.transform(final_tr)
        final_adv = pca_model.transform(result_j.cpu().detach().numpy())
    else:
        final_adv = result_j.cpu().detach().numpy()
        
    decision = np.zeros(final_adv.shape[0])
    for i in range(final_adv.shape[0]):
        data_train_sample = final_tr[predicted_tr == int(predicted[i])] 
        ind = np.random.choice(data_train_sample.shape[0],min(sample_size, data_train_sample.shape[0]),replace=False)
        data_train_sample_i = data_train_sample[ind,]
        dist = np.zeros(data_train_sample_i.shape[0])
        for k in range(data_train_sample_i.shape[0]):
            dist[k] = wasserstein_distance(final_adv[i,:], data_train_sample_i[k,:])
        if criteria == 'mean':
            dis_adv = dist.mean()
        elif criteria == 'min':
            dis_adv = dist.min()
        else:
            dis_adv = np.median(dist)
        if dis_adv > thrd[int(predicted[i])]:
            decision[i] = 1
    return decision

def get_dist(outf, layer,result_j, predicted, sample_size, criteria):
    final_tr = np.load(outf + "emp_bnn_train_"+str(layer) + ".npy")
    predicted_tr = np.load(outf + "labels_bnn_train.npy")
    #print(layer)
    if layer < 44:
        pca_model = pk.load(open(outf + "pca_bnn"+str(layer)+".pkl","rb"))
        #final_tr = pca_model.transform(final_tr)
        #print(result_j.cpu().detach().numpy().shape)
        final_adv = pca_model.transform(result_j.cpu().detach().numpy())
    else:
        final_adv = result_j.cpu().detach().numpy()
        
    distance = np.zeros(final_adv.shape[0])
    for i in range(final_adv.shape[0]):
        data_train_sample = final_tr[predicted_tr == int(predicted[i])]
        #print(predicted[i], data_train_sample.shape)
        ind = np.random.choice(data_train_sample.shape[0],min(sample_size, data_train_sample.shape[0]),replace=False)
        data_train_sample_i = data_train_sample[ind,]
        dist = np.zeros(data_train_sample_i.shape[0])
        for k in range(data_train_sample_i.shape[0]):
            dist[k] = wasserstein_distance(final_adv[i,:], data_train_sample_i[k,:])
        #print(dist)
        if len(dist) == 0:   
            dis_adv = 0
        elif criteria == 'mean':
            dis_adv = dist.mean()
        elif criteria == 'min':
            dis_adv = dist.min()
        else:
            dis_adv = np.median(dist)
        distance[i] = dis_adv
    return distance


class VGG(nn.Module):
    def __init__(self, sigma_0, N, init_s, vgg_name, nclass, criteria='mean', sample_size=20,
                 layer_list=[23,33,43], pass_num=4, img_width=32):
        super(VGG, self).__init__()
        self.sigma_0 = sigma_0
        self.N = N
        self.init_s = init_s
        self.img_width = img_width
        self.classifier = RandLinear(sigma_0, N, init_s, 512, nclass)
        self.features = self._make_layers(cfg[vgg_name])
        self.pass_num = pass_num
        self.layer_list = layer_list
        self.sample_size = sample_size
        self.criteria = criteria

    def forward(self, x):
        kl_sum = 0
        out = x
        for l in self.features:
            if type(l).__name__.startswith("Rand"):
                out, kl = l.forward(out)
                if kl is not None:
                    kl_sum += kl
            else:
                out = l.forward(out)
        out = out.view(out.size(0), -1)
        out, kl = self.classifier.forward(out)
        kl_sum += kl
        return out, kl

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        width = self.img_width
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
                width = width // 2
            else:
                layers += [
                        RandConv2d(self.sigma_0, self.N, self.init_s, in_channels, x, kernel_size=3, padding=1),
                        RandBatchNorm2d(self.sigma_0, self.N, self.init_s, x),
                        nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=width, stride=1)]
        return nn.Sequential(*layers)

    def forward_hidden(self, x):
        out = x
        result = []
        for l in self.features:
            if type(l).__name__.startswith("Rand"):
                out, _ = l.forward(out)
            else:
                out = l.forward(out)
            result.append(out.view(out.size(0), -1))
        return result

    def detect(self, x):
        outputs, _ = self.forward(x)
        _, predicted = outputs.max(1)
        # Get hidden layer output and thrds
        m = len(self.layer_list)
        result = [[] for _ in range(m)]
        layer_names = ''
        for i in range(m):
            layer_names += '_'+str(self.layer_list[i])
        thrds = np.load('./data/Thrds_Res_'+self.criteria+'_bnn'+layer_names+'.npy')
        
        for i in range(self.pass_num):
            inter_output = self.forward_hidden(x)
            for j in range(m):
                result[j].append(inter_output[self.layer_list[j]])
        
        # Get vote from each layer      
        decision_matrix = np.zeros((x.shape[0],m))
        for j in range(m):
            result[j] = torch.cat(result[j],1)
            decision_matrix[:,j] = get_decision(self.layer_list[j],result[j],predicted, 
                           thrds[j],self.sample_size, self.criteria)
        decision = np.sum(decision_matrix,1)
        
        for i in range(x.shape[0]):
            if decision[i] > m/2:
                temp = torch.tensor([-1])
                if torch.cuda.is_available():
                    temp = temp.cuda()
                predicted[i] = temp       
        return predicted, outputs

    def dist(self, x, outf):
        outputs_agg = 0
        for _ in range(20):
            outputs, _ = self.forward(x)
            outputs_agg += outputs           
        _, predicted = outputs_agg.max(1)
        # Get hidden layer output and thrds
        m = len(self.layer_list)
        result = [[] for _ in range(m)]
        layer_names = ''
        for i in range(m):
            layer_names += '_'+str(self.layer_list[i])
        
        for i in range(self.pass_num):
            inter_output = self.forward_hidden(x)
            for j in range(m):
                result[j].append(inter_output[self.layer_list[j]])
        
        # Get distance from each layer      
        dist_matrix = np.zeros((x.shape[0],m))
        for j in range(m):
            result[j] = torch.cat(result[j],1)
            dist_matrix[:,j] = get_dist(outf, self.layer_list[j],result[j],predicted,
                       self.sample_size, self.criteria)
  
        return dist_matrix