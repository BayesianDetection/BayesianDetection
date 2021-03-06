import torch
import torch.nn as nn
from .layers.batchnorm2d import RandBatchNorm2d
from .layers.conv2d import RandConv2d
from .layers.linear import RandLinear

import numpy as np
from scipy.stats import wasserstein_distance

from sklearn.decomposition import PCA
import pickle as pk



def get_dist(outf, layer,result_j, predicted, sample_size, criteria):
    final_tr = np.load(outf + "emp_bnn_train_"+str(layer) + ".npy")
    predicted_tr = np.load(outf + "labels_bnn_train.npy")
    if layer < 6:
        pca_model = pk.load(open(outf + "pca_bnn"+str(layer)+".pkl","rb"))
        #final_tr = pca_model.transform(final_tr)
        final_adv = pca_model.transform(result_j.cpu().detach().numpy())
    else:
        final_adv = result_j.cpu().detach().numpy()
 
    #final_adv = result_j.cpu().detach().numpy()
    distance = np.zeros(final_adv.shape[0])
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
        distance[i] = dis_adv
    return distance

class BasicCNN(nn.Module):
    def __init__(self, sigma_0, N, init_s, criteria='mean', sample_size=20, layer_list=[3,6], pass_num=4):
        super(BasicCNN, self).__init__()        
        self.main = nn.Sequential(
            RandConv2d(sigma_0, N, init_s, 1, 20, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            RandConv2d(sigma_0, N, init_s, 20, 50, kernel_size=5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            )
        self.fc = nn.Sequential(
            RandLinear(sigma_0, N, init_s, 4*4*50, 500),    
            nn.ReLU(),            
            RandLinear(sigma_0, N, init_s, 500, 10),
            )
        self.pass_num = pass_num
        self.layer_list = layer_list
        self.sample_size = sample_size
        self.criteria = criteria


    def forward(self, x):
        kl_sum = 0
        out = x
        #out = self.features(x)
        for l in self.main:
            if type(l).__name__.startswith("Rand"):
                out, kl = l.forward(out)
                if kl is not None:
                    kl_sum += kl
            else:
                out = l.forward(out)
        out = out.view(out.size(0), -1)
        for l in self.fc:
            if type(l).__name__.startswith("Rand"):
                out, kl = l.forward(out)
                if kl is not None:
                    kl_sum += kl
            else:
                out = l.forward(out)
        # out, kl = self.classifier(out)
        # kl_sum += kl
        return out, kl_sum


    def forward_hidden(self, x):
        out = x
        result = []
        for l in self.main:
            if type(l).__name__.startswith("Rand"):
                out, _ = l.forward(out)
            else:
                out = l.forward(out)
            result.append(out.view(out.size(0), -1))
        return result

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
                if self.layer_list[j] >= len(inter_output):
                    result[j].append(outputs)
                else:
                    result[j].append(inter_output[self.layer_list[j]])
                #result[j].append(inter_output[self.layer_list[j]])
        
        # Get distance from each layer      
        dist_matrix = np.zeros((x.shape[0],m))
        for j in range(m):
            result[j] = torch.cat(result[j],1)
            dist_matrix[:,j] = get_dist(outf, self.layer_list[j],result[j],predicted,
                       self.sample_size, self.criteria)
  
        return dist_matrix


"""
        x = self.main(x)
        x = x.view(-1, 4*4*50)
        x = self.fc(x)
        return x, None
"""