import torch
import torch.nn.functional as F
from .linf_sgd import Linf_SGD
from torch.optim import Adam


def Linf_Detect(x_in, y_true, z_true, net, steps, eps, L):
    if eps == 0:
        return x_in
    training = net.training
    if training:
        net.eval()
    x_adv = x_in.clone().requires_grad_()
    optimizer = Linf_SGD([x_adv], lr=0.007)
    for _ in range(steps):
        optimizer.zero_grad()
        net.zero_grad()
        decision, out = net(x_adv)
        loss = -F.cross_entropy(out, y_true)-L*F.cross_entropy(decision, z_true)
        loss.backward()
        optimizer.step()
        diff = x_adv - x_in
        diff.clamp_(-eps, eps)
        x_adv.detach().copy_((diff + x_in).clamp_(0, 1))
    net.zero_grad()
    # reset to the original state
    if training:
        net.train()
    return x_adv


# performs Linf-constraint PGD attack w/o noise
# @epsilon: radius of Linf-norm ball
def Linf_PGD(x_in, y_true, net, steps, eps):
    if eps == 0:
        return x_in
    training = net.training
    if training:
        net.eval()
    x_adv = x_in.clone().requires_grad_()
    optimizer = Linf_SGD([x_adv], lr=0.007)
    for _ in range(steps):
        optimizer.zero_grad()
        net.zero_grad()
        out, _ = net(x_adv)
        loss = -F.cross_entropy(out, y_true)
        loss.backward()
        optimizer.step()
        diff = x_adv - x_in
        diff.clamp_(-eps, eps)
        x_adv.detach().copy_((diff + x_in).clamp_(0, 1))
    net.zero_grad()
    # reset to the original state
    if training:
        net.train()
    return x_adv


# performs L2-constraint PGD attack w/o noise
# @epsilon: radius of L2-norm ball
def L2_PGD(x_in, y_true, net, steps, eps):
    if eps == 0:
        return x_in
    training = net.training
    if training:
        net.eval()
    x_adv = x_in.clone().requires_grad_()
    optimizer = Adam([x_adv], lr=0.01)
    eps = torch.tensor(eps).view(1,1,1,1).cuda()
    #print('====================')
    for _ in range(steps):
        optimizer.zero_grad()
        net.zero_grad()
        out, _ = net(x_adv)
        loss = -F.cross_entropy(out, y_true)
        loss.backward()
        #print(loss.item())
        optimizer.step()
        diff = x_adv - x_in
        norm = torch.sqrt(torch.sum(diff * diff, (1, 2, 3)))
        norm = norm.view(norm.size(0), 1, 1, 1)
        norm_out = torch.min(norm, eps)
        diff = diff / norm * norm_out
        x_adv.detach().copy_((diff + x_in).clamp_(0, 1))
    net.zero_grad()
    # reset to the original state
    if training :
        net.train()
    return x_adv

# performs L2-constraint PGD attack w/o noise
# @epsilon: radius of L2-norm ball
def FGSM(x_in, y_true, net, eps):
    if eps == 0:
        return x_in
    training = net.training
    if training:
        net.eval()
    criterion = torch.nn.CrossEntropyLoss().cuda()
    #print(x_in[0,0,0,0:10])
    x_adv = x_in.clone().requires_grad_()
    optimizer = Linf_SGD([x_adv], lr=0.01)
    #inputs = Variable(data.data, requires_grad=True)
    for _ in range(1):
        #optimizer.zero_grad()
        net.zero_grad()
        out, _ = net(x_adv)
        loss = -F.cross_entropy(out, y_true)
        loss.backward()
        optimizer.step()
        diff = torch.ge(x_adv - x_in, 0)
        diff = (diff.float()-0.5)*2
        x_adv.detach().copy_((diff *eps+ x_in).clamp_(0, 1))
    net.zero_grad()
        
        #gradient = torch.ge(x_copy.grad.data, 0)
        #gradient = (gradient.float()-0.5)*2
        #gradient.index_copy_(1, torch.LongTensor([0]).cuda(), \
        #                             gradient.index_select(1, torch.LongTensor([0]).cuda()) / (63.0/255.0))
        #gradient.index_copy_(1, torch.LongTensor([1]).cuda(), \
        #                             gradient.index_select(1, torch.LongTensor([1]).cuda()) / (62.1/255.0))
        #gradient.index_copy_(1, torch.LongTensor([2]).cuda(), \
        #                             gradient.index_select(1, torch.LongTensor([2]).cuda()) / (66.7/255.0))


    #x_adv = torch.add(x_in.data, eps, gradient)
    #x_adv = torch.clamp(x_adv, 0, 1)
    #net.zero_grad()
    # reset to the original state
    if training:
        net.train()
    return x_adv

# performs Linf-constraint PGD attack with output variance restricted, w/o noise
# @epsilon: radius of Linf-norm ball
def pair_dist(input1):
    dist = 0
    for i in range(input1.shape[0]):
        for j in range(i+1,input1.shape[0]):
            dist += (input1[i]-input1[j])**2
    return dist**(0.5)

def res_var_loss(out, y_true, l):
    dist = 0
    for i in range(out.shape[0]):
        dist += pair_dist(out[i])
    loss = -F.cross_entropy(out, y_true) + l * dist
    return loss

def Linf_Res_PGD(x_in, y_true, net, steps, eps, l):
    if eps == 0:
        return x_in
    training = net.training
    if training:
        net.eval()
    x_adv = x_in.clone().requires_grad_()
    optimizer = Linf_SGD([x_adv], lr=0.007)
    for _ in range(steps):
        optimizer.zero_grad()
        net.zero_grad()
        out, _ = net(x_adv)
        loss = res_var_loss(out, y_true, l)
        loss.backward()
        optimizer.step()
        diff = x_adv - x_in
        diff.clamp_(-eps, eps)
        x_adv.detach().copy_((diff + x_in).clamp_(0, 1))
    net.zero_grad()
    # reset to the original state
    if training:
        net.train()
    return x_adv

#import torch
#import torch.nn as nn
#
#pdist = nn.PairwiseDistance(p=2)
#input1 = torch.randn(100,10)
#output = pdist(input1,input1)
#
#input1 = torch.randn(1,10)
#input2 = torch.randn(1,10)
#output = pdist(input1,input2)
#
#def check_dist(input1):
#    dist = 0
#    count = 0
#    for i in range(input1.shape[0]):
#        for j in range(i+1,input1.shape[0]):
#            temp = (input1[i]-input1[j])**2
#            print('Number {} distantce {}'.format(count, temp))
#            dist += temp
#            count += 1
#    return dist
#
#dist = 0            
#for i in range(10):
#    dist += (input1[0][i]-input2[0][i])**2
#
#
#tensor([[-0.5937, -1.0120],
#        [-0.6408, -0.3050]])
#torch.nn.functional.pdist(input1, p=2)
#tensor([0.7086])

