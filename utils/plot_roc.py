import os
import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

data_dir = '' #Specify the path to test scores and labels

def plot_roc(dataset, attack):
    methods = ['kd','lid','odds','rebel','bnn']
    scores, labels = [], []
    for i in range(5):
        score_file = os.path.join(data_dir,methods[i],'scores_'+methods[i]+'_'+dataset+'_'+attack+'_pred.npy')
        label_file = os.path.join(data_dir,methods[i],'scores_'+methods[i]+'_'+dataset+'_'+attack+'_test.npy')
        scores.append(np.load(score_file))
        labels.append(np.load(label_file))

    labels = ['KD','LID','ODD','ReBeL','BATector']
    FP,TP,AUC = [],[],[]
    for i in range(5):
        fpr, tpr, _ = metrics.roc_curve(labels[i], scores[i])
        auc = metrics.roc_auc_score(labels[i], scores[i])
        print("AUC of {} is {:.3f}".format(labels[i],auc))
        AUC.append(round(auc,3))
        FP.append(fpr)
        TP.append(tpr)
    
    if dataset == 'imagenet-sub':
        dataset_name = 'Imagenet-sub'
    else:
        dataset_name = dataset.upper()
    for i in range(5):
        plt.plot(FP[i],TP[i],label="{}({:.3f})".format(labels[i],AUC[i]))
        plt.title('ROC Curves against {} on {}'.format(attack,dataset_name))
        plt.legend()
    plt.savefig(os.path.join(data_dir,'pics','roc_'+dataset+'_'+attack))
    return