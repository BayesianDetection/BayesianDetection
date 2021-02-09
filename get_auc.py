import matplotlib
matplotlib.use('Agg')
import argparse
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics

parser = argparse.ArgumentParser(description='PyTorch Getting Probabilities')
parser.add_argument('--model', default='bnn', type=str, help='model name')
parser.add_argument('--data', default='cifar10', type=str, help='dataset name')
parser.add_argument('--stat', default='min', type=str, choices=['min','mean','median'],help='test statistic')
parser.add_argument('--res', default='True', type=str, help='whether comes from restricted attack')
parser.add_argument('--adv_type', default='PGD', type=str, help='dataset name')
parser.add_argument('--net', default='vgg', type=str, help='model name')

opt = parser.parse_args()
print(opt)

opt.outf  = './adv_output/' + opt.net +  '_' + opt.data + '/'


test = np.load(opt.outf + f"dist_{opt.model}_{opt.data}_{opt.stat}_{opt.adv_type}_test.npy")
label = np.load(f"{opt.outf}label_{opt.net}_{opt.data}_{opt.adv_type}_{opt.model}.npy")
adv = np.load(opt.outf + f"dist_{opt.model}_{opt.data}_{opt.stat}_{opt.adv_type}_adv.npy")
adv_label = np.load(f"{opt.outf}adv_label_{opt.net}_{opt.data}_{opt.adv_type}_{opt.model}.npy")


from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import train_test_split

def ROC_cv(test_sta, adv_sta):
    X = np.concatenate((test_sta, adv_sta))
    y = np.concatenate((np.zeros(test_sta.shape[0]), np.ones(adv_sta.shape[0])))
    X_train, X_test, Y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    lr = LogisticRegressionCV(n_jobs=-1).fit(X_train, Y_train)
    y_pred = lr.predict_proba(X_test)[:, 1]
    np.save(f"{opt.outf}{opt.model}_{opt.stat}_{opt.adv_type}_test", y_test)
    np.save(f"{opt.outf}{opt.model}_{opt.stat}_{opt.adv_type}_pred", y_pred)
    return metrics.roc_auc_score(y_test, y_pred)


auroc = ROC_cv(test, adv)

print(f"auroc of {opt.model} for {opt.stat}:", round(auroc, 4))
np.save(f"{opt.outf}auroc_{opt.model}_{opt.stat}_{opt.adv_type}", np.array(auroc))




