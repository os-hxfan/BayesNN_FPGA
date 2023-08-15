import numpy as np
import torch
from torch.nn import functional as F
from train.loss.loss_utils import binary_accuracy,multiclass_accuracies

# Both classes are from: https://github.com/mary-phuong/multiexit-distillation/blob/master/main.py
class _Loss:
    def __call__(self, net, *args):
        raise NotImplementedError

    def metrics(self, net, *args):
        "Returns list. First element is used for comparison (higher = better)"
        raise NotImplementedError

    def trn_metrics(self):
        "Metrics from last call."
        raise NotImplementedError

    metric_names = []

    
class _MultiExitAccuracy(_Loss):
    def __init__(self, n_exits, acc_tops=(1,), _binary_clf=False):
        # Hyperparameters
        self.n_exits = n_exits
        self._binary_clf = _binary_clf
        self._acc_tops = acc_tops
        self._cache = dict()
        # Creating metric for each top accuracy for each classifier
        self.metric_names = [f'acc{i}_avg' for i in acc_tops]
        for i in acc_tops:
            self.metric_names += [f'acc{i}_clf{k}' for k in range(n_exits)]
            self.metric_names += [f'acc{i}_ens{k}' for k in range(1, n_exits)]
        self.metric_names += ['avg_maxprob']

    def __call__(self, net, *args):
        raise NotImplementedError

    def _metrics(self, logits_list, y):
        # Init results arrays
        ensemble = torch.zeros_like(logits_list[0])
        acc_clf = np.zeros((self.n_exits, len(self._acc_tops)))
        acc_ens = np.zeros((self.n_exits, len(self._acc_tops)))
        for i, logits in enumerate(logits_list):
            if self.n_exits == 1 and i != len(logits_list)-1:
                continue
            else:
                i = 0
            if self._binary_clf:
                ensemble = ensemble*i/(i+1) + F.sigmoid(logits)/(i+1)
                acc_clf[i] = binary_accuracy(logits, y)
                acc_ens[i] = binary_accuracy(ensemble, y, apply_sigmoid=False)
            else:
                ensemble += F.softmax(logits, dim=1)
                # print(type(multiclass_accuracies(logits, y, self._acc_tops)))
                # print(type(multiclass_accuracies(logits, y, self._acc_tops)[0]))
                acc_clf[i] = multiclass_accuracies(logits, y, self._acc_tops)
                acc_ens[i] = multiclass_accuracies(ensemble, y, self._acc_tops)
                
        maxprob = F.softmax(logits_list[-1].data, dim=1).max(dim=1)[0].mean()

        out = list(acc_clf.mean(axis=0))
        for i in range(acc_clf.shape[1]):
            out += list(acc_clf[:, i])
            out += list(acc_ens[1:, i])
        return out + [maxprob]

    def metrics(self, net, X, y, *args):
        logits_list = net.train(False)(X)
        return self._metrics(logits_list, y)

    def trn_metrics(self):
        return self._metrics(self._cache['logits_list'], self._cache['y'])