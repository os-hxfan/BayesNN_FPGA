import torch
import torch.nn as nn
from train.loss.base_classes import _MultiExitAccuracy

# Below class is slightly modified from: https://github.com/hjdw2/Exit-Ensemble-Distillation/blob/main/train.py
# Modifications include an option to return immediately if a single exit model
class ExitEnsembleDistillation(_MultiExitAccuracy):
    def __init__(self, n_exits, acc_tops=(1,), use_EED = True, loss_output = "MSE", use_feature_dist = False, temperature = 3):
        super().__init__(n_exits, acc_tops)
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.use_EED = use_EED
        self.loss_output = loss_output
        self.use_feature_dist = use_feature_dist
        self.temperature = temperature

    def __call__(self, net, X, y):
        _ = net(X)
        output, middle_outputs, \
            final_fea, middle_feas = net.intermediary_output_list
        target = y
        final_output_loss = self.criterion(output, target)
        if net.n_exits == 1:
            return final_output_loss
        loss = 0
        for middle_output in middle_outputs:
            loss += self.criterion(middle_output, target)
        L_C = loss + final_output_loss

        if self.use_EED:
            target_output = ((sum(middle_outputs)+output)/(len(middle_outputs)+1)).detach()
            if self.use_feature_dist:
                target_fea = ((sum(middle_feas)+output)/(len(middle_feas)+1)).detach()
        else:
            target_output = output.detach()
            if self.use_feature_dist:
                target_fea = final_fea.detach()

        if self.loss_output == 'KL':
            temp = target_output / self.temperature
            temp = torch.softmax(temp, dim=1)
            lossxbyn = 0
            for middle_output in middle_outputs:
                lossxbyn += self.kd_loss_function(middle_output, temp) * (self.temperature**2)
            L_O = 0.1 * (lossxbyn)
            if self.use_EED:
                loss4by4 = self.kd_loss_function(output, temp) * (self.temperature**2)
                L_O += 0.1 * loss4by4

        elif self.loss_output == 'MSE':
            MSEloss = nn.MSELoss(reduction='mean').cuda()
            loss_mse_n = 0
            for middle_output in middle_outputs:
                loss_mse_n += MSEloss(middle_output, target_output)
            L_O = loss_mse_n
            if self.use_EED:
                loss_mse_4 = MSEloss(output, target_output)
                L_O += loss_mse_4
        total_loss = L_C + L_O

        if self.use_feature_dist:
            feature_loss_n = 0
            for middle_fea in middle_feas:
                feature_loss_n += self.feature_loss_function(middle_fea, target_fea)
            L_F = feature_loss_n
            if self.use_EED:
                feature_loss_4 = self.feature_loss_function(final_fea, target_fea)
                L_F += feature_loss_4
            total_loss += L_F
        return total_loss

    def kd_loss_function(self, output, target_output):
        """Compute kd loss"""
        """
        para: output: middle ouptput logits.
        para: target_output: final output has divided by temperature and softmax.
        """

        output = output / self.temperature
        output_log_softmax = torch.log_softmax(output, dim=1)
        loss_kd = -torch.mean(torch.sum(output_log_softmax * target_output, dim=1))
        return loss_kd

    def feature_loss_function(self, fea, target_fea):
        loss = (fea - target_fea)**2 * ((fea > 0) | (target_fea > 0)).float()
        return torch.abs(loss).mean()

    def accuracy(self, output, target, topk=(1,)):
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul(100.0 / batch_size))

        return res

    def validate(self, val_loader, model, device):
        model.eval()
        top1_acc = 0
        avg_acc = 0
        for (x,y) in val_loader:
            x = x.to(device)
            y = y.to(device, dtype=torch.long)
            target = y
            with torch.no_grad():
                _ = model(x)
                output, middle_outputs, \
                final_fea, middle_feas = model.intermediary_output_list
                loss = self.criterion(output, target)
                prec1 = self.accuracy(output.data, target, topk=(1,))
                avg_acc_batch = prec1[0].clone().detach()
                for middle_output in middle_outputs:
                    avg_acc_batch += self.accuracy(middle_output.data, target, topk=(1,))[0]
                avg_acc_batch /= (len(middle_outputs)+1)
                top1_acc += prec1[0]
                avg_acc += avg_acc_batch
        top1_acc /= len(val_loader)*100
        avg_acc /= len(val_loader)*100
        model.train()
        return avg_acc, top1_acc 

class MultiExitAccuracy(_MultiExitAccuracy): 
    def __init__(self, n_exits, acc_tops=(1,)):
        super().__init__(n_exits, acc_tops)